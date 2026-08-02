"""QLoRA SFT entry point for the frozen CodeGuide dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .sft_data import (
    AssistantOnlyDataCollator,
    load_id_list,
    split_records,
    stratified_sample_ids,
    tokenize_assistant_only,
)

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_attention_backend(requested: str) -> str:
    if requested not in {"auto", "flash_attention_2", "sdpa"}:
        raise ValueError(f"unsupported attention backend: {requested}")
    if requested == "sdpa":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        if requested == "flash_attention_2":
            raise RuntimeError("flash_attention_2 requested but flash-attn is unavailable")
        return "sdpa"


class TokenizedDataset:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {key: item[key] for key in ("input_ids", "attention_mask", "labels")}


class LossOnlyPredictionMixin:
    """Evaluate causal-LM loss without retaining vocabulary-sized logits."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import torch

        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return loss.detach().mean(), None, None


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    try:
        return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodeGuide QLoRA SFT")
    parser.add_argument("--config", default="configs/sft/qwen25_coder_7b_qlora_8k.yaml")
    parser.add_argument("--mode", choices=("calibration", "full"))
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--tokenizer-name-or-path")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)
    assert config_path is not None
    cfg = load_config(config_path)
    mode = args.mode or cfg["mode"]
    model_cfg, data_cfg, qlora_cfg, train_cfg = (
        cfg["model"], cfg["data"], cfg["qlora"], cfg["training"]
    )

    canonical = resolve_path(data_cfg["canonical"])
    train_ids_path = resolve_path(data_cfg["train_ids"])
    dev_ids_path = resolve_path(data_cfg["dev_ids"])
    assert canonical and train_ids_path and dev_ids_path
    actual_hash = sha256(canonical)
    if actual_hash != model_cfg["expected_canonical_sha256"]:
        raise RuntimeError(f"canonical SHA256 mismatch: {actual_hash}")

    train_records, dev_records = split_records(canonical, train_ids_path, dev_ids_path)
    if len(train_records) != data_cfg["expected_train_count"] or len(dev_records) != data_cfg["expected_dev_count"]:
        raise RuntimeError(f"frozen split count mismatch: train={len(train_records)}, dev={len(dev_records)}")

    calibration_ids_path = resolve_path(data_cfg["calibration_ids"])
    assert calibration_ids_path
    calibration_ids = load_id_list(calibration_ids_path)
    train_by_id = {record["id"]: record for record in train_records}
    if not set(calibration_ids) <= set(train_by_id):
        raise RuntimeError("calibration IDs are not a subset of frozen SFT train")
    selected_train = [train_by_id[item] for item in calibration_ids] if mode == "calibration" else train_records
    eval_ids = stratified_sample_ids(dev_records, min(data_cfg["calibration_eval_size"], len(dev_records)), cfg["seed"])
    dev_by_id = {record["id"]: record for record in dev_records}
    selected_eval = [dev_by_id[item] for item in eval_ids] if mode == "calibration" else dev_records

    if args.validate_only:
        print(json.dumps({"canonical_hash": actual_hash, "train": len(selected_train), "eval": len(selected_eval), "mode": mode}))
        return

    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("missing cloud training dependencies; install requirements-sft.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    if not qlora_cfg["load_in_4bit"]:
        raise RuntimeError("this pipeline requires 4-bit QLoRA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    cache_dir = args.cache_dir or model_cfg.get("cache_dir")
    model_name = args.model_name_or_path or model_cfg["model_name_or_path"]
    tokenizer_name = args.tokenizer_name_or_path or model_cfg["tokenizer_name_or_path"]
    attention_backend = select_attention_backend(model_cfg["attention_backend"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, trust_remote_code=model_cfg["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_length = model_cfg["max_seq_length"]
    tokenized_train = [tokenize_assistant_only(record, tokenizer, max_length) for record in selected_train]
    tokenized_eval = [tokenize_assistant_only(record, tokenizer, max_length) for record in selected_eval]
    supervised = sum(sum(label != -100 for label in item["labels"]) for item in tokenized_train)
    total = sum(len(item["labels"]) for item in tokenized_train)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora_cfg["quant_type"],
        bnb_4bit_use_double_quant=qlora_cfg["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=model_cfg["trust_remote_code"],
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": train_cfg["gradient_checkpointing_use_reentrant"]},
    )
    model = get_peft_model(model, LoraConfig(
        r=qlora_cfg["r"], lora_alpha=qlora_cfg["alpha"], lora_dropout=qlora_cfg["dropout"],
        bias=qlora_cfg["bias"], task_type="CAUSAL_LM", target_modules=qlora_cfg["target_modules"],
    ))
    non_lora_trainable = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name]
    if non_lora_trainable:
        raise RuntimeError(f"non-LoRA trainable parameters detected: {non_lora_trainable[:5]}")

    output_dir = resolve_path(args.output_dir or train_cfg["output_dir"])
    assert output_dir
    training_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=train_cfg["num_train_epochs"], max_steps=args.max_steps or -1,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"], per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"], learning_rate=train_cfg["learning_rate"],
        optim=train_cfg["optimizer"], lr_scheduler_type=train_cfg["lr_scheduler_type"], warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"], max_grad_norm=train_cfg["max_grad_norm"], bf16=train_cfg["bf16"], fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": train_cfg["gradient_checkpointing_use_reentrant"]},
        use_liger_kernel=train_cfg.get("use_liger_kernel", False),
        liger_kernel_config=train_cfg.get("liger_kernel_config"),
        logging_steps=train_cfg["logging_steps"], eval_strategy="steps", eval_steps=train_cfg["eval_steps"],
        save_strategy="steps", save_steps=train_cfg["save_steps"], save_total_limit=train_cfg["save_total_limit"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"], group_by_length=data_cfg["group_by_length"],
        ddp_find_unused_parameters=train_cfg["ddp_find_unused_parameters"], report_to=[] if train_cfg["report_to"] == "none" else [train_cfg["report_to"]],
        seed=cfg["seed"], data_seed=cfg["data_seed"], remove_unused_columns=False,
        label_names=["labels"], prediction_loss_only=True,
    )
    class LossOnlyTrainer(LossOnlyPredictionMixin, Trainer):
        pass

    trainer = LossOnlyTrainer(
        model=model, args=training_args, train_dataset=TokenizedDataset(tokenized_train),
        eval_dataset=TokenizedDataset(tokenized_eval), data_collator=AssistantOnlyDataCollator(tokenizer),
    )
    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir / "adapter")
        manifest = {
            "schema_version": "codeguide-sft-run-v1", "mode": mode, "config": cfg,
            "canonical_sha256": actual_hash, "train_count": len(selected_train), "eval_count": len(selected_eval),
            "attention_backend": attention_backend, "supervised_tokens": supervised, "total_unpadded_tokens": total,
            "supervised_token_ratio": supervised / total, "elapsed_seconds": time.time() - started,
            "use_liger_kernel": train_cfg.get("use_liger_kernel", False),
            "liger_kernel_config": train_cfg.get("liger_kernel_config"),
            "train_metrics": result.metrics, "git": _git_state(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__, "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
