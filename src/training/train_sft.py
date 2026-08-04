"""QLoRA SFT entry point for the frozen CodeGuide dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

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
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid config: {path}")
    return payload


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
        inputs = dict(self._prepare_inputs(inputs))
        if getattr(self.args, "use_liger_kernel", False):
            inputs["skip_logits"] = True
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return loss.detach().mean(), None, None


def parse_checkpoint_milestones(value: Any) -> list[int]:
    """Normalize an optional list/CSV of optimizer-step checkpoints."""
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        raw_items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise ValueError("checkpoint_milestones must be a list or comma-separated string")

    milestones = sorted({int(item) for item in raw_items if str(item).strip()})
    if any(step <= 0 for step in milestones):
        raise ValueError("checkpoint milestones must be positive optimizer steps")
    return milestones


def make_milestone_save_callback(TrainerCallback, milestones: list[int]):
    """Request full Trainer checkpoints only at the frozen milestone steps."""
    frozen = frozenset(milestones)

    class MilestoneSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step in frozen:
                control.should_save = True
                print(
                    f"[milestone-checkpoint] request save at optimizer step "
                    f"{state.global_step}",
                    flush=True,
                )
            return control

    return MilestoneSaveCallback()


def make_stop_after_step_callback(TrainerCallback, stop_after_step: int):
    """Pause at one milestone without redefining the planned LR schedule."""

    class StopAfterStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step >= stop_after_step:
                control.should_training_stop = True
                print(
                    f"[phase-stop] pause training at optimizer step {state.global_step}",
                    flush=True,
                )
            return control

    return StopAfterStepCallback()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
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
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="pause at this optimizer step while keeping the full LR schedule",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--checkpoint-milestones",
        help="comma-separated optimizer steps; overrides training.checkpoint_milestones",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        raise ValueError("--stop-after-step must be positive")

    config_path = resolve_path(args.config)
    assert config_path is not None
    cfg = load_config(config_path)
    mode = args.mode or cfg["mode"]
    model_cfg, data_cfg, qlora_cfg = cfg["model"], cfg["data"], cfg["qlora"]
    train_cfg = dict(cfg["training"])

    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        if args.gradient_accumulation_steps <= 0:
            raise ValueError("gradient accumulation must be positive")
        train_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.checkpoint_milestones is not None:
        train_cfg["checkpoint_milestones"] = parse_checkpoint_milestones(
            args.checkpoint_milestones
        )
    cfg["training"] = train_cfg

    milestones = parse_checkpoint_milestones(train_cfg.get("checkpoint_milestones"))
    save_total_limit = int(train_cfg["save_total_limit"])
    if milestones and save_total_limit < len(milestones):
        raise RuntimeError(
            "save_total_limit would delete frozen trajectory checkpoints: "
            f"limit={save_total_limit}, milestones={len(milestones)}"
        )
    if args.stop_after_step is not None and args.stop_after_step not in milestones:
        raise RuntimeError(
            "stop-after-step must be one of checkpoint_milestones so the phase is resumable: "
            f"stop={args.stop_after_step}, milestones={milestones}"
        )

    canonical = resolve_path(data_cfg["canonical"])
    train_ids_path = resolve_path(data_cfg["train_ids"])
    dev_ids_path = resolve_path(data_cfg["dev_ids"])
    assert canonical and train_ids_path and dev_ids_path
    actual_hash = sha256(canonical)
    if actual_hash != model_cfg["expected_canonical_sha256"]:
        raise RuntimeError(f"canonical SHA256 mismatch: {actual_hash}")

    train_records, dev_records = split_records(canonical, train_ids_path, dev_ids_path)
    if (
        len(train_records) != data_cfg["expected_train_count"]
        or len(dev_records) != data_cfg["expected_dev_count"]
    ):
        raise RuntimeError(
            f"frozen split count mismatch: train={len(train_records)}, "
            f"dev={len(dev_records)}"
        )

    calibration_ids_path = resolve_path(data_cfg["calibration_ids"])
    assert calibration_ids_path
    calibration_ids = load_id_list(calibration_ids_path)
    train_by_id = {record["id"]: record for record in train_records}
    if not set(calibration_ids) <= set(train_by_id):
        raise RuntimeError("calibration IDs are not a subset of frozen SFT train")
    selected_train = (
        [train_by_id[item] for item in calibration_ids]
        if mode == "calibration"
        else train_records
    )
    eval_ids = stratified_sample_ids(
        dev_records,
        min(data_cfg["calibration_eval_size"], len(dev_records)),
        cfg["seed"],
    )
    dev_by_id = {record["id"]: record for record in dev_records}
    selected_eval = (
        [dev_by_id[item] for item in eval_ids] if mode == "calibration" else dev_records
    )

    if args.validate_only:
        print(
            json.dumps(
                {
                    "canonical_hash": actual_hash,
                    "train": len(selected_train),
                    "eval": len(selected_eval),
                    "mode": mode,
                    "learning_rate": float(train_cfg["learning_rate"]),
                    "gradient_accumulation_steps": int(
                        train_cfg["gradient_accumulation_steps"]
                    ),
                    "checkpoint_milestones": milestones,
                    "stop_after_step": args.stop_after_step,
                    "expected_world_size": train_cfg.get("expected_world_size"),
                },
                ensure_ascii=False,
            )
        )
        return

    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "missing cloud training dependencies; install requirements-sft.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    if not qlora_cfg["load_in_4bit"]:
        raise RuntimeError("this pipeline requires 4-bit QLoRA")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world_size = train_cfg.get("expected_world_size")
    if expected_world_size is not None and world_size != int(expected_world_size):
        raise RuntimeError(
            f"world-size mismatch: expected={expected_world_size}, actual={world_size}. "
            "The LR trajectory configs are single-GPU runs with matched global batch."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    cache_dir = args.cache_dir or model_cfg.get("cache_dir")
    model_name = args.model_name_or_path or model_cfg["model_name_or_path"]
    tokenizer_name = args.tokenizer_name_or_path or model_cfg["tokenizer_name_or_path"]
    attention_backend = select_attention_backend(model_cfg["attention_backend"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_length = model_cfg["max_seq_length"]
    tokenized_train = [
        tokenize_assistant_only(record, tokenizer, max_length)
        for record in selected_train
    ]
    tokenized_eval = [
        tokenize_assistant_only(record, tokenizer, max_length)
        for record in selected_eval
    ]
    supervised = sum(
        sum(label != -100 for label in item["labels"]) for item in tokenized_train
    )
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
        gradient_checkpointing_kwargs={
            "use_reentrant": train_cfg["gradient_checkpointing_use_reentrant"]
        },
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=qlora_cfg["r"],
            lora_alpha=qlora_cfg["alpha"],
            lora_dropout=qlora_cfg["dropout"],
            bias=qlora_cfg["bias"],
            task_type="CAUSAL_LM",
            target_modules=qlora_cfg["target_modules"],
        ),
    )
    non_lora_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]
    if non_lora_trainable:
        raise RuntimeError(
            f"non-LoRA trainable parameters detected: {non_lora_trainable[:5]}"
        )

    output_dir = resolve_path(args.output_dir or train_cfg["output_dir"])
    assert output_dir
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        optim=train_cfg["optimizer"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={
            "use_reentrant": train_cfg["gradient_checkpointing_use_reentrant"]
        },
        use_liger_kernel=train_cfg.get("use_liger_kernel", False),
        liger_kernel_config=train_cfg.get("liger_kernel_config"),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=save_total_limit,
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        group_by_length=data_cfg["group_by_length"],
        ddp_find_unused_parameters=train_cfg["ddp_find_unused_parameters"],
        report_to=[]
        if train_cfg["report_to"] == "none"
        else [train_cfg["report_to"]],
        seed=cfg["seed"],
        data_seed=cfg["data_seed"],
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
    )

    class LossOnlyTrainer(LossOnlyPredictionMixin, Trainer):
        pass

    trainer = LossOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(tokenized_train),
        eval_dataset=TokenizedDataset(tokenized_eval),
        data_collator=AssistantOnlyDataCollator(tokenizer),
    )
    if milestones:
        trainer.add_callback(make_milestone_save_callback(TrainerCallback, milestones))
    if args.stop_after_step is not None:
        trainer.add_callback(
            make_stop_after_step_callback(TrainerCallback, args.stop_after_step)
        )

    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir / "adapter")
        manifest = {
            "schema_version": "codeguide-sft-run-v1",
            "mode": mode,
            "config": cfg,
            "canonical_sha256": actual_hash,
            "train_count": len(selected_train),
            "eval_count": len(selected_eval),
            "attention_backend": attention_backend,
            "supervised_tokens": supervised,
            "total_unpadded_tokens": total,
            "supervised_token_ratio": supervised / total,
            "elapsed_seconds": time.time() - started,
            "checkpoint_milestones": milestones,
            "stop_after_step": args.stop_after_step,
            "use_liger_kernel": train_cfg.get("use_liger_kernel", False),
            "liger_kernel_config": train_cfg.get("liger_kernel_config"),
            "train_metrics": result.metrics,
            "git": _git_state(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "world_size": world_size,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
