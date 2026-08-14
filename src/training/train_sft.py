"""TRL SFTTrainer entry point for the frozen CodeGuide QLoRA dataset."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from .common import (
    load_config,
    report_targets,
    resolve_path,
    select_attention_backend,
    sha256_file,
)
from .sft_data import (
    AssistantOnlyDataCollator,
    load_id_list,
    split_records,
    stratified_sample_ids,
    tokenize_assistant_only,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--mode", choices=("calibration", "full"), default="full")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def prepare_records(
    config: dict[str, Any], mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    data = config["data"]
    canonical = resolve_path(data["canonical"])
    train_ids = resolve_path(data["train_ids"])
    dev_ids = resolve_path(data["dev_ids"])
    assert canonical and train_ids and dev_ids

    actual_hash = sha256_file(canonical)
    if actual_hash != data["expected_sha256"]:
        raise RuntimeError(f"canonical SHA256 mismatch: {actual_hash}")

    train_records, dev_records = split_records(canonical, train_ids, dev_ids)
    expected = (int(data["expected_train_count"]), int(data["expected_dev_count"]))
    if (len(train_records), len(dev_records)) != expected:
        raise RuntimeError(
            f"frozen split count mismatch: train={len(train_records)}, dev={len(dev_records)}"
        )
    if mode == "full":
        return train_records, dev_records, actual_hash

    calibration_ids_path = resolve_path(data["calibration_ids"])
    assert calibration_ids_path
    calibration_ids = load_id_list(calibration_ids_path)
    train_by_id = {record["id"]: record for record in train_records}
    if not set(calibration_ids) <= set(train_by_id):
        raise RuntimeError("calibration IDs are not a subset of frozen SFT train")
    eval_ids = stratified_sample_ids(
        dev_records,
        min(int(data["calibration_eval_size"]), len(dev_records)),
        int(config["seed"]),
    )
    dev_by_id = {record["id"]: record for record in dev_records}
    return (
        [train_by_id[item] for item in calibration_ids],
        [dev_by_id[item] for item in eval_ids],
        actual_hash,
    )


def validation_summary(config: dict[str, Any], mode: str) -> dict[str, Any]:
    train_records, eval_records, canonical_hash = prepare_records(config, mode)
    return {
        "stage": "sft",
        "mode": mode,
        "canonical_sha256": canonical_hash,
        "train": len(train_records),
        "eval": len(eval_records),
        "framework": "trl.SFTTrainer",
    }


def main() -> None:
    cli = build_parser().parse_args()
    config = load_config(cli.config, "sft")
    train_records, eval_records, canonical_hash = prepare_records(config, cli.mode)
    if cli.validate_only:
        print(
            json.dumps(
                {
                    "stage": "sft",
                    "mode": cli.mode,
                    "canonical_sha256": canonical_hash,
                    "train": len(train_records),
                    "eval": len(eval_records),
                    "framework": "trl.SFTTrainer",
                },
                ensure_ascii=False,
            )
        )
        return

    try:
        import torch
        import transformers
        from accelerate import PartialState
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install the single project requirements.txt before training") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    train_cfg = dict(config["training"])
    model_name = cli.model_name_or_path or model_cfg["name_or_path"]
    tokenizer_name = model_cfg["tokenizer_name_or_path"]
    attention_backend = select_attention_backend(model_cfg["attention_backend"])
    max_length = int(model_cfg["max_length"])

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cli.cache_dir,
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized_train = [
        tokenize_assistant_only(record, tokenizer, max_length) for record in train_records
    ]
    tokenized_eval = [
        tokenize_assistant_only(record, tokenizer, max_length) for record in eval_records
    ]
    supervised_tokens = sum(
        sum(label != -100 for label in item["labels"]) for item in tokenized_train
    )
    total_tokens = sum(len(item["labels"]) for item in tokenized_train)

    compute_dtype = getattr(torch, str(quant_cfg["compute_dtype"]))
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=quant_cfg["quant_type"],
        bnb_4bit_use_double_quant=bool(quant_cfg["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    state = PartialState()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cli.cache_dir,
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        attn_implementation=attention_backend,
        device_map={"": state.local_process_index},
    )
    model.config.use_cache = False

    output_dir = resolve_path(cli.output_dir or train_cfg.pop("output_dir"))
    assert output_dir
    if cli.max_steps is not None:
        train_cfg["max_steps"] = cli.max_steps
    if cli.learning_rate is not None:
        train_cfg["learning_rate"] = cli.learning_rate
    train_cfg["report_to"] = report_targets(train_cfg.get("report_to"))

    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        **train_cfg,
    )
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(tokenized_train),
        eval_dataset=Dataset.from_list(tokenized_eval),
        data_collator=AssistantOnlyDataCollator(tokenizer),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    unexpected = [
        name
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(f"non-LoRA trainable parameters detected: {unexpected[:5]}")

    started = time.time()
    result = trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_dir)
        manifest = {
            "schema_version": "codeguide-training-run-v2",
            "stage": "sft",
            "mode": cli.mode,
            "framework": "trl.SFTTrainer",
            "canonical_sha256": canonical_hash,
            "train_count": len(train_records),
            "eval_count": len(eval_records),
            "attention_backend": attention_backend,
            "supervised_tokens": supervised_tokens,
            "total_tokens": total_tokens,
            "elapsed_seconds": time.time() - started,
            "metrics": result.metrics,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
