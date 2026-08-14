"""TRL GRPOTrainer entry point, warm-started from the frozen SFT adapter."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from src.reward.grpo import build_reward_functions
from src.training.common import (
    load_config,
    load_jsonl,
    report_targets,
    resolve_path,
    select_attention_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--sft-adapter-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _prompt_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record {record.get('id')} has no messages")
    prompt = [message for message in messages if message.get("role") != "assistant"]
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError(f"record {record.get('id')} has no final user prompt")

    metadata = dict(record.get("metadata") or {})
    test_cases = metadata.pop("test_cases", None)
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError(f"record {record.get('id')} has no executable test cases")
    contract = {
        key: metadata.get(key)
        for key in ("io_mode", "fn_name", "starter_code", "difficulty")
    }
    return {
        "problem_id": record.get("id") or record.get("problem_id"),
        "prompt": prompt,
        "test_cases": json.dumps(test_cases, ensure_ascii=False),
        "metadata": json.dumps(contract, ensure_ascii=False),
    }


def prepare_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = config["data"]
    train_path, eval_path = resolve_path(data["train"]), resolve_path(data["eval"])
    assert train_path and eval_path
    train = [_prompt_record(item) for item in load_jsonl(train_path)]
    evaluation = [_prompt_record(item) for item in load_jsonl(eval_path)]
    expected = (int(data["expected_train_count"]), int(data["expected_eval_count"]))
    if (len(train), len(evaluation)) != expected:
        raise RuntimeError(
            f"frozen GRPO count mismatch: train={len(train)}, eval={len(evaluation)}"
        )
    train_ids = {item["problem_id"] for item in train}
    eval_ids = {item["problem_id"] for item in evaluation}
    if train_ids & eval_ids:
        raise RuntimeError("GRPO train/eval problem_id overlap")
    return train, evaluation


def validation_summary(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    manifest_path = resolve_path(data["manifest"])
    assert manifest_path
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    counts = manifest["counts"]
    train_count = int(counts["grpo_train"]["samples"])
    eval_count = int(counts["grpo_dev"]["samples"])
    if train_count != int(data["expected_train_count"]):
        raise RuntimeError("GRPO manifest train count mismatch")
    if eval_count != int(data["expected_eval_count"]):
        raise RuntimeError("GRPO manifest eval count mismatch")
    adapter = resolve_path(config["model"]["sft_adapter_path"])
    return {
        "stage": "grpo",
        "framework": "trl.GRPOTrainer",
        "train": train_count,
        "eval": eval_count,
        "sft_adapter": str(adapter),
        "sft_adapter_exists": bool(adapter and adapter.exists()),
        "reward_functions": ["correctness", "teaching_contract"],
    }


def main() -> None:
    cli = build_parser().parse_args()
    config = load_config(cli.config, "grpo")
    if cli.validate_only:
        print(json.dumps(validation_summary(config), ensure_ascii=False))
        return

    try:
        import torch
        import transformers
        from accelerate import PartialState
        from datasets import Dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install the single project requirements.txt before training") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    train_records, eval_records = prepare_records(config)
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    generation_cfg = config["generation"]
    reward_cfg = config["reward"]
    train_cfg = dict(config["training"])
    model_name = cli.model_name_or_path or model_cfg["name_or_path"]
    adapter_path = resolve_path(cli.sft_adapter_path or model_cfg["sft_adapter_path"])
    if not adapter_path or not adapter_path.exists():
        raise RuntimeError(f"missing SFT adapter for GRPO warm start: {adapter_path}")

    attention_backend = select_attention_backend(model_cfg["attention_backend"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cli.cache_dir,
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = getattr(torch, str(quant_cfg["compute_dtype"]))
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=quant_cfg["quant_type"],
        bnb_4bit_use_double_quant=bool(quant_cfg["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    state = PartialState()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cli.cache_dir,
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        attn_implementation=attention_backend,
        device_map={"": state.local_process_index},
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)

    output_dir = resolve_path(cli.output_dir or train_cfg.pop("output_dir"))
    assert output_dir
    if cli.max_steps is not None:
        train_cfg["max_steps"] = cli.max_steps
    train_cfg["report_to"] = report_targets(train_cfg.get("report_to"))
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_prompt_length=int(generation_cfg["max_prompt_length"]),
        max_completion_length=int(generation_cfg["max_completion_length"]),
        num_generations=int(generation_cfg["num_generations"]),
        temperature=float(generation_cfg["temperature"]),
        top_p=float(generation_cfg["top_p"]),
        reward_weights=list(reward_cfg["weights"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        **train_cfg,
    )
    backend = str(reward_cfg["execution_backend"])
    image = os.environ.get("CODEGUIDE_EXECUTION_IMAGE") or reward_cfg.get("container_image")
    reward_functions = build_reward_functions(
        backend=backend,
        container_image=str(image) if image else None,
        timeout=float(reward_cfg["timeout"]),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=list(reward_functions),
        args=args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(eval_records),
        processing_class=tokenizer,
    )

    started = time.time()
    result = trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(adapter_dir)
        manifest = {
            "schema_version": "codeguide-training-run-v2",
            "stage": "grpo",
            "framework": "trl.GRPOTrainer",
            "warm_start_adapter": str(adapter_path),
            "train_count": len(train_records),
            "eval_count": len(eval_records),
            "reward_functions": ["correctness", "teaching_contract"],
            "reward_weights": list(reward_cfg["weights"]),
            "execution_backend": backend,
            "attention_backend": attention_backend,
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
