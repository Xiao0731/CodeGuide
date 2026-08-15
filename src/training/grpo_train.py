"""Formal TRL GRPO curriculum, warm-started from a selected SFT adapter."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from src.reward.grpo import RewardWeights, build_composite_reward
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
    difficulty = str(metadata.get("difficulty") or "").lower()
    contract = {
        key: metadata.get(key)
        for key in ("io_mode", "fn_name", "starter_code", "difficulty")
    }
    return {
        "problem_id": record.get("id") or record.get("problem_id"),
        "difficulty": difficulty,
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
    if len(train_ids) != len(train) or len(eval_ids) != len(evaluation):
        raise RuntimeError("duplicate problem_id in frozen GRPO data")
    if train_ids & eval_ids:
        raise RuntimeError("GRPO train/dev problem_id overlap")
    return train, evaluation


def partition_curriculum(
    train_records: list[dict[str, Any]], curriculum: dict[str, Any]
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if curriculum.get("enabled") is not True:
        raise RuntimeError("formal GRPO requires curriculum.enabled=true")
    stages = list(curriculum.get("stages") or [])
    if [stage.get("difficulty") for stage in stages] != ["easy", "medium", "hard"]:
        raise RuntimeError("formal curriculum order must be easy -> medium -> hard")

    grouped: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    assigned = 0
    for stage in stages:
        records = [
            record
            for record in train_records
            if record.get("difficulty") == stage["difficulty"]
        ]
        expected = int(stage["expected_count"])
        if len(records) != expected:
            raise RuntimeError(
                f"curriculum {stage['difficulty']} count mismatch: {len(records)} != {expected}"
            )
        if float(stage["epochs"]) != 1.0:
            raise RuntimeError("each formal curriculum stage must run exactly one epoch")
        grouped.append((stage, records))
        assigned += len(records)
    if assigned != len(train_records):
        unknown = sorted({record.get("difficulty") for record in train_records})
        raise RuntimeError(f"curriculum does not cover all train records: {unknown}")
    return grouped


def _validate_formal_config(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    data = config["data"]
    generation = config["generation"]
    training = config["training"]
    stages = list(config["curriculum"].get("stages") or [])
    expected_stages = [
        ("easy", 3228, 512),
        ("medium", 1735, 768),
        ("hard", 1488, 1024),
    ]
    observed_stages = [
        (
            str(stage.get("difficulty")),
            int(stage.get("expected_count", -1)),
            int(stage.get("max_completion_length", -1)),
        )
        for stage in stages
    ]
    if config["curriculum"].get("enabled") is not True or observed_stages != expected_stages:
        raise RuntimeError("formal easy/medium/hard curriculum configuration changed")
    if any(float(stage.get("epochs", 0)) != 1.0 for stage in stages):
        raise RuntimeError("formal curriculum stage epochs changed")

    exact_values = {
        "num_generations": (generation.get("num_generations"), 4),
        "temperature": (generation.get("temperature"), 0.8),
        "top_p": (generation.get("top_p"), 0.95),
        "learning_rate": (training.get("learning_rate"), 1.0e-5),
        "per_device_train_batch_size": (training.get("per_device_train_batch_size"), 1),
        "per_device_eval_batch_size": (training.get("per_device_eval_batch_size"), 2),
        "gradient_accumulation_steps": (training.get("gradient_accumulation_steps"), 8),
        "beta": (training.get("beta"), 0.05),
        "save_steps": (training.get("save_steps"), 100),
        "loss_type": (training.get("loss_type"), "grpo"),
        "scale_rewards": (training.get("scale_rewards"), False),
        "bf16": (training.get("bf16"), True),
        "gradient_checkpointing": (training.get("gradient_checkpointing"), True),
    }
    changed = [name for name, (actual, expected) in exact_values.items() if actual != expected]
    if changed:
        raise RuntimeError(f"formal GRPO settings changed: {', '.join(changed)}")
    if config["reward"].get("execution_backend") != "subprocess":
        raise RuntimeError("formal online reward backend must be subprocess")
    RewardWeights.from_mapping(config["reward"])

    counts = manifest["counts"]
    if int(counts["grpo_train"]["samples"]) != int(data["expected_train_count"]):
        raise RuntimeError("GRPO manifest train count mismatch")
    if int(counts["grpo_dev"]["samples"]) != int(data["expected_eval_count"]):
        raise RuntimeError("GRPO manifest dev count mismatch")
    if int(counts["taco515_ids"]) != int(data["expected_taco515_count"]):
        raise RuntimeError("TACO-515 manifest count mismatch")
    manifest_difficulty = counts["grpo_train"]["by_difficulty"]
    if any(int(manifest_difficulty[name]) != count for name, count, _ in expected_stages):
        raise RuntimeError("GRPO manifest curriculum distribution mismatch")
    overlap = manifest.get("overlap") or {}
    if any(int(overlap.get(key, -1)) != 0 for key in ("train_dev", "train_taco515", "dev_taco515")):
        raise RuntimeError("train/dev/TACO-515 must be pairwise disjoint")
    if data.get("checkpoint_selection_split") != "grpo_dev":
        raise RuntimeError("only GRPO dev50 may select checkpoints")
    if data.get("final_evaluation_split") != "taco515":
        raise RuntimeError("TACO-515 must remain a final-only evaluation split")


def validation_summary(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_path(config["data"]["manifest"])
    assert manifest_path
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_formal_config(config, manifest)

    adapter_value = str(config["model"].get("sft_adapter_path") or "").strip()
    adapter = resolve_path(adapter_value) if adapter_value else None
    stages = config["curriculum"]["stages"]
    return {
        "stage": "grpo",
        "framework": "trl.GRPOTrainer",
        "trl_version": "0.22.2",
        "train": int(config["data"]["expected_train_count"]),
        "dev": int(config["data"]["expected_eval_count"]),
        "taco515_final_only": int(config["data"]["expected_taco515_count"]),
        "pairwise_overlap": manifest["overlap"],
        "checkpoint_selection": "grpo_dev",
        "sft_adapter": str(adapter) if adapter else None,
        "sft_adapter_configured": bool(adapter),
        "sft_adapter_exists": bool(adapter and adapter.exists()),
        "curriculum": [
            {
                "difficulty": stage["difficulty"],
                "samples": int(stage["expected_count"]),
                "epochs": float(stage["epochs"]),
                "max_completion_length": int(stage["max_completion_length"]),
            }
            for stage in stages
        ],
        "reward_function": "formal_composite_reward",
        "execution_backend": "subprocess",
        "loss_type": "grpo",
        "scale_rewards": False,
    }


def main() -> None:
    cli = build_parser().parse_args()
    config = load_config(cli.config, "grpo")
    if cli.sft_adapter_path:
        config["model"]["sft_adapter_path"] = cli.sft_adapter_path
    summary = validation_summary(config)
    if cli.validate_only:
        print(json.dumps(summary, ensure_ascii=False))
        return

    try:
        import torch
        import transformers
        import trl
        from accelerate import PartialState
        from datasets import Dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install the single project requirements.txt before training") from exc
    if trl.__version__ != "0.22.2":
        raise RuntimeError(f"formal GRPO requires TRL 0.22.2, found {trl.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    train_records, eval_records = prepare_records(config)
    curriculum = partition_curriculum(train_records, config["curriculum"])
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    generation_cfg = config["generation"]
    reward_cfg = config["reward"]
    base_train_cfg = dict(config["training"])
    model_name = cli.model_name_or_path or model_cfg["name_or_path"]
    adapter_value = model_cfg.get("sft_adapter_path")
    adapter_path = resolve_path(adapter_value) if adapter_value else None
    if not adapter_path or not adapter_path.exists():
        raise RuntimeError(
            "missing selected best SFT adapter; pass --sft-adapter-path explicitly"
        )

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
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)

    output_dir = resolve_path(cli.output_dir or base_train_cfg.pop("output_dir"))
    assert output_dir
    base_train_cfg["report_to"] = report_targets(base_train_cfg.get("report_to"))
    reward_function = build_composite_reward(reward_config=reward_cfg)
    eval_dataset = Dataset.from_list(eval_records)
    started = time.time()
    stage_runs: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None

    for stage_number, (stage, records) in enumerate(curriculum, 1):
        stage_name = str(stage["difficulty"])
        stage_output = output_dir / f"stage_{stage_number}_{stage_name}"
        train_cfg = dict(base_train_cfg)
        train_cfg["num_train_epochs"] = float(stage["epochs"])
        if cli.max_steps is not None:
            train_cfg["max_steps"] = cli.max_steps
        args = GRPOConfig(
            output_dir=str(stage_output),
            max_prompt_length=int(generation_cfg["max_prompt_length"]),
            max_completion_length=int(stage["max_completion_length"]),
            num_generations=int(generation_cfg["num_generations"]),
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            seed=int(config["seed"]),
            data_seed=int(config["seed"]),
            **train_cfg,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=args,
            train_dataset=Dataset.from_list(records),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        result = trainer.train(
            resume_from_checkpoint=(
                cli.resume_from_checkpoint if stage_number == 1 else None
            )
        )
        stage_adapter = stage_output / "adapter"
        trainer.save_model(str(stage_adapter))
        eval_metrics = trainer.evaluate()
        eval_reward = float(eval_metrics.get("eval_reward", float("-inf")))
        candidate = {
            "difficulty": stage_name,
            "path": str(stage_adapter),
            "eval_reward": eval_reward,
            "source": "stage_final",
        }
        if best_candidate is None or eval_reward > best_candidate["eval_reward"]:
            best_candidate = candidate
        if trainer.state.best_model_checkpoint and trainer.state.best_metric is not None:
            periodic = {
                "difficulty": stage_name,
                "path": str(trainer.state.best_model_checkpoint),
                "eval_reward": float(trainer.state.best_metric),
                "source": "periodic_dev50",
            }
            if periodic["eval_reward"] > best_candidate["eval_reward"]:
                best_candidate = periodic
        stage_runs.append(
            {
                "difficulty": stage_name,
                "samples": len(records),
                "epochs": float(stage["epochs"]),
                "max_completion_length": int(stage["max_completion_length"]),
                "train_metrics": result.metrics,
                "dev50_metrics": eval_metrics,
                "adapter": str(stage_adapter),
            }
        )
        model = trainer.model

    final_adapter = output_dir / "adapter"
    trainer.save_model(str(final_adapter))
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(final_adapter)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "codeguide-training-run-v2",
            "stage": "grpo",
            "framework": "trl.GRPOTrainer",
            "trl_version": trl.__version__,
            "warm_start_adapter": str(adapter_path),
            "train_count": len(train_records),
            "dev_count": len(eval_records),
            "taco515_role": "final_evaluation_only",
            "curriculum": stage_runs,
            "best_checkpoint": best_candidate,
            "final_adapter": str(final_adapter),
            "reward_function": "formal_composite_reward",
            "reward_formula": {
                "code_reward": "0.05*static_validity + 0.70*pass_rate + 0.25*strict",
                "contract_gate": "0.25 + 0.75*pass_rate",
                "total": "0.60*code_reward + 0.40*gated_contract",
            },
            "execution_backend": "subprocess",
            "loss_type": "grpo",
            "scale_rewards": False,
            "attention_backend": attention_backend,
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
