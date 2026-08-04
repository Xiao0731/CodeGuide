#!/usr/bin/env python3
"""Prepare deterministic, mode-pure SFT expert datasets and online probes.

For each IO mode, this script:
1. starts from the frozen global SFT train split;
2. reserves a mode-pure execution probe from that train split;
3. removes probe IDs from expert training;
4. writes a pure-mode train/dev canonical for loss monitoring;
5. preserves the original 515 dev subset as confirmation-only IDs;
6. emits one SFT config and one execution-eval protocol per expert.

The online probe is used only for checkpoint selection. The original 515 dev set
is not used to choose expert checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.sft_data import load_id_list, stratified_sample_ids

MODES = ("standard_input", "call_based")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("problem_id")
    if not value:
        raise ValueError("record has no id/problem_id")
    return str(value)


def io_mode(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("io_mode") or record.get("io_mode") or "unknown")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exact(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(
                f"frozen expert artifact differs: {path}; use --force only intentionally"
            )
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any, force: bool) -> None:
    write_exact(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        force,
    )


def load_canonical_in_order(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record_id(record)
            if problem_id in seen:
                raise ValueError(f"duplicate problem id at line {line_no}: {problem_id}")
            seen.add(problem_id)
            records.append(record)
    return records


def milestone_steps(total_steps: int) -> list[int]:
    if total_steps <= 0:
        raise ValueError("expert has no optimizer steps")
    candidates = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500]
    return sorted({step for step in candidates if step < total_steps} | {total_steps})


def compact_protocol(
    *,
    mode: str,
    online_dev_ids_path: Path,
    canonical_source: Path,
    source_bank: Path,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "codeguide-eval-protocol-v1",
        "protocol_name": f"compact-code-first-expert-online-{mode}-v1",
        "model": {
            "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "dtype": "bfloat16",
        },
        "dataset": {
            "canonical": repo_relative(canonical_source),
            "dev_ids": repo_relative(online_dev_ids_path),
            "source_bank": repo_relative(source_bank),
            "use_all_dev": True,
            "seed": seed,
        },
        "generation": {
            "max_new_tokens": 2048,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "num_beams": 1,
            "batch_size": batch_size,
            "length_bucket": True,
        },
        "verification": {"workers": 4, "timeout_seconds": 5.0},
        "required_sections": [
            "```python",
            "题意",
            "关键观察",
            "步骤",
            "复杂度",
            "常见错误",
        ],
        "system_prompt": (
            "你是 CodeGuide，一位算法教学助手。本次任务首先考察最终代码的正确性，\n"
            "其次考察能否提供紧凑、清晰的教学说明。\n\n"
            "请严格遵守以下输出协议：\n"
            "1. 回答开头必须直接给出唯一一个完整的 Python 代码块。\n"
            "2. 代码必须符合题目指定的 standard_input 或 call_based 接口，可以直接提交运行。\n"
            "3. 代码块之后依次输出：题意、关键观察、步骤、复杂度、常见错误。\n"
            "禁止展开暴力解法，禁止输出伪代码，禁止示例调用，禁止输出第二个代码块。"
        ),
        "user_suffix": (
            "\n\n【本次评测再次提醒】\n"
            "请先输出唯一的完整 Python 代码块，再给紧凑说明。"
        ),
    }


def build_config(
    *,
    mode: str,
    canonical_path: Path,
    train_ids_path: Path,
    online_dev_ids_path: Path,
    calibration_ids_path: Path,
    canonical_hash: str,
    train_count: int,
    online_dev_count: int,
    total_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    milestones = milestone_steps(total_steps)
    return {
        "schema_version": "codeguide-sft-training-v1",
        "mode": "full",
        "seed": seed,
        "data_seed": seed,
        "expert": {
            "axis": "io_mode",
            "value": mode,
            "checkpoint_selection": "mode-pure execution probe",
        },
        "model": {
            "model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "tokenizer_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "cache_dir": None,
            "trust_remote_code": False,
            "max_seq_length": 8192,
            "expected_canonical_sha256": canonical_hash,
            "attention_backend": "auto",
        },
        "data": {
            "canonical": repo_relative(canonical_path),
            "train_ids": repo_relative(train_ids_path),
            "dev_ids": repo_relative(online_dev_ids_path),
            "calibration_ids": repo_relative(calibration_ids_path),
            "calibration_eval_size": online_dev_count,
            "expected_train_count": train_count,
            "expected_dev_count": online_dev_count,
            "packing": False,
            "group_by_length": True,
        },
        "qlora": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "training": {
            "output_dir": (
                "outputs/sft/qwen25_coder_7b_qlora_8k/"
                f"expert_{mode}_lr1e4_seed{seed}"
            ),
            "expected_world_size": 1,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": learning_rate,
            "optimizer": "paged_adamw_8bit",
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "use_liger_kernel": True,
            "liger_kernel_config": {"fused_linear_cross_entropy": True},
            "logging_steps": 1,
            "eval_steps": 25,
            "save_steps": 1000000,
            "save_total_limit": len(milestones) + 2,
            "checkpoint_milestones": milestones,
            "dataloader_num_workers": 2,
            "ddp_find_unused_parameters": False,
            "report_to": "none",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--source-bank", default="data/final/taco_verified_source_bank.jsonl.zst")
    parser.add_argument("--train-ids", default="data/splits/sft_train_ids.json")
    parser.add_argument("--dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--output-root", default="data/experts/io_mode")
    parser.add_argument("--config-dir", default="configs/sft")
    parser.add_argument("--eval-config-dir", default="configs/eval")
    parser.add_argument("--online-dev-per-mode", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.online_dev_per_mode <= 0:
        raise ValueError("--online-dev-per-mode must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")

    canonical_source = resolve_path(args.canonical)
    source_bank = resolve_path(args.source_bank)
    train_ids_source = resolve_path(args.train_ids)
    dev_ids_source = resolve_path(args.dev_ids)
    output_root = resolve_path(args.output_root)
    config_dir = resolve_path(args.config_dir)
    eval_config_dir = resolve_path(args.eval_config_dir)

    records = load_canonical_in_order(canonical_source)
    by_id = {record_id(record): record for record in records}
    train_ids = load_id_list(train_ids_source)
    dev_ids = load_id_list(dev_ids_source)
    if set(train_ids) & set(dev_ids):
        raise RuntimeError("source train/dev overlap")
    if set(train_ids) | set(dev_ids) != set(by_id):
        raise RuntimeError("source train/dev do not cover canonical exactly")

    summary: dict[str, Any] = {
        "schema_version": "codeguide-io-expert-preparation-v2",
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "online_dev_per_mode": args.online_dev_per_mode,
        "selection_rule": (
            "maximize strict Pass@1 on the expert's own mode; tie-break by mean test "
            "pass rate, fewer runtime failures, shorter output, then earlier step"
        ),
        "modes": {},
    }

    for mode in MODES:
        mode_root = output_root / mode
        source_mode_train_ids = [pid for pid in train_ids if io_mode(by_id[pid]) == mode]
        final_eval_ids = [pid for pid in dev_ids if io_mode(by_id[pid]) == mode]
        if len(source_mode_train_ids) <= args.online_dev_per_mode:
            raise RuntimeError(
                f"not enough {mode} train IDs for a held-out online probe: "
                f"available={len(source_mode_train_ids)} requested={args.online_dev_per_mode}"
            )
        if not final_eval_ids:
            raise RuntimeError(f"empty final 515 subset for {mode}")

        online_dev_ids = stratified_sample_ids(
            [by_id[pid] for pid in source_mode_train_ids],
            args.online_dev_per_mode,
            args.seed,
        )
        online_dev_set = set(online_dev_ids)
        expert_train_ids = [pid for pid in source_mode_train_ids if pid not in online_dev_set]
        if set(expert_train_ids) & online_dev_set:
            raise RuntimeError(f"online probe leaked into {mode} expert train")

        expert_canonical_ids = set(expert_train_ids) | online_dev_set
        expert_records = [
            record for record in records if record_id(record) in expert_canonical_ids
        ]
        if any(io_mode(record) != mode for record in expert_records):
            raise RuntimeError(f"cross-mode record leaked into {mode} expert")

        canonical_path = mode_root / "sft_accepted.jsonl"
        train_ids_path = mode_root / "sft_train_ids.json"
        online_dev_ids_path = mode_root / "online_dev_ids.json"
        final_eval_ids_path = mode_root / "taco515_confirmation_ids.json"
        calibration_ids_path = mode_root / "sft_calibration_ids.json"

        canonical_text = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in expert_records
        )
        write_exact(canonical_path, canonical_text, args.force)
        write_json(train_ids_path, {"ids": expert_train_ids}, args.force)
        write_json(online_dev_ids_path, {"ids": online_dev_ids}, args.force)
        write_json(final_eval_ids_path, {"ids": final_eval_ids}, args.force)

        calibration_size = min(500, len(expert_train_ids))
        calibration_ids = stratified_sample_ids(
            [by_id[pid] for pid in expert_train_ids],
            calibration_size,
            args.seed,
        )
        write_json(calibration_ids_path, {"ids": calibration_ids}, args.force)

        effective_batch = 16
        total_steps = math.ceil(len(expert_train_ids) / effective_batch)
        canonical_hash = sha256_file(canonical_path)
        config = build_config(
            mode=mode,
            canonical_path=canonical_path,
            train_ids_path=train_ids_path,
            online_dev_ids_path=online_dev_ids_path,
            calibration_ids_path=calibration_ids_path,
            canonical_hash=canonical_hash,
            train_count=len(expert_train_ids),
            online_dev_count=len(online_dev_ids),
            total_steps=total_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        config_path = config_dir / f"qwen25_coder_7b_qlora_8k_expert_{mode}.yaml"
        write_exact(
            config_path,
            yaml.safe_dump(
                config,
                allow_unicode=True,
                sort_keys=False,
                width=1000,
            ),
            args.force,
        )

        eval_protocol = compact_protocol(
            mode=mode,
            online_dev_ids_path=online_dev_ids_path,
            canonical_source=canonical_source,
            source_bank=source_bank,
            batch_size=args.eval_batch_size,
            seed=args.seed,
        )
        eval_config_path = eval_config_dir / f"expert_online_{mode}_v1.yaml"
        write_exact(
            eval_config_path,
            yaml.safe_dump(
                eval_protocol,
                allow_unicode=True,
                sort_keys=False,
                width=1000,
            ),
            args.force,
        )

        summary["modes"][mode] = {
            "source_train_count": len(source_mode_train_ids),
            "expert_train_count": len(expert_train_ids),
            "online_dev_count": len(online_dev_ids),
            "taco515_confirmation_count": len(final_eval_ids),
            "optimizer_steps": total_steps,
            "checkpoint_milestones": config["training"]["checkpoint_milestones"],
            "canonical_sha256": canonical_hash,
            "canonical": repo_relative(canonical_path),
            "train_ids": repo_relative(train_ids_path),
            "online_dev_ids": repo_relative(online_dev_ids_path),
            "taco515_confirmation_ids": repo_relative(final_eval_ids_path),
            "config": repo_relative(config_path),
            "eval_protocol": repo_relative(eval_config_path),
            "output_dir": config["training"]["output_dir"],
        }

    write_json(output_root / "prepare_manifest.json", summary, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
