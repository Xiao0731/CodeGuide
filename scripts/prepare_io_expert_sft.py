#!/usr/bin/env python3
"""Prepare deterministic standard_input and call_based SFT expert datasets/configs.

This is an upper-bound experiment, not a learned MoE router. Each expert sees only
one IO mode in both train and dev. If specialists cannot improve their own mode,
there is little evidence that task interference is the main bottleneck.
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


def build_config(
    *,
    mode: str,
    canonical_path: Path,
    train_ids_path: Path,
    dev_ids_path: Path,
    calibration_ids_path: Path,
    canonical_hash: str,
    train_count: int,
    dev_count: int,
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
            "purpose": "specialist upper-bound before routed dual-adapter/MoE-like experiments",
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
            "canonical": str(canonical_path.relative_to(ROOT)),
            "train_ids": str(train_ids_path.relative_to(ROOT)),
            "dev_ids": str(dev_ids_path.relative_to(ROOT)),
            "calibration_ids": str(calibration_ids_path.relative_to(ROOT)),
            "calibration_eval_size": min(100, dev_count),
            "expected_train_count": train_count,
            "expected_dev_count": dev_count,
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
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
    parser.add_argument("--train-ids", default="data/splits/sft_train_ids.json")
    parser.add_argument("--dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--output-root", default="data/experts/io_mode")
    parser.add_argument("--config-dir", default="configs/sft")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    canonical_source = resolve_path(args.canonical)
    train_ids_source = resolve_path(args.train_ids)
    dev_ids_source = resolve_path(args.dev_ids)
    output_root = resolve_path(args.output_root)
    config_dir = resolve_path(args.config_dir)

    records = load_canonical_in_order(canonical_source)
    by_id = {record_id(record): record for record in records}
    train_ids = load_id_list(train_ids_source)
    dev_ids = load_id_list(dev_ids_source)
    if set(train_ids) & set(dev_ids):
        raise RuntimeError("source train/dev overlap")
    if set(train_ids) | set(dev_ids) != set(by_id):
        raise RuntimeError("source train/dev do not cover canonical exactly")

    summary: dict[str, Any] = {
        "schema_version": "codeguide-io-expert-preparation-v1",
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "modes": {},
    }

    for mode in MODES:
        mode_root = output_root / mode
        mode_train_ids = [pid for pid in train_ids if io_mode(by_id[pid]) == mode]
        mode_dev_ids = [pid for pid in dev_ids if io_mode(by_id[pid]) == mode]
        selected_set = set(mode_train_ids) | set(mode_dev_ids)
        mode_records = [record for record in records if record_id(record) in selected_set]

        if not mode_train_ids or not mode_dev_ids:
            raise RuntimeError(
                f"empty expert split for {mode}: train={len(mode_train_ids)} dev={len(mode_dev_ids)}"
            )
        if any(io_mode(record) != mode for record in mode_records):
            raise RuntimeError(f"cross-mode record leaked into {mode} expert")

        canonical_path = mode_root / "sft_accepted.jsonl"
        train_ids_path = mode_root / "sft_train_ids.json"
        dev_ids_path = mode_root / "sft_dev_ids.json"
        calibration_ids_path = mode_root / "sft_calibration_ids.json"

        canonical_text = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in mode_records
        )
        write_exact(canonical_path, canonical_text, args.force)
        write_exact(
            train_ids_path,
            json.dumps({"ids": mode_train_ids}, ensure_ascii=False, indent=2) + "\n",
            args.force,
        )
        write_exact(
            dev_ids_path,
            json.dumps({"ids": mode_dev_ids}, ensure_ascii=False, indent=2) + "\n",
            args.force,
        )

        calibration_size = min(500, len(mode_train_ids))
        calibration_ids = stratified_sample_ids(
            [by_id[pid] for pid in mode_train_ids],
            calibration_size,
            args.seed,
        )
        write_exact(
            calibration_ids_path,
            json.dumps({"ids": calibration_ids}, ensure_ascii=False, indent=2) + "\n",
            args.force,
        )

        effective_batch = 16
        total_steps = len(mode_train_ids) // effective_batch
        canonical_hash = sha256_file(canonical_path)
        config = build_config(
            mode=mode,
            canonical_path=canonical_path,
            train_ids_path=train_ids_path,
            dev_ids_path=dev_ids_path,
            calibration_ids_path=calibration_ids_path,
            canonical_hash=canonical_hash,
            train_count=len(mode_train_ids),
            dev_count=len(mode_dev_ids),
            total_steps=total_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        config_path = config_dir / f"qwen25_coder_7b_qlora_8k_expert_{mode}.yaml"
        config_text = yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        )
        write_exact(config_path, config_text, args.force)

        summary["modes"][mode] = {
            "train_count": len(mode_train_ids),
            "dev_count": len(mode_dev_ids),
            "optimizer_steps": total_steps,
            "checkpoint_milestones": config["training"]["checkpoint_milestones"],
            "canonical_sha256": canonical_hash,
            "canonical": str(canonical_path.relative_to(ROOT)),
            "config": str(config_path.relative_to(ROOT)),
            "output_dir": config["training"]["output_dir"],
        }

    manifest_path = output_root / "prepare_manifest.json"
    write_exact(
        manifest_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
