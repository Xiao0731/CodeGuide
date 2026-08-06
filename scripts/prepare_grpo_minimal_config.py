#!/usr/bin/env python3
"""从基础配置生成正式 GRPO 与精确 20-step smoke 配置。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是对象")
            records.append(item)
    return records


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("problem_id") or record.get("task_id")
    if not value:
        raise ValueError("GRPO 记录缺少 ID")
    return str(value)


def stable_rank(record: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{record_id(record)}".encode("utf-8")).hexdigest()


def select_smoke(
    records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """按正式约 3:1 IO 比例选 smoke，确保两种执行链路都会被覆盖。"""
    if samples <= 0:
        raise ValueError("smoke_samples 必须为正整数")
    standard_target = round(samples * 0.75)
    call_target = samples - standard_target

    pools = {
        mode: sorted(
            [
                item
                for item in records
                if (item.get("metadata") or {}).get("io_mode") == mode
            ],
            key=lambda item: stable_rank(item, seed),
        )
        for mode in ("standard_input", "call_based")
    }
    if len(pools["standard_input"]) < standard_target:
        raise RuntimeError(
            f"smoke standard_input 不足：需要 {standard_target}，"
            f"实际 {len(pools['standard_input'])}"
        )
    if len(pools["call_based"]) < call_target:
        raise RuntimeError(
            f"smoke call_based 不足：需要 {call_target}，"
            f"实际 {len(pools['call_based'])}"
        )

    selected = (
        pools["standard_input"][:standard_target]
        + pools["call_based"][:call_target]
    )
    return sorted(selected, key=lambda item: stable_rank(item, seed + 1))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def optimizer_steps(samples: int, batch: int, accumulation: int, epochs: int = 1) -> int:
    return math.ceil(samples / (batch * accumulation)) * epochs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/train_config.yaml")
    parser.add_argument("--output", default="configs/grpo_minimal_v1.yaml")
    parser.add_argument("--smoke-output", default="configs/grpo_smoke_v1.yaml")
    parser.add_argument(
        "--sft-adapter",
        default=(
            "outputs/sft/qwen25_coder_7b_qlora_8k/"
            "full_lr1e4_seed20260728/checkpoint-200"
        ),
    )
    parser.add_argument(
        "--train-data",
        default="data/splits/grpo_minimal_v1/grpo_train.jsonl",
    )
    parser.add_argument(
        "--eval-data",
        default="data/splits/grpo_minimal_v1/grpo_dev.jsonl",
    )
    parser.add_argument(
        "--smoke-train-data",
        default="data/splits/grpo_minimal_v1/grpo_smoke_train.jsonl",
    )
    parser.add_argument("--smoke-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    base_path = resolve(args.base_config)
    output_path = resolve(args.output)
    smoke_output_path = resolve(args.smoke_output)
    train_path = resolve(args.train_data)
    eval_path = resolve(args.eval_data)
    smoke_train_path = resolve(args.smoke_train_data)

    for required in (base_path, train_path, eval_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少必要文件：{required}")

    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"配置格式错误：{base_path}")

    train_records = load_jsonl(train_path)
    eval_records = load_jsonl(eval_path)
    if len(eval_records) != 50:
        raise ValueError(f"GRPO dev 应为 50 条，实际 {len(eval_records)}")

    cfg.setdefault("model", {})["max_seq_length"] = 8192
    grpo = cfg.setdefault("grpo", {})
    grpo.update(
        {
            "run_name": "grpo-codeguide-minimal-v1",
            "sft_adapter_path": args.sft_adapter,
            "max_seq_length": 8192,
            "train_data": args.train_data,
            "eval_data": args.eval_data,
            "output_dir": "outputs/grpo/codeguide_minimal_v1",
            "best_model_dir": "outputs/grpo/codeguide_minimal_v1_best",
            "merged_dir": "outputs/grpo/codeguide_minimal_v1_merged",
            "logging_dir": "outputs/grpo/codeguide_minimal_v1_logs",
            "num_generations": 4,
            "max_new_tokens": 1024,
            "max_online_tests": 16,
            "temperature": 0.8,
            "top_p": 0.95,
            "learning_rate": 1.0e-5,
            "num_train_epochs": 1,
            "max_steps": -1,
            "logging_steps": 10,
            "save_steps": 100,
            "normalize_rewards": False,
            "save_best": True,
            "eval_steps": 100,
            "checkpoint_eval_max_samples": 50,
            "merge_model": False,
            "seed": args.seed,
        }
    )

    reward = cfg.setdefault("reward", {})
    reward.update(
        {
            "alpha": 0.60,
            "execution_backend": "subprocess",
            "container_image": "",
            "exec_timeout": 5.0,
            "code_weight": 0.60,
            "contract_weight": 0.40,
            "static_validity_weight": 0.05,
            "partial_pass_weight": 0.70,
            "strict_pass_weight": 0.25,
            "contract_gate_floor": 0.25,
            "teaching_completeness": 0.0,
            "code_correctness": 0.60,
            "format_compliance": 0.40,
            "teaching_mode": "local",
        }
    )

    curriculum = cfg.setdefault("curriculum", {})
    curriculum["enabled"] = True
    write_yaml(output_path, cfg)

    smoke_records = select_smoke(
        train_records,
        samples=args.smoke_samples,
        seed=args.seed,
    )
    write_jsonl(smoke_train_path, smoke_records)

    smoke_cfg = copy.deepcopy(cfg)
    smoke_grpo = smoke_cfg["grpo"]
    smoke_grpo.update(
        {
            "run_name": "grpo-codeguide-smoke-v1",
            "train_data": args.smoke_train_data,
            "output_dir": "outputs/grpo/codeguide_smoke_v1",
            "best_model_dir": "outputs/grpo/codeguide_smoke_v1_best",
            "merged_dir": "outputs/grpo/codeguide_smoke_v1_merged",
            "logging_dir": "outputs/grpo/codeguide_smoke_v1_logs",
            "max_new_tokens": 512,
            "max_online_tests": 4,
            "logging_steps": 1,
            "save_steps": 20,
            "save_best": False,
            "num_train_epochs": 1,
            "max_steps": 20,
            "merge_model": False,
        }
    )
    smoke_cfg.setdefault("curriculum", {})["enabled"] = False
    write_yaml(smoke_output_path, smoke_cfg)

    batch = int(grpo["per_device_train_batch_size"])
    accumulation = int(grpo["gradient_accumulation_steps"])
    by_difficulty = Counter(
        str((item.get("metadata") or {}).get("difficulty") or "unknown")
        for item in train_records
    )
    formal_stage_steps = {
        difficulty: optimizer_steps(count, batch, accumulation)
        for difficulty, count in sorted(by_difficulty.items())
    }
    total_steps = sum(formal_stage_steps.values())
    rollouts = (
        total_steps
        * batch
        * accumulation
        * int(grpo["num_generations"])
    )
    smoke_modes = Counter(
        str((item.get("metadata") or {}).get("io_mode") or "unknown")
        for item in smoke_records
    )

    print(f"[完成] 正式 GRPO 配置：{output_path}")
    print(f"[完成] Smoke GRPO 配置：{smoke_output_path}")
    print(f"[完成] Smoke 数据：{smoke_train_path}")
    print(f"[正式训练] samples={len(train_records)}，stage_steps={formal_stage_steps}")
    print(f"[正式训练] total_optimizer_steps≈{total_steps}，rollouts≈{rollouts}")
    print(f"[开发集] samples={len(eval_records)}，每 100 step 做 Pass@1 选优")
    print(f"[Smoke] samples={len(smoke_records)}，io_mode={dict(smoke_modes)}，max_steps=20")
    print("[奖励] 0.6 Code + 0.4 门控 Format；在线测试上限正式16/Smoke4")
    print("[保存] 只保存 LoRA adapter；merge_model=false")


if __name__ == "__main__":
    main()
