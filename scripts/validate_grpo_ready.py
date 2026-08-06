#!/usr/bin/env python3
"""GRPO 上云前无 GPU 门禁：验证数据、配置、重叠和步数。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reward.execution import supports_verification


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("problem_id") or record.get("task_id")
    if not value:
        raise ValueError("记录缺少 ID")
    return str(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 错误：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是对象")
            records.append(item)
    return records


def validate_records(
    records: list[dict[str, Any]],
    *,
    label: str,
    require_heldout: bool,
) -> tuple[set[str], dict[str, Any], list[str]]:
    ids: set[str] = set()
    modes = Counter()
    difficulties = Counter()
    test_counts: list[int] = []
    errors: list[str] = []

    for index, record in enumerate(records, 1):
        try:
            item_id = record_id(record)
        except ValueError:
            errors.append(f"{label} 第 {index} 条缺少 ID")
            continue
        if item_id in ids:
            errors.append(f"{label} 重复 ID：{item_id}")
        ids.add(item_id)

        messages = record.get("messages")
        user_ok = isinstance(messages, list) and any(
            isinstance(message, dict)
            and message.get("role") == "user"
            and bool(str(message.get("content") or "").strip())
            for message in messages
        )
        if not user_ok:
            errors.append(f"{label} {item_id} 缺少 user prompt")

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"{label} {item_id} 缺少 metadata")
            continue
        tests = metadata.get("test_cases") or []
        if not isinstance(tests, list) or len(tests) < 4:
            errors.append(f"{label} {item_id} test_cases 少于 4")
            continue
        if metadata.get("reward_compatible") is not True:
            errors.append(f"{label} {item_id} reward_compatible 不为 true")
        if not supports_verification(metadata):
            errors.append(f"{label} {item_id} verifier 不支持")

        heldout = metadata.get("heldout_tests") or []
        if require_heldout and (not isinstance(heldout, list) or not heldout):
            errors.append(f"{label} {item_id} 缺少 heldout_tests")

        modes[str(metadata.get("io_mode") or "unknown")] += 1
        difficulties[str(metadata.get("difficulty") or "unknown")] += 1
        test_counts.append(len(tests))

    stats = {
        "samples": len(records),
        "unique_ids": len(ids),
        "by_io_mode": dict(modes),
        "by_difficulty": dict(difficulties),
        "test_count_min": min(test_counts, default=0),
        "test_count_max": max(test_counts, default=0),
        "test_count_avg": (
            round(sum(test_counts) / len(test_counts), 3)
            if test_counts
            else 0.0
        ),
    }
    return ids, stats, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default="data/splits/grpo_minimal_v1/grpo_train.jsonl",
    )
    parser.add_argument(
        "--dev",
        default="data/splits/grpo_minimal_v1/grpo_dev.jsonl",
    )
    parser.add_argument(
        "--smoke",
        default="data/splits/grpo_minimal_v1/grpo_smoke_train.jsonl",
    )
    parser.add_argument(
        "--taco515",
        default="data/experts/io_mode/taco515_oracle_source_bank.jsonl",
    )
    parser.add_argument("--config", default="configs/grpo_minimal_v1.yaml")
    parser.add_argument("--smoke-config", default="configs/grpo_smoke_v1.yaml")
    args = parser.parse_args()

    paths = {
        "train": resolve(args.train),
        "dev": resolve(args.dev),
        "smoke": resolve(args.smoke),
        "taco515": resolve(args.taco515),
        "config": resolve(args.config),
        "smoke_config": resolve(args.smoke_config),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少文件：\n" + "\n".join(missing))

    train = load_jsonl(paths["train"])
    dev = load_jsonl(paths["dev"])
    smoke = load_jsonl(paths["smoke"])
    taco515 = load_jsonl(paths["taco515"])

    train_ids, train_stats, train_errors = validate_records(
        train,
        label="GRPO train",
        require_heldout=False,
    )
    dev_ids, dev_stats, dev_errors = validate_records(
        dev,
        label="GRPO dev",
        require_heldout=True,
    )
    smoke_ids, smoke_stats, smoke_errors = validate_records(
        smoke,
        label="GRPO smoke",
        require_heldout=False,
    )
    taco515_ids = {record_id(item) for item in taco515}

    errors = train_errors + dev_errors + smoke_errors
    if len(train) != 6451:
        errors.append(f"GRPO train 预期 6451，实际 {len(train)}")
    if len(dev) != 50:
        errors.append(f"GRPO dev 预期 50，实际 {len(dev)}")
    if len(smoke) != 160:
        errors.append(f"GRPO smoke 预期 160，实际 {len(smoke)}")
    if train_ids & dev_ids:
        errors.append(f"train/dev 重叠 {len(train_ids & dev_ids)}")
    if train_ids & taco515_ids:
        errors.append(f"train/TACO-515 重叠 {len(train_ids & taco515_ids)}")
    if dev_ids & taco515_ids:
        errors.append(f"dev/TACO-515 重叠 {len(dev_ids & taco515_ids)}")
    if not smoke_ids <= train_ids:
        errors.append(f"smoke 中有 {len(smoke_ids - train_ids)} 条不属于 train")

    cfg = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    smoke_cfg = yaml.safe_load(paths["smoke_config"].read_text(encoding="utf-8"))
    grpo = cfg["grpo"]
    smoke_grpo = smoke_cfg["grpo"]

    config_checks = {
        "formal_train_data": grpo.get("train_data"),
        "formal_eval_data": grpo.get("eval_data"),
        "formal_curriculum": bool(cfg.get("curriculum", {}).get("enabled")),
        "formal_max_online_tests": grpo.get("max_online_tests"),
        "formal_save_best": grpo.get("save_best"),
        "formal_merge_model": grpo.get("merge_model"),
        "smoke_train_data": smoke_grpo.get("train_data"),
        "smoke_max_steps": smoke_grpo.get("max_steps"),
        "smoke_max_online_tests": smoke_grpo.get("max_online_tests"),
        "smoke_curriculum": bool(smoke_cfg.get("curriculum", {}).get("enabled")),
        "smoke_merge_model": smoke_grpo.get("merge_model"),
    }

    if int(grpo.get("max_online_tests", 0)) != 16:
        errors.append("正式 max_online_tests 应为 16")
    if grpo.get("save_best") is not True:
        errors.append("正式 save_best 应为 true")
    if grpo.get("merge_model") is not False:
        errors.append("正式 merge_model 应为 false")
    if int(smoke_grpo.get("max_steps", 0)) != 20:
        errors.append("Smoke max_steps 应为 20")
    if int(smoke_grpo.get("max_online_tests", 0)) != 4:
        errors.append("Smoke max_online_tests 应为 4")
    if smoke_cfg.get("curriculum", {}).get("enabled") is not False:
        errors.append("Smoke curriculum 应关闭")

    batch = int(grpo["per_device_train_batch_size"])
    accumulation = int(grpo["gradient_accumulation_steps"])
    by_difficulty = Counter(
        str((item.get("metadata") or {}).get("difficulty") or "unknown")
        for item in train
    )
    stage_steps = {
        difficulty: math.ceil(count / (batch * accumulation))
        for difficulty, count in sorted(by_difficulty.items())
    }

    report = {
        "status": "ok" if not errors else "failed",
        "train": train_stats,
        "dev": dev_stats,
        "smoke": smoke_stats,
        "overlap": {
            "train_dev": len(train_ids & dev_ids),
            "train_taco515": len(train_ids & taco515_ids),
            "dev_taco515": len(dev_ids & taco515_ids),
            "smoke_outside_train": len(smoke_ids - train_ids),
        },
        "steps": {
            "by_difficulty": stage_steps,
            "formal_total": sum(stage_steps.values()),
            "smoke": int(smoke_grpo.get("max_steps", -1)),
        },
        "config": config_checks,
        "errors": errors[:100],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    print("[通过] GRPO 数据与配置已满足上云 smoke 条件")


if __name__ == "__main__":
    main()
