#!/usr/bin/env python3
"""统计完整 SFT accepted 中可进入当前 GRPO 主链路的样本数量。

本脚本只读取数据并输出统计，不修改任何训练集或配置。
判定口径与 src/training/grpo_train.py 保持一致：
- 存在 user 消息；
- metadata.reward_compatible == true；
- 至少 4 条测试；
- 当前 execution verifier 支持该 metadata。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reward.execution import supports_verification


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def get_id(record: dict[str, Any]) -> str | None:
    value = record.get("id") or record.get("problem_id") or record.get("task_id")
    return str(value) if value else None


def has_user_message(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def infer_io_mode(metadata: dict[str, Any]) -> str:
    mode = metadata.get("io_mode")
    if mode in {"standard_input", "call_based"}:
        return str(mode)
    tests = metadata.get("test_cases") or []
    if metadata.get("fn_name") or any(
        isinstance(case, dict) and "input_args" in case for case in tests
    ):
        return "call_based"
    return "standard_input"


def normalized_label(value: Any, default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    return text or default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/final/sft_accepted.jsonl")
    parser.add_argument(
        "--out",
        default="reports/data/grpo_dataset_audit.json",
        help="统计 JSON 输出路径；传空字符串可只打印",
    )
    args = parser.parse_args()

    data_path = resolve(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"未找到 SFT accepted：{data_path}")

    counts = Counter()
    eligible_io = Counter()
    eligible_difficulty = Counter()
    eligible_source = Counter()
    recomputed_io = Counter()
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    malformed_lines: list[int] = []

    with data_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counts["nonempty_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_json"] += 1
                malformed_lines.append(line_no)
                continue
            if not isinstance(record, dict):
                counts["non_object_records"] += 1
                continue

            counts["records"] += 1
            item_id = get_id(record)
            if item_id:
                if item_id in seen_ids:
                    counts["duplicate_records"] += 1
                    if len(duplicate_ids) < 20:
                        duplicate_ids.append(item_id)
                seen_ids.add(item_id)
            else:
                counts["missing_id"] += 1

            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                counts["missing_metadata"] += 1

            user_ok = has_user_message(record)
            tests = metadata.get("test_cases") or []
            tests_ok = isinstance(tests, list) and bool(tests)
            enough_tests = tests_ok and len(tests) >= 4
            reward_flag = metadata.get("reward_compatible") is True
            schema_supported = supports_verification(metadata)
            io_mode = infer_io_mode(metadata)

            if user_ok:
                counts["with_user_message"] += 1
            else:
                counts["missing_user_message"] += 1

            if tests_ok:
                counts["with_test_cases"] += 1
            else:
                counts["missing_test_cases"] += 1

            if enough_tests:
                counts["with_at_least_4_tests"] += 1
            elif tests_ok:
                counts["with_1_to_3_tests"] += 1

            if reward_flag:
                counts["reward_flag_true"] += 1
            else:
                counts["reward_flag_not_true"] += 1

            if schema_supported:
                counts["schema_supported"] += 1
            else:
                counts["schema_unsupported"] += 1

            # 当前 grpo_train.py 的真实准入口径。
            eligible = user_ok and reward_flag and enough_tests and schema_supported
            if eligible:
                counts["eligible_current_grpo"] += 1
                eligible_io[io_mode] += 1
                eligible_difficulty[normalized_label(metadata.get("difficulty"))] += 1
                eligible_source[normalized_label(metadata.get("source"))] += 1

            # 诊断旧数据中的 reward_compatible 标记是否过时；不改变正式口径。
            recomputed_eligible = user_ok and enough_tests and schema_supported
            if recomputed_eligible:
                counts["eligible_if_recomputed"] += 1
                recomputed_io[io_mode] += 1
                if not reward_flag:
                    counts["supported_but_flag_not_true"] += 1

    report = {
        "data": str(data_path.relative_to(ROOT)),
        "definition": {
            "eligible_current_grpo": (
                "存在 user 消息 + reward_compatible=true + 测试数>=4 + verifier 支持"
            ),
            "eligible_if_recomputed": (
                "存在 user 消息 + 测试数>=4 + verifier 支持；仅用于发现旧标记问题"
            ),
        },
        "counts": dict(counts),
        "eligible_current_grpo": {
            "by_io_mode": dict(eligible_io),
            "by_difficulty": dict(eligible_difficulty),
            "by_source": dict(eligible_source),
        },
        "eligible_if_recomputed": {
            "by_io_mode": dict(recomputed_io),
        },
        "diagnostics": {
            "duplicate_id_examples": duplicate_ids,
            "malformed_line_examples": malformed_lines[:20],
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        out_path = resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[完成] 审计报告：{out_path}")


if __name__ == "__main__":
    main()
