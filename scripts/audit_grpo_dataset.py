#!/usr/bin/env python3
"""审计完整 SFT accepted 恢复 TACO 测试后可进入 GRPO 的样本数量。

本脚本只读取数据并输出统计，不修改训练集或配置。

背景：data/final/sft_accepted.jsonl 是 SFT 标签文件，可能只保留
reward_compatible 标记而不携带 test_cases。GRPO 需要真实可执行测试，因此
必须按题目 ID 与本地原始 TACO 对齐，恢复 io_mode、fn_name 与 test_cases。

真实准入口径：
- 样本存在于 SFT accepted；
- 存在 user 消息；
- 能在本地 TACO 中按 ID 找回原题；
- 测试不少于 4 条；
- 当前 execution verifier 支持恢复后的 metadata。

同时报告排除 TACO-515 后的正式 GRPO train 候选数。
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

from src.data.loader import load_problems
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


def normalized_label(value: Any, default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    return text or default


def load_jsonl_records(path: Path) -> tuple[dict[str, dict[str, Any]], Counter, list[str]]:
    records: dict[str, dict[str, Any]] = {}
    counts = Counter()
    duplicate_examples: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counts["nonempty_lines"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_json"] += 1
                continue
            if not isinstance(item, dict):
                counts["non_object_records"] += 1
                continue
            item_id = get_id(item)
            if item_id is None:
                counts["missing_id"] += 1
                continue
            counts["records"] += 1
            if item_id in records:
                counts["duplicate_records"] += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append(item_id)
            records[item_id] = item
    return records, counts, duplicate_examples


def load_id_set(path: Path) -> set[str]:
    result: set[str] = set()
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = get_id(item)
            if item_id:
                result.add(item_id)
    return result


def restored_metadata(problem: Any, accepted: dict[str, Any]) -> dict[str, Any]:
    old_meta = accepted.get("metadata")
    metadata = dict(old_meta) if isinstance(old_meta, dict) else {}
    input_output = problem.input_output if isinstance(problem.input_output, dict) else {}
    fn_name = input_output.get("fn_name")
    io_mode = "call_based" if fn_name else "standard_input"
    metadata.update(
        {
            "source": problem.source,
            "difficulty": normalized_label(problem.difficulty),
            "tags": list(problem.tags or []),
            "raw_tags": list(problem.raw_tags or []),
            "skill_types": list(problem.skill_types or []),
            "io_mode": io_mode,
            "fn_name": fn_name,
            "starter_code": problem.starter_code or "",
            "test_cases": list(problem.public_tests or []),
        }
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--taco-data-root", default="data/raw/TACO/ALL")
    parser.add_argument(
        "--taco515-source-bank",
        default="data/experts/io_mode/taco515_oracle_source_bank.jsonl",
    )
    parser.add_argument("--max-load", type=int, default=30000)
    parser.add_argument(
        "--out",
        default="reports/data/grpo_dataset_audit.json",
        help="统计 JSON 输出路径；传空字符串可只打印",
    )
    args = parser.parse_args()

    accepted_path = resolve(args.data)
    taco_root = resolve(args.taco_data_root)
    taco515_path = resolve(args.taco515_source_bank)
    if not accepted_path.is_file():
        raise FileNotFoundError(f"未找到 SFT accepted：{accepted_path}")
    if not taco_root.exists():
        raise FileNotFoundError(f"未找到本地 TACO：{taco_root}")

    accepted_by_id, input_counts, duplicate_examples = load_jsonl_records(accepted_path)
    taco515_ids = load_id_set(taco515_path)

    problems = load_problems(
        source="taco",
        split="train",
        max_items=args.max_load,
        deduplicate=True,
        taco_data_root=taco_root,
    )
    problem_by_id = {problem.id: problem for problem in problems}

    counts = Counter(input_counts)
    counts["accepted_unique_ids"] = len(accepted_by_id)
    counts["taco_loaded"] = len(problem_by_id)
    counts["taco515_ids"] = len(taco515_ids)

    eligible_io = Counter()
    eligible_difficulty = Counter()
    eligible_source = Counter()
    train_io = Counter()
    train_difficulty = Counter()
    missing_taco_examples: list[str] = []
    unsupported_examples: list[str] = []

    for item_id, accepted in accepted_by_id.items():
        if has_user_message(accepted):
            counts["with_user_message"] += 1
        else:
            counts["missing_user_message"] += 1
            continue

        old_meta = accepted.get("metadata")
        if isinstance(old_meta, dict) and old_meta.get("reward_compatible") is True:
            counts["stored_reward_flag_true"] += 1
        else:
            counts["stored_reward_flag_not_true"] += 1

        problem = problem_by_id.get(item_id)
        if problem is None:
            counts["missing_from_loaded_taco"] += 1
            if len(missing_taco_examples) < 20:
                missing_taco_examples.append(item_id)
            continue
        counts["matched_to_taco"] += 1

        metadata = restored_metadata(problem, accepted)
        tests = metadata["test_cases"]
        enough_tests = len(tests) >= 4
        supported = supports_verification(metadata)

        if tests:
            counts["with_restored_tests"] += 1
        else:
            counts["without_restored_tests"] += 1
        if enough_tests:
            counts["with_at_least_4_restored_tests"] += 1
        elif tests:
            counts["with_1_to_3_restored_tests"] += 1
        if supported:
            counts["restored_schema_supported"] += 1
        else:
            counts["restored_schema_unsupported"] += 1
            if len(unsupported_examples) < 20:
                unsupported_examples.append(item_id)

        eligible = enough_tests and supported
        if not eligible:
            continue

        counts["eligible_after_restore"] += 1
        mode = str(metadata["io_mode"])
        difficulty = normalized_label(metadata.get("difficulty"))
        source = normalized_label(metadata.get("source"))
        eligible_io[mode] += 1
        eligible_difficulty[difficulty] += 1
        eligible_source[source] += 1

        if item_id in taco515_ids:
            counts["eligible_in_taco515"] += 1
            continue

        counts["eligible_grpo_train_after_taco515_exclusion"] += 1
        train_io[mode] += 1
        train_difficulty[difficulty] += 1

    report = {
        "accepted_data": str(accepted_path.relative_to(ROOT)),
        "taco_data_root": str(taco_root.relative_to(ROOT)),
        "taco515_source_bank": (
            str(taco515_path.relative_to(ROOT)) if taco515_path.exists() else None
        ),
        "definition": {
            "eligible_after_restore": (
                "SFT accepted + user 消息 + 按 ID 找回 TACO + 恢复测试数>=4 + verifier 支持"
            ),
            "eligible_grpo_train_after_taco515_exclusion": (
                "eligible_after_restore 再排除全部 TACO-515 ID"
            ),
        },
        "counts": dict(counts),
        "eligible_after_restore": {
            "by_io_mode": dict(eligible_io),
            "by_difficulty": dict(eligible_difficulty),
            "by_source": dict(eligible_source),
        },
        "eligible_grpo_train_after_taco515_exclusion": {
            "by_io_mode": dict(train_io),
            "by_difficulty": dict(train_difficulty),
        },
        "diagnostics": {
            "duplicate_id_examples": duplicate_examples,
            "missing_from_taco_examples": missing_taco_examples,
            "unsupported_schema_examples": unsupported_examples,
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
