#!/usr/bin/env python3
"""从 SFT accepted 恢复测试用例并冻结正式 GRPO train/dev。

数据口径：
- data/final/sft_accepted.jsonl 是母集，保证 GRPO 只继续训练 SFT 已使用的题目；
- 根据题目 ID 回到 data/raw/TACO/ALL 恢复 test_cases、接口和难度信息；
- TACO-515 永久作为 Base/SFT/GRPO 共用的自研回归测试集，不进入 GRPO train/dev；
- 恢复后要求测试不少于 4 条且当前 verifier 支持；
- 从 eligible 中固定 40 道 standard_input + 10 道 call_based 作为 GRPO dev，
  只用于 checkpoint 选择；其余全部进入 GRPO train，不做人为数量截断。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = (
    "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。"
    "请先理解题意，从朴素思路逐步优化，解释为什么这样做，"
    "最后给出带注释的完整 Python 代码和复杂度分析。"
)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("problem_id") or record.get("task_id")
    if not value:
        raise ValueError("记录缺少 id/problem_id/task_id")
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{problem_id}".encode("utf-8")).hexdigest()


def normalize_difficulty(value: Any) -> str:
    difficulty = str(value or "unknown").strip().lower().replace("-", "_")
    if "easy" in difficulty:
        return "easy"
    if "hard" in difficulty:
        return "hard"
    return "medium"


def load_jsonl_records(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                item_id = record_id(item)
            except ValueError as exc:
                raise ValueError(f"{label} 第 {line_no} 行缺少 ID") from exc
            if item_id in records:
                raise ValueError(f"{label} 存在重复 ID：{item_id}")
            records[item_id] = item
    return records


def load_ids(path: Path) -> set[str]:
    return set(load_jsonl_records(path, label="TACO-515 source bank"))


def prompt_messages(accepted: dict[str, Any], problem: Any) -> list[dict[str, str]]:
    """保留 SFT 时的 system/user prompt，移除 assistant 标签。"""
    messages = accepted.get("messages")
    kept: list[dict[str, str]] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "user"} and isinstance(content, str) and content.strip():
                kept.append({"role": str(role), "content": content})

    if any(message["role"] == "user" for message in kept):
        if not any(message["role"] == "system" for message in kept):
            kept.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return kept

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请按教学格式讲解以下算法题：\n\n"
                f"【题目描述】\n{str(problem.description).strip()}"
            ),
        },
    ]


def restore_record(accepted: dict[str, Any], problem: Any) -> dict[str, Any] | None:
    from src.reward.execution import supports_verification

    input_output = problem.input_output if isinstance(problem.input_output, dict) else {}
    fn_name = input_output.get("fn_name")
    io_mode = "call_based" if fn_name else "standard_input"
    tests = list(problem.public_tests or [])

    accepted_meta = accepted.get("metadata")
    if not isinstance(accepted_meta, dict):
        accepted_meta = {}

    metadata = {
        **accepted_meta,
        "source": problem.source,
        "difficulty": normalize_difficulty(problem.difficulty),
        "original_difficulty": str(problem.difficulty or "unknown"),
        "tags": list(problem.tags or []),
        "raw_tags": list(problem.raw_tags or []),
        "skill_types": list(problem.skill_types or []),
        "io_mode": io_mode,
        "fn_name": fn_name,
        "starter_code": problem.starter_code or "",
        "test_cases": tests,
    }
    metadata["reward_compatible"] = len(tests) >= 4 and supports_verification(metadata)
    if metadata["reward_compatible"] is not True:
        return None

    return {
        "id": problem.id,
        "messages": prompt_messages(accepted, problem),
        "metadata": metadata,
    }


def select_dev(
    records: list[dict[str, Any]],
    *,
    standard_count: int,
    call_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_mode = {
        mode: sorted(
            [item for item in records if item["metadata"]["io_mode"] == mode],
            key=lambda item: stable_rank(item["id"], seed),
        )
        for mode in ("standard_input", "call_based")
    }
    if len(by_mode["standard_input"]) < standard_count:
        raise RuntimeError(
            f"standard_input GRPO dev 候选不足：需要 {standard_count}，"
            f"实际 {len(by_mode['standard_input'])}"
        )
    if len(by_mode["call_based"]) < call_count:
        raise RuntimeError(
            f"call_based GRPO dev 候选不足：需要 {call_count}，"
            f"实际 {len(by_mode['call_based'])}"
        )

    dev = (
        by_mode["standard_input"][:standard_count]
        + by_mode["call_based"][:call_count]
    )
    # GRPO dev 完全不参与优化，因此全部测试均作为 checkpoint held-out tests。
    for item in dev:
        item["metadata"]["heldout_tests"] = list(item["metadata"]["test_cases"])

    dev_ids = {item["id"] for item in dev}
    train = [item for item in records if item["id"] not in dev_ids]
    train = sorted(train, key=lambda item: stable_rank(item["id"], seed + 1))
    dev = sorted(dev, key=lambda item: stable_rank(item["id"], seed + 2))
    return train, dev


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_ids(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"ids": [item["id"] for item in records]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def counts_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "by_io_mode": dict(Counter(item["metadata"]["io_mode"] for item in records)),
        "by_difficulty": dict(
            Counter(item["metadata"]["difficulty"] for item in records)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-accepted", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--taco-data-root", default="data/raw/TACO/ALL")
    parser.add_argument(
        "--taco515-source-bank",
        default="data/experts/io_mode/taco515_oracle_source_bank.jsonl",
    )
    parser.add_argument("--output-dir", default="data/splits/grpo_minimal_v1")
    parser.add_argument("--dev-standard", type=int, default=40)
    parser.add_argument("--dev-call", type=int, default=10)
    parser.add_argument("--max-load", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    accepted_path = resolve(args.sft_accepted)
    taco_root = resolve(args.taco_data_root)
    taco515_path = resolve(args.taco515_source_bank)
    output_dir = resolve(args.output_dir)
    for required in (accepted_path, taco_root, taco515_path):
        if not required.exists():
            raise FileNotFoundError(f"缺少必要文件：{required}")

    accepted = load_jsonl_records(accepted_path, label="SFT accepted")
    taco515_ids = load_ids(taco515_path)

    from src.data.loader import load_problems

    problems = load_problems(
        source="taco",
        split="train",
        max_items=args.max_load,
        deduplicate=True,
        taco_data_root=taco_root,
    )
    problem_map = {problem.id: problem for problem in problems}

    stats = Counter()
    eligible: list[dict[str, Any]] = []
    missing_raw_examples: list[str] = []
    for item_id, accepted_record in accepted.items():
        stats["sft_accepted"] += 1
        if item_id in taco515_ids:
            stats["excluded_taco515"] += 1
            continue
        problem = problem_map.get(item_id)
        if problem is None:
            stats["missing_in_raw_taco"] += 1
            if len(missing_raw_examples) < 20:
                missing_raw_examples.append(item_id)
            continue
        stats["matched_to_raw_taco"] += 1
        restored = restore_record(accepted_record, problem)
        if restored is None:
            stats["rejected_after_restore"] += 1
            continue
        stats["eligible_after_restore"] += 1
        eligible.append(restored)

    train, dev = select_dev(
        eligible,
        standard_count=args.dev_standard,
        call_count=args.dev_call,
        seed=args.seed,
    )

    train_ids = {item["id"] for item in train}
    dev_ids = {item["id"] for item in dev}
    if train_ids & dev_ids:
        raise RuntimeError("GRPO train/dev 存在重叠")
    if (train_ids | dev_ids) & taco515_ids:
        raise RuntimeError("GRPO 数据与 TACO-515 存在重叠")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "grpo_train.jsonl"
    dev_path = output_dir / "grpo_dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    write_ids(output_dir / "grpo_train.ids.json", train)
    write_ids(output_dir / "grpo_dev.ids.json", dev)

    report = {
        "schema_version": "codeguide-grpo-from-sft-accepted-v1",
        "seed": args.seed,
        "sources": {
            "sft_accepted": str(accepted_path.relative_to(ROOT)),
            "sft_accepted_sha256": sha256_file(accepted_path),
            "taco_data_root": str(taco_root.relative_to(ROOT)),
            "taco515_source_bank": str(taco515_path.relative_to(ROOT)),
            "taco515_source_bank_sha256": sha256_file(taco515_path),
        },
        "policy": {
            "candidate_pool": "SFT accepted ID 与原始 TACO 的交集",
            "taco515": "永久冻结为 Base/SFT/GRPO 共用自研回归测试集，不参与 train/dev",
            "eligibility": "恢复测试后 test_cases>=4 且当前 verifier 支持",
            "grpo_dev": (
                f"{args.dev_standard} standard_input + {args.dev_call} call_based，"
                "只用于 checkpoint 选择"
            ),
            "grpo_train": "eligible 排除 GRPO dev 后全部使用，不做人为截断",
        },
        "counts": {
            **dict(stats),
            "raw_taco_loaded": len(problems),
            "taco515_ids": len(taco515_ids),
            "grpo_train": counts_for(train),
            "grpo_dev": counts_for(dev),
        },
        "overlap": {
            "train_dev": len(train_ids & dev_ids),
            "train_taco515": len(train_ids & taco515_ids),
            "dev_taco515": len(dev_ids & taco515_ids),
        },
        "diagnostics": {
            "missing_in_raw_taco_examples": missing_raw_examples,
        },
        "artifacts": {
            "grpo_train": str(train_path.relative_to(ROOT)),
            "grpo_train_sha256": sha256_file(train_path),
            "grpo_dev": str(dev_path.relative_to(ROOT)),
            "grpo_dev_sha256": sha256_file(dev_path),
        },
    }
    manifest_path = output_dir / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"[完成] GRPO train={len(train)}，dev={len(dev)}；"
        f"排除 TACO-515={stats['excluded_taco515']}"
    )


if __name__ == "__main__":
    main()
