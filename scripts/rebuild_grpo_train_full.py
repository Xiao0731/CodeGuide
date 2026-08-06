#!/usr/bin/env python3
"""从完整 TACO + 已有 reference cache 重建正式 GRPO train。

TACO-515 只保留为 dev / TeachingEval；全部 515 ID 从训练集排除。
默认固定抽取 2000 条，避免把最小项目扩成超长训练。
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


def load_verified_ids(path: Path) -> tuple[set[str], int]:
    verified: set[str] = set()
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1
            try:
                item_id = record_id(item)
            except ValueError as exc:
                raise ValueError(f"reference cache 第 {line_no} 行缺少 ID") from exc
            if item.get("reference_verified") is True:
                verified.add(item_id)
    return verified, total


def load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                ids.add(record_id(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"TACO-515 source bank 第 {line_no} 行缺少 ID") from exc
    return ids


def problem_to_record(problem: Any) -> dict[str, Any] | None:
    from src.reward.execution import supports_verification

    input_output = problem.input_output if isinstance(problem.input_output, dict) else {}
    fn_name = input_output.get("fn_name")
    io_mode = "call_based" if fn_name else "standard_input"
    tests = list(problem.public_tests or [])
    difficulty = normalize_difficulty(problem.difficulty)
    metadata = {
        "source": problem.source,
        "difficulty": difficulty,
        "original_difficulty": str(problem.difficulty or "unknown"),
        "tags": list(problem.tags or []),
        "raw_tags": list(problem.raw_tags or []),
        "skill_types": list(problem.skill_types or []),
        "io_mode": io_mode,
        "fn_name": fn_name,
        "starter_code": problem.starter_code or "",
        "test_cases": tests,
        "reward_compatible": True,
    }
    metadata["reward_compatible"] = len(tests) >= 4 and supports_verification(metadata)
    if metadata["reward_compatible"] is not True:
        return None

    tags = ", ".join(metadata["tags"][:12]) or "无"
    user_content = (
        "请按教学格式讲解以下算法题：\n\n"
        f"【题目描述】\n{problem.description.strip()}\n\n"
        f"【难度】{difficulty}\n"
        f"【标签】{tags}\n\n"
        "【判题接口】\n"
        f"- io_mode: {io_mode}\n"
        f"- fn_name: {fn_name or 'None'}\n\n"
        "【starter_code】\n"
        f"```python\n{problem.starter_code or ''}\n```"
    )
    return {
        "id": problem.id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "metadata": metadata,
    }


def select_balanced(
    records: list[dict[str, Any]],
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for difficulty in ("easy", "medium", "hard"):
        pool = [
            item for item in records
            if item["metadata"]["difficulty"] == difficulty
        ]
        pools[difficulty] = sorted(
            pool,
            key=lambda item: stable_rank(item["id"], seed),
        )

    offsets = {key: 0 for key in pools}
    selected: list[dict[str, Any]] = []
    while len(selected) < max_samples:
        added = False
        for difficulty in ("easy", "medium", "hard"):
            index = offsets[difficulty]
            if index < len(pools[difficulty]):
                selected.append(pools[difficulty][index])
                offsets[difficulty] += 1
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-cache",
        default="data/cache/taco_reference_verification_train_full.jsonl",
    )
    parser.add_argument("--taco-data-root", default="data/raw/TACO/ALL")
    parser.add_argument(
        "--taco515-source-bank",
        default="data/experts/io_mode/taco515_oracle_source_bank.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/grpo_minimal_v1",
    )
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-load", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    cache_path = resolve(args.reference_cache)
    taco_root = resolve(args.taco_data_root)
    taco515_path = resolve(args.taco515_source_bank)
    output_dir = resolve(args.output_dir)
    manifest_path = output_dir / "freeze_manifest.json"
    dev_path = output_dir / "grpo_dev.jsonl"
    teaching_path = output_dir / "teaching_eval_50.jsonl"

    for required in (cache_path, taco515_path, dev_path, teaching_path):
        if not required.exists():
            raise FileNotFoundError(f"缺少必要文件：{required}")

    verified_ids, cache_records = load_verified_ids(cache_path)
    taco515_ids = load_ids(taco515_path)

    from src.data.loader import load_problems

    problems = load_problems(
        source="taco",
        split="train",
        max_items=args.max_load,
        deduplicate=True,
        taco_data_root=taco_root,
    )

    eligible: list[dict[str, Any]] = []
    for problem in problems:
        if problem.id not in verified_ids or problem.id in taco515_ids:
            continue
        item = problem_to_record(problem)
        if item is not None:
            eligible.append(item)

    selected = select_balanced(
        eligible,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError(
            "未构造出 GRPO train；请检查 reference cache 与本地 TACO 是否同一版本"
        )

    train_path = output_dir / "grpo_train.jsonl"
    ids_path = output_dir / "grpo_train.ids.json"
    write_jsonl(train_path, selected)
    ids_path.write_text(
        json.dumps({"ids": [item["id"] for item in selected]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.update(
        {
            "schema_version": "codeguide-grpo-minimal-freeze-v3",
            "reference_cache": str(cache_path.relative_to(ROOT)),
            "reference_cache_sha256": sha256_file(cache_path),
            "reference_cache_records": cache_records,
            "reference_verified_ids": len(verified_ids),
            "taco_loaded": len(problems),
            "train_eligible_before_sampling": len(eligible),
            "train_max_samples": args.max_train_samples,
        }
    )
    manifest.setdefault("policy", {})["grpo_train"] = (
        "完整 TACO train 中 reference_verified、可执行、测试不少于4条，"
        "排除全部 TACO-515 后固定抽取"
    )
    manifest.setdefault("counts", {})["grpo_train"] = {
        "samples": len(selected),
        "by_io_mode": dict(Counter(item["metadata"]["io_mode"] for item in selected)),
        "by_difficulty": dict(Counter(item["metadata"]["difficulty"] for item in selected)),
        "jsonl": str(train_path.relative_to(ROOT)),
        "jsonl_sha256": sha256_file(train_path),
    }
    selected_ids = {item["id"] for item in selected}
    manifest["overlap"] = {
        **dict(manifest.get("overlap") or {}),
        "train_taco515": len(selected_ids & taco515_ids),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"]["grpo_train"], ensure_ascii=False, indent=2))
    print(
        f"[完成] 从 eligible={len(eligible)} 中固定抽取 GRPO train={len(selected)}；"
        "TACO-515 仅保留为 dev/TeachingEval"
    )


if __name__ == "__main__":
    main()
