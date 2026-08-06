#!/usr/bin/env python3
"""冻结最小版 GRPO 数据：train、dev 与 TeachingEval-50。

目标是尽快完成项目，而不是扩展成新的数据工程分支：
- TeachingEval-50 直接从既有 SFT dev 冻结，不要求存在可执行测试；
- GRPO dev 优先从 reward-compatible 的 SFT dev 选择；数量不足时，
  从 reward-compatible 的 SFT train 划出并从 GRPO train 中移除；
- 所有选择使用固定 seed 的哈希排序，重复运行结果一致；
- 产物包含 JSONL、ID 文件和 SHA256 manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("problem_id")
    if not value:
        raise ValueError("样本缺少 id/problem_id")
    return str(value)


def load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError(f"ID 文件格式错误：{path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"ID 文件存在重复项：{path}")
    return ids


def load_canonical(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record_id(record)
            if problem_id in by_id:
                raise ValueError(f"canonical 重复 ID：{problem_id}，行 {line_no}")
            ordered.append(record)
            by_id[problem_id] = record
    return ordered, by_id


def io_mode(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("io_mode") or record.get("io_mode") or "unknown")


def is_reward_compatible(record: dict[str, Any]) -> bool:
    """与现有 GRPO 入口保持同一最低合同：可执行，且至少 4 条测试。"""
    metadata = record.get("metadata") or {}
    if metadata.get("reward_compatible") is not True:
        return False
    test_cases = metadata.get("test_cases") or []
    if not isinstance(test_cases, list) or len(test_cases) < 4:
        return False

    mode = io_mode(record)
    if mode == "standard_input":
        return all(
            isinstance(case, dict) and "input" in case and "output" in case
            for case in test_cases
        )
    if mode != "call_based":
        return False

    fn_name = metadata.get("fn_name") or next(
        (
            case.get("fn_name")
            for case in test_cases
            if isinstance(case, dict) and case.get("fn_name")
        ),
        None,
    )
    if not isinstance(fn_name, str) or not fn_name.strip():
        return False
    return all(
        isinstance(case, dict)
        and isinstance(case.get("input_args"), list)
        and "expected_output" in case
        for case in test_cases
    )


def stable_rank(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{problem_id}".encode("utf-8")).hexdigest()


def ids_for_mode(
    candidates: list[str],
    by_id: dict[str, dict[str, Any]],
    mode: str,
    seed: int,
) -> list[str]:
    result = [problem_id for problem_id in candidates if io_mode(by_id[problem_id]) == mode]
    result.sort(key=lambda problem_id: stable_rank(problem_id, seed))
    return result


def select_by_mode(
    candidates: list[str],
    by_id: dict[str, dict[str, Any]],
    *,
    standard_count: int,
    call_count: int,
    seed: int,
) -> list[str]:
    targets = {
        "standard_input": standard_count,
        "call_based": call_count,
    }
    selected: list[str] = []
    for mode, count in targets.items():
        mode_ids = ids_for_mode(candidates, by_id, mode, seed)
        if len(mode_ids) < count:
            raise RuntimeError(
                f"{mode} 候选不足：需要 {count}，实际 {len(mode_ids)}"
            )
        selected.extend(mode_ids[:count])
    selected_set = set(selected)
    return [problem_id for problem_id in candidates if problem_id in selected_set]


def select_dev_with_train_fallback(
    *,
    eligible_dev: list[str],
    eligible_train: list[str],
    by_id: dict[str, dict[str, Any]],
    standard_count: int,
    call_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """优先从 SFT dev 选择；不足部分从 SFT train 划出。"""
    targets = {
        "standard_input": standard_count,
        "call_based": call_count,
    }
    selected: list[str] = []
    borrowed_from_train: list[str] = []

    for offset, (mode, count) in enumerate(targets.items()):
        dev_mode = ids_for_mode(eligible_dev, by_id, mode, seed + offset)
        train_mode = ids_for_mode(eligible_train, by_id, mode, seed + 100 + offset)

        take_dev = dev_mode[:count]
        need = count - len(take_dev)
        if len(train_mode) < need:
            raise RuntimeError(
                f"{mode} 的 GRPO dev 候选不足：需要 {count}，"
                f"SFT dev 可用 {len(dev_mode)}，SFT train 可补 {len(train_mode)}"
            )

        take_train = train_mode[:need]
        selected.extend(take_dev)
        selected.extend(take_train)
        borrowed_from_train.extend(take_train)

    selected_set = set(selected)
    ordered = [
        problem_id
        for problem_id in [*eligible_dev, *eligible_train]
        if problem_id in selected_set
    ]
    return ordered, borrowed_from_train


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exact(path: Path, content: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(
                f"冻结文件已存在且内容不同：{path}；确认重建时再加 --force"
            )
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any, *, force: bool) -> None:
    write_exact(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        force=force,
    )


def write_jsonl(
    path: Path,
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> None:
    content = "".join(
        json.dumps(by_id[problem_id], ensure_ascii=False, separators=(",", ":")) + "\n"
        for problem_id in ids
    )
    write_exact(path, content, force=force)


def mode_counts(ids: list[str], by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(io_mode(by_id[problem_id]) for problem_id in ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--sft-train-ids", default="data/splits/sft_train_ids.json")
    parser.add_argument("--sft-dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--output-dir", default="data/splits/grpo_minimal_v1")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dev-standard", type=int, default=75)
    parser.add_argument("--dev-call", type=int, default=25)
    parser.add_argument("--teaching-standard", type=int, default=40)
    parser.add_argument("--teaching-call", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    canonical_path = resolve(args.canonical)
    train_ids_path = resolve(args.sft_train_ids)
    dev_ids_path = resolve(args.sft_dev_ids)
    output_dir = resolve(args.output_dir)

    ordered, by_id = load_canonical(canonical_path)
    train_ids = load_ids(train_ids_path)
    dev_ids = load_ids(dev_ids_path)
    if set(train_ids) & set(dev_ids):
        raise RuntimeError("原 SFT train/dev 存在重叠")

    missing = sorted((set(train_ids) | set(dev_ids)) - set(by_id))
    if missing:
        raise RuntimeError(f"canonical 缺少冻结 ID：{missing[:5]}")

    eligible_train = [
        problem_id for problem_id in train_ids if is_reward_compatible(by_id[problem_id])
    ]
    eligible_dev = [
        problem_id for problem_id in dev_ids if is_reward_compatible(by_id[problem_id])
    ]

    # 教学盲评只看回答质量，不需要可执行测试，因此直接从完整 SFT dev 冻结。
    teaching_ids = select_by_mode(
        dev_ids,
        by_id,
        standard_count=args.teaching_standard,
        call_count=args.teaching_call,
        seed=args.seed + 1,
    )
    teaching_set = set(teaching_ids)
    remaining_eligible_dev = [
        problem_id for problem_id in eligible_dev if problem_id not in teaching_set
    ]

    grpo_dev_ids, borrowed_from_train = select_dev_with_train_fallback(
        eligible_dev=remaining_eligible_dev,
        eligible_train=eligible_train,
        by_id=by_id,
        standard_count=args.dev_standard,
        call_count=args.dev_call,
        seed=args.seed + 2,
    )
    borrowed_set = set(borrowed_from_train)
    grpo_train_ids = [
        problem_id for problem_id in eligible_train if problem_id not in borrowed_set
    ]

    if set(grpo_train_ids) & set(grpo_dev_ids):
        raise RuntimeError("GRPO train/dev 重叠")
    if set(grpo_train_ids) & set(teaching_ids):
        raise RuntimeError("GRPO train/TeachingEval 重叠")
    if set(grpo_dev_ids) & set(teaching_ids):
        raise RuntimeError("GRPO dev/TeachingEval 重叠")

    artifacts = {
        "grpo_train": (output_dir / "grpo_train.jsonl", grpo_train_ids),
        "grpo_dev": (output_dir / "grpo_dev.jsonl", grpo_dev_ids),
        "teaching_eval_50": (output_dir / "teaching_eval_50.jsonl", teaching_ids),
    }
    for _, (path, ids) in artifacts.items():
        write_jsonl(path, ids, by_id, force=args.force)
        write_json(path.with_suffix(".ids.json"), {"ids": ids}, force=args.force)

    manifest = {
        "schema_version": "codeguide-grpo-minimal-freeze-v1",
        "seed": args.seed,
        "source": {
            "canonical": str(canonical_path.relative_to(ROOT)),
            "canonical_sha256": sha256_file(canonical_path),
            "sft_train_ids": str(train_ids_path.relative_to(ROOT)),
            "sft_dev_ids": str(dev_ids_path.relative_to(ROOT)),
        },
        "policy": {
            "grpo_train": "SFT train 中 reward-compatible 且测试不少于 4 条；扣除借给 GRPO dev 的样本",
            "grpo_dev": "优先取 SFT dev 的 75 standard_input + 25 call_based；不足时从 SFT train 划出",
            "teaching_eval": "完整 SFT dev 中冻结 40 standard_input + 10 call_based，不要求可执行测试",
            "note": "TeachingEval 属于训练域教学回归集，不宣称完全未见泛化",
        },
        "source_counts": {
            "sft_train": len(train_ids),
            "sft_dev": len(dev_ids),
            "eligible_sft_train": len(eligible_train),
            "eligible_sft_dev": len(eligible_dev),
            "eligible_sft_train_by_io_mode": mode_counts(eligible_train, by_id),
            "eligible_sft_dev_by_io_mode": mode_counts(eligible_dev, by_id),
            "grpo_dev_borrowed_from_sft_train": len(borrowed_from_train),
            "borrowed_by_io_mode": mode_counts(borrowed_from_train, by_id),
        },
        "counts": {
            name: {
                "samples": len(ids),
                "by_io_mode": mode_counts(ids, by_id),
                "jsonl": str(path.relative_to(ROOT)),
                "jsonl_sha256": sha256_file(path),
                "ids": str(path.with_suffix(".ids.json").relative_to(ROOT)),
            }
            for name, (path, ids) in artifacts.items()
        },
        "overlap": {
            "train_dev": len(set(grpo_train_ids) & set(grpo_dev_ids)),
            "train_teaching": len(set(grpo_train_ids) & set(teaching_ids)),
            "dev_teaching": len(set(grpo_dev_ids) & set(teaching_ids)),
        },
        "canonical_records": len(ordered),
    }
    write_json(output_dir / "freeze_manifest.json", manifest, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[完成] 最小版 GRPO 数据已冻结：{output_dir}")


if __name__ == "__main__":
    main()
