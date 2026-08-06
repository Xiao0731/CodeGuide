#!/usr/bin/env python3
"""复用既有 TACO-515，冻结最小版 GRPO 数据。

为了尽快完成项目，不再重新构造独立数据集。脚本将已经用于 SFT 回归评测的
TACO-515 确认集固定拆分为：
- GRPO train：剩余 415 题；
- GRPO dev：40 standard_input + 10 call_based；
- TeachingEval-50：40 standard_input + 10 call_based。

三部分在 GRPO 阶段互不重叠。由于这些题曾参与 SFT 数据与 checkpoint 分析，
它们只被称为训练域开发/回归集，不宣称完全未见泛化。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。"
    "请先理解题意，从朴素思路逐步优化，解释为什么这样做，"
    "最后给出带注释的完整 Python 代码和复杂度分析。"
)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("problem_id") or record.get("task_id")
    if not value:
        raise ValueError("TACO-515 source bank 中存在缺少 id/problem_id/task_id 的样本")
    return str(value)


def extract_user_content(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content

    for key in ("question", "description", "problem_statement", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return (
                "请按教学格式讲解以下算法题：\n\n"
                "【题目描述】\n"
                + value.strip()
            )
    raise ValueError(f"样本 {record_id(record)} 缺少题目文本")


def normalize_tests(record: dict[str, Any], metadata: dict[str, Any]) -> tuple[list[dict], str, str | None]:
    """兼容已归一化 source bank 与原始 TACO input_output。"""
    direct = metadata.get("test_cases") or record.get("test_cases")
    if isinstance(direct, list) and direct:
        fn_name = metadata.get("fn_name") or record.get("fn_name")
        io_mode = metadata.get("io_mode") or record.get("io_mode")
        if io_mode not in {"standard_input", "call_based"}:
            io_mode = "call_based" if fn_name else "standard_input"
        return direct, str(io_mode), str(fn_name) if fn_name else None

    public_tests = parse_jsonish(metadata.get("public_tests") or record.get("public_tests"))
    if isinstance(public_tests, list) and public_tests:
        fn_name = metadata.get("fn_name") or record.get("fn_name")
        io_mode = "call_based" if fn_name else "standard_input"
        return public_tests, io_mode, str(fn_name) if fn_name else None

    input_output = parse_jsonish(metadata.get("input_output") or record.get("input_output"))
    if not isinstance(input_output, dict):
        return [], "unknown", None

    inputs = input_output.get("inputs") or []
    outputs = input_output.get("outputs") or []
    fn_name = input_output.get("fn_name") or metadata.get("fn_name") or record.get("fn_name")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return [], "unknown", str(fn_name) if fn_name else None

    cases: list[dict[str, Any]] = []
    if fn_name:
        for raw_input, raw_output in zip(inputs, outputs):
            args = parse_jsonish(raw_input)
            expected = parse_jsonish(raw_output)
            if not isinstance(args, list):
                args = [args]
            cases.append(
                {
                    "input_args": args,
                    "expected_output": expected,
                    "fn_name": str(fn_name),
                }
            )
        return cases, "call_based", str(fn_name)

    for raw_input, raw_output in zip(inputs, outputs):
        cases.append(
            {
                "input": str(raw_input),
                "output": str(raw_output),
            }
        )
    return cases, "standard_input", None


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    from src.reward.execution import supports_verification

    metadata = dict(record.get("metadata") or {})
    test_cases, io_mode, fn_name = normalize_tests(record, metadata)
    metadata.update(
        {
            "io_mode": io_mode,
            "fn_name": fn_name,
            "starter_code": metadata.get("starter_code") or record.get("starter_code") or "",
            "test_cases": test_cases,
        }
    )
    metadata["reward_compatible"] = supports_verification(metadata) and len(test_cases) >= 4
    if metadata["reward_compatible"] is not True:
        return None

    return {
        "id": record_id(record),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": extract_user_content(record)},
        ],
        "metadata": metadata,
    }


def stable_rank(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{problem_id}".encode("utf-8")).hexdigest()


def split_mode(records: list[dict[str, Any]], mode: str, seed: int) -> list[dict[str, Any]]:
    selected = [record for record in records if record["metadata"]["io_mode"] == mode]
    return sorted(selected, key=lambda item: stable_rank(item["id"], seed))


def write_text(path: Path, content: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        old = path.read_text(encoding="utf-8")
        if old != content:
            raise RuntimeError(f"冻结文件已存在且内容不同：{path}；重建请加 --force")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any, *, force: bool) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", force=force)


def write_jsonl(path: Path, records: list[dict[str, Any]], *, force: bool) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    write_text(path, content, force=force)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mode_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(record["metadata"]["io_mode"] for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bank",
        default="data/experts/io_mode/taco515_oracle_source_bank.jsonl",
        help="此前严格验证使用的 TACO-515 source bank",
    )
    parser.add_argument("--output-dir", default="data/splits/grpo_minimal_v1")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dev-standard", type=int, default=40)
    parser.add_argument("--dev-call", type=int, default=10)
    parser.add_argument("--teaching-standard", type=int, default=40)
    parser.add_argument("--teaching-call", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_path = resolve(args.source_bank)
    output_dir = resolve(args.output_dir)
    if not source_path.exists():
        raise FileNotFoundError(
            f"未找到 TACO-515 source bank：{source_path}\n"
            "该文件就是此前双专家严格验证使用的 515 题源文件。"
        )

    normalized: list[dict[str, Any]] = []
    rejected = 0
    seen: set[str] = set()
    with source_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            item = normalize_record(raw)
            if item is None:
                rejected += 1
                continue
            if item["id"] in seen:
                raise ValueError(f"source bank 存在重复 ID：{item['id']}，行 {line_no}")
            seen.add(item["id"])
            normalized.append(item)

    standard = split_mode(normalized, "standard_input", args.seed)
    call = split_mode(normalized, "call_based", args.seed)
    need_standard = args.dev_standard + args.teaching_standard
    need_call = args.dev_call + args.teaching_call
    if len(standard) < need_standard or len(call) < need_call:
        raise RuntimeError(
            "TACO-515 可用样本不足："
            f"standard_input={len(standard)}（至少需要 {need_standard}），"
            f"call_based={len(call)}（至少需要 {need_call}），rejected={rejected}"
        )

    teaching = (
        standard[: args.teaching_standard]
        + call[: args.teaching_call]
    )
    dev = (
        standard[args.teaching_standard : need_standard]
        + call[args.teaching_call : need_call]
    )
    heldout_ids = {item["id"] for item in teaching + dev}
    train = [item for item in normalized if item["id"] not in heldout_ids]

    if len(train) + len(dev) + len(teaching) != len(normalized):
        raise RuntimeError("冻结拆分数量校验失败")
    if {item["id"] for item in train} & {item["id"] for item in dev}:
        raise RuntimeError("GRPO train/dev 重叠")
    if {item["id"] for item in train} & {item["id"] for item in teaching}:
        raise RuntimeError("GRPO train/TeachingEval 重叠")
    if {item["id"] for item in dev} & {item["id"] for item in teaching}:
        raise RuntimeError("GRPO dev/TeachingEval 重叠")

    artifacts = {
        "grpo_train": (output_dir / "grpo_train.jsonl", train),
        "grpo_dev": (output_dir / "grpo_dev.jsonl", dev),
        "teaching_eval_50": (output_dir / "teaching_eval_50.jsonl", teaching),
    }
    for _, (path, records) in artifacts.items():
        write_jsonl(path, records, force=args.force)
        write_json(
            path.with_suffix(".ids.json"),
            {"ids": [record["id"] for record in records]},
            force=args.force,
        )

    manifest = {
        "schema_version": "codeguide-grpo-minimal-taco515-freeze-v1",
        "seed": args.seed,
        "source_bank": str(source_path.relative_to(ROOT)),
        "source_bank_sha256": sha256_file(source_path),
        "normalized_samples": len(normalized),
        "rejected_samples": rejected,
        "policy": {
            "grpo_train": "TACO-515 中排除 dev 与 TeachingEval 后的全部样本",
            "grpo_dev": "40 standard_input + 10 call_based",
            "teaching_eval": "40 standard_input + 10 call_based",
            "disclosure": "该集合曾参与 SFT 与 checkpoint 分析，仅作为训练域开发/回归集",
        },
        "counts": {
            name: {
                "samples": len(records),
                "by_io_mode": mode_counts(records),
                "jsonl": str(path.relative_to(ROOT)),
                "jsonl_sha256": sha256_file(path),
            }
            for name, (path, records) in artifacts.items()
        },
        "overlap": {
            "train_dev": 0,
            "train_teaching": 0,
            "dev_teaching": 0,
        },
    }
    write_json(output_dir / "freeze_manifest.json", manifest, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[完成] 已复用 TACO-515 冻结最小版 GRPO 数据：{output_dir}")


if __name__ == "__main__":
    main()
