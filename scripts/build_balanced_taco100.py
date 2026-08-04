#!/usr/bin/env python3
"""Build a deterministic 50 standard_input + 50 call_based TACO dev probe."""

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

from src.training.sft_data import load_canonical, load_id_list, stratified_sample_ids


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def io_mode(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("io_mode") or record.get("io_mode") or "unknown")


def ids_sha256(ids: list[str]) -> str:
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument(
        "--output",
        default="data/splits/sft_checkpoint_dev_100_ids.json",
    )
    parser.add_argument("--per-mode", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.per_mode <= 0:
        raise ValueError("--per-mode must be positive")

    canonical_path = resolve_path(args.canonical)
    dev_ids_path = resolve_path(args.dev_ids)
    output_path = resolve_path(args.output)

    canonical = load_canonical(canonical_path)
    dev_ids = load_id_list(dev_ids_path)
    missing = [problem_id for problem_id in dev_ids if problem_id not in canonical]
    if missing:
        raise RuntimeError(f"canonical misses frozen dev IDs: {missing[:5]}")

    dev_records = [canonical[problem_id] for problem_id in dev_ids]
    mode_counts = Counter(io_mode(record) for record in dev_records)
    expected_modes = ("standard_input", "call_based")
    unexpected = sorted(set(mode_counts) - set(expected_modes))
    if unexpected:
        raise RuntimeError(
            f"unexpected io_mode values in frozen dev: {unexpected}; counts={dict(mode_counts)}"
        )

    selected_by_mode: dict[str, list[str]] = {}
    for mode in expected_modes:
        records = [record for record in dev_records if io_mode(record) == mode]
        if len(records) < args.per_mode:
            raise RuntimeError(
                f"not enough {mode} records: required={args.per_mode}, available={len(records)}"
            )
        selected_by_mode[mode] = stratified_sample_ids(
            records,
            args.per_mode,
            args.seed,
        )

    selected_set = set().union(*map(set, selected_by_mode.values()))
    selected = [problem_id for problem_id in dev_ids if problem_id in selected_set]
    if len(selected) != args.per_mode * 2:
        raise RuntimeError(f"selection size mismatch: {len(selected)}")

    selected_counts = Counter(io_mode(canonical[problem_id]) for problem_id in selected)
    expected_counts = {mode: args.per_mode for mode in expected_modes}
    if dict(selected_counts) != expected_counts:
        raise RuntimeError(
            f"balanced selection mismatch: expected={expected_counts}, actual={dict(selected_counts)}"
        )

    payload = {
        "schema_version": "codeguide-balanced-taco-probe-v1",
        "seed": args.seed,
        "parent_dev_ids": str(dev_ids_path.relative_to(ROOT))
        if dev_ids_path.is_relative_to(ROOT)
        else str(dev_ids_path),
        "canonical": str(canonical_path.relative_to(ROOT))
        if canonical_path.is_relative_to(ROOT)
        else str(canonical_path),
        "samples": len(selected),
        "counts": expected_counts,
        "ids_sha256": ids_sha256(selected),
        "ids": selected,
    }

    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output_path.exists() and not args.force:
        existing = output_path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                f"frozen probe already exists with different contents: {output_path}; "
                "use --force only when intentionally creating a new protocol"
            )
        print(f"[balanced-taco100] existing selection verified: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(f"[balanced-taco100] wrote: {output_path}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
