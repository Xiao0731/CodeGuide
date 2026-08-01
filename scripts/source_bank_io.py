#!/usr/bin/env python3
"""Streaming helpers and CLI for the compressed TACO verified source bank."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterator


def iter_source_bank(path: str | Path) -> Iterator[dict]:
    """Yield JSON records from a .jsonl or .jsonl.zst source bank."""
    bank_path = Path(path)
    if bank_path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Reading .zst requires: pip install zstandard") from exc
        with bank_path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    for line in text:
                        if line.strip():
                            yield json.loads(line)
        return

    with bank_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def get_source_record(path: str | Path, problem_id: str) -> dict | None:
    for record in iter_source_bank(path):
        if record.get("problem_id") == problem_id:
            return record
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a CodeGuide verified source bank")
    parser.add_argument("bank", type=Path)
    parser.add_argument("--id", dest="problem_id")
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()

    if args.problem_id:
        record = get_source_record(args.bank, args.problem_id)
        if record is None:
            raise SystemExit(f"problem_id not found: {args.problem_id}")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    if args.count:
        print(sum(1 for _ in iter_source_bank(args.bank)))
        return
    parser.error("pass --id or --count")


if __name__ == "__main__":
    main()
