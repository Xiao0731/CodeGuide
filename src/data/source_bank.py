"""Streaming access to the compressed verified TACO source bank."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator


def iter_source_bank(path: str | Path) -> Iterator[dict]:
    bank_path = Path(path)
    if bank_path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("install zstandard to read a .zst source bank") from exc
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
    return next(
        (row for row in iter_source_bank(path) if row.get("problem_id") == problem_id),
        None,
    )
