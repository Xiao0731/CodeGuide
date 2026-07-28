#!/usr/bin/env python3
"""Frozen-manifest validation entry point for model evaluation.

G0 uses ``--validate-only`` to prove that evaluation cannot silently read a
different split. Model inference and metric computation are added only after
the evaluation suites are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "split_name",
        "records_path",
        "records_sha256",
        "record_count",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest missing fields: {missing}")

    records_path = (path.parent / manifest["records_path"]).resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    actual_hash = _sha256(records_path)
    if actual_hash != manifest["records_sha256"]:
        raise ValueError(
            f"records hash mismatch: expected {manifest['records_sha256']}, got {actual_hash}"
        )
    with records_path.open(encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    if count != int(manifest["record_count"]):
        raise ValueError(
            f"record_count mismatch: expected {manifest['record_count']}, got {count}"
        )
    return {**manifest, "resolved_records_path": str(records_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    validated = validate_manifest(args.manifest.resolve())
    print(json.dumps(validated, ensure_ascii=False, indent=2))
    if not args.validate_only:
        raise SystemExit(
            "model evaluation is intentionally blocked until ExplainBench/"
            "TutorBench and the model adapter are frozen"
        )


if __name__ == "__main__":
    main()
