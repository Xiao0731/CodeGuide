#!/usr/bin/env python3
"""Randomly reverify source-bank references without reading TACO parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.source_bank_io import iter_source_bank
from src.reward.execution import verify_code

DEFAULT_IMAGE = (
    "python:3.11.9-slim-bookworm@"
    "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("data/final/taco_verified_source_bank.jsonl.zst"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/source_bank_verification.json"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    args = parser.parse_args()

    records = list(iter_source_bank(args.bank))
    rng = random.Random(args.seed)
    sampled = rng.sample(records, min(args.sample_size, len(records)))
    results = []
    for index, record in enumerate(sampled, 1):
        reference = str(record.get("reference_solution") or "")
        actual_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest()
        hash_ok = actual_hash == record.get("reference_hash")
        metadata = {
            "io_mode": record.get("io_mode"),
            "fn_name": record.get("fn_name"),
            "starter_code": record.get("starter_code") or "",
            "test_cases": record.get("test_cases") or [],
            "tags": record.get("tags") or [],
            "raw_tags": record.get("raw_tags") or [],
            "skill_types": record.get("skill_types") or [],
        }
        verification = verify_code(
            reference,
            metadata,
            timeout=args.timeout,
            backend="docker",
            container_image=args.container_image,
        )
        passed = bool(
            hash_ok
            and not verification.unsupported
            and not verification.error
            and verification.total_cases > 0
            and verification.pass_rate >= 1.0
        )
        result = {
            "problem_id": record.get("problem_id"),
            "io_mode": record.get("io_mode"),
            "difficulty": record.get("difficulty"),
            "reference_hash_match": hash_ok,
            "passed": passed,
            "pass_rate": verification.pass_rate,
            "passed_cases": verification.passed_cases,
            "total_cases": verification.total_cases,
            "unsupported": verification.unsupported,
            "error": verification.error,
            "first_failure": verification.first_failure,
        }
        results.append(result)
        print(f"[{index}/{len(sampled)}] {record.get('problem_id')} pass={passed} rate={verification.pass_rate:.3f}")

    payload = {
        "schema_version": "codeguide-source-bank-verification-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bank": str(args.bank).replace("\\", "/"),
        "seed": args.seed,
        "sample_size": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "container_image": args.container_image,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if payload["failed"]:
        raise SystemExit(f"Source-bank sample verification failed: {payload['failed']}/{len(results)}")


if __name__ == "__main__":
    main()
