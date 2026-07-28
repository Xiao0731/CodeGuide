#!/usr/bin/env python3
"""Re-execute generated SFT code and fail if any accepted label is wrong."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.code_validator import extract_code
from src.reward.execution import verify_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["docker", "subprocess"],
        default="docker",
    )
    parser.add_argument("--container-image", default="")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    totals = {
        "records": 0,
        "full_pass": 0,
        "failed": 0,
        "unsupported": 0,
        "missing_code": 0,
    }
    failures: list[dict] = []
    with args.data.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            totals["records"] += 1
            record = json.loads(line)
            assistant = next(
                (
                    message.get("content", "")
                    for message in record.get("messages", [])
                    if message.get("role") == "assistant"
                ),
                "",
            )
            code = extract_code(assistant)
            if not code:
                totals["missing_code"] += 1
                failures.append(
                    {
                        "line": line_number,
                        "id": record.get("id"),
                        "error": "missing code block",
                    }
                )
                continue

            result = verify_code(
                code,
                record.get("metadata", {}),
                timeout=args.timeout,
                backend=args.backend,
                container_image=args.container_image or None,
            )
            if result.unsupported:
                totals["unsupported"] += 1
            if (
                not result.unsupported
                and not result.error
                and result.total_cases > 0
                and result.pass_rate >= 1.0
            ):
                totals["full_pass"] += 1
            else:
                totals["failed"] += 1
                failures.append(
                    {
                        "line": line_number,
                        "id": record.get("id"),
                        "pass_rate": result.pass_rate,
                        "error": result.error,
                        "first_failure": result.first_failure,
                        "unsupported": result.unsupported,
                    }
                )

    print(json.dumps({"summary": totals, "failures": failures}, ensure_ascii=False, indent=2))
    if totals["records"] == 0 or totals["full_pass"] != totals["records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
