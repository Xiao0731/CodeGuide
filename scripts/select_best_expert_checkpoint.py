#!/usr/bin/env python3
"""Rank mode-pure expert checkpoints from strict execution reports.

Selection is lexicographic and data-driven:
1. more strict passes;
2. higher mean test-case pass rate;
3. fewer runtime/timeout failures;
4. fewer missing-code failures;
5. shorter average generation;
6. earlier optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"(?:step|checkpoint)[_-]?(\d+)")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def step_from_variant(variant: str) -> int:
    match = STEP_RE.search(variant)
    if not match:
        raise ValueError(f"cannot infer optimizer step from variant: {variant}")
    return int(match.group(1))


def normalize_report(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    variant = str(payload["variant"])
    metrics = payload.get("overall") or payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"invalid metrics in {path}")
    failures = metrics.get("failure_types") or {}
    return {
        "variant": variant,
        "step": step_from_variant(variant),
        "passed": int(metrics["passed"]),
        "pass_at_1": float(metrics["pass_at_1"]),
        "mean_test_pass_rate": float(metrics["mean_test_pass_rate"]),
        "runtime_or_timeout": int(failures.get("runtime_or_timeout", 0)),
        "missing_code": int(failures.get("missing_code", 0)),
        "average_generated_tokens": float(metrics["average_generated_tokens"]),
        "report": str(path),
    }


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["passed"],
        -row["mean_test_pass_rate"],
        row["runtime_or_timeout"],
        row["missing_code"],
        row["average_generated_tokens"],
        row["step"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=("standard_input", "call_based"), required=True)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    report_paths = sorted((run_dir / "reports").glob("strict_expert_*step*.json"))
    rows = [normalize_report(path) for path in report_paths]
    if not rows:
        raise RuntimeError(f"no expert checkpoint reports found under {run_dir / 'reports'}")
    rows.sort(key=rank_key)
    best = rows[0]

    output = Path(args.output) if args.output else run_dir / "reports" / "best_checkpoint.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "codeguide-expert-checkpoint-selection-v1",
        "mode": args.mode,
        "selection_rule": [
            "max strict passed",
            "max mean test pass rate",
            "min runtime_or_timeout",
            "min missing_code",
            "min average generated tokens",
            "min optimizer step",
        ],
        "best": best,
        "ranking": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
