#!/usr/bin/env python3
"""Evaluate a text-routed two-expert system on one shared frozen problem set.

The router prediction decides which already-verified expert answer is selected.
No metadata/oracle routing score is computed. The report includes classifier
accuracy, routed overall/per-mode Pass@1, and whether the routed system beats a
specified mixed-SFT baseline on every claimed dimension.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row.get("problem_id")
            if not problem_id:
                raise ValueError(f"missing problem_id at {path}:{line_no}")
            rows[str(problem_id)] = row
    return rows


def selection_ids(path: Path) -> list[str]:
    payload = read_json(path)
    ids = payload.get("problem_ids") or payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RuntimeError("invalid selection IDs")
    return ids


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "passed": 0, "pass_at_1": None, "mean_test_pass_rate": None}
    passed = sum(bool(row["selected_strict_pass"]) for row in rows)
    return {
        "samples": len(rows),
        "passed": passed,
        "pass_at_1": passed / len(rows),
        "mean_test_pass_rate": sum(float(row["selected_pass_rate"]) for row in rows)
        / len(rows),
    }


def verification_mode_metrics(
    selected: list[str],
    rows: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    subset = [rows[pid] for pid in selected if rows[pid].get("io_mode") == mode]
    if not subset:
        return {"samples": 0, "passed": 0, "pass_at_1": None}
    passed = sum(bool(row.get("strict_pass")) for row in subset)
    return {"samples": len(subset), "passed": passed, "pass_at_1": passed / len(subset)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--router-predictions", required=True)
    parser.add_argument("--standard-verification", required=True)
    parser.add_argument("--call-verification", required=True)
    parser.add_argument("--mixed-verification", required=True)
    parser.add_argument("--output-dir", default="outputs/router/routed_expert_eval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = selection_ids(Path(args.selection))
    selected_set = set(selected)
    router = read_jsonl(Path(args.router_predictions))
    standard = read_jsonl(Path(args.standard_verification))
    call = read_jsonl(Path(args.call_verification))
    mixed = read_jsonl(Path(args.mixed_verification))

    for name, rows in {
        "router": router,
        "standard expert": standard,
        "call expert": call,
        "mixed baseline": mixed,
    }.items():
        if set(rows) != selected_set:
            raise RuntimeError(
                f"{name} IDs differ from selection: expected={len(selected_set)} actual={len(rows)}"
            )

    routed_rows: list[dict[str, Any]] = []
    for pid in selected:
        prediction = router[pid]
        predicted_mode = str(prediction["predicted_io_mode"])
        true_mode = str(prediction["true_io_mode"])
        if predicted_mode == "standard_input":
            selected_verification = standard[pid]
            selected_expert = "standard_input"
        elif predicted_mode == "call_based":
            selected_verification = call[pid]
            selected_expert = "call_based"
        else:
            raise RuntimeError(f"invalid predicted mode for {pid}: {predicted_mode}")
        if str(selected_verification.get("io_mode")) != true_mode:
            raise RuntimeError(f"true mode mismatch across artifacts for {pid}")

        routed_rows.append(
            {
                "problem_id": pid,
                "true_io_mode": true_mode,
                "predicted_io_mode": predicted_mode,
                "router_correct": predicted_mode == true_mode,
                "router_confidence": float(prediction.get("confidence", 0.0)),
                "selected_expert": selected_expert,
                "selected_strict_pass": bool(selected_verification.get("strict_pass")),
                "selected_pass_rate": float(selected_verification.get("pass_rate", 0.0)),
                "mixed_strict_pass": bool(mixed[pid].get("strict_pass")),
                "standard_expert_strict_pass": bool(standard[pid].get("strict_pass")),
                "call_expert_strict_pass": bool(call[pid].get("strict_pass")),
            }
        )

    by_true_mode = {
        mode: subset_metrics([row for row in routed_rows if row["true_io_mode"] == mode])
        for mode in ("standard_input", "call_based")
    }
    overall = subset_metrics(routed_rows)
    correct_route = subset_metrics([row for row in routed_rows if row["router_correct"]])
    wrong_route = subset_metrics([row for row in routed_rows if not row["router_correct"]])
    router_accuracy = sum(bool(row["router_correct"]) for row in routed_rows) / len(routed_rows)

    mixed_modes = {
        mode: verification_mode_metrics(selected, mixed, mode)
        for mode in ("standard_input", "call_based")
    }
    mixed_passed = sum(bool(mixed[pid].get("strict_pass")) for pid in selected)
    mixed_overall = {
        "samples": len(selected),
        "passed": mixed_passed,
        "pass_at_1": mixed_passed / len(selected),
        "by_io_mode": mixed_modes,
    }
    specialist_own_mode = {
        "standard_input": verification_mode_metrics(selected, standard, "standard_input"),
        "call_based": verification_mode_metrics(selected, call, "call_based"),
    }

    gates = {
        "standard_expert_beats_mixed_standard": (
            specialist_own_mode["standard_input"]["pass_at_1"]
            > mixed_modes["standard_input"]["pass_at_1"]
        ),
        "call_expert_beats_mixed_call": (
            specialist_own_mode["call_based"]["pass_at_1"]
            > mixed_modes["call_based"]["pass_at_1"]
        ),
        "text_routed_overall_beats_mixed": overall["pass_at_1"] > mixed_overall["pass_at_1"],
        "text_routed_standard_beats_mixed": (
            by_true_mode["standard_input"]["pass_at_1"]
            > mixed_modes["standard_input"]["pass_at_1"]
        ),
        "text_routed_call_beats_mixed": (
            by_true_mode["call_based"]["pass_at_1"]
            > mixed_modes["call_based"]["pass_at_1"]
        ),
    }
    gates["moe_like_claim_supported"] = all(gates.values())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "routed_predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(routed_rows[0]))
        writer.writeheader()
        writer.writerows(routed_rows)

    payload = {
        "schema_version": "codeguide-text-routed-expert-report-v1",
        "samples": len(selected),
        "true_mode_distribution": dict(Counter(row["true_io_mode"] for row in routed_rows)),
        "predicted_mode_distribution": dict(
            Counter(row["predicted_io_mode"] for row in routed_rows)
        ),
        "router_accuracy": router_accuracy,
        "router_correct_count": sum(bool(row["router_correct"]) for row in routed_rows),
        "router_wrong_count": sum(not bool(row["router_correct"]) for row in routed_rows),
        "routed": {
            "overall": overall,
            "by_true_io_mode": by_true_mode,
            "correct_route_subset": correct_route,
            "wrong_route_subset": wrong_route,
        },
        "specialist_own_mode": specialist_own_mode,
        "mixed_baseline": mixed_overall,
        "claim_gates": gates,
        "row_level_csv": str(csv_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
