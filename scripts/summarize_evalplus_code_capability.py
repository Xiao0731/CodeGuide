#!/usr/bin/env python3
"""Summarize EvalPlus Pass@1 and frozen generation statistics."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

VARIANTS = (
    "base",
    "mixed_lr1e4_step020",
    "mixed_lr1e4_step200",
)
DATASETS = ("humaneval", "mbpp")
EXPECTED = {"humaneval": 164, "mbpp": 378}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_pass_dict(value: str) -> float:
    payload = ast.literal_eval(value.strip())
    if not isinstance(payload, dict) or "pass@1" not in payload:
        raise RuntimeError(f"invalid EvalPlus metric dictionary: {value}")
    return float(payload["pass@1"])


def parse_evalplus_log(path: Path) -> tuple[float, float]:
    """Parse the stable EvalPlus v0.3.1 CLI output."""
    text = path.read_text(encoding="utf-8", errors="replace")
    base_match = re.search(
        r"(?m)^Base\s*\r?\n\s*(\{[^\r\n]*['\"]pass@1['\"][^\r\n]*\})",
        text,
    )
    plus_match = re.search(
        r"(?m)^Base \+ Extra\s*\r?\n\s*(\{[^\r\n]*['\"]pass@1['\"][^\r\n]*\})",
        text,
    )
    if base_match is None or plus_match is None:
        raise RuntimeError(
            f"cannot parse EvalPlus Base/Base+Extra metrics from {path}"
        )
    return (
        parse_pass_dict(base_match.group(1)),
        parse_pass_dict(plus_match.group(1)),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def echarts_option(wide_rows: list[dict[str, Any]]) -> str:
    labels = {
        "base": "Base",
        "mixed_lr1e4_step020": "SFT step20",
        "mixed_lr1e4_step200": "SFT step200",
    }
    metric_fields = (
        ("HumanEval", "humaneval_pass_at_1"),
        ("HumanEval+", "humaneval_plus_pass_at_1"),
        ("MBPP", "mbpp_pass_at_1"),
        ("MBPP+", "mbpp_plus_pass_at_1"),
    )
    option = {
        "title": {
            "text": "CodeGuide 外源代码能力评测",
            "subtext": "EvalPlus v0.3.1；同提示、同精度、同解码、同执行器",
            "left": "center",
        },
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 48},
        "grid": {"left": 65, "right": 30, "top": 95, "bottom": 55},
        "xAxis": {
            "type": "category",
            "data": [labels[row["variant"]] for row in wide_rows],
        },
        "yAxis": {
            "type": "value",
            "name": "Pass@1",
            "min": 0,
            "max": 100,
            "axisLabel": {"formatter": "{value}%"},
        },
        "series": [
            {
                "name": label,
                "type": "bar",
                "data": [
                    round(float(row[field]) * 100, 2)
                    for row in wide_rows
                ],
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{c}%",
                },
            }
            for label, field in metric_fields
        ],
    }
    return "option = " + json.dumps(option, ensure_ascii=False, indent=2) + ";\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="outputs/eval/evalplus_code_capability_v1",
    )
    args = parser.parse_args()
    root = Path(args.run_root)

    long_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            log_path = root / "evaluation_logs" / f"{dataset}_{variant}.log"
            stats_path = root / "stats" / dataset / f"{variant}.json"
            sample_path = root / "samples" / dataset / f"{variant}.jsonl"
            for path in (log_path, stats_path, sample_path):
                if not path.is_file():
                    raise FileNotFoundError(path)

            base_pass, plus_pass = parse_evalplus_log(log_path)
            stats = read_json(stats_path)
            samples = int(stats["samples"])
            if samples != EXPECTED[dataset]:
                raise RuntimeError(
                    f"{dataset}/{variant} sample mismatch: "
                    f"expected={EXPECTED[dataset]} actual={samples}"
                )

            long_rows.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "samples": samples,
                    "base_pass_at_1": base_pass,
                    "plus_pass_at_1": plus_pass,
                    "average_generated_tokens": float(
                        stats["average_generated_tokens"]
                    ),
                    "hit_generation_limit": int(
                        stats["hit_generation_limit"]
                    ),
                    "syntax_failures": int(stats["syntax_failures"]),
                    "syntax_failure_rate": (
                        int(stats["syntax_failures"]) / samples
                    ),
                    "average_solution_characters": float(
                        stats["average_solution_characters"]
                    ),
                    "sample_sha256": str(stats["sample_sha256"]),
                }
            )

    by_key = {
        (row["variant"], row["dataset"]): row
        for row in long_rows
    }
    reproduced_base = {
        dataset: by_key[("base", dataset)]
        for dataset in DATASETS
    }

    for row in long_rows:
        base_row = reproduced_base[row["dataset"]]
        row["base_pass_delta_vs_reproduced_base"] = (
            float(row["base_pass_at_1"])
            - float(base_row["base_pass_at_1"])
        )
        row["plus_pass_delta_vs_reproduced_base"] = (
            float(row["plus_pass_at_1"])
            - float(base_row["plus_pass_at_1"])
        )

    wide_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        human = by_key[(variant, "humaneval")]
        mbpp = by_key[(variant, "mbpp")]
        wide_rows.append(
            {
                "variant": variant,
                "humaneval_pass_at_1": human["base_pass_at_1"],
                "humaneval_plus_pass_at_1": human["plus_pass_at_1"],
                "mbpp_pass_at_1": mbpp["base_pass_at_1"],
                "mbpp_plus_pass_at_1": mbpp["plus_pass_at_1"],
                "humaneval_delta_vs_base": (
                    human["base_pass_delta_vs_reproduced_base"]
                ),
                "humaneval_plus_delta_vs_base": (
                    human["plus_pass_delta_vs_reproduced_base"]
                ),
                "mbpp_delta_vs_base": (
                    mbpp["base_pass_delta_vs_reproduced_base"]
                ),
                "mbpp_plus_delta_vs_base": (
                    mbpp["plus_pass_delta_vs_reproduced_base"]
                ),
                "humaneval_average_generated_tokens": (
                    human["average_generated_tokens"]
                ),
                "mbpp_average_generated_tokens": (
                    mbpp["average_generated_tokens"]
                ),
                "total_hit_generation_limit": (
                    int(human["hit_generation_limit"])
                    + int(mbpp["hit_generation_limit"])
                ),
                "total_syntax_failures": (
                    int(human["syntax_failures"])
                    + int(mbpp["syntax_failures"])
                ),
            }
        )

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "evalplus_code_summary_long.csv", long_rows)
    write_csv(report_dir / "evalplus_code_summary_wide.csv", wide_rows)
    write_json(
        report_dir / "evalplus_code_summary.json",
        {
            "schema_version": "codeguide-evalplus-code-summary-v1",
            "evaluation_module": "code_capability",
            "teaching_metrics_in_scope": False,
            "dataset_counts": EXPECTED,
            "comparison_rule": (
                "All experimental deltas use the reproduced Base under the "
                "same frozen protocol."
            ),
            "long_rows": long_rows,
            "wide_rows": wide_rows,
        },
    )
    (report_dir / "evalplus_code_echarts_option.js").write_text(
        echarts_option(wide_rows),
        encoding="utf-8",
    )

    print(json.dumps(wide_rows, ensure_ascii=False, indent=2))
    print(f"[done] {report_dir / 'evalplus_code_summary_wide.csv'}")


if __name__ == "__main__":
    main()
