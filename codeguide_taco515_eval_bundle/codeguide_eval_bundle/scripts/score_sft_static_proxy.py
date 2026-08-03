#!/usr/bin/env python3
"""Offline external AST-proxy rescoring for saved TACO matrix generations.

This script NEVER loads a language model. It reads saved generation JSONL files,
extracts the contract-aware code block, applies the external project's exact
five-dimensional proxy, and optionally joins strict Docker verification.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.source_bank_io import iter_source_bank
from src.data.code_validator import extract_code
from src.evaluation.static_proxy import score_external_static_proxy


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            problem_id = item.get("problem_id")
            if not problem_id:
                raise ValueError(f"missing problem_id at {path}:{line_no}")
            rows[str(problem_id)] = item
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rankdata(values: list[float]) -> list[float]:
    """Average ranks for ties; standard-library replacement for scipy.rankdata."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def discover_variants(run_dir: Path) -> list[str]:
    return sorted(path.stem for path in (run_dir / "generations").glob("*.jsonl"))


def load_source_subset(path: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    source = {
        item["problem_id"]: item
        for item in iter_source_bank(path)
        if item.get("problem_id") in selected
    }
    missing = selected - set(source)
    if missing:
        raise RuntimeError(f"source bank misses selected IDs: {sorted(missing)[:5]}")
    return source


def summarize_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores = [float(row["static_score"]) for row in rows]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "mean_static_score": statistics.fmean(scores) if scores else None,
        "static_acceptable": sum(score >= threshold for score in scores),
        "static_acceptable_rate": (
            sum(score >= threshold for score in scores) / len(scores)
            if scores else None
        ),
        "static_full_score": sum(score == 1.0 for score in scores),
        "syntax_valid": sum(bool(row["syntax_valid"]) for row in rows),
    }

    strict_rows = [row for row in rows if row.get("strict_pass") is not None]
    if strict_rows:
        strict_pass = [1.0 if row["strict_pass"] else 0.0 for row in strict_rows]
        pass_rates = [float(row.get("docker_pass_rate") or 0.0) for row in strict_rows]
        strict_scores = [float(row["static_score"]) for row in strict_rows]
        false_positive = sum(
            row["static_score"] >= threshold and not row["strict_pass"]
            for row in strict_rows
        )
        summary.update(
            {
                "docker_pass_at_1": statistics.fmean(strict_pass),
                "mean_docker_test_pass_rate": statistics.fmean(pass_rates),
                "static_false_positive": false_positive,
                "static_false_positive_rate_among_all": false_positive
                / len(strict_rows),
                "spearman_static_vs_docker_pass_rate": spearman(
                    strict_scores, pass_rates
                ),
            }
        )
    return summary


def score_variant(
    run_dir: Path,
    source: dict[str, dict[str, Any]],
    selected: list[str],
    variant: str,
    threshold: float,
) -> dict[str, Any]:
    generations = read_jsonl(run_dir / "generations" / f"{variant}.jsonl")
    missing = [problem_id for problem_id in selected if problem_id not in generations]
    if missing:
        raise RuntimeError(f"{variant} generation incomplete: {missing[:5]}")
    verification = read_jsonl(run_dir / "verification" / f"{variant}.jsonl")

    output_path = run_dir / "static_proxy" / f"{variant}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for problem_id in selected:
            generation = generations[problem_id]
            item = source[problem_id]

            # 【改动核心】这就是用户问到的那段循环，但完整脚本还必须负责
            # 读取文件、接口感知代码提取、关联 Docker 结果、落盘和汇总。
            code = extract_code(
                str(generation.get("text") or ""),
                io_mode=item.get("io_mode"),
                fn_name=item.get("fn_name"),
                starter_code=item.get("starter_code"),
            )
            proxy = score_external_static_proxy(code or "")
            strict = verification.get(problem_id)
            row = {
                "schema_version": "codeguide-external-static-proxy-v1",
                "problem_id": problem_id,
                "variant": variant,
                "io_mode": item.get("io_mode"),
                "static_score": proxy.score,
                "static_acceptable": proxy.score >= threshold,
                **{
                    key: value
                    for key, value in proxy.to_dict().items()
                    if key != "score"
                },
                "strict_pass": strict.get("strict_pass") if strict else None,
                "docker_pass_rate": strict.get("pass_rate") if strict else None,
                "docker_failure_type": strict.get("failure_type") if strict else None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    overall = summarize_rows(rows, threshold)
    by_io_mode: dict[str, Any] = {}
    for io_mode in sorted({str(row.get("io_mode") or "unknown") for row in rows}):
        by_io_mode[io_mode] = summarize_rows(
            [row for row in rows if str(row.get("io_mode") or "unknown") == io_mode],
            threshold,
        )
    return {"overall": overall, "by_io_mode": by_io_mode}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/sft/taco515_compact_code_first_v1",
    )
    parser.add_argument(
        "--source-bank",
        default="data/final/taco_verified_source_bank.jsonl.zst",
    )
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    source_bank = Path(args.source_bank)
    if not source_bank.is_absolute():
        source_bank = ROOT / source_bank

    selection_path = run_dir / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("missing selection.json; prepare matrix evaluation first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["problem_ids"]
    variants = args.variants or discover_variants(run_dir)
    if not variants:
        raise RuntimeError("no variants found")

    source = load_source_subset(source_bank, set(selected))
    report = {
        "schema_version": "codeguide-external-static-proxy-summary-v1",
        "warning": "Static proxy is not correctness and is not Pass@1.",
        "threshold": args.threshold,
        "samples": len(selected),
        "variants": {},
    }
    for variant in variants:
        report["variants"][variant] = score_variant(
            run_dir,
            source,
            selected,
            variant,
            args.threshold,
        )
        print(f"[static-proxy] scored {variant}", flush=True)

    output = run_dir / "reports" / "static_proxy_summary.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
