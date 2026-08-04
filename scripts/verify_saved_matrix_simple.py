#!/usr/bin/env python3
"""Verify any saved CodeGuide generation matrix with the strict Docker verifier.

Unlike the generation evaluator, this offline verifier ignores platform-specific
manifest fields. It validates only the selected IDs, protocol content, saved
answers and digest-pinned Docker image, then writes per-variant and per-mode data.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_sft_matrix as matrix

DEFAULT_IMAGE = (
    "python:3.11.9-slim-bookworm@sha256:"
    "8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def selected_ids(run_dir: Path) -> list[str]:
    payload = read_json(run_dir / "selection.json")
    ids = payload.get("problem_ids") or payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RuntimeError("selection.json has invalid IDs")
    if len(ids) != len(set(ids)):
        raise RuntimeError("selection.json contains duplicate IDs")
    return ids


def discover_variants(run_dir: Path) -> list[str]:
    return sorted(path.stem for path in (run_dir / "generations").glob("*.jsonl"))


def docker_preflight(image: str, pull_image: bool) -> None:
    if "@sha256:" not in image:
        raise ValueError("Docker image must be pinned by digest")
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        cwd=ROOT,
        check=True,
    )
    if pull_image:
        subprocess.run(["docker", "pull", image], cwd=ROOT, check=True)
    else:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            raise RuntimeError("Pinned image is missing; add --pull-image")


def exception_row(
    problem_id: str,
    variant: str,
    source: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "codeguide-verification-v1",
        "problem_id": problem_id,
        "variant": variant,
        "strict_pass": False,
        "passed_cases": 0,
        "total_cases": 0,
        "pass_rate": 0.0,
        "io_mode": source.get("io_mode"),
        "interface_match": False,
        "template_complete": False,
        "has_code": False,
        "failure_type": "verifier_exception",
        "error": f"{type(exc).__name__}: {exc}",
        "first_failure": None,
        "execution_backend": "docker",
    }


def mode_summary(
    selected: list[str],
    rows: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    subset = [rows[pid] for pid in selected if rows[pid].get("io_mode") == mode]
    if not subset:
        return {"samples": 0, "passed": 0, "pass_at_1": None, "mean_test_pass_rate": None}
    passed = sum(bool(row.get("strict_pass")) for row in subset)
    return {
        "samples": len(subset),
        "passed": passed,
        "pass_at_1": passed / len(subset),
        "mean_test_pass_rate": sum(float(row.get("pass_rate", 0.0)) for row in subset)
        / len(subset),
    }


def verify_variant(
    *,
    variant: str,
    selected: list[str],
    protocol: dict[str, Any],
    source: dict[str, dict[str, Any]],
    run_dir: Path,
    image: str,
    workers: int,
) -> dict[str, Any]:
    generations = matrix.read_jsonl(run_dir / "generations" / f"{variant}.jsonl")
    if set(generations) != set(selected):
        raise RuntimeError(
            f"generation IDs mismatch for {variant}: "
            f"expected={len(selected)} actual={len(generations)}"
        )

    output_path = run_dir / "verification" / f"{variant}.jsonl"
    completed = matrix.read_jsonl(output_path)
    unknown = set(completed) - set(selected)
    if unknown:
        raise RuntimeError(f"verification has unknown IDs for {variant}: {sorted(unknown)[:5]}")
    pending = [pid for pid in selected if pid not in completed]

    required_sections = [str(item) for item in protocol.get("required_sections", [])]
    timeout = float(protocol.get("verification", {}).get("timeout_seconds", 5.0))
    print(
        f"[verify:{variant}] completed={len(completed)} pending={len(pending)} workers={workers}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(
                    matrix.verify_one,
                    pid,
                    generations[pid],
                    source[pid],
                    required_sections=required_sections,
                    container_image=image,
                    timeout=timeout,
                    variant=variant,
                ): pid
                for pid in pending
            }
            for index, future in enumerate(as_completed(future_to_id), 1):
                pid = future_to_id[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = exception_row(pid, variant, source[pid], exc)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed[pid] = row
                if index % 10 == 0 or index == len(pending):
                    print(f"[verify:{variant}] {len(completed)}/{len(selected)}", flush=True)

    overall = matrix.summarize_variant(selected, generations, completed)
    modes = sorted({str(source[pid].get("io_mode")) for pid in selected})
    report = {
        "schema_version": "codeguide-offline-variant-report-v1",
        "variant": variant,
        "container_image": image,
        "overall": overall,
        "by_io_mode": {mode: mode_summary(selected, completed, mode) for mode in modes},
    }
    write_json(run_dir / "reports" / f"strict_{variant}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def summarize(run_dir: Path, variants: list[str]) -> None:
    reports = [read_json(run_dir / "reports" / f"strict_{variant}.json") for variant in variants]
    rows: list[dict[str, Any]] = []
    for report in reports:
        row: dict[str, Any] = {
            "variant": report["variant"],
            "samples": report["overall"]["samples"],
            "passed": report["overall"]["passed"],
            "pass_at_1": report["overall"]["pass_at_1"],
            "mean_test_pass_rate": report["overall"]["mean_test_pass_rate"],
            "average_generated_tokens": report["overall"]["average_generated_tokens"],
            "hit_generation_limit": report["overall"]["hit_generation_limit"],
        }
        for mode, metrics in report["by_io_mode"].items():
            row[f"{mode}_samples"] = metrics["samples"]
            row[f"{mode}_passed"] = metrics["passed"]
            row[f"{mode}_pass_at_1"] = metrics["pass_at_1"]
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["pass_at_1"]), row["variant"]))
    output = run_dir / "reports" / "matrix_summary.json"
    write_json(output, {"schema_version": "codeguide-offline-matrix-v1", "rows": rows})
    csv_path = run_dir / "reports" / "matrix_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--pull-image", action="store_true")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verify_workers <= 0:
        raise ValueError("--verify-workers must be positive")
    run_dir = resolve(args.run_dir)
    protocol_path = resolve(args.protocol_config)
    protocol = matrix.load_protocol(protocol_path)
    selected = selected_ids(run_dir)
    source_path = matrix.resolve_repo_path(str(protocol["dataset"]["source_bank"]))
    source = matrix.load_source_subset(source_path, set(selected))
    variants = args.variant or discover_variants(run_dir)
    if not variants:
        raise RuntimeError("no generation variants found")

    if args.summarize_only:
        summarize(run_dir, variants)
        return

    docker_preflight(args.container_image, args.pull_image)
    for variant in variants:
        verify_variant(
            variant=variant,
            selected=selected,
            protocol=protocol,
            source=source,
            run_dir=run_dir,
            image=args.container_image,
            workers=args.verify_workers,
        )
    summarize(run_dir, variants)


if __name__ == "__main__":
    main()
