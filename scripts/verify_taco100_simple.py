#!/usr/bin/env python3
"""Minimal, resumable Docker verification for the downloaded TACO-100 trajectory.

This script treats evaluation outputs as the primary artifact. It checks only
semantically relevant invariants, verifies saved answers with the existing
Docker verifier, and writes trajectory tables for checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_sft_matrix as matrix

DEFAULT_RUN_DIR = "outputs/eval/taco100_lr_trajectory_bs16"
DEFAULT_PROTOCOL = "configs/eval/taco100_balanced_code_first_v1_cloud_b3714792.yaml"
DEFAULT_IMAGE = (
    "python:3.11.9-slim-bookworm@sha256:"
    "8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)
STEPS = (5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def expected_variants() -> list[str]:
    variants = ["base"]
    variants.extend(f"lr2e4_step{step:03d}" for step in STEPS)
    variants.append("lr2e4_final611")
    variants.extend(f"lr1e4_step{step:03d}" for step in STEPS)
    variants.append("lr1e4_final611")
    return variants


def validate_inputs(
    run_dir: Path,
    protocol_path: Path,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any]]:
    selection_path = run_dir / "selection.json"
    manifest_path = run_dir / "manifest.json"
    for path in (selection_path, manifest_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = read_json(selection_path)
    manifest = read_json(manifest_path)
    selected = selection.get("problem_ids")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise RuntimeError("selection.json has invalid problem_ids")
    if len(selected) != 100 or len(set(selected)) != 100:
        raise RuntimeError(
            f"TACO-100 must contain 100 unique IDs: rows={len(selected)} unique={len(set(selected))}"
        )

    actual_hash = sha256_file(protocol_path)
    frozen_hash = str(manifest.get("protocol_config_sha256") or "")
    if actual_hash != frozen_hash:
        raise RuntimeError(
            "protocol content differs from the generation run:\n"
            f"  local : {actual_hash}\n"
            f"  frozen: {frozen_hash}"
        )

    protocol = matrix.load_protocol(protocol_path)
    variants = expected_variants()
    actual_variants = sorted(
        path.stem for path in (run_dir / "generations").glob("*.jsonl")
    )
    missing = sorted(set(variants) - set(actual_variants))
    extra = sorted(set(actual_variants) - set(variants))
    if missing or extra:
        raise RuntimeError(f"generation files mismatch: missing={missing}, extra={extra}")

    selected_set = set(selected)
    for variant in variants:
        rows = matrix.read_jsonl(run_dir / "generations" / f"{variant}.jsonl")
        if set(rows) != selected_set:
            raise RuntimeError(
                f"{variant} generation IDs mismatch: expected=100 actual={len(rows)}"
            )

    source_path = matrix.resolve_repo_path(str(protocol["dataset"]["source_bank"]))
    source = matrix.load_source_subset(source_path, selected_set)
    mode_counts = Counter(str(source[pid].get("io_mode")) for pid in selected)
    if mode_counts != Counter({"standard_input": 50, "call_based": 50}):
        raise RuntimeError(f"TACO-100 is not balanced: {dict(mode_counts)}")

    print(
        json.dumps(
            {
                "inputs": "accepted",
                "variants": len(variants),
                "answers": len(variants) * len(selected),
                "io_modes": dict(mode_counts),
                "protocol_sha256": actual_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return selected, variants, protocol, source


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
            raise RuntimeError(
                "Pinned Docker image is not available locally. Add --pull-image."
            )
    subprocess.run(
        ["docker", "run", "--rm", image, "python", "--version"],
        cwd=ROOT,
        check=True,
    )


def verifier_exception_row(
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
    output_path = run_dir / "verification" / f"{variant}.jsonl"
    completed = matrix.read_jsonl(output_path)
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
                    problem_id,
                    generations[problem_id],
                    source[problem_id],
                    required_sections=required_sections,
                    container_image=image,
                    timeout=timeout,
                    variant=variant,
                ): problem_id
                for problem_id in pending
            }
            for index, future in enumerate(as_completed(future_to_id), 1):
                problem_id = future_to_id[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = verifier_exception_row(
                        problem_id, variant, source[problem_id], exc
                    )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed[problem_id] = row
                if index % 10 == 0 or index == len(pending):
                    print(
                        f"[verify:{variant}] {len(completed)}/{len(selected)}",
                        flush=True,
                    )

    if set(completed) != set(selected):
        raise RuntimeError(f"verification incomplete for {variant}")

    report = {
        "schema_version": "codeguide-strict-variant-report-v1",
        "protocol_name": protocol["protocol_name"],
        "variant": variant,
        "container_image": image,
        "metrics": matrix.summarize_variant(selected, generations, completed),
    }
    write_json(run_dir / "reports" / f"strict_{variant}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def variant_axis(variant: str) -> tuple[str, int]:
    if variant == "base":
        return "base", 0
    family = "1e-4" if variant.startswith("lr1e4_") else "2e-4"
    if "final611" in variant:
        return family, 611
    return family, int(variant.rsplit("step", 1)[1])


def mode_metrics(
    selected: list[str],
    verifications: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    rows = [verifications[pid] for pid in selected if verifications[pid]["io_mode"] == mode]
    passed = sum(bool(row["strict_pass"]) for row in rows)
    return {
        "samples": len(rows),
        "passed": passed,
        "pass_at_1": passed / len(rows),
        "mean_test_pass_rate": sum(float(row["pass_rate"]) for row in rows) / len(rows),
    }


def summarize_trajectory(
    *,
    selected: list[str],
    variants: list[str],
    run_dir: Path,
    image: str,
) -> None:
    matrix_payload: dict[str, Any] = {
        "schema_version": "codeguide-checkpoint-matrix-v1",
        "samples": len(selected),
        "container_image": image,
        "variants": {},
    }
    rows: list[dict[str, Any]] = []
    pass_sets: dict[str, set[str]] = {}

    for variant in variants:
        generations = matrix.read_jsonl(run_dir / "generations" / f"{variant}.jsonl")
        verifications = matrix.read_jsonl(run_dir / "verification" / f"{variant}.jsonl")
        if set(verifications) != set(selected):
            raise RuntimeError(f"verification is incomplete for {variant}")
        overall = matrix.summarize_variant(selected, generations, verifications)
        standard = mode_metrics(selected, verifications, "standard_input")
        call = mode_metrics(selected, verifications, "call_based")
        family, step = variant_axis(variant)
        pass_sets[variant] = {
            pid for pid in selected if bool(verifications[pid].get("strict_pass"))
        }
        payload = {
            "status": "complete",
            **overall,
            "standard_input": standard,
            "call_based": call,
        }
        matrix_payload["variants"][variant] = payload
        rows.append(
            {
                "variant": variant,
                "learning_rate": family,
                "step": step,
                "overall_passed": overall["passed"],
                "overall_pass_at_1": overall["pass_at_1"],
                "standard_passed": standard["passed"],
                "standard_pass_at_1": standard["pass_at_1"],
                "call_passed": call["passed"],
                "call_pass_at_1": call["pass_at_1"],
                "mean_test_pass_rate": overall["mean_test_pass_rate"],
                "template_complete": overall["template_complete"],
                "average_generated_tokens": overall["average_generated_tokens"],
                "hit_generation_limit": overall["hit_generation_limit"],
                "failure_types": json.dumps(overall["failure_types"], ensure_ascii=False),
            }
        )

    base_pass = pass_sets["base"]
    for variant, passed in pass_sets.items():
        payload = matrix_payload["variants"][variant]
        payload["rescued_vs_base"] = len(passed - base_pass)
        payload["lost_vs_base"] = len(base_pass - passed)
        payload["rescued_problem_ids"] = sorted(passed - base_pass)
        payload["lost_problem_ids"] = sorted(base_pass - passed)

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "checkpoint_matrix.json", matrix_payload)

    rows.sort(key=lambda row: (row["learning_rate"], int(row["step"])))
    csv_path = reports_dir / "taco100_lr_trajectory.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def best(key: str) -> dict[str, Any]:
        ranked = sorted(rows, key=lambda row: (-float(row[key]), row["variant"]))
        return {
            "metric": key,
            "best": ranked[0],
            "top5": ranked[:5],
        }

    candidates = {
        "schema_version": "codeguide-taco100-checkpoint-candidates-v1",
        "best_overall": best("overall_pass_at_1"),
        "best_standard_input": best("standard_pass_at_1"),
        "best_call_based": best("call_pass_at_1"),
    }
    write_json(reports_dir / "taco100_checkpoint_candidates.json", candidates)
    print(json.dumps(candidates, ensure_ascii=False, indent=2))
    print(f"[done] trajectory CSV: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--protocol-config", default=DEFAULT_PROTOCOL)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--pull-image", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="verify only base")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        help="verify only selected variant(s); may be repeated",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verify_workers <= 0:
        raise ValueError("--verify-workers must be positive")

    run_dir = resolve(args.run_dir)
    protocol_path = resolve(args.protocol_config)
    selected, variants, protocol, source = validate_inputs(run_dir, protocol_path)
    if args.check_only:
        return

    if args.summarize_only:
        summarize_trajectory(
            selected=selected,
            variants=variants,
            run_dir=run_dir,
            image=args.container_image,
        )
        return

    docker_preflight(args.container_image, args.pull_image)
    if args.variant:
        unknown = sorted(set(args.variant) - set(variants))
        if unknown:
            raise ValueError(f"unknown variants: {unknown}")
        targets = args.variant
    elif args.smoke:
        targets = ["base"]
    else:
        targets = variants

    for index, variant in enumerate(targets, 1):
        print(f"\n=== [{index}/{len(targets)}] {variant} ===", flush=True)
        verify_variant(
            variant=variant,
            selected=selected,
            protocol=protocol,
            source=source,
            run_dir=run_dir,
            image=args.container_image,
            workers=args.verify_workers,
        )

    if not args.smoke and not args.variant:
        summarize_trajectory(
            selected=selected,
            variants=variants,
            run_dir=run_dir,
            image=args.container_image,
        )


if __name__ == "__main__":
    main()
