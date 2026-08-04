#!/usr/bin/env python3
"""Run resumable strict-Docker verification for the 35-variant TACO-100 trajectory.

The cloud run was prepared from an unpacked repository and therefore froze
``git_commit: null`` in its manifest. A normal local Git checkout has a commit,
which would make the original evaluator reject the frozen run. This wrapper
preserves the manifest and masks Git discovery only inside evaluator subprocesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = "outputs/eval/taco100_lr_trajectory_bs16"
DEFAULT_PROTOCOL = "configs/eval/taco100_balanced_code_first_v1.yaml"
DEFAULT_IMAGE = (
    "python:3.11.9-slim-bookworm@sha256:"
    "8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)
EXPECTED_STEPS = (5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600)


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


def read_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row.get("problem_id")
            if not problem_id:
                raise RuntimeError(f"missing problem_id at {path}:{line_no}")
            ids.add(str(problem_id))
    return ids


def expected_variants() -> list[str]:
    variants = ["base"]
    variants.extend(f"lr2e4_step{step:03d}" for step in EXPECTED_STEPS)
    variants.append("lr2e4_final611")
    variants.extend(f"lr1e4_step{step:03d}" for step in EXPECTED_STEPS)
    variants.append("lr1e4_final611")
    return variants


def run_checked(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def docker_preflight(image: str, pull: bool) -> None:
    run_checked(["docker", "info", "--format", "{{.ServerVersion}}"])
    if pull:
        run_checked(["docker", "pull", image])
    else:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            raise RuntimeError(
                "pinned Docker image is missing. Rerun with --pull-image or execute:\n"
                f"docker pull {image}"
            )
    run_checked(["docker", "run", "--rm", image, "python", "--version"])


def validate_frozen_inputs(run_dir: Path, protocol_path: Path) -> tuple[list[str], dict[str, Any]]:
    selection_path = run_dir / "selection.json"
    manifest_path = run_dir / "manifest.json"
    for path in (selection_path, manifest_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = read_json(selection_path)
    manifest = read_json(manifest_path)
    selected = selection.get("problem_ids")
    if not isinstance(selected, list) or len(selected) != 100 or len(set(selected)) != 100:
        raise RuntimeError("selection.json must contain exactly 100 unique problem IDs")
    if int(manifest.get("samples", -1)) != 100:
        raise RuntimeError("manifest samples must equal 100")
    if int(manifest.get("generation", {}).get("batch_size", -1)) != 16:
        raise RuntimeError("this downloaded run was expected to use batch_size=16")

    actual_config_hash = sha256_file(protocol_path)
    frozen_config_hash = str(manifest.get("protocol_config_sha256") or "")
    if actual_config_hash != frozen_config_hash:
        raise RuntimeError(
            "local protocol config differs from the cloud-frozen config:\n"
            f"  local : {actual_config_hash}\n"
            f"  frozen: {frozen_config_hash}\n"
            "Restore configs/eval/taco100_balanced_code_first_v1.yaml from the generation commit."
        )

    variants = expected_variants()
    actual_files = sorted(path.stem for path in (run_dir / "generations").glob("*.jsonl"))
    missing_files = sorted(set(variants) - set(actual_files))
    extra_files = sorted(set(actual_files) - set(variants))
    if missing_files or extra_files:
        raise RuntimeError(
            f"generation variant mismatch: missing={missing_files}, extra={extra_files}"
        )

    selected_set = set(selected)
    for variant in variants:
        ids = read_jsonl_ids(run_dir / "generations" / f"{variant}.jsonl")
        if ids != selected_set:
            raise RuntimeError(
                f"generation IDs differ for {variant}: expected={len(selected_set)} actual={len(ids)}"
            )

    source_bank = ROOT / "data/final/taco_verified_source_bank.jsonl.zst"
    if not source_bank.is_file():
        raise FileNotFoundError(
            f"source bank required for strict verification is missing: {source_bank}"
        )

    print(
        json.dumps(
            {
                "frozen_inputs": "accepted",
                "run_dir": str(run_dir),
                "variants": len(variants),
                "answers": len(variants) * len(selected),
                "protocol_sha256": actual_config_hash,
                "manifest_git_commit": manifest.get("git_commit"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return variants, manifest


def evaluator_env(manifest: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    if manifest.get("git_commit") is None:
        env["GIT_DIR"] = str(ROOT / ".codeguide_no_git_for_frozen_eval")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--protocol-config", default=DEFAULT_PROTOCOL)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--pull-image", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="verify only the shared base variant; rerunning without this resumes the remaining variants",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate files and frozen hashes without invoking Docker",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="verify only selected variant(s); may be supplied multiple times",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verify_workers <= 0:
        raise ValueError("--verify-workers must be positive")
    if "@sha256:" not in args.container_image:
        raise ValueError("container image must be pinned by digest")

    run_dir = resolve(args.run_dir)
    protocol_path = resolve(args.protocol_config)
    variants, manifest = validate_frozen_inputs(run_dir, protocol_path)
    if args.check_only:
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

    env = evaluator_env(manifest)
    batch_size = int(manifest["generation"]["batch_size"])
    for index, variant in enumerate(targets, 1):
        print(f"\n=== [{index}/{len(targets)}] verify {variant} ===", flush=True)
        command = [
            sys.executable,
            "scripts/evaluate_sft_matrix.py",
            "--stage",
            "verify",
            "--protocol-config",
            str(protocol_path),
            "--run-dir",
            str(run_dir),
            "--variant",
            variant,
            "--batch-size",
            str(batch_size),
            "--container-image",
            args.container_image,
            "--verify-workers",
            str(args.verify_workers),
        ]
        run_checked(
            command,
            env=env,
            log_path=run_dir / "logs" / f"local_verify_{variant}.log",
        )

    if not args.smoke and not args.variant:
        run_checked(
            [
                sys.executable,
                "scripts/evaluate_sft_matrix.py",
                "--stage",
                "summarize",
                "--protocol-config",
                str(protocol_path),
                "--run-dir",
                str(run_dir),
                "--batch-size",
                str(batch_size),
            ],
            env=env,
            log_path=run_dir / "logs" / "local_summarize.log",
        )
        run_checked(
            [
                sys.executable,
                "scripts/summarize_taco100_lr_trajectory.py",
                "--run-dir",
                str(run_dir),
            ],
            env=env,
            log_path=run_dir / "logs" / "local_trajectory_summary.log",
        )

        selected = set(read_json(run_dir / "selection.json")["problem_ids"])
        acceptance: dict[str, Any] = {
            "schema_version": "codeguide-taco100-local-verification-acceptance-v1",
            "expected_variants": len(variants),
            "expected_rows_per_variant": len(selected),
            "container_image": args.container_image,
            "variants": {},
        }
        for variant in variants:
            path = run_dir / "verification" / f"{variant}.jsonl"
            ids = read_jsonl_ids(path)
            acceptance["variants"][variant] = {
                "rows": len(ids),
                "complete": ids == selected,
            }
        acceptance["complete_variants"] = sum(
            bool(item["complete"]) for item in acceptance["variants"].values()
        )
        output = run_dir / "reports" / "local_verification_acceptance.json"
        output.write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(acceptance, ensure_ascii=False, indent=2))
        if acceptance["complete_variants"] != len(variants):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
