#!/usr/bin/env python3
"""Record a reproducible, secret-free CodeGuide environment snapshot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "unsloth",
    "vllm",
    "omegaconf",
    "PyYAML",
    "pytest",
]


def _command_output(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False}
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_snapshot() -> dict:
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            # Keep committed snapshots portable and free of usernames/workspace paths.
            "executable_name": Path(sys.executable).name,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "docker": _command_output(["docker", "version", "--format", "{{json .}}"]),
        "git": _command_output(["git", "status", "--short", "--branch"]),
    }
    try:
        import torch

        snapshot["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        snapshot["torch_runtime"] = {"available": False, "error": str(exc)}
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    payload = build_snapshot()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
