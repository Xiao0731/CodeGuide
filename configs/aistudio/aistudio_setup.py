#!/usr/bin/env python3
"""Install dependencies in an already-cloned AI Studio checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="requirements.lock.txt")
    parser.add_argument(
        "--index-url",
        default="https://mirror.baidu.com/pypi/simple",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    requirements = root / args.requirements
    if not requirements.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            args.index_url,
            "-r",
            str(requirements),
        ],
        check=True,
        cwd=root,
    )


if __name__ == "__main__":
    main()
