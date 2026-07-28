#!/usr/bin/env python3
"""Canonical command-line entry point for CodeGuide GRPO training.

The implementation lives in :mod:`src.training.grpo_train`. Keeping this
file as a thin wrapper prevents historical entry points from diverging.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.grpo_train import main


if __name__ == "__main__":
    main()
