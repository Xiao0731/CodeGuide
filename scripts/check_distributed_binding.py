#!/usr/bin/env python3
"""Verify that each torchrun rank uses the active interpreter and its own GPU."""

from __future__ import annotations

import json
import os
import sys

import torch

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
if torch.cuda.current_device() != local_rank:
    raise RuntimeError(f"rank {local_rank} is bound to CUDA {torch.cuda.current_device()}")

# Import a venv-installed package so a launcher that silently falls back to
# the base interpreter fails before model download or training begins.
import transformers

print(json.dumps({
    "rank": local_rank,
    "python": sys.executable,
    "device": torch.cuda.get_device_name(local_rank),
    "transformers": transformers.__version__,
}))
