#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - <<'PY'
import hashlib, importlib, json, os, shutil, subprocess, sys
from pathlib import Path

expected = "08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b"
canonical = Path("data/final/sft_accepted.jsonl")
if not canonical.exists():
    raise SystemExit("missing canonical SFT; sync data/final/sft_accepted.jsonl first")
actual = hashlib.sha256(canonical.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"canonical hash mismatch: {actual}")

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit(f"exactly two CUDA GPUs required, found {torch.cuda.device_count()}")
gpus = []
for index in range(2):
    props = torch.cuda.get_device_properties(index)
    gpus.append({"index": index, "name": props.name, "vram_gib": round(props.total_memory / 2**30, 2), "bf16": torch.cuda.is_bf16_supported()})
    if "4090" not in props.name or props.total_memory < 23 * 2**30 or not torch.cuda.is_bf16_supported():
        raise SystemExit(f"GPU {index} does not satisfy RTX 4090 24GB/bf16 contract: {gpus[-1]}")

versions = {name: importlib.import_module(name).__version__ for name in ("transformers", "peft", "accelerate", "bitsandbytes")}
try:
    import flash_attn
    flash = {"available": True, "version": flash_attn.__version__}
except ImportError:
    flash = {"available": False, "fallback": "sdpa"}
free = shutil.disk_usage(".").free
if free < 40 * 2**30:
    raise SystemExit(f"insufficient disk space: {free / 2**30:.1f} GiB free")
output = Path("outputs/sft/preflight-write-test")
output.mkdir(parents=True, exist_ok=True)
(output / "ok").write_text("ok")
(output / "ok").unlink()

report = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpus": gpus,
          "dependencies": versions, "flash_attention": flash, "canonical_sha256": actual,
          "disk_free_gib": round(free / 2**30, 2)}
Path("artifacts/sft").mkdir(parents=True, exist_ok=True)
Path("artifacts/sft/preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

torchrun --standalone --nproc_per_node=2 - <<'PY'
import os, torch
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
assert torch.cuda.current_device() == rank
print(f"rank={rank} device={torch.cuda.get_device_name(rank)}")
PY

python -m src.training.train_sft --validate-only --mode calibration
echo "dual-4090 SFT preflight passed"

