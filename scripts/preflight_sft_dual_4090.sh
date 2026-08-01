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
import bitsandbytes as bnb
from bitsandbytes.cextension import lib as bnb_lib
if not getattr(bnb_lib, "compiled_with_cuda", False):
    raise SystemExit("bitsandbytes CUDA native library is unavailable")

# Import success is insufficient: older bitsandbytes catches native-library
# errors internally. Exercise the exact NF4 and paged optimizer primitives
# required by this QLoRA configuration.
probe = bnb.nn.Linear4bit(
    16, 16, bias=False, compute_dtype=torch.bfloat16,
    compress_statistics=True, quant_type="nf4",
).cuda(0)
x = torch.randn(2, 16, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
probe(x).float().sum().backward()
parameter = torch.nn.Parameter(torch.ones(16, device="cuda:0"))
optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=1e-3)
(parameter.square().sum()).backward()
optimizer.step()
del probe, x, parameter, optimizer
torch.cuda.empty_cache()
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
          "bitsandbytes_cuda_probe": "passed", "disk_free_gib": round(free / 2**30, 2)}
Path("artifacts/sft").mkdir(parents=True, exist_ok=True)
Path("artifacts/sft/preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

echo "active_python=$(python -c 'import sys; print(sys.executable)')"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/check_distributed_binding.py

python -m src.training.train_sft --validate-only --mode calibration
echo "dual-4090 SFT preflight passed"
