#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/preflight_sft_dual_4090.sh
python -m torch.distributed.run --standalone --nproc_per_node=2 -m src.training.train_sft \
  --config configs/sft/qwen25_coder_7b_qlora_8k.yaml \
  --mode calibration "$@"
