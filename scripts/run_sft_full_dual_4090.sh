#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
mkdir -p logs

bash scripts/preflight_sft_dual_4090.sh
python -m torch.distributed.run --standalone --nproc_per_node=2 -m src.training.train_sft \
  --config configs/sft/qwen25_coder_7b_qlora_8k.yaml \
  --mode full \
  --output-dir outputs/sft/qwen25_coder_7b_qlora_8k/full_seed20260728 "$@"
