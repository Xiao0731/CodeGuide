#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of the caller's working directory.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-outputs/sft/taco515_compact_code_first_v1}"
CONFIG="${CONFIG:-configs/eval/taco515_compact_code_first_v1.yaml}"
BATCH_SIZE="${BATCH_SIZE:-4}"

: "${CALIBRATION_ADAPTER:?Set CALIBRATION_ADAPTER to the 500-sample adapter path}"
: "${CHECKPOINT_ADAPTER:?Set CHECKPOINT_ADAPTER to one full-training checkpoint path}"
: "${FULL_ADAPTER:?Set FULL_ADAPTER to the full adapter path}"

mkdir -p "${RUN_DIR}/logs"

# 【改动 1】Freeze the same 515 IDs and protocol once.
python scripts/evaluate_sft_matrix.py \
  --stage prepare \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  --batch-size "${BATCH_SIZE}"

# 【改动 2】Round 1: one model per 4090; each model internally uses batch_size=4.
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant base \
  --batch-size "${BATCH_SIZE}" \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/base.log" 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant calibration500 \
  --adapter-path "${CALIBRATION_ADAPTER}" \
  --batch-size "${BATCH_SIZE}" \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/calibration500.log" 2>&1 &
PID_CALIBRATION=$!

wait "${PID_BASE}"
wait "${PID_CALIBRATION}"

# 【改动 3】Round 2: checkpoint + full adapter.
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant checkpoint_step_xxx \
  --adapter-path "${CHECKPOINT_ADAPTER}" \
  --batch-size "${BATCH_SIZE}" \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/checkpoint_step_xxx.log" 2>&1 &
PID_CHECKPOINT=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant full \
  --adapter-path "${FULL_ADAPTER}" \
  --batch-size "${BATCH_SIZE}" \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/full.log" 2>&1 &
PID_FULL=$!

wait "${PID_CHECKPOINT}"
wait "${PID_FULL}"

echo "All four generation variants completed."
echo "Next: run strict Docker verification for each variant, then static rescoring."
