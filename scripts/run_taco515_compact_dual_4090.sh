#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_DIR="outputs/sft/taco515_compact_code_first_v1"
CONFIG="configs/eval/taco515_compact_code_first_v1.yaml"

: "${CALIBRATION_ADAPTER:?set CALIBRATION_ADAPTER}"
: "${CHECKPOINT_ADAPTER:?set CHECKPOINT_ADAPTER}"
: "${FULL_ADAPTER:?set FULL_ADAPTER}"

mkdir -p "${RUN_DIR}/logs"

python scripts/evaluate_sft_matrix.py \
  --stage prepare \
  --all-dev \
  --variant base \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}"

# 第一轮：Base + 500 Adapter
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant base \
  --batch-size 4 \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/base.log" 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant calibration500 \
  --adapter-path "${CALIBRATION_ADAPTER}" \
  --batch-size 4 \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/calibration500.log" 2>&1 &
PID_CALIBRATION=$!

wait "${PID_BASE}"
wait "${PID_CALIBRATION}"

# 第二轮：中间 checkpoint + Full
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant checkpoint_step_xxx \
  --adapter-path "${CHECKPOINT_ADAPTER}" \
  --batch-size 4 \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/checkpoint_step_xxx.log" 2>&1 &
PID_CHECKPOINT=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_sft_matrix.py \
  --stage generate \
  --variant full \
  --adapter-path "${FULL_ADAPTER}" \
  --batch-size 4 \
  --run-dir "${RUN_DIR}" \
  --protocol-config "${CONFIG}" \
  > "${RUN_DIR}/logs/full.log" 2>&1 &
PID_FULL=$!

wait "${PID_CHECKPOINT}"
wait "${PID_FULL}"

echo "All generations completed."