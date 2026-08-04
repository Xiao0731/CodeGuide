#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-20260728}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ONLINE_DEV_PER_MODE="${ONLINE_DEV_PER_MODE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"
CACHE_DIR="${CACHE_DIR:-}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

mkdir -p outputs

PREPARE_ARGS=(
  scripts/prepare_io_expert_sft.py
  --learning-rate "$LEARNING_RATE"
  --online-dev-per-mode "$ONLINE_DEV_PER_MODE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --seed "$SEED"
)
if [[ "$PREPARE_FORCE" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
"$PYTHON_BIN" "${PREPARE_ARGS[@]}" | tee outputs/io_expert_prepare.log

# Online selection executes generated code, so Docker is mandatory.
docker info --format '{{.ServerVersion}}'
docker pull "$CONTAINER_IMAGE"

manifest_value() {
  local mode="$1"
  local key="$2"
  "$PYTHON_BIN" - "$mode" "$key" <<'PY'
import json, sys
from pathlib import Path
mode, key = sys.argv[1:]
payload = json.loads(Path("data/experts/io_mode/prepare_manifest.json").read_text(encoding="utf-8"))
value = payload["modes"][mode][key]
if isinstance(value, list):
    print(",".join(str(item) for item in value))
else:
    print(value)
PY
}

run_worker() {
  local mode="$1"
  local gpu="$2"
  local config
  local protocol
  local output_dir
  local milestones_csv
  config="$(manifest_value "$mode" config)"
  protocol="$(manifest_value "$mode" eval_protocol)"
  output_dir="$(manifest_value "$mode" output_dir)"
  milestones_csv="$(manifest_value "$mode" checkpoint_milestones)"

  local run_dir="outputs/eval/expert_online_${mode}_seed${SEED}"
  local log_dir="$run_dir/logs"
  mkdir -p "$log_dir"

  "$PYTHON_BIN" scripts/evaluate_sft_matrix.py \
    --stage prepare \
    --protocol-config "$protocol" \
    --run-dir "$run_dir" \
    --batch-size "$EVAL_BATCH_SIZE" \
    >"$log_dir/prepare.log" 2>&1

  IFS=',' read -r -a milestones <<< "$milestones_csv"
  local previous_checkpoint=""

  for step in "${milestones[@]}"; do
    printf -v variant "expert_%s_step%03d" "$mode" "$step"
    local checkpoint="$output_dir/checkpoint-$step"
    local report="$run_dir/reports/strict_${variant}.json"

    if [[ -f "$report" ]]; then
      echo "[$mode] step=$step already evaluated; skip"
      previous_checkpoint="$checkpoint"
      continue
    fi

    if [[ ! -f "$checkpoint/adapter_config.json" ]]; then
      local train_args=(
        -m src.training.train_sft
        --config "$config"
        --stop-after-step "$step"
      )
      if [[ -n "$CACHE_DIR" ]]; then
        train_args+=(--cache-dir "$CACHE_DIR")
      fi
      if [[ -n "$previous_checkpoint" ]]; then
        train_args+=(--resume-from-checkpoint "$previous_checkpoint")
      fi

      echo "[$(date '+%F %T')] [$mode] GPU${gpu} train -> step $step" \
        | tee "$log_dir/train_to_step${step}.log"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${train_args[@]}" \
        2>&1 | tee -a "$log_dir/train_to_step${step}.log"
    fi

    if [[ ! -f "$checkpoint/adapter_config.json" ]]; then
      echo "[fatal] checkpoint was not saved: $checkpoint" >&2
      return 1
    fi

    echo "[$(date '+%F %T')] [$mode] GPU${gpu} generate step $step" \
      | tee "$log_dir/generate_${variant}.log"
    local eval_args=(
      scripts/evaluate_sft_matrix.py
      --stage generate
      --protocol-config "$protocol"
      --run-dir "$run_dir"
      --batch-size "$EVAL_BATCH_SIZE"
      --variant "$variant"
      --adapter-path "$checkpoint"
    )
    if [[ -n "$CACHE_DIR" ]]; then
      eval_args+=(--cache-dir "$CACHE_DIR")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${eval_args[@]}" \
      2>&1 | tee -a "$log_dir/generate_${variant}.log"

    "$PYTHON_BIN" scripts/verify_saved_matrix_simple.py \
      --run-dir "$run_dir" \
      --protocol-config "$protocol" \
      --container-image "$CONTAINER_IMAGE" \
      --verify-workers "$VERIFY_WORKERS" \
      --variant "$variant" \
      2>&1 | tee "$log_dir/verify_${variant}.log"

    "$PYTHON_BIN" scripts/select_best_expert_checkpoint.py \
      --run-dir "$run_dir" \
      --mode "$mode" \
      2>&1 | tee "$log_dir/select_after_step${step}.log"

    previous_checkpoint="$checkpoint"
  done

  echo "[$mode] online checkpoint selection complete"
  cat "$run_dir/reports/best_checkpoint.json"
}

run_worker standard_input 0 &
PID_STANDARD=$!
run_worker call_based 1 &
PID_CALL=$!

set +e
wait "$PID_STANDARD"; STATUS_STANDARD=$?
wait "$PID_CALL"; STATUS_CALL=$?
set -e

if [[ "$STATUS_STANDARD" -ne 0 || "$STATUS_CALL" -ne 0 ]]; then
  echo "[fatal] expert workers failed: standard=$STATUS_STANDARD call=$STATUS_CALL" >&2
  exit 1
fi

echo "[success] both experts trained and selected on mode-pure execution probes"
echo "standard best: outputs/eval/expert_online_standard_input_seed${SEED}/reports/best_checkpoint.json"
echo "call best: outputs/eval/expert_online_call_based_seed${SEED}/reports/best_checkpoint.json"
