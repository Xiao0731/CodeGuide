#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/eval/taco100_balanced_code_first_v1.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_DIR="${RUN_DIR:-outputs/eval/taco100_lr_trajectory_bs${BATCH_SIZE}}"
MODEL_ROOT="${MODEL_ROOT:-outputs/sft/qwen25_coder_7b_qlora_8k}"
LR2_DIR="${LR2_DIR:-${MODEL_ROOT}/full_lr2e4_seed20260728}"
LR1_DIR="${LR1_DIR:-${MODEL_ROOT}/full_lr1e4_seed20260728}"
CACHE_DIR="${CACHE_DIR:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

STEPS=(5 10 20 30 40 50 75 100 150 200 250 300 350 400 500 600)
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR" "${RUN_DIR}/generations"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[fatal] missing file: $1" >&2
    exit 1
  fi
}

require_adapter() {
  local path="$1"
  require_file "$path/adapter_config.json"
  if [[ ! -s "$path/adapter_model.safetensors" && ! -s "$path/adapter_model.bin" ]]; then
    echo "[fatal] adapter weights missing: $path" >&2
    exit 1
  fi
}

require_file "$PROTOCOL_CONFIG"
require_file "scripts/evaluate_sft_matrix.py"
require_file "data/splits/sft_checkpoint_dev_100_ids.json"
require_file "data/final/sft_accepted.jsonl"

# Hard acceptance gate: exactly 100 unique IDs, 50 per io_mode.
"$PYTHON_BIN" - <<'PY'
import json
from collections import Counter
from pathlib import Path

ids_payload = json.loads(Path("data/splits/sft_checkpoint_dev_100_ids.json").read_text(encoding="utf-8"))
ids = ids_payload.get("ids") if isinstance(ids_payload, dict) else ids_payload
if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
    raise SystemExit("invalid TACO100 ID file")
if len(ids) != 100 or len(set(ids)) != 100:
    raise SystemExit(f"TACO100 must contain 100 unique IDs, got total={len(ids)} unique={len(set(ids))}")
selected = set(ids)
counts = Counter()
with Path("data/final/sft_accepted.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        problem_id = record.get("id") or record.get("problem_id")
        if problem_id not in selected:
            continue
        metadata = record.get("metadata") or {}
        counts[str(metadata.get("io_mode") or record.get("io_mode") or "unknown")] += 1
expected = {"standard_input": 50, "call_based": 50}
if dict(counts) != expected:
    raise SystemExit(f"TACO100 io_mode mismatch: expected={expected}, actual={dict(counts)}")
print(f"[accept] balanced TACO100 verified: {dict(counts)}")
PY

COMMON_ARGS=(
  --protocol-config "$PROTOCOL_CONFIG"
  --run-dir "$RUN_DIR"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$CACHE_DIR" ]]; then
  COMMON_ARGS+=(--cache-dir "$CACHE_DIR")
fi

# Freeze selection/protocol before any concurrent generation process starts.
"$PYTHON_BIN" scripts/evaluate_sft_matrix.py \
  --stage prepare \
  "${COMMON_ARGS[@]}" \
  >"$LOG_DIR/prepare.log" 2>&1
cat "$LOG_DIR/prepare.log"

# Validate all expected adapters before spending GPU time.
for step in "${STEPS[@]}"; do
  require_adapter "$LR2_DIR/checkpoint-$step"
  require_adapter "$LR1_DIR/checkpoint-$step"
done
require_adapter "$LR2_DIR/adapter"
require_adapter "$LR1_DIR/adapter"

{
  printf "variant\tlearning_rate\tstep\tadapter_path\n"
  printf "base\tbase\t0\t\n"
  for step in "${STEPS[@]}"; do
    printf "lr2e4_step%03d\t2e-4\t%d\t%s/checkpoint-%d\n" "$step" "$step" "$LR2_DIR" "$step"
  done
  printf "lr2e4_final611\t2e-4\t611\t%s/adapter\n" "$LR2_DIR"
  for step in "${STEPS[@]}"; do
    printf "lr1e4_step%03d\t1e-4\t%d\t%s/checkpoint-%d\n" "$step" "$step" "$LR1_DIR" "$step"
  done
  printf "lr1e4_final611\t1e-4\t611\t%s/adapter\n" "$LR1_DIR"
} >"$RUN_DIR/trajectory_variants.tsv"

run_eval() {
  local gpu="$1"
  local stage="$2"
  local variant="$3"
  local adapter_path="${4:-}"
  local log_path="$LOG_DIR/${stage}_${variant}.log"

  local args=(
    scripts/evaluate_sft_matrix.py
    --stage "$stage"
    "${COMMON_ARGS[@]}"
    --variant "$variant"
  )
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter-path "$adapter_path")
  fi

  echo "[$(date '+%F %T')] GPU${gpu} ${stage} ${variant} start" | tee -a "$log_path"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log_path"
  echo "[$(date '+%F %T')] GPU${gpu} ${stage} ${variant} done" | tee -a "$log_path"
}

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "[preflight] batch_size=$BATCH_SIZE on both GPUs"
  run_eval 0 preflight lr2e4_final611 "$LR2_DIR/adapter" &
  PREFLIGHT0=$!
  run_eval 1 preflight lr1e4_final611 "$LR1_DIR/adapter" &
  PREFLIGHT1=$!

  set +e
  wait "$PREFLIGHT0"; STATUS0=$?
  wait "$PREFLIGHT1"; STATUS1=$?
  set -e
  if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
    echo "[fatal] preflight failed: gpu0=$STATUS0 gpu1=$STATUS1" >&2
    echo "If this is OOM, rerun with: BATCH_SIZE=8 RUN_DIR=outputs/eval/taco100_lr_trajectory_bs8 ..." >&2
    exit 1
  fi
fi

worker_lr2() {
  # Base is shared by both plots and generated only once.
  run_eval 0 generate base
  for step in "${STEPS[@]}"; do
    printf -v variant "lr2e4_step%03d" "$step"
    run_eval 0 generate "$variant" "$LR2_DIR/checkpoint-$step"
  done
  run_eval 0 generate lr2e4_final611 "$LR2_DIR/adapter"
}

worker_lr1() {
  for step in "${STEPS[@]}"; do
    printf -v variant "lr1e4_step%03d" "$step"
    run_eval 1 generate "$variant" "$LR1_DIR/checkpoint-$step"
  done
  run_eval 1 generate lr1e4_final611 "$LR1_DIR/adapter"
}

worker_lr2 &
PID_LR2=$!
worker_lr1 &
PID_LR1=$!
printf "%s\n" "$PID_LR2" >"$RUN_DIR/gpu0_worker.pid"
printf "%s\n" "$PID_LR1" >"$RUN_DIR/gpu1_worker.pid"

echo "[launch] gpu0 worker pid=$PID_LR2 (base + 2e-4 trajectory)"
echo "[launch] gpu1 worker pid=$PID_LR1 (1e-4 trajectory)"

set +e
wait "$PID_LR2"; STATUS_LR2=$?
wait "$PID_LR1"; STATUS_LR1=$?
set -e

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
run_dir = Path(${RUN_DIR@Q})
expected = ["base"]
steps = [5,10,20,30,40,50,75,100,150,200,250,300,350,400,500,600]
expected += [f"lr2e4_step{s:03d}" for s in steps] + ["lr2e4_final611"]
expected += [f"lr1e4_step{s:03d}" for s in steps] + ["lr1e4_final611"]
status = {}
for variant in expected:
    path = run_dir / "generations" / f"{variant}.jsonl"
    ids = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(str(json.loads(line)["problem_id"]))
    status[variant] = {"path": str(path), "unique_generations": len(ids), "complete": len(ids) == 100}
summary = {
    "schema_version": "codeguide-taco100-generation-acceptance-v1",
    "batch_size": int(${BATCH_SIZE@Q}),
    "gpu0_exit": int(${STATUS_LR2}),
    "gpu1_exit": int(${STATUS_LR1}),
    "expected_variants": len(expected),
    "complete_variants": sum(item["complete"] for item in status.values()),
    "variants": status,
}
(run_dir / "generation_acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["complete_variants"] != len(expected) or summary["gpu0_exit"] or summary["gpu1_exit"]:
    raise SystemExit(1)
PY

echo "[success] all 35 variants complete: base + 17 checkpoints for each learning rate"
