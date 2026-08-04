#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/eval/taco515_selected_code_first_v1.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_DIR="${RUN_DIR:-outputs/eval/taco515_selected_bs${BATCH_SIZE}}"
MODEL_ROOT="${MODEL_ROOT:-outputs/sft/qwen25_coder_7b_qlora_8k}"
LR2_DIR="${LR2_DIR:-${MODEL_ROOT}/full_lr2e4_seed20260728}"
LR1_DIR="${LR1_DIR:-${MODEL_ROOT}/full_lr1e4_seed20260728}"
CACHE_DIR="${CACHE_DIR:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

VARIANT_A="mixed_lr2e4_step050"
ADAPTER_A="${LR2_DIR}/checkpoint-50"
VARIANT_B="mixed_lr1e4_step020"
ADAPTER_B="${LR1_DIR}/checkpoint-20"
VARIANT_C="mixed_lr1e4_step200"
ADAPTER_C="${LR1_DIR}/checkpoint-200"

LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR" "${RUN_DIR}/generations"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_file() {
  [[ -f "$1" ]] || { echo "[fatal] missing file: $1" >&2; exit 1; }
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
require_file "data/splits/sft_dev_ids.json"
require_adapter "$ADAPTER_A"
require_adapter "$ADAPTER_B"
require_adapter "$ADAPTER_C"

COMMON_ARGS=(
  --protocol-config "$PROTOCOL_CONFIG"
  --run-dir "$RUN_DIR"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$CACHE_DIR" ]]; then
  COMMON_ARGS+=(--cache-dir "$CACHE_DIR")
fi

"$PYTHON_BIN" scripts/evaluate_sft_matrix.py \
  --stage prepare \
  "${COMMON_ARGS[@]}" \
  | tee "$LOG_DIR/prepare.log"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
run_dir = Path(${RUN_DIR@Q})
selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
ids = selection["problem_ids"]
if len(ids) != 515 or len(set(ids)) != 515:
    raise SystemExit(f"expected 515 unique IDs, got total={len(ids)} unique={len(set(ids))}")
print("[accept] TACO-515 selection verified")
PY

run_eval() {
  local gpu="$1"
  local stage="$2"
  local variant="$3"
  local adapter_path="${4:-}"
  local log="$LOG_DIR/${stage}_${variant}.log"
  local args=(scripts/evaluate_sft_matrix.py --stage "$stage" "${COMMON_ARGS[@]}" --variant "$variant")
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter-path "$adapter_path")
  fi
  echo "[$(date '+%F %T')] GPU${gpu} ${stage} ${variant}" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log"
}

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  run_eval 0 preflight "$VARIANT_A" "$ADAPTER_A" & P0=$!
  run_eval 1 preflight "$VARIANT_C" "$ADAPTER_C" & P1=$!
  wait "$P0"
  wait "$P1"
fi

worker0() {
  run_eval 0 generate base
  run_eval 0 generate "$VARIANT_A" "$ADAPTER_A"
}

worker1() {
  run_eval 1 generate "$VARIANT_B" "$ADAPTER_B"
  run_eval 1 generate "$VARIANT_C" "$ADAPTER_C"
}

worker0 & PID0=$!
worker1 & PID1=$!
wait "$PID0"
wait "$PID1"

cat > "$RUN_DIR/selected_variants.tsv" <<EOF
variant	role	adapter_path
base	base	
${VARIANT_A}	best_overall	${ADAPTER_A}
${VARIANT_B}	no_forgetting_candidate	${ADAPTER_B}
${VARIANT_C}	low_lr_late_peak	${ADAPTER_C}
EOF

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
run_dir = Path(${RUN_DIR@Q})
expected = ["base", ${VARIANT_A@Q}, ${VARIANT_B@Q}, ${VARIANT_C@Q}]
status = {}
for variant in expected:
    path = run_dir / "generations" / f"{variant}.jsonl"
    ids = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(str(json.loads(line)["problem_id"]))
    status[variant] = {"rows": len(ids), "complete": len(ids) == 515}
summary = {
    "schema_version": "codeguide-taco515-selected-generation-v1",
    "expected_variants": expected,
    "complete_variants": sum(item["complete"] for item in status.values()),
    "variants": status,
}
(run_dir / "generation_acceptance.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["complete_variants"] != len(expected):
    raise SystemExit(1)
PY

echo "[success] Base + three selected checkpoints generated on TACO-515"
echo "[next] strict Docker verification:"
echo "  python scripts/verify_saved_matrix_simple.py --run-dir '$RUN_DIR' --protocol-config '$PROTOCOL_CONFIG' --verify-workers 8"
