#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-20260728}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/eval/taco515_selected_code_first_v1.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_DIR="${RUN_DIR:-outputs/eval/taco515_best_experts_bs${BATCH_SIZE}}"
CACHE_DIR="${CACHE_DIR:-}"
STANDARD_SELECTION="${STANDARD_SELECTION:-outputs/eval/expert_online_standard_input_seed${SEED}/reports/best_checkpoint.json}"
CALL_SELECTION="${CALL_SELECTION:-outputs/eval/expert_online_call_based_seed${SEED}/reports/best_checkpoint.json}"
MANIFEST="${MANIFEST:-data/experts/io_mode/prepare_manifest.json}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read_value() {
  local file="$1"
  local expression="$2"
  "$PYTHON_BIN" - "$file" "$expression" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

STANDARD_STEP="$(read_value "$STANDARD_SELECTION" best.step)"
CALL_STEP="$(read_value "$CALL_SELECTION" best.step)"
STANDARD_OUTPUT="$(read_value "$MANIFEST" modes.standard_input.output_dir)"
CALL_OUTPUT="$(read_value "$MANIFEST" modes.call_based.output_dir)"
STANDARD_ADAPTER="$STANDARD_OUTPUT/checkpoint-$STANDARD_STEP"
CALL_ADAPTER="$CALL_OUTPUT/checkpoint-$CALL_STEP"
printf -v STANDARD_VARIANT "expert_standard_best_step%03d" "$STANDARD_STEP"
printf -v CALL_VARIANT "expert_call_best_step%03d" "$CALL_STEP"

for adapter in "$STANDARD_ADAPTER" "$CALL_ADAPTER"; do
  [[ -f "$adapter/adapter_config.json" ]] || {
    echo "[fatal] missing adapter: $adapter" >&2
    exit 1
  }
done

mkdir -p "$RUN_DIR/logs"
COMMON_ARGS=(
  --protocol-config "$PROTOCOL_CONFIG"
  --run-dir "$RUN_DIR"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$CACHE_DIR" ]]; then
  COMMON_ARGS+=(--cache-dir "$CACHE_DIR")
fi

"$PYTHON_BIN" scripts/evaluate_sft_matrix.py --stage prepare "${COMMON_ARGS[@]}" \
  | tee "$RUN_DIR/logs/prepare.log"

run_generate() {
  local gpu="$1"
  local variant="$2"
  local adapter="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_sft_matrix.py \
    --stage generate \
    "${COMMON_ARGS[@]}" \
    --variant "$variant" \
    --adapter-path "$adapter" \
    2>&1 | tee "$RUN_DIR/logs/generate_${variant}.log"
}

run_generate 0 "$STANDARD_VARIANT" "$STANDARD_ADAPTER" & PID0=$!
run_generate 1 "$CALL_VARIANT" "$CALL_ADAPTER" & PID1=$!
wait "$PID0"
wait "$PID1"

cat > "$RUN_DIR/expert_variants.tsv" <<EOF
variant	mode	step	adapter_path
${STANDARD_VARIANT}	standard_input	${STANDARD_STEP}	${STANDARD_ADAPTER}
${CALL_VARIANT}	call_based	${CALL_STEP}	${CALL_ADAPTER}
EOF

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
run_dir = Path(${RUN_DIR@Q})
variants = [${STANDARD_VARIANT@Q}, ${CALL_VARIANT@Q}]
status = {}
for variant in variants:
    path = run_dir / "generations" / f"{variant}.jsonl"
    ids = {
        str(json.loads(line)["problem_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    status[variant] = {"rows": len(ids), "complete": len(ids) == 515}
summary = {"variants": status, "complete": all(item["complete"] for item in status.values())}
(run_dir / "generation_acceptance.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not summary["complete"]:
    raise SystemExit(1)
PY

echo "[success] best standard and call experts generated on TACO-515"
echo "[next] verify both variants with scripts/verify_saved_matrix_simple.py"
