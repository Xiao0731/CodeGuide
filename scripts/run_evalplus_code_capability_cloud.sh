#!/usr/bin/env bash
# Generate HumanEval(+)/MBPP(+) samples for Base, step20 and step200.
# One free GPU: GPU_IDS=1 bash scripts/run_evalplus_code_capability_cloud.sh
# Two free GPUs: GPU_IDS=0,1 bash scripts/run_evalplus_code_capability_cloud.sh
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/eval/evalplus_code_capability_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/evalplus_code_capability_v1}"
GPU_IDS="${GPU_IDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

mkdir -p "$OUTPUT_ROOT/logs"

require_file() { [[ -f "$1" ]] || { echo "[fatal] missing file: $1" >&2; exit 1; }; }
require_adapter() {
  local path="$1"
  require_file "$path/adapter_config.json"
  if [[ ! -s "$path/adapter_model.safetensors" && ! -s "$path/adapter_model.bin" ]]; then
    echo "[fatal] adapter weights missing: $path" >&2
    exit 1
  fi
}

require_file "$CONFIG"
require_file "scripts/generate_evalplus_code_capability.py"
require_adapter "outputs/sft/qwen25_coder_7b_qlora_8k/full_lr1e4_seed20260728/checkpoint-20"
require_adapter "outputs/sft/qwen25_coder_7b_qlora_8k/full_lr1e4_seed20260728/checkpoint-200"

"$PYTHON_BIN" - <<'PY'
import importlib.metadata
version = importlib.metadata.version("evalplus")
if version != "0.3.1":
    raise SystemExit(f"EvalPlus version mismatch: expected=0.3.1 actual={version}")
print("[accept] evalplus", version)
PY

run_variant() {
  local gpu="$1" variant="$2" log="$OUTPUT_ROOT/logs/generate_${variant}.log"
  echo "[$(date '+%F %T')] GPU${gpu} start ${variant}" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/generate_evalplus_code_capability.py \
    --config "$CONFIG" --variant "$variant" --batch-size "$BATCH_SIZE" \
    2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] GPU${gpu} done ${variant}" | tee -a "$log"
}

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 1 ]]; then
  GPU="${GPUS[0]}"
  run_variant "$GPU" base
  run_variant "$GPU" mixed_lr1e4_step200
  run_variant "$GPU" mixed_lr1e4_step020
elif [[ "${#GPUS[@]}" -eq 2 ]]; then
  run_variant "${GPUS[0]}" base & PID_BASE=$!
  run_variant "${GPUS[1]}" mixed_lr1e4_step200 & PID_STEP200=$!
  set +e
  wait "$PID_BASE"; STATUS_BASE=$?
  wait "$PID_STEP200"; STATUS_STEP200=$?
  set -e
  if [[ "$STATUS_BASE" -ne 0 || "$STATUS_STEP200" -ne 0 ]]; then
    echo "[fatal] first wave failed: base=$STATUS_BASE step200=$STATUS_STEP200" >&2
    exit 1
  fi
  run_variant "${GPUS[0]}" mixed_lr1e4_step020
else
  echo "[fatal] GPU_IDS must contain one or two IDs, e.g. 1 or 0,1" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path("outputs/eval/evalplus_code_capability_v1")
variants = ["base", "mixed_lr1e4_step020", "mixed_lr1e4_step200"]
expected = {"humaneval": 164, "mbpp": 378}
status = {}
for dataset, expected_count in expected.items():
    status[dataset] = {}
    for variant in variants:
        sample_path = root / "samples" / dataset / f"{variant}.jsonl"
        raw_path = root / "raw" / dataset / f"{variant}.jsonl"
        stats_path = root / "stats" / dataset / f"{variant}.json"
        def ids(path):
            if not path.is_file(): return set()
            return {str(json.loads(line)["task_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        sample_ids, raw_ids = ids(sample_path), ids(raw_path)
        complete = len(sample_ids) == expected_count and sample_ids == raw_ids and stats_path.is_file()
        status[dataset][variant] = {"expected": expected_count, "sample_rows": len(sample_ids), "raw_rows": len(raw_ids), "stats_exists": stats_path.is_file(), "complete": complete}
payload = {"schema_version": "codeguide-evalplus-cloud-acceptance-v1", "evaluation_module": "code_capability", "status": status, "complete": all(item["complete"] for dataset in status.values() for item in dataset.values())}
(root / "cloud_generation_acceptance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["complete"]: raise SystemExit(1)
PY

ARCHIVE="codeguide_evalplus_code_generations_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" "$OUTPUT_ROOT/samples" "$OUTPUT_ROOT/raw" "$OUTPUT_ROOT/stats" "$OUTPUT_ROOT/manifests" "$OUTPUT_ROOT/cloud_generation_acceptance.json" "$CONFIG"
echo "[success] Base + step20 + step200 generation complete"
echo "[download] $ARCHIVE"
