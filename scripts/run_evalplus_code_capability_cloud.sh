#!/usr/bin/env bash
# 为 Base、step20、step200 生成 HumanEval(+)/MBPP(+) 代码答案。
# 单张空闲显卡：GPU_IDS=1 bash scripts/run_evalplus_code_capability_cloud.sh
# 两张空闲显卡：GPU_IDS=0,1 bash scripts/run_evalplus_code_capability_cloud.sh
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/eval/evalplus_code_capability_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/evalplus_code_capability_v1}"
EVALPLUS_DATA_ROOT="${EVALPLUS_DATA_ROOT:-data/external/evalplus}"
EVALPLUS_GITHUB_MIRROR="${EVALPLUS_GITHUB_MIRROR:-https://gh.llkk.cc/}"
GPU_IDS="${GPU_IDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

HUMANEVAL_DATA="$EVALPLUS_DATA_ROOT/HumanEvalPlus-v0.1.10.jsonl.gz"
MBPP_DATA="$EVALPLUS_DATA_ROOT/MbppPlus-v0.2.0.jsonl.gz"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

mkdir -p "$OUTPUT_ROOT/logs"

require_file() {
  [[ -f "$1" ]] || {
    echo "[致命] 缺少文件：$1" >&2
    exit 1
  }
}

require_adapter() {
  local path="$1"
  require_file "$path/adapter_config.json"
  if [[ ! -s "$path/adapter_model.safetensors" && ! -s "$path/adapter_model.bin" ]]; then
    echo "[致命] 缺少适配器权重：$path" >&2
    exit 1
  fi
}

require_file "$CONFIG"
require_file "scripts/generate_evalplus_code_capability.py"
require_file "scripts/prepare_evalplus_datasets_offline.py"
require_adapter "outputs/sft/qwen25_coder_7b_qlora_8k/full_lr1e4_seed20260728/checkpoint-20"
require_adapter "outputs/sft/qwen25_coder_7b_qlora_8k/full_lr1e4_seed20260728/checkpoint-200"

"$PYTHON_BIN" - <<'PY'
import importlib.metadata
version = importlib.metadata.version("evalplus")
if version != "0.3.1":
    raise SystemExit(f"EvalPlus 版本不匹配：expected=0.3.1 actual={version}")
print("[验收] evalplus", version)
PY

# 每次启动都执行数据验收。有效文件会被直接复用；旧的 Hugging Face
# Parquet 转换文件因缺少 contract 字段，会被自动识别并替换。
"$PYTHON_BIN" scripts/prepare_evalplus_datasets_offline.py \
  --output-dir "$EVALPLUS_DATA_ROOT" \
  --github-mirror "$EVALPLUS_GITHUB_MIRROR"

require_file "$HUMANEVAL_DATA"
require_file "$MBPP_DATA"
require_file "$EVALPLUS_DATA_ROOT/manifest.json"
export HUMANEVAL_OVERRIDE_PATH="$(realpath "$HUMANEVAL_DATA")"
export MBPP_OVERRIDE_PATH="$(realpath "$MBPP_DATA")"

echo "[数据] HUMANEVAL_OVERRIDE_PATH=$HUMANEVAL_OVERRIDE_PATH"
echo "[数据] MBPP_OVERRIDE_PATH=$MBPP_OVERRIDE_PATH"

run_variant() {
  local gpu="$1"
  local variant="$2"
  local log="$OUTPUT_ROOT/logs/generate_${variant}.log"
  echo "[$(date '+%F %T')] GPU${gpu} 开始 ${variant}" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/generate_evalplus_code_capability.py \
    --config "$CONFIG" --variant "$variant" --batch-size "$BATCH_SIZE" \
    2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] GPU${gpu} 完成 ${variant}" | tee -a "$log"
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
    echo "[致命] 第一轮失败：base=$STATUS_BASE step200=$STATUS_STEP200" >&2
    exit 1
  fi
  run_variant "${GPUS[0]}" mixed_lr1e4_step020
else
  echo "[致命] GPU_IDS 只能包含一张或两张显卡，例如 1 或 0,1" >&2
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
            if not path.is_file():
                return set()
            return {
                str(json.loads(line)["task_id"])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        sample_ids, raw_ids = ids(sample_path), ids(raw_path)
        complete = (
            len(sample_ids) == expected_count
            and sample_ids == raw_ids
            and stats_path.is_file()
        )
        status[dataset][variant] = {
            "expected": expected_count,
            "sample_rows": len(sample_ids),
            "raw_rows": len(raw_ids),
            "stats_exists": stats_path.is_file(),
            "complete": complete,
        }
payload = {
    "schema_version": "codeguide-evalplus-cloud-acceptance-v1",
    "evaluation_module": "code_capability",
    "status": status,
    "complete": all(
        item["complete"]
        for dataset in status.values()
        for item in dataset.values()
    ),
}
(root / "cloud_generation_acceptance.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["complete"]:
    raise SystemExit(1)
PY

ARCHIVE="codeguide_evalplus_code_generations_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" \
  "$OUTPUT_ROOT/samples" \
  "$OUTPUT_ROOT/raw" \
  "$OUTPUT_ROOT/stats" \
  "$OUTPUT_ROOT/manifests" \
  "$OUTPUT_ROOT/cloud_generation_acceptance.json" \
  "$EVALPLUS_DATA_ROOT" \
  "$CONFIG"

echo "[成功] Base、step20、step200 全部生成完成"
echo "[下载] $ARCHIVE"
