#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <adapter_path> [extra args...]" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
adapter="$1"
shift
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
python scripts/evaluate_sft_adapter.py "$adapter" --stage generate "$@"
