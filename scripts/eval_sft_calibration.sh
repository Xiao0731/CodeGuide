#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <adapter_path> [extra args...]" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
adapter="$1"
shift
: "${CODEGUIDE_EXECUTION_IMAGE:?set CODEGUIDE_EXECUTION_IMAGE to the fixed Docker image digest}"
python scripts/evaluate_sft_adapter.py "$adapter" \
  --container-image "$CODEGUIDE_EXECUTION_IMAGE" "$@"

