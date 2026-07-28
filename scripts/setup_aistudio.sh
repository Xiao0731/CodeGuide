#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -f requirements.lock.txt ]]; then
  echo "requirements.lock.txt 不存在；先完成并验证训练环境锁定。"
  exit 1
fi

python -m pip install \
  --index-url https://mirror.baidu.com/pypi/simple \
  -r requirements.lock.txt
python scripts/check_environment.py
