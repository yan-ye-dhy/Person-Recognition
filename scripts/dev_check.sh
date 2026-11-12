#!/usr/bin/env bash
# 一键环境自检：建 venv、装依赖、跑全部离线测试
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "==> 创建虚拟环境 .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "==> 安装依赖（首次较慢）"
pip install -q -e ".[dev]"

echo "==> 运行全部离线测试"
pytest -q

echo "==> CLI 冒烟"
vcrb --help >/dev/null && echo "vcrb CLI OK"
vcrb models list >/dev/null && echo "models list OK"

echo "✔ 全部通过"
