#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --disable-pip-version-check --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --disable-pip-version-check \
  -r "${ROOT_DIR}/requirements-lock.txt"

echo "[成功] MuJoCo Python 环境：${VENV_DIR}"
echo "运行可视化：${ROOT_DIR}/run_viewer.sh"
