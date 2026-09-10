#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MUJOCO_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[失败] Python 环境不存在：${PYTHON}；请先运行 ./setup.sh" >&2
  exit 2
fi

exec "${PYTHON}" "${ROOT_DIR}/pc_tools/a3_serve_receive_mujoco.py" \
  --view --realtime "$@"
