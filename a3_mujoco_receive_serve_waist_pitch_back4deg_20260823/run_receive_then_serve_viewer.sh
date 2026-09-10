#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MUJOCO_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[失败] Python 环境不存在：${PYTHON}；请先运行 ./setup.sh" >&2
  exit 2
fi

exec "${PYTHON}" "${ROOT_DIR}/pc_tools/a3_serve_receive_mujoco.py" \
  --auto-cycle \
  --auto-receive-first \
  --cycles "${CYCLES:-3}" \
  --receive-balls 1 \
  --duration "${DURATION:-120}" \
  --auto-delay-s 0.50 \
  --ready-stable-s 0.50 \
  --ready-stable-velocity-rad-s 0.15 \
  --ready-stable-timeout-s 15 \
  --ready-dwell-s 1.0 \
  --serve-home-s 1.35 \
  --serve-waist-pitch-deg=-4 \
  --view \
  --realtime \
  "$@"
