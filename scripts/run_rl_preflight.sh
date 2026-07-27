#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${A3_RL_VENV:-${ROOT_DIR}/.venv_rl}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "RL Python environment is missing; run:" >&2
  echo "  ${ROOT_DIR}/scripts/setup_rl_venv.sh" >&2
  exit 66
fi

"${VENV_DIR}/bin/python" "${ROOT_DIR}/pc_tools/a3_rl_self_check.py"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/pc_tools/a3_rl_mujoco_check.py"
