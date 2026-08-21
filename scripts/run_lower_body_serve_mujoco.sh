#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_PYTHON="${HOPE_DEPLOY_PYTHON:-${REPO_ROOT}/.venv_rl/bin/python}"

if [[ ! -x "${DEPLOY_PYTHON}" ]]; then
  echo "MuJoCo Python环境不存在或不可执行: ${DEPLOY_PYTHON}" >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  set -- --view --realtime
fi

exec "${DEPLOY_PYTHON}" \
  "${REPO_ROOT}/pc_tools/a3_lower_body_serve_mujoco.py" "$@"
