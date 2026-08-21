#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_PYTHON="${HOPE_DEPLOY_PYTHON:-${REPO_ROOT}/.venv_rl/bin/python}"

if [[ ! -x "${DEPLOY_PYTHON}" ]]; then
  echo "MuJoCo Python 环境不存在或不可执行: ${DEPLOY_PYTHON}" >&2
  echo "请先运行 ${REPO_ROOT}/scripts/setup_rl_venv.sh" >&2
  exit 2
fi

# No arguments means safe manual visualization.  Automatic V/C/F playback is
# available only when --auto-cycle is explicitly supplied by a regression test.
if [[ "$#" -eq 0 ]]; then
  set -- --view --realtime
fi

exec "${DEPLOY_PYTHON}" "${REPO_ROOT}/pc_tools/a3_serve_receive_mujoco.py" "$@"
