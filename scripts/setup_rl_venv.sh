#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${A3_RL_VENV:-${ROOT_DIR}/.venv_rl}"
PYTHON_BIN="${A3_RL_PYTHON:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]] \
    || ! grep -Eq '^include-system-site-packages = true$' "${VENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --clear --system-site-packages --python "${PYTHON_BIN}" "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv --clear --system-site-packages "${VENV_DIR}"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${VENV_DIR}/bin/python" \
    -r "${ROOT_DIR}/requirements-rl.txt"
else
  "${VENV_DIR}/bin/python" -m pip install \
    --disable-pip-version-check \
    -r "${ROOT_DIR}/requirements-rl.txt"
fi

echo "RL Python ready: ${VENV_DIR}/bin/python"
