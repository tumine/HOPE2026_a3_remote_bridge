#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_PYTHON="${HOPE_DEPLOY_PYTHON:-python3}"
export PYTHONPATH="${BUNDLE_DIR}/reference${PYTHONPATH:+:${PYTHONPATH}}"

exec "${DEPLOY_PYTHON}" -m a3_deploy_onnx_ref_pingpong \
  --config "${BUNDLE_DIR}/config/runtime.yaml" \
  "$@"
