#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_PYTHON="${HOPE_DEPLOY_PYTHON:-python3}"
export HOPE_BALL_PHYSICS_CONFIG="${HOPE_BALL_PHYSICS_CONFIG:-${BUNDLE_DIR}/config/ball_physics.yaml}"

exec "${DEPLOY_PYTHON}" "${BUNDLE_DIR}/scripts/mujoco_eval_onnx.py" \
  --onnx "${BUNDLE_DIR}/policy/hope_pingpong.onnx" \
  --runtime-config "${BUNDLE_DIR}/config/runtime.yaml" \
  --model-xml "${BUNDLE_DIR}/robot_model/agibot_a3/xml/agibot_a3.xml" \
  --reference-dir "${BUNDLE_DIR}/reference" \
  --eval-mode continuous \
  --serve-side-pattern coverage \
  --num-serves 20 \
  --seed 0 \
  --diagnostics-out "${BUNDLE_DIR}/validation/continuous_20_diagnostics.json" \
  --json-out "${BUNDLE_DIR}/validation/continuous_20_result.json" \
  "$@"
