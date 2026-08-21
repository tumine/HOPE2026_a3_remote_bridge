#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HOPE_ROOT="$(cd "${BUNDLE_DIR}/../.." && pwd)"
HOPE_ROOT="${HOPE_ROOT:-${DEFAULT_HOPE_ROOT}}"
DEPLOY_PYTHON="${HOPE_DEPLOY_PYTHON:-python3}"
export HOPE_BALL_PHYSICS_CONFIG="${HOPE_BALL_PHYSICS_CONFIG:-${BUNDLE_DIR}/config/ball_physics.yaml}"

exec "${DEPLOY_PYTHON}" "${HOPE_ROOT}/hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py" \
  --onnx "${BUNDLE_DIR}/policy/hope_pingpong.onnx" \
  --runtime-config "${BUNDLE_DIR}/config/runtime.yaml" \
  --model-xml "${BUNDLE_DIR}/robot_model/agibot_a3/xml/agibot_a3.xml" \
  --reference-dir "${BUNDLE_DIR}/reference" \
  --forehand-strike-forward-min 0.38 \
  --forehand-strike-forward-max 0.40 \
  --forehand-strike-lateral-min=-0.755 \
  --forehand-strike-lateral-max=-0.735 \
  --forehand-strike-height-min 1.10 \
  --forehand-strike-height-max 1.18 \
  --backhand-strike-forward-min 0.70 \
  --backhand-strike-forward-max 0.74 \
  --backhand-strike-lateral-min 0.075 \
  --backhand-strike-lateral-max 0.095 \
  --backhand-strike-height-min 0.86 \
  --backhand-strike-height-max 0.94 \
  --serve-side-pattern alternating \
  --eval-mode continuous \
  --num-serves 12 \
  --serve-flight-time-min 0.8 \
  --serve-flight-time-max 1.0 \
  --hold-min-seconds 0.0 \
  --hold-max-seconds 0.5 \
  --stand-min-hold-seconds 0.5 \
  --start-base-x-jitter 0.03 \
  --start-base-y-jitter 0.20 \
  --seed 0 \
  --diagnostics-out "${BUNDLE_DIR}/validation/hard_alternating_12_model_64500.json" \
  "$@"
