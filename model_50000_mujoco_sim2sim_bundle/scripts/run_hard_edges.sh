#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export HOPE_BALL_PHYSICS_CONFIG="${BUNDLE_DIR}/configs/ball_physics.yaml"

exec "${PYTHON_BIN}" \
  "${BUNDLE_DIR}/hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py" \
  --onnx "${BUNDLE_DIR}/policy/hope_pingpong.onnx" \
  --runtime-config "${BUNDLE_DIR}/config/hope_pingpong_runtime.yaml" \
  --model-xml "${BUNDLE_DIR}/gjc_robot_model/agibot_a3/xml/agibot_a3.xml" \
  --eval-mode continuous \
  --num-serves 20 \
  --serve-side-pattern alternating \
  --forehand-strike-lateral-min=-0.75 \
  --forehand-strike-lateral-max=-0.70 \
  --backhand-strike-lateral-min=0.70 \
  --backhand-strike-lateral-max=0.75 \
  --forehand-strike-height-min=0.84 \
  --forehand-strike-height-max=1.21 \
  --backhand-strike-height-min=0.84 \
  --backhand-strike-height-max=1.21 \
  --serve-flight-time-min=0.8 \
  --serve-flight-time-max=1.0 \
  --max-rest-seconds=1.0 \
  --max-trial-seconds=2.5 \
  --hold-min-seconds=0.0 \
  --hold-max-seconds=0.0 \
  --stand-min-hold-seconds=0.0 \
  --start-base-x-jitter=0.05 \
  --start-base-y-jitter=0.20 \
  --seed=7 \
  "$@"
