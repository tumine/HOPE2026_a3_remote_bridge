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
  --legacy-training-contract "${BUNDLE_DIR}/config/model_50000_training_contract.yaml" \
  --model-xml "${BUNDLE_DIR}/gjc_robot_model/agibot_a3/xml/agibot_a3.xml" \
  "$@"
