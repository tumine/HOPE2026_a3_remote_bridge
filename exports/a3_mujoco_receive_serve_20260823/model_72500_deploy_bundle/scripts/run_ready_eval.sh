#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_PYTHON="${HOPE_DEPLOY_PYTHON:-python3}"

exec "${DEPLOY_PYTHON}" "${BUNDLE_DIR}/scripts/mujoco_ready_eval_onnx.py" \
  --onnx "${BUNDLE_DIR}/policy/hope_pingpong.onnx" \
  --runtime-config "${BUNDLE_DIR}/config/runtime.yaml" \
  --model-xml "${BUNDLE_DIR}/robot_model/agibot_a3/xml/agibot_a3.xml" \
  --reference-dir "${BUNDLE_DIR}/reference" \
  --ready-reference "${BUNDLE_DIR}/reference_motion/hope_neutral_ready_crouch2cm_flatfeet_compact_waist_pitch_roll_locked0.npz" \
  --duration 10 \
  --json-out "${BUNDLE_DIR}/validation/ready_no_command_10s.json" \
  "$@"
