#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PACKAGE_DIR}/../.." && pwd)"
TRAINING_DIR="${REPO_DIR}/hope_training/whole_body_tracking"

ONNX="${PACKAGE_DIR}/policy/hope_pingpong.onnx"
RUNTIME_CONFIG="${PACKAGE_DIR}/config/hope_pingpong_runtime.yaml"
CHECKPOINT_CONTRACT="${PACKAGE_DIR}/config/model_48000_training_contract.yaml"
MODEL_XML="${REPO_DIR}/gjc_robot_model/agibot_a3/xml/agibot_a3.xml"
EVALUATOR="${TRAINING_DIR}/scripts/mujoco_eval_onnx.py"

for required_file in "${ONNX}" "${RUNTIME_CONFIG}" "${CHECKPOINT_CONTRACT}" "${MODEL_XML}" "${EVALUATOR}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required file: ${required_file}" >&2
    exit 1
  fi
done

exec python3 "${EVALUATOR}" \
  --onnx "${ONNX}" \
  --runtime-config "${RUNTIME_CONFIG}" \
  --legacy-training-contract "${CHECKPOINT_CONTRACT}" \
  --model-xml "${MODEL_XML}" \
  "$@"

