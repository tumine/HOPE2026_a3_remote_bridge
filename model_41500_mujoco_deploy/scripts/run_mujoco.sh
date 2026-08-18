#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PACKAGE_DIR}/../.." && pwd)"

# The original HOPE layout is <repo>/a3_deploy/<package>.  This workspace also
# keeps the package directly under a3_remote_bridge, so accept that one-level
# layout when the two-level candidate has no training source.
if [[ ! -d "${REPO_DIR}/hope_training/whole_body_tracking" &&
      ! -d "${REPO_DIR}/26.7.25发球部署/pingpang_ustc-srf/hope_training/whole_body_tracking" &&
      -d "${PACKAGE_DIR}/../26.7.25发球部署/pingpang_ustc-srf/hope_training/whole_body_tracking" ]]; then
  REPO_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
fi

if [[ -n "${A3_TRAINING_SOURCE:-}" ]]; then
  TRAINING_SOURCE="${A3_TRAINING_SOURCE}"
elif [[ -d "${REPO_DIR}/hope_training/whole_body_tracking" ]]; then
  TRAINING_SOURCE="${REPO_DIR}"
elif [[ -d "${REPO_DIR}/26.7.25发球部署/pingpang_ustc-srf/hope_training/whole_body_tracking" ]]; then
  TRAINING_SOURCE="${REPO_DIR}/26.7.25发球部署/pingpang_ustc-srf"
else
  echo "unable to locate HOPE training source; set A3_TRAINING_SOURCE" >&2
  exit 1
fi

TRAINING_DIR="${TRAINING_SOURCE}/hope_training/whole_body_tracking"

ONNX="${PACKAGE_DIR}/policy/hope_pingpong.onnx"
RUNTIME_CONFIG="${PACKAGE_DIR}/config/hope_pingpong_runtime.yaml"
LEGACY_CONTRACT="${PACKAGE_DIR}/config/model_41500_training_contract.yaml"
if [[ -n "${A3_MODEL_XML:-}" ]]; then
  MODEL_XML="${A3_MODEL_XML}"
elif [[ -f "${REPO_DIR}/gjc_robot_model/agibot_a3/xml/agibot_a3.xml" ]]; then
  MODEL_XML="${REPO_DIR}/gjc_robot_model/agibot_a3/xml/agibot_a3.xml"
elif [[ -f "${REPO_DIR}/a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml" ]]; then
  MODEL_XML="${REPO_DIR}/a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
elif [[ -f "${REPO_DIR}/26.7.25发球部署/pingpang_ustc-srf/a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml" ]]; then
  MODEL_XML="${REPO_DIR}/26.7.25发球部署/pingpang_ustc-srf/a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
elif [[ -f "${REPO_DIR}/../hope_deploy/sim/agibot_a3/xml/agibot_a3.xml" ]]; then
  MODEL_XML="${REPO_DIR}/../hope_deploy/sim/agibot_a3/xml/agibot_a3.xml"
else
  echo "unable to locate agibot_a3.xml; set A3_MODEL_XML" >&2
  exit 1
fi
EVALUATOR="${TRAINING_DIR}/scripts/mujoco_eval_onnx.py"
REFERENCE_DIR="${TRAINING_SOURCE}/a3_deploy/a3_deploy_example/reference"

for required_file in "${ONNX}" "${RUNTIME_CONFIG}" "${LEGACY_CONTRACT}" "${MODEL_XML}" "${EVALUATOR}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required file: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "missing required directory: ${REFERENCE_DIR}" >&2
  exit 1
fi

exec python3 "${EVALUATOR}" \
  --onnx "${ONNX}" \
  --runtime-config "${RUNTIME_CONFIG}" \
  --legacy-training-contract "${LEGACY_CONTRACT}" \
  --reference-dir "${REFERENCE_DIR}" \
  --model-xml "${MODEL_XML}" \
  "$@"
