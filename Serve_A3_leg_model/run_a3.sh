#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
A3_DEPLOY_PACKAGE_DIR="${A3_DEPLOY_PACKAGE_DIR:-${SCRIPT_DIR}/hope_deploy/dist/a3_deploy_rockchip}"

if [[ ! -x "${A3_DEPLOY_PACKAGE_DIR}/a3_deploy_onnx_ref" ]]; then
  echo "A3 deploy package is missing: ${A3_DEPLOY_PACKAGE_DIR}" >&2
  echo "initialize the submodule and run: ${SCRIPT_DIR}/compile.sh" >&2
  exit 66
fi

A3_SOURCE_ROBOT_ENV_DEFAULT="1"
A3_ROBOT_ENV="${A3_ROBOT_ENV:-/agibot/software/v0/entry/env/env.sh}"
A3_SOURCE_ROBOT_ENV_ENABLED="${A3_SOURCE_ROBOT_ENV:-${A3_SOURCE_ROBOT_ENV_DEFAULT}}"
if [[ "${A3_SOURCE_ROBOT_ENV_ENABLED}" != "0" ]]; then
  if [[ ! -f "${A3_ROBOT_ENV}" ]]; then
    echo "required robot env not found: ${A3_ROBOT_ENV} (set A3_SOURCE_ROBOT_ENV=0 to skip)" >&2
    exit 66
  fi
  set +u
  # shellcheck disable=SC1090
  source "${A3_ROBOT_ENV}"
  set -u
  echo "[a3] sourced robot env: ${A3_ROBOT_ENV}"
fi

DEFAULT_A3_TRANSPORT="iceoryx"
A3_TRANSPORT="${A3_TRANSPORT:-${DEFAULT_A3_TRANSPORT}}"
case "${A3_TRANSPORT}" in
  iceoryx)
    A3_AIMRT_CFG="${SCRIPT_DIR}/config/a3_aimrt_config.iceoryx.yaml"
    ;;
  ros2)
    A3_AIMRT_CFG="${SCRIPT_DIR}/config/a3_aimrt_config.ros2.yaml"
    ;;
  *)
    echo "invalid A3_TRANSPORT='${A3_TRANSPORT}'; expected one of: iceoryx, ros2" >&2
    exit 64
    ;;
esac
if [[ ! -f "${A3_AIMRT_CFG}" ]]; then
  echo "selected AimRT config does not exist: ${A3_AIMRT_CFG}" >&2
  exit 66
fi

mkdir -p logs
export LD_LIBRARY_PATH="${A3_DEPLOY_PACKAGE_DIR}:${A3_DEPLOY_PACKAGE_DIR}/lib:${SCRIPT_DIR}:${LD_LIBRARY_PATH:-}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/config/fastrtps_profile.xml"

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

echo "[a3] transport=${A3_TRANSPORT} aimrt_cfg=${A3_AIMRT_CFG}"
exec "${A3_DEPLOY_PACKAGE_DIR}/a3_deploy_onnx_ref" \
  --runtime-cfg="${SCRIPT_DIR}/config/a3_runtime_config.yaml" \
  --aimrt-cfg="${A3_AIMRT_CFG}" \
  "$@"
