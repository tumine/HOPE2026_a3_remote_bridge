#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
A3_DEPLOY_PACKAGE_DIR="${A3_DEPLOY_PACKAGE_DIR:-${SCRIPT_DIR}/hope_deploy/dist/a3_deploy_rockchip}"

if [[ ! -x "${A3_DEPLOY_PACKAGE_DIR}/hope_lower_body_deploy" ]]; then
  echo "A3 deploy package is missing: ${A3_DEPLOY_PACKAGE_DIR}" >&2
  echo "initialize the submodule and run: ${SCRIPT_DIR}/compile.sh" >&2
  exit 66
fi

CHECK_ONLY=0
for arg in "$@"; do
  if [[ "${arg}" == "--check" ]]; then
    CHECK_ONLY=1
  fi
done

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
  if [[ "${A3_ACTUATION_CONFIRM:-}" != "ENABLE_A3_ACTUATION" ||
        "${A3_ROBOT_SAFETY_READY:-}" != "1" ]]; then
    echo "real control is locked; set A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION and A3_ROBOT_SAFETY_READY=1" >&2
    exit 77
  fi
  if systemctl is-active --quiet agibot_pm; then
    echo "refusing to start while agibot_pm is active" >&2
    exit 78
  fi
  if ! pgrep -f './aimrt_main_hal' >/dev/null; then
    echo "required EtherCAT HAL process is not running" >&2
    exit 69
  fi
  if ! pgrep -f 'iox-roudi' >/dev/null; then
    echo "required iceoryx RouDi process is not running" >&2
    exit 69
  fi
fi

A3_ROBOT_ENV="${A3_ROBOT_ENV:-/agibot/software/v0/entry/env/env.sh}"
if [[ -f "${A3_ROBOT_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${A3_ROBOT_ENV}"
  set -u
fi

TRANSPORT="${A3_TRANSPORT:-iceoryx}"
if [[ "${TRANSPORT}" == "ros2" ]]; then
  sed -i 's#a3_aimrt_config.iceoryx.yaml#a3_aimrt_config.ros2.yaml#' \
    "${SCRIPT_DIR}/config/a3_lower_body.yaml"
elif [[ "${TRANSPORT}" == "iceoryx" ]]; then
  sed -i 's#a3_aimrt_config.ros2.yaml#a3_aimrt_config.iceoryx.yaml#' \
    "${SCRIPT_DIR}/config/a3_lower_body.yaml"
else
  echo "A3_TRANSPORT must be iceoryx or ros2" >&2
  exit 64
fi
if [[ -f "${SCRIPT_DIR}/setup_ros2_msgs.bash" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/setup_ros2_msgs.bash"
fi
export LD_LIBRARY_PATH="${A3_DEPLOY_PACKAGE_DIR}:${A3_DEPLOY_PACKAGE_DIR}/lib:${SCRIPT_DIR}:${LD_LIBRARY_PATH:-}"
exec "${A3_DEPLOY_PACKAGE_DIR}/hope_lower_body_deploy" \
  --config "${SCRIPT_DIR}/config/a3_lower_body.yaml" \
  --ik-serve-config "${SCRIPT_DIR}/config/a3_serve_ik.yaml" \
  "$@"
