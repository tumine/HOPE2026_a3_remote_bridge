#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
VENV_DIR="${A3_RL_VENV:-${ROOT_DIR}/.venv_rl}"
PC_WIRED_ADDRESS="${A3_PC_WIRED_ADDRESS:-192.168.123.123}"
MDU_ADDRESS="${A3_MDU_ADDRESS:-10.42.10.12}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 setup not found: ${ROS_SETUP}" >&2
  exit 66
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "RL Python environment is missing; run:" >&2
  echo "  ${ROOT_DIR}/scripts/setup_rl_venv.sh" >&2
  exit 66
fi
if ! ip -4 -o addr show | grep -Fq " ${PC_WIRED_ADDRESS}/"; then
  echo "wired address ${PC_WIRED_ADDRESS} is not configured on this PC" >&2
  exit 69
fi
if ! ip route get "${MDU_ADDRESS}" 2>/dev/null | grep -Fq "src ${PC_WIRED_ADDRESS}"; then
  echo "route to MDU ${MDU_ADDRESS} does not use source ${PC_WIRED_ADDRESS}" >&2
  ip route get "${MDU_ADDRESS}" >&2 || true
  exit 69
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup_pc_joint_msgs.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT_DIR}/config/fastrtps_pc_direct_mdu.xml"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI

exec "${VENV_DIR}/bin/python" \
  "${ROOT_DIR}/pc_tools/a3_rl_deploy.py" \
  "$@"
