#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
MOCAP_SETUP="${A3_MOCAP_ROS_SETUP:-/home/bth/workspace/Mocap/install/setup.bash}"
VENV_DIR="${A3_RL_VENV:-${ROOT_DIR}/.venv_rl}"
PC_WIRED_ADDRESS="${A3_PC_WIRED_ADDRESS:-192.168.1.11}"
MDU_ADDRESS="${A3_MDU_ADDRESS:-192.168.1.100}"

for setup in "${ROS_SETUP}" "${MOCAP_SETUP}"; do
  if [[ ! -r "${setup}" ]]; then
    echo "required ROS setup not found: ${setup}" >&2
    exit 66
  fi
done
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "RL Python environment missing: ${VENV_DIR}" >&2
  exit 66
fi
if ! ip -4 -o addr show | grep -Fq " ${PC_WIRED_ADDRESS}/"; then
  echo "wired address ${PC_WIRED_ADDRESS} is not configured on this PC" >&2
  exit 69
fi
if ! ip route get "${MDU_ADDRESS}" 2>/dev/null | grep -Fq "src ${PC_WIRED_ADDRESS}"; then
  echo "route to MDU ${MDU_ADDRESS} does not use ${PC_WIRED_ADDRESS}" >&2
  ip route get "${MDU_ADDRESS}" >&2 || true
  exit 69
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${MOCAP_SETUP}"
set -u

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_pc_planner_mdu.xml}"
export ROS_LOG_DIR="${A3_ROS_LOG_DIR:-${ROOT_DIR}/log/ros}"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
mkdir -p -- "${ROS_LOG_DIR}"

exec "${VENV_DIR}/bin/python" \
  "${ROOT_DIR}/pc_tools/a3_receive_input_gateway.py" "$@"
