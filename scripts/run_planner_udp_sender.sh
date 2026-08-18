#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
HOPE_SETUP="${A3_HOPE_ROS_SETUP:-${ROOT_DIR}/.hope_ros2/install/setup.bash}"
VENV_DIR="${A3_RL_VENV:-${ROOT_DIR}/.venv_rl}"

for setup in "${ROS_SETUP}" "${HOPE_SETUP}"; do
  if [[ ! -r "${setup}" ]]; then
    echo "required ROS setup not found: ${setup}" >&2
    exit 66
  fi
done
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "RL Python environment missing: ${VENV_DIR}" >&2
  exit 66
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${HOPE_SETUP}"
set -u

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY="${A3_ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${A3_ROS_DISCOVERY_RANGE:-LOCALHOST}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_pc_local.xml}"
export ROS_LOG_DIR="${A3_ROS_LOG_DIR:-${ROOT_DIR}/log/ros}"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
mkdir -p -- "${ROS_LOG_DIR}"

exec "${VENV_DIR}/bin/python" "${ROOT_DIR}/pc_tools/a3_planner_udp_sender.py" \
  --source-address "${A3_PC_WIRED_ADDRESS:-192.168.1.11}" \
  --destination-address "${A3_MDU_ADDRESS:-192.168.1.100}" \
  --port "${A3_PLANNER_UDP_PORT:-15001}" \
  --rate-hz "${A3_PLANNER_UDP_HZ:-50}" \
  --pose-timeout-ms "${A3_BASE_POSE_TIMEOUT_MS:-100}" \
  --expected-frame "${A3_CANONICAL_FRAME:-hope_table}" \
  --status-period-s "${A3_STATUS_PERIOD_S:-5}" \
  "$@"
