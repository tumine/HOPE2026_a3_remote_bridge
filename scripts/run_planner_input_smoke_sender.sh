#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
SENDER="${A3_PLANNER_SMOKE_SENDER:-${ROOT_DIR}/build/src/a3_pingpong_deploy/a3_planner_input_smoke_sender}"

if [[ ! -r "${ROS_SETUP}" ]]; then
  echo "required ROS setup not found: ${ROS_SETUP}" >&2
  exit 66
fi
if [[ ! -x "${SENDER}" ]]; then
  echo "planner input smoke sender not found: ${SENDER}" >&2
  echo "build the local test targets first" >&2
  exit 66
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_pc_planner_mdu.xml}"
export LD_LIBRARY_PATH="${ROOT_DIR}/build/thirdparty/hope_msgs:${LD_LIBRARY_PATH:-}"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI

echo "sending pose and RacketCommand smoke data; joint command publishers=0"
exec "${SENDER}" "$@"
