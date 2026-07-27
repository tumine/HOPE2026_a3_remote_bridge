#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
HOPE_SETUP="${A3_HOPE_ROS_SETUP:-${ROOT_DIR}/.hope_ros2/install/setup.bash}"
TRAINING_SOURCE="${A3_TRAINING_SOURCE:-${ROOT_DIR}/26.7.25发球部署/pingpang_ustc-srf}"
PLANNER_CONFIG="${A3_HOPE_PLANNER_CONFIG:-${TRAINING_SOURCE}/hope_ws/src/hope_planner/config/hope_planner.yaml}"
# Live-ball experiment: the policy training clip starts at +1.0 s, but normal
# physical arrivals are first predictable much later. Keep the training YAML
# unchanged and make the deployment window explicit and easy to override.
LIVE_TTS_MIN_S="${A3_NEW_TASK_TTS_MIN_S:-0.25}"
LIVE_TTS_MAX_S="${A3_NEW_TASK_TTS_MAX_S:-1.00}"
LIVE_STRIKE_Y_MARGIN_M="${A3_STRIKE_Y_MARGIN_M:-0.07}"
LIVE_STRIKE_Z_MARGIN_M="${A3_STRIKE_Z_MARGIN_M:-0.04}"
LIVE_RACKET_VELOCITY_MARGIN_MPS="${A3_RACKET_VELOCITY_MARGIN_MPS:-0.15}"

for setup in "${ROS_SETUP}" "${HOPE_SETUP}"; do
  if [[ ! -r "${setup}" ]]; then
    echo "required ROS setup not found: ${setup}" >&2
    echo "run scripts/setup_hope_planner.sh once after a training/planner update" >&2
    exit 66
  fi
done

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${HOPE_SETUP}"
set -u

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT_DIR}/config/fastrtps_pc_direct_mdu.xml"
export ROS_LOG_DIR="${A3_ROS_LOG_DIR:-${ROOT_DIR}/log/ros}"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
mkdir -p -- "${ROS_LOG_DIR}"

PLANNER_PREFIX="$(ros2 pkg prefix hope_planner)"
PLANNER_EXECUTABLE="${A3_HOPE_PLANNER_EXECUTABLE:-${PLANNER_PREFIX}/lib/hope_planner/hope_planner_node}"
if [[ ! -x "${PLANNER_EXECUTABLE}" ]]; then
  echo "HOPE planner executable not found: ${PLANNER_EXECUTABLE}" >&2
  exit 66
fi

# Execute the installed node directly.  `ros2 run` starts the node as a child
# process; killing only the wrapper can otherwise leave an orphan publisher
# behind after S/X, Ctrl+C, or a failed startup.
exec "${PLANNER_EXECUTABLE}" --ros-args \
  --params-file "${PLANNER_CONFIG}" \
  -p poses_topic:=/a3_mocap/ball_poses \
  -p command_topic:=/racket/command \
  -p frame_id:=hope_table \
  -p new_task_tts_min_s:="${LIVE_TTS_MIN_S}" \
  -p new_task_tts_max_s:="${LIVE_TTS_MAX_S}" \
  -p strike_y_margin_m:="${LIVE_STRIKE_Y_MARGIN_M}" \
  -p strike_z_margin_m:="${LIVE_STRIKE_Z_MARGIN_M}" \
  -p racket_velocity_margin_mps:="${LIVE_RACKET_VELOCITY_MARGIN_MPS}" \
  -p ball_physics_path:="${TRAINING_SOURCE}/configs/ball_physics.yaml"
