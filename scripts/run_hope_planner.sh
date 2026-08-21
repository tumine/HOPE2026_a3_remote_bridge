#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
HOPE_SETUP="${A3_HOPE_ROS_SETUP:-${ROOT_DIR}/.hope_ros2/install/setup.bash}"
TRAINING_SOURCE="${A3_TRAINING_SOURCE:-${ROOT_DIR}/26.7.25发球部署/pingpang_ustc-srf}"
PLANNER_CONFIG="${A3_HOPE_PLANNER_CONFIG:-${TRAINING_SOURCE}/hope_ws/src/hope_planner/config/hope_planner.yaml}"
MODEL_72500_BUNDLE="${A3_MODEL_72500_BUNDLE:-${ROOT_DIR}/model_72500_deploy_bundle}"
BALL_PHYSICS_CONFIG="${HOPE_BALL_PHYSICS_CONFIG:-${MODEL_72500_BUNDLE}/config/ball_physics.yaml}"
# Keep the explicitly requested live commissioning window. MuJoCo starts new
# tasks near 1.0 s; values below that are a documented real-planner exception.
LIVE_TTS_MIN_S="${A3_NEW_TASK_TTS_MIN_S:-0.20}"
LIVE_TTS_MAX_S="${A3_NEW_TASK_TTS_MAX_S:-1.00}"
# Bounded real-robot admission margins. These admit the near-edge misses seen
# in the 2026-08-19 ball logs while still rejecting large trajectory outliers.
LIVE_STRIKE_Y_MARGIN_M="${A3_STRIKE_Y_MARGIN_M:-0.08}"
LIVE_STRIKE_Z_MARGIN_M="${A3_STRIKE_Z_MARGIN_M:-0.08}"
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
export ROS_LOCALHOST_ONLY="${A3_ROS_LOCALHOST_ONLY:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${A3_ROS_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_pc_planner_mdu.xml}"
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
  -p ball_physics_path:="${BALL_PHYSICS_CONFIG}"
