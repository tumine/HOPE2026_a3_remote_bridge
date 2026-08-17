#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_ENV="${A3_ROBOT_ENV:-/agibot/software/v0/entry/env/env.sh}"

if [[ ! -f "${ROBOT_ENV}" ]]; then
  echo "robot environment not found: ${ROBOT_ENV}" >&2
  exit 66
fi

set +u
# shellcheck disable=SC1090
source "${ROBOT_ENV}"
set -u

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_mdu_planner.xml}"
export LD_LIBRARY_PATH="${ROOT_DIR}/dist:/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI

ROBOT_IO_ARGS=()
if [[ "${A3_ENABLE_ROBOT_IO_PROBE:-0}" == "1" ||
      "${A3_ENABLE_OBSERVATION_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ONNX_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ||
      "${A3_ENABLE_MANUAL_CONTROL:-0}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(
    --aimrt-cfg "${A3_AIMRT_CONFIG:-${ROOT_DIR}/config/a3_mdu_iceoryx.yaml}"
    --state-timeout-ms "${A3_ROBOT_STATE_TIMEOUT_MS:-50}"
    --control-hz "${A3_POLICY_HZ:-50}"
  )
fi
if [[ "${A3_ENABLE_OBSERVATION_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ONNX_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ||
      "${A3_ENABLE_MANUAL_CONTROL:-0}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--observation-probe)
fi
if [[ "${A3_ENABLE_ONNX_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(
    --onnx-model "${A3_ONNX_MODEL:-${ROOT_DIR}/models/hope_pingpong.onnx}"
  )
fi
if [[ "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--action-dry-run)
fi
if [[ "${A3_ENABLE_MANUAL_CONTROL:-0}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--manual-control)
fi
if [[ "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  if [[ "${A3_ACTUATION_CONFIRM:-}" != "ENABLE_A3_ACTUATION" ||
        "${A3_ROBOT_SAFETY_READY:-0}" != "1" ]]; then
    echo "command publishing refused: set A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION and A3_ROBOT_SAFETY_READY=1" >&2
    exit 78
  fi
  if systemctl is-active --quiet agibot_pm; then
    echo "command publishing refused: agibot_pm is active" >&2
    echo "stop it only after the robot is suspended/supported" >&2
    exit 73
  fi
  if ! pgrep -f './aimrt_main_hal([[:space:]]|$)' >/dev/null; then
    echo "command publishing refused: aimrt_main_hal is not running" >&2
    exit 69
  fi
  if ! pgrep -f '(^|/)iox-roudi([[:space:]]|$)' >/dev/null; then
    echo "command publishing refused: iceoryx RouDi is not running" >&2
    exit 69
  fi
  ROBOT_IO_ARGS+=(--publish-commands)
fi

exec "${ROOT_DIR}/dist/a3_mdu_planner_receiver" \
  --command-topic "${A3_RACKET_COMMAND_TOPIC:-/racket/command}" \
  --base-pose-topic "${A3_BASE_POSE_TOPIC:-/a3_mocap/pelvis_pose}" \
  --expected-frame "${A3_CANONICAL_FRAME:-hope_table}" \
  --command-timeout-ms "${A3_PLANNER_COMMAND_TIMEOUT_MS:-150}" \
  --base-pose-timeout-ms "${A3_BASE_POSE_TIMEOUT_MS:-100}" \
  "${ROBOT_IO_ARGS[@]}" \
  "$@"
