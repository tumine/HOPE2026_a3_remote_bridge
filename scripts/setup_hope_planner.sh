#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_SOURCE="${A3_TRAINING_SOURCE:-${ROOT_DIR}/26.7.25发球部署/pingpang_ustc-srf}"
OUTPUT_ROOT="${A3_HOPE_ROS_ROOT:-${ROOT_DIR}/.hope_ros2}"
BUILD_SCRIPT="${TRAINING_SOURCE}/hope_ws/scripts/build_ros_ascii.sh"

if [[ ! -x "${BUILD_SCRIPT}" ]]; then
  echo "HOPE workspace build script not found: ${BUILD_SCRIPT}" >&2
  exit 66
fi

exec "${BUILD_SCRIPT}" "${OUTPUT_ROOT}" --packages-up-to hope_planner
