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

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT_DIR}/config/fastrtps_mdu.xml"
export LD_LIBRARY_PATH="${ROOT_DIR}/dist:/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"

COMMAND_ENABLE=0
for arg in "$@"; do
  if [[ "${arg}" == "--command-enable" ]]; then
    COMMAND_ENABLE=1
    break
  fi
done

if (( COMMAND_ENABLE )); then
  MOTION_CONFLICTS="$(
    pgrep -af '(^|/)motion_control([[:space:]]|$)|start_motion_control\.sh' || true
  )"
  BRIDGE_CONFLICTS="$(pgrep -af '(^|/)a3_mdu_state_bridge([[:space:]]|$)' || true)"
  if [[ -n "${MOTION_CONFLICTS}" || -n "${BRIDGE_CONFLICTS}" ]]; then
    echo "refusing real control: another command source or bridge is running" >&2
    if [[ -n "${MOTION_CONFLICTS}" ]]; then
      echo "motion_control conflicts:" >&2
      printf '%s\n' "${MOTION_CONFLICTS}" >&2
    fi
    if [[ -n "${BRIDGE_CONFLICTS}" ]]; then
      echo "bridge conflicts:" >&2
      printf '%s\n' "${BRIDGE_CONFLICTS}" >&2
    fi
    echo "stop both the motion_control child and start_motion_control.sh wrapper, then stop any existing state bridge before retrying" >&2
    exit 73
  fi
fi

# The MDU process receives a constrained CPU set. A 55 Hz internal schedule
# provides a measured direct-host delivery rate above the required 50 Hz.
exec "${ROOT_DIR}/dist/a3_mdu_state_bridge" \
  --aimrt-cfg "${ROOT_DIR}/config/a3_mdu_iceoryx.yaml" \
  --topic-prefix /a3_internal \
  --publish-hz "${A3_STATE_PUBLISH_HZ:-55}" \
  "$@"
