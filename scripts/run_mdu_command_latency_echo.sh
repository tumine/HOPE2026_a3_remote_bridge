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

exec python3 "${ROOT_DIR}/mdu_tools/a3_command_latency_echo.py"

