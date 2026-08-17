#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_PID=""
PLANNER_PID=""

cleanup() {
  local pid
  for pid in "${PLANNER_PID}" "${GATEWAY_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${ROOT_DIR}/scripts/run_receive_input_gateway.sh" &
GATEWAY_PID=$!
"${ROOT_DIR}/scripts/run_hope_planner.sh" &
PLANNER_PID=$!

echo "local receive pipeline started: gateway=${GATEWAY_PID} planner=${PLANNER_PID}"
echo "publishing /racket/command and /a3_mocap/pelvis_pose for the MDU"

set +e
wait -n "${GATEWAY_PID}" "${PLANNER_PID}"
STATUS=$?
set -e
exit "${STATUS}"
