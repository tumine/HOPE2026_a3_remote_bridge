#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_PID=""
PLANNER_PID=""
UDP_SENDER_PID=""

export A3_ROS_LOCALHOST_ONLY="${A3_ROS_LOCALHOST_ONLY:-1}"
export A3_ROS_DISCOVERY_RANGE="${A3_ROS_DISCOVERY_RANGE:-LOCALHOST}"
export A3_PLANNER_FASTRTPS_PROFILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_pc_local.xml}"

cleanup() {
  local pid
  for pid in "${UDP_SENDER_PID}" "${PLANNER_PID}" "${GATEWAY_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${ROOT_DIR}/scripts/run_receive_input_gateway.sh" &
GATEWAY_PID=$!
"${ROOT_DIR}/scripts/run_hope_planner.sh" \
  > >(python3 -u "${ROOT_DIR}/pc_tools/a3_planner_event_cn.py" --passthrough) 2>&1 &
PLANNER_PID=$!
"${ROOT_DIR}/scripts/run_planner_udp_sender.sh" &
UDP_SENDER_PID=$!

echo "local receive pipeline started: gateway=${GATEWAY_PID} planner=${PLANNER_PID} udp_sender=${UDP_SENDER_PID}"
echo "ROS2 stays local; A3PP UDP -> ${A3_MDU_ADDRESS:-192.168.1.100}:${A3_PLANNER_UDP_PORT:-15001} at ${A3_PLANNER_UDP_HZ:-50} Hz"
echo "击球允许边界扩展: y=±${A3_STRIKE_Y_MARGIN_M:-0.08}m z=±${A3_STRIKE_Z_MARGIN_M:-0.08}m 目标拍速=±${A3_RACKET_VELOCITY_MARGIN_MPS:-0.15}m/s"
echo "球事件中文日志已启用；原始 planner 日志同时保留"

set +e
wait -n "${GATEWAY_PID}" "${PLANNER_PID}" "${UDP_SENDER_PID}"
STATUS=$?
set -e
exit "${STATUS}"
