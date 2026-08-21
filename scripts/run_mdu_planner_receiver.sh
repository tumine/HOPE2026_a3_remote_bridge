#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_ENV="${A3_ROBOT_ENV:-/agibot/software/v0/entry/env/env.sh}"
INPUT_TRANSPORT="${A3_PLANNER_INPUT_TRANSPORT:-udp}"
SERVE_VCF_ENABLED="${A3_ENABLE_SERVE_VCF:-${A3_ENABLE_COMMAND_PUBLISH:-0}}"
HAL_SOFTWARE_ROOT="${A3_HAL_SOFTWARE_ROOT:-/agibot/software/v0}"
HAL_START_ATTEMPTS="${A3_HAL_START_ATTEMPTS:-100}"
HAL_LOG_DIR="${A3_HAL_LOG_DIR:-${ROOT_DIR}/logs/hal_services}"

if [[ ! "${HAL_START_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[失败] A3_HAL_START_ATTEMPTS 必须是正整数" >&2
  exit 64
fi

ethercat_ready() {
  pgrep -f './aimrt_main_hal([[:space:]]|$)' >/dev/null
}

gripper_ready() {
  local port="${A3_GRIPPER_PORT:-56422}"
  ss -ltn 2>/dev/null | awk -v suffix=":${port}" '$4 ~ suffix "$" { found=1 } END { exit !found }'
}

print_service_log_tail() {
  local log_file="$1"
  if [[ -s "${log_file}" ]]; then
    echo "----- 服务日志末尾：${log_file} -----" >&2
    tail -n 20 "${log_file}" >&2
    echo "----- 服务日志结束 -----" >&2
  fi
}

ensure_hal_service() {
  local name_cn="$1"
  local health_fn="$2"
  local relative_script="$3"
  local log_file="$4"
  local script_path="${HAL_SOFTWARE_ROOT}/${relative_script}"
  local launcher_pid=""

  if "${health_fn}"; then
    echo "[成功] ${name_cn} 已在运行，无需重复启动"
    return 0
  fi
  if [[ ! -f "${script_path}" ]]; then
    echo "[失败] ${name_cn} 启动脚本不存在：${script_path}" >&2
    return 69
  fi
  mkdir -p "${HAL_LOG_DIR}"
  echo "[启动] 正在启动 ${name_cn}，日志：${log_file}"
  launcher_pid="$({
    cd "${HAL_SOFTWARE_ROOT}"
    nohup bash "${relative_script}" </dev/null >"${log_file}" 2>&1 &
    echo $!
  })"
  for ((attempt = 1; attempt <= HAL_START_ATTEMPTS; ++attempt)); do
    if "${health_fn}"; then
      echo "[成功] ${name_cn} 启动成功"
      return 0
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      echo "[失败] ${name_cn} 启动进程已提前退出" >&2
      print_service_log_tail "${log_file}"
      return 69
    fi
    sleep 0.2
  done
  echo "[失败] ${name_cn} 在等待时间内未就绪" >&2
  print_service_log_tail "${log_file}"
  return 69
}

ensure_required_hal_services() {
  ensure_hal_service \
    "EtherCAT HAL" ethercat_ready \
    "scripts/hal_ethercat/start_hal_ethercat.sh" \
    "${HAL_LOG_DIR}/hal_ethercat.log"
  ensure_hal_service \
    "夹爪服务 hal_elink" gripper_ready \
    "scripts/hal_elink/start_hal_elink.sh" \
    "${HAL_LOG_DIR}/hal_elink.log"
}

if [[ ! -f "${ROBOT_ENV}" ]]; then
  echo "robot environment not found: ${ROBOT_ENV}" >&2
  exit 66
fi

set +u
# shellcheck disable=SC1090
source "${ROBOT_ENV}"
set -u

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${ROOT_DIR}/dist:/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"
if [[ "${INPUT_TRANSPORT}" == "udp" ]]; then
  export ROS_LOCALHOST_ONLY=1
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
elif [[ "${INPUT_TRANSPORT}" == "ros2" ]]; then
  export ROS_LOCALHOST_ONLY=0
  export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
  export FASTRTPS_DEFAULT_PROFILES_FILE="${A3_PLANNER_FASTRTPS_PROFILE:-${ROOT_DIR}/config/fastrtps_mdu_planner.xml}"
  unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
else
  echo "A3_PLANNER_INPUT_TRANSPORT must be udp or ros2" >&2
  exit 64
fi

ROBOT_IO_ARGS=()
if [[ "${A3_ENABLE_ROBOT_IO_PROBE:-0}" == "1" ||
      "${A3_ENABLE_OBSERVATION_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ONNX_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ||
      "${A3_ENABLE_UPPER_BODY_SERVE_DRY_RUN:-0}" == "1" ||
      "${SERVE_VCF_ENABLED}" == "1" ||
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
      "${A3_ENABLE_UPPER_BODY_SERVE_DRY_RUN:-0}" == "1" ||
      "${SERVE_VCF_ENABLED}" == "1" ||
      "${A3_ENABLE_MANUAL_CONTROL:-0}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--observation-probe)
fi
if [[ "${A3_ENABLE_ONNX_PROBE:-0}" == "1" ||
      "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ||
      "${SERVE_VCF_ENABLED}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(
    --onnx-model "${A3_ONNX_MODEL:-${ROOT_DIR}/models/hope_pingpong.onnx}"
  )
fi
if [[ "${A3_ENABLE_ACTION_DRY_RUN:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--action-dry-run)
fi
if [[ "${A3_ENABLE_UPPER_BODY_SERVE_DRY_RUN:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--upper-body-serve-dry-run)
fi
if [[ "${SERVE_VCF_ENABLED}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--serve-vcf)
fi
if [[ "${A3_ENABLE_MANUAL_CONTROL:-0}" == "1" ||
      "${A3_ENABLE_UPPER_BODY_SERVE_DRY_RUN:-0}" == "1" ||
      "${SERVE_VCF_ENABLED}" == "1" ||
      "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  ROBOT_IO_ARGS+=(--manual-control)
fi
if [[ "${A3_ENABLE_COMMAND_PUBLISH:-0}" == "1" ]]; then
  if [[ "${A3_ACTUATION_CONFIRM:-}" != "ENABLE_A3_ACTUATION" ||
        "${A3_ROBOT_SAFETY_READY:-0}" != "1" ]]; then
    echo "command publishing refused: set A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION and A3_ROBOT_SAFETY_READY=1" >&2
    exit 78
  fi
  if [[ "${A3_GRIPPER_ACTUATION_CONFIRM:-}" != "ENABLE_A3_GRIPPER" ]]; then
    echo "[失败] 自动启动夹爪服务需要设置 A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER" >&2
    exit 78
  fi
  if systemctl is-active --quiet agibot_pm; then
    echo "command publishing refused: agibot_pm is active" >&2
    echo "stop it only after the robot is suspended/supported" >&2
    exit 73
  fi
  ensure_required_hal_services
  if ! pgrep -f './aimrt_main_hal([[:space:]]|$)' >/dev/null; then
    echo "[失败] 无法进入部署：EtherCAT HAL 未运行" >&2
    exit 69
  fi
  if ! pgrep -f '(^|/)iox-roudi([[:space:]]|$)' >/dev/null; then
    echo "[失败] 无法进入部署：iceoryx RouDi 未运行" >&2
    exit 69
  fi
  if [[ "${SERVE_VCF_ENABLED}" == "1" ]]; then
    ROBOT_IO_ARGS+=(
      --gripper-http
      --gripper-host "${A3_GRIPPER_HOST:-10.42.10.12}"
      --gripper-port "${A3_GRIPPER_PORT:-56422}"
      --gripper-open-position "${A3_GRIPPER_OPEN_POSITION:-4096}"
      --gripper-close-position "${A3_GRIPPER_CLOSE_POSITION:-1200}"
    )
  fi
  ROBOT_IO_ARGS+=(--publish-commands)
fi

WAIST_PITCH_GUARD_ARGS=(
  --waist-pitch-enter "${A3_WAIST_PITCH_GUARD_ENTER_RAD:-0.35}"
  --waist-pitch-release "${A3_WAIST_PITCH_GUARD_RELEASE_RAD:-0.27}"
  --waist-pitch-target "${A3_WAIST_PITCH_GUARD_TARGET_RAD:-0.00}"
  --waist-pitch-kp "${A3_WAIST_PITCH_GUARD_KP:-400}"
  --waist-pitch-kd "${A3_WAIST_PITCH_GUARD_KD:-8}"
)
case "${A3_WAIST_PITCH_GUARD_ENABLED:-0}" in
  1) WAIST_PITCH_GUARD_ARGS+=(--waist-pitch-guard) ;;
  0) WAIST_PITCH_GUARD_ARGS+=(--no-waist-pitch-guard) ;;
  *)
    echo "A3_WAIST_PITCH_GUARD_ENABLED must be 0 or 1" >&2
    exit 64
    ;;
esac

exec "${ROOT_DIR}/dist/a3_mdu_planner_receiver" \
  --input-transport "${INPUT_TRANSPORT}" \
  --command-topic "${A3_RACKET_COMMAND_TOPIC:-/racket/command}" \
  --base-pose-topic "${A3_BASE_POSE_TOPIC:-/a3_mocap/pelvis_pose}" \
  --expected-frame "${A3_CANONICAL_FRAME:-hope_table}" \
  --udp-bind-address "${A3_MDU_ADDRESS:-192.168.1.100}" \
  --udp-source-address "${A3_PC_WIRED_ADDRESS:-192.168.1.11}" \
  --udp-port "${A3_PLANNER_UDP_PORT:-15001}" \
  --command-timeout-ms "${A3_PLANNER_COMMAND_TIMEOUT_MS:-150}" \
  --base-pose-timeout-ms "${A3_BASE_POSE_TIMEOUT_MS:-100}" \
  --status-period-s "${A3_STATUS_PERIOD_S:-5}" \
  "${WAIST_PITCH_GUARD_ARGS[@]}" \
  "${ROBOT_IO_ARGS[@]}" \
  "$@"
