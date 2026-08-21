#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_a3_teach_and_fetch.sh [keyframes|trajectory]

Without an argument, interactively select one of:
  1  Record two keyframes (windup and contact; HOME is configured)
  2  Record one continuous right-arm path

The selected C++ recorder runs on agi@192.168.1.101 in an interactive SSH
terminal. After Q saves and exits, this script downloads the current run to:

  artifacts/a3_teach_downloads/<mode>/<MMDD_HHMM>/

Environment overrides:
  A3_TEACH_ROBOT       SSH destination (default: agi@192.168.1.101)
  A3_TEACH_REMOTE_DIR  Robot deploy directory (default: /agibot/a3_deploy)
  A3_TEACH_LOCAL_ROOT  Local download root
  A3_TEACH_ALIGN_S     Left-arm alignment duration (default: 8)
  A3_TEACH_DAMPING     Right-arm damping (default: 4)
  A3_TEACH_MAX_S       Maximum continuous recording duration (default: 120)
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 64
fi

if [[ $# -eq 0 ]]; then
  echo "请选择录制模式："
  echo "  1) 记录两个关键点：蓄力位、挥过/击球位（HOME 使用配置值）"
  echo "  2) 记录完整连续轨迹"
  read -r -p "请输入 1 或 2：" SELECTION
else
  SELECTION="$1"
fi

case "${SELECTION,,}" in
  1|keyframe|keyframes|point|points)
    MODE="keyframes"
    ;;
  2|trajectory|traj)
    MODE="trajectory"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "无效选择：${SELECTION}；请输入 1 或 2" >&2
    exit 64
    ;;
esac

for program in ssh rsync tee; do
  if ! command -v "${program}" >/dev/null 2>&1; then
    echo "required local program is missing: ${program}" >&2
    exit 69
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOT="${A3_TEACH_ROBOT:-agi@192.168.1.101}"
REMOTE_DIR="${A3_TEACH_REMOTE_DIR:-/agibot/a3_deploy}"
LOCAL_ROOT="${A3_TEACH_LOCAL_ROOT:-${SCRIPT_DIR}/artifacts/a3_teach_downloads}"
ALIGN_S="${A3_TEACH_ALIGN_S:-8}"
DAMPING="${A3_TEACH_DAMPING:-4}"
MAX_S="${A3_TEACH_MAX_S:-120}"
RUN_ID="$(date +%m%d_%H%M)"
LOCAL_DIR="${LOCAL_ROOT}/${MODE}/${RUN_ID}"
REMOTE_OUTPUT_DIR="${REMOTE_DIR}/artifacts/a3_teach"

if [[ "${MODE}" == "trajectory" ]]; then
  RUNNER="run_hope_a3_teach_trajectory.sh"
  BASENAME="trajectory_${RUN_ID}"
else
  RUNNER="run_hope_a3_teach_keyframes.sh"
  BASENAME="keyframes_${RUN_ID}"
fi

REMOTE_YAML="${REMOTE_OUTPUT_DIR}/${BASENAME}.yaml"
REMOTE_CSV="${REMOTE_OUTPUT_DIR}/${BASENAME}.csv"
mkdir -p "${LOCAL_DIR}"

CONTROL_SOCKET="${TMPDIR:-/tmp}/a3-teach-ssh-${UID}-$$"
SSH_OPTIONS=(
  -o ControlMaster=auto
  -o ControlPersist=60
  -o "ControlPath=${CONTROL_SOCKET}"
)

close_ssh() {
  ssh "${SSH_OPTIONS[@]}" -O exit "${ROBOT}" >/dev/null 2>&1 || true
  rm -f -- "${CONTROL_SOCKET}"
}
trap close_ssh EXIT

printf 'Mode: %s\nRobot: %s\nLocal output: %s\n' \
  "${MODE}" "${ROBOT}" "${LOCAL_DIR}"

ssh "${SSH_OPTIONS[@]}" "${ROBOT}" \
  "cd '${REMOTE_DIR}' && './${RUNNER}' --check"

REMOTE_COMMAND="cd '${REMOTE_DIR}' && \
export A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION && \
export A3_ROBOT_SAFETY_READY=1 && \
export A3_TRANSPORT=iceoryx && \
exec './${RUNNER}' --enable-command \
--align-duration-s '${ALIGN_S}' \
--right-damping '${DAMPING}' \
--output '${REMOTE_YAML}'"
if [[ "${MODE}" == "trajectory" ]]; then
  REMOTE_COMMAND+=" --max-duration-s '${MAX_S}'"
fi

echo "Starting the robot recorder. Use Q to save and exit."
set +e
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT}" "${REMOTE_COMMAND}" 2>&1 \
  | tee "${LOCAL_DIR}/session.log"
REMOTE_STATUS=${PIPESTATUS[0]}
set -e

if ! ssh "${SSH_OPTIONS[@]}" "${ROBOT}" "test -f '${REMOTE_YAML}'"; then
  echo "robot recorder did not produce ${REMOTE_YAML}" >&2
  echo "session log retained at ${LOCAL_DIR}/session.log" >&2
  if [[ ${REMOTE_STATUS} -ne 0 ]]; then
    exit "${REMOTE_STATUS}"
  fi
  exit 1
fi

RSYNC_SHELL="ssh -o ControlMaster=no -o ControlPath=${CONTROL_SOCKET}"
rsync -azP -e "${RSYNC_SHELL}" \
  "${ROBOT}:${REMOTE_YAML}" "${LOCAL_DIR}/"
if [[ "${MODE}" == "trajectory" ]]; then
  rsync -azP -e "${RSYNC_SHELL}" \
    "${ROBOT}:${REMOTE_CSV}" "${LOCAL_DIR}/"
fi

echo "Download complete: ${LOCAL_DIR}"
find "${LOCAL_DIR}" -maxdepth 1 -type f -printf '  %f\n' | sort

if [[ ${REMOTE_STATUS} -ne 0 ]]; then
  echo "warning: recorder exited with status ${REMOTE_STATUS}; files were still downloaded" >&2
  exit "${REMOTE_STATUS}"
fi
