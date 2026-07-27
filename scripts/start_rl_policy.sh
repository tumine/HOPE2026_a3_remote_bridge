#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MDU_HOST="${A3_MDU_SSH_HOST:-agi@10.42.10.12}"
ROBOT_SUSPENDED=0
DRY_RUN=0
LINE_CONSOLE=0
START_RVIZ=0
PLANNER_PID=""
PLANNER_EVENT_PID=""
RVIZ_PID=""

usage() {
  cat <<'EOF'
Usage:
  start_rl_policy.sh --dry-run [--rviz]
  start_rl_policy.sh --robot-suspended [--rviz]

Options:
  --dry-run          run mocap/planner/inference without a command publisher
  --robot-suspended  required for real control after support/e-stop checks
  --line-console     use text commands + Enter instead of single-key hotkeys
  --rviz             open the aligned live A3 model and physical-ground view
  --mdu USER@HOST    default agi@10.42.10.12

Hotkeys: P=training default pose, D=live policy (only after every gate passes),
         I=status, C=recalibrate in OBSERVE, S=normal stop, X=halt, H=help.
Live balls are automatic through PPMocap -> HOPE planner; F/B injection is removed.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --robot-suspended) ROBOT_SUSPENDED=1; shift ;;
    --line-console) LINE_CONSOLE=1; shift ;;
    --rviz) START_RVIZ=1; shift ;;
    --mdu)
      [[ $# -ge 2 ]] || { echo "--mdu needs USER@HOST" >&2; exit 64; }
      MDU_HOST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if ((DRY_RUN && ROBOT_SUSPENDED)); then
  echo "--dry-run and --robot-suspended are mutually exclusive" >&2
  exit 64
fi
if (( ! DRY_RUN && ! ROBOT_SUSPENDED )); then
  echo "refusing real control: pass --robot-suspended only after physical checks" >&2
  exit 77
fi

PLANNER_LOG="${A3_HOPE_PLANNER_LOG:-${ROOT_DIR}/log/hope_planner.log}"
mkdir -p -- "$(dirname -- "${PLANNER_LOG}")"
"${ROOT_DIR}/scripts/run_hope_planner.sh" >"${PLANNER_LOG}" 2>&1 &
PLANNER_PID=$!

SSH_TMP=""
SSH_SOCKET=""
MASTER_READY=0
REMOTE_STARTED=0

stop_local_process() {
  local pid="$1"
  local label="$2"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi

  kill -INT "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.05
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.05
    done
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    echo "warning: ${label} PID ${pid} ignored INT/TERM; sending KILL" >&2
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local result=$?
  trap - EXIT INT TERM
  if [[ -n "${RVIZ_PID}" ]]; then
    stop_local_process "${RVIZ_PID}" "A3 RViz"
  fi
  if [[ -n "${PLANNER_EVENT_PID}" ]]; then
    stop_local_process "${PLANNER_EVENT_PID}" "planner event monitor"
  fi
  if [[ -n "${PLANNER_PID}" ]]; then
    stop_local_process "${PLANNER_PID}" "HOPE planner"
  fi
  if ((MASTER_READY)); then
    if ((REMOTE_STARTED)); then
      ssh -S "${SSH_SOCKET}" "${MDU_HOST}" \
        '/agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh stop' || true
    else
      ssh -S "${SSH_SOCKET}" "${MDU_HOST}" \
        '/agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh shutdown-owned' || true
    fi
    ssh -S "${SSH_SOCKET}" -O exit "${MDU_HOST}" >/dev/null 2>&1 || true
  fi
  [[ -z "${SSH_TMP}" ]] || rmdir "${SSH_TMP}" 2>/dev/null || true
  exit "${result}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_rviz() {
  ((START_RVIZ)) || return 0
  "${ROOT_DIR}/scripts/run_a3_rviz.sh" &
  RVIZ_PID=$!
  sleep 1
  if ! kill -0 "${RVIZ_PID}" 2>/dev/null; then
    wait "${RVIZ_PID}" || true
    echo "A3 RViz failed to start; run scripts/run_a3_rviz.sh --check" >&2
    exit 70
  fi
  echo "A3 RViz started: live PPMocap pelvis + /a3_internal/joint_states + ground"
}

sleep 1
if ! kill -0 "${PLANNER_PID}" 2>/dev/null; then
  echo "HOPE planner failed to start; inspect ${PLANNER_LOG}" >&2
  exit 70
fi
echo "HOPE planner started (log: ${PLANNER_LOG})"
echo "Live new-task TTS window: [${A3_NEW_TASK_TTS_MIN_S:-0.25}, ${A3_NEW_TASK_TTS_MAX_S:-1.00}] s"
echo "Live strike margins: y=+/-${A3_STRIKE_Y_MARGIN_M:-0.07}m z=+/-${A3_STRIKE_Z_MARGIN_M:-0.04}m racket_v=+/-${A3_RACKET_VELOCITY_MARGIN_MPS:-0.15}m/s"
echo "The PPMocap driver must already run in ROS_DOMAIN_ID=${A3_ROS_DOMAIN_ID:-232}."
# Show lifecycle, timing, publication, physical-plane, and throttled rejection
# events in the interactive terminal. Full planner output remains in PLANNER_LOG.
# GNU tail exits automatically when this parent script exits.
tail --pid="$$" -n 0 -F "${PLANNER_LOG}" 2>/dev/null |
  grep --line-buffered -E 'BALL_REARM|TTS_GATE|PLANNER_TASK_PUBLISHED|STRIKE_PLANE_CROSS|command not published|incoming ball reached' &
PLANNER_EVENT_PID=$!

pc_args=()
((LINE_CONSOLE)) && pc_args+=(--line-console)
if ((DRY_RUN)); then
  start_rviz
  "${ROOT_DIR}/scripts/run_pc_rl.sh" "${pc_args[@]}"
  exit $?
fi

SSH_TMP="$(mktemp -d "${TMPDIR:-/tmp}/a3_rl_ssh.XXXXXX")"
SSH_SOCKET="${SSH_TMP}/control.sock"
echo "opening one reusable SSH connection to ${MDU_HOST}"
echo "enter the MDU password when prompted; it is not stored"
ssh -M -S "${SSH_SOCKET}" -o ControlMaster=yes -o ControlPersist=no \
  -o StrictHostKeyChecking=accept-new -fnNT "${MDU_HOST}"
MASTER_READY=1

scp -o ControlPath="${SSH_SOCKET}" \
  "${ROOT_DIR}/scripts/run_mdu_rl_control.sh" \
  "${ROOT_DIR}/scripts/mdu_rl_stack.sh" \
  "${MDU_HOST}:/agibot/a3_remote_bridge/scripts/"
ssh -S "${SSH_SOCKET}" "${MDU_HOST}" \
  'chmod +x /agibot/a3_remote_bridge/scripts/run_mdu_rl_control.sh /agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh'

ssh -tt -S "${SSH_SOCKET}" "${MDU_HOST}" \
  'A3_PHYSICAL_SAFETY_CONFIRM=ROBOT_SUSPENDED A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION /agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh start'
REMOTE_STARTED=1

echo "MDU armed. PC remains in OBSERVE until P, then D."
start_rviz
"${ROOT_DIR}/scripts/run_pc_rl.sh" \
  --enable-actuation \
  --i-understand-this-can-move-the-robot \
  "${pc_args[@]}"
