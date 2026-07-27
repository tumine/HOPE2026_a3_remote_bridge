#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_ENV="${A3_ROBOT_ENV:-/agibot/software/v0/entry/env/env.sh}"
ROUDI_SESSION="a3_rl_roudi"
HAL_SESSION="a3_rl_hal"
BRIDGE_SESSION="a3_rl_bridge"

usage() {
  cat <<'EOF'
Usage: mdu_rl_stack.sh start|status|logs|stop|shutdown-owned

start requires both:
  A3_PHYSICAL_SAFETY_CONFIRM=ROBOT_SUSPENDED
  A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION

stop             stops only the RL control bridge
shutdown-owned   also stops RouDi/HAL only when this script started them
EOF
}

has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

start_session() {
  local session="$1"
  local command="$2"
  has_session "${session}" && tmux kill-session -t "${session}"
  tmux new-session -d -s "${session}" bash -lc "${command}"
}

wait_for_process() {
  local pattern="$1"
  local seconds="$2"
  local count
  for ((count = 0; count < seconds * 10; ++count)); do
    if pgrep -f "${pattern}" >/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

bridge_pids() {
  pgrep -f '(^|/)a3_mdu_state_bridge([[:space:]]|$)' || true
}

stop_read_only_bridges() {
  local pid command
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    command="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${command}" == *"--command-enable"* ]]; then
      echo "refusing start: a real-control bridge already exists: ${pid} ${command}" >&2
      return 73
    fi
    echo "stopping old read-only bridge: ${pid} ${command}"
    kill -TERM "${pid}"
  done < <(bridge_pids)

  for _ in {1..30}; do
    [[ -z "$(bridge_pids)" ]] && return 0
    sleep 0.1
  done
  echo "read-only bridge did not stop; refusing to force-kill it" >&2
  return 73
}

status() {
  printf 'host='; hostname
  printf 'agibot_pm='; systemctl is-active agibot_pm || true
  echo "managed tmux sessions:"
  tmux list-sessions 2>/dev/null | grep -E '^a3_rl_(roudi|hal|bridge):' || true
  echo "relevant processes:"
  pgrep -af 'iox-roudi|aimrt_main_hal|start_motion_control|(^|/)motion_control([[:space:]]|$)|a3_mdu_state_bridge' || true
}

logs() {
  if ! has_session "${BRIDGE_SESSION}"; then
    echo "${BRIDGE_SESSION} is not running" >&2
    return 69
  fi
  tmux capture-pane -p -t "${BRIDGE_SESSION}" -S -80
}

stop_bridge() {
  if has_session "${BRIDGE_SESSION}"; then
    tmux send-keys -t "${BRIDGE_SESSION}" C-c
    for _ in {1..30}; do
      has_session "${BRIDGE_SESSION}" || break
      sleep 0.1
    done
    has_session "${BRIDGE_SESSION}" && tmux kill-session -t "${BRIDGE_SESSION}"
    echo "RL control bridge stopped"
  else
    echo "RL control bridge is not managed by this script"
  fi
}

start_stack() {
  if [[ "${A3_PHYSICAL_SAFETY_CONFIRM:-}" != "ROBOT_SUSPENDED" ]]; then
    echo "refusing start: set A3_PHYSICAL_SAFETY_CONFIRM=ROBOT_SUSPENDED" >&2
    exit 77
  fi
  if [[ "${A3_ACTUATION_CONFIRM:-}" != "ENABLE_A3_ACTUATION" ]]; then
    echo "refusing start: set A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION" >&2
    exit 77
  fi
  if [[ ! -f "${ROBOT_ENV}" || ! -x "${ROOT_DIR}/scripts/run_mdu_rl_control.sh" ]]; then
    echo "MDU RL runtime files are incomplete under ${ROOT_DIR}" >&2
    exit 66
  fi
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is required on MDU" >&2
    exit 69
  }

  if systemctl is-active --quiet agibot_pm; then
    echo "stopping agibot_pm (sudo may ask for the MDU password)"
    sudo systemctl stop agibot_pm
  fi
  if systemctl is-active --quiet agibot_pm; then
    echo "agibot_pm is still active" >&2
    exit 73
  fi

  local motion_conflicts
  motion_conflicts="$(
    pgrep -af 'start_motion_control\.sh|(^|/)motion_control([[:space:]]|$)' || true
  )"
  if [[ -n "${motion_conflicts}" ]]; then
    echo "motion_control conflict remains; refusing RL control:" >&2
    printf '%s\n' "${motion_conflicts}" >&2
    exit 73
  fi

  if has_session "${BRIDGE_SESSION}"; then
    echo "${BRIDGE_SESSION} is already running"
    logs
    return 0
  fi
  stop_read_only_bridges

  if ! pgrep -f '(^|/)iox-roudi([[:space:]]|$)' >/dev/null; then
    echo "starting RouDi in tmux session ${ROUDI_SESSION}"
    start_session "${ROUDI_SESSION}" \
      "exec /usr/local/bin/roudi/iox-roudi --config-file=/usr/local/bin/roudi/cfg/roudi_config.toml -m off"
    wait_for_process '(^|/)iox-roudi([[:space:]]|$)' 10 || {
      echo "RouDi failed to start" >&2
      exit 70
    }
  fi

  if ! pgrep -f './aimrt_main_hal([[:space:]]|$)' >/dev/null; then
    echo "starting HAL in tmux session ${HAL_SESSION}"
    start_session "${HAL_SESSION}" \
      "source '${ROBOT_ENV}'; cd /agibot/software/v0; exec bash scripts/hal_ethercat/start_hal_ethercat.sh"
    wait_for_process './aimrt_main_hal([[:space:]]|$)' 20 || {
      echo "HAL failed to start; inspect: tmux capture-pane -p -t ${HAL_SESSION}" >&2
      exit 70
    }
  fi

  echo "starting the unique RL control bridge in tmux session ${BRIDGE_SESSION}"
  start_session "${BRIDGE_SESSION}" \
    "cd '${ROOT_DIR}'; export A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION; exec ./scripts/run_mdu_rl_control.sh"
  wait_for_process 'a3_mdu_state_bridge.*--command-enable' 10 || {
    tmux capture-pane -p -t "${BRIDGE_SESSION}" -S -80 2>/dev/null || true
    echo "RL control bridge failed to start" >&2
    exit 70
  }
  sleep 0.5
  logs
}

action="${1:-}"
case "${action}" in
  start)
    start_stack
    ;;
  status)
    status
    ;;
  logs)
    logs
    ;;
  stop)
    stop_bridge
    ;;
  shutdown-owned)
    stop_bridge
    for session in "${HAL_SESSION}" "${ROUDI_SESSION}"; do
      if has_session "${session}"; then
        tmux send-keys -t "${session}" C-c
        sleep 0.5
        has_session "${session}" && tmux kill-session -t "${session}"
        echo "stopped owned session ${session}"
      fi
    done
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
