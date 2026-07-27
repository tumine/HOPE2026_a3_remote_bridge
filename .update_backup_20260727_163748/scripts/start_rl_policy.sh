#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MDU_HOST="${A3_MDU_SSH_HOST:-agi@10.42.10.12}"
POLICY_BLEND="1.0"
ROBOT_SUSPENDED=0
DRY_RUN=0
ALLOW_FULL=1
LINE_CONSOLE=0

usage() {
  cat <<'EOF'
Usage:
  start_rl_policy.sh --dry-run
  start_rl_policy.sh --robot-suspended [--policy-blend VALUE]

Options:
  --dry-run              PC inference only; do not change MDU or publish commands
  --robot-suspended      required for real control; robot is physically supported
  --policy-blend VALUE   default 1.0; real control rejects any other value
  --allow-full-policy    retained for direct-run compatibility (already enabled here)
  --line-console         use text commands + Enter instead of hotkeys
  --mdu USER@HOST        default agi@10.42.10.12

Hotkeys: P ramps to and holds the training default pose,
         D enters the no-ball READY policy only after pose_ready=yes,
         F inject one forehand ball,
         B inject one backhand ball, I status,
         S normal stop, X immediate halt, H help, Q quit in OBSERVE.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --robot-suspended)
      ROBOT_SUSPENDED=1
      shift
      ;;
    --policy-blend)
      [[ $# -ge 2 ]] || { echo "--policy-blend needs a value" >&2; exit 64; }
      POLICY_BLEND="$2"
      shift 2
      ;;
    --allow-full-policy)
      ALLOW_FULL=1
      shift
      ;;
    --line-console)
      LINE_CONSOLE=1
      shift
      ;;
    --mdu)
      [[ $# -ge 2 ]] || { echo "--mdu needs USER@HOST" >&2; exit 64; }
      MDU_HOST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

pc_args=(--policy-blend "${POLICY_BLEND}")
((LINE_CONSOLE)) && pc_args+=(--line-console)
((ALLOW_FULL)) && pc_args+=(--allow-full-policy)

if ((DRY_RUN)); then
  if ((ROBOT_SUSPENDED)); then
    echo "--dry-run and --robot-suspended are mutually exclusive" >&2
    exit 64
  fi
  exec "${ROOT_DIR}/scripts/run_pc_rl.sh" "${pc_args[@]}"
fi

if (( ! ROBOT_SUSPENDED )); then
  echo "refusing real control: pass --robot-suspended only after physical support and e-stop checks" >&2
  exit 77
fi

SSH_TMP="$(mktemp -d "${TMPDIR:-/tmp}/a3_rl_ssh.XXXXXX")"
SSH_SOCKET="${SSH_TMP}/control.sock"
MASTER_READY=0
REMOTE_STARTED=0

cleanup() {
  local result=$?
  trap - EXIT INT TERM
  if ((MASTER_READY)); then
    if ((REMOTE_STARTED)); then
      echo "stopping the MDU RL bridge; HAL/RouDi remain available"
      ssh -S "${SSH_SOCKET}" "${MDU_HOST}" \
        '/agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh stop' || true
    else
      ssh -S "${SSH_SOCKET}" "${MDU_HOST}" \
        '/agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh shutdown-owned' || true
    fi
    ssh -S "${SSH_SOCKET}" -O exit "${MDU_HOST}" >/dev/null 2>&1 || true
  fi
  rmdir "${SSH_TMP}" 2>/dev/null || true
  exit "${result}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

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

echo "preparing the MDU RL stack"
ssh -tt -S "${SSH_SOCKET}" "${MDU_HOST}" \
  'A3_PHYSICAL_SAFETY_CONFIRM=ROBOT_SUSPENDED A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION /agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh start'
REMOTE_STARTED=1

echo
echo "MDU is armed. PC remains in OBSERVE until you explicitly press P."
echo "Hotkeys: P=default-pose D=enter-policy F=one-forehand-ball B=one-backhand-ball I=status S=stop X=halt H=help"
set +e
"${ROOT_DIR}/scripts/run_pc_rl.sh" \
  --enable-actuation \
  --i-understand-this-can-move-the-robot \
  "${pc_args[@]}"
runner_status=$?
set -e
exit "${runner_status}"
