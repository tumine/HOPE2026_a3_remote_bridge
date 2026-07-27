#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if systemctl is-active --quiet agibot_pm; then
  echo "refusing RL control: agibot_pm is active" >&2
  echo "stop it only after the robot is suspended/supported:" >&2
  echo "  sudo systemctl stop agibot_pm" >&2
  exit 73
fi
if ! pgrep -f './aimrt_main_hal([[:space:]]|$)' >/dev/null; then
  echo "refusing RL control: aimrt_main_hal is not running" >&2
  echo "start only hal_ethercat before this gateway" >&2
  exit 69
fi
if ! pgrep -f '(^|/)iox-roudi([[:space:]]|$)' >/dev/null; then
  echo "refusing RL control: iceoryx RouDi is not running" >&2
  exit 69
fi
if [[ "${A3_ACTUATION_CONFIRM:-}" != "ENABLE_A3_ACTUATION" ]]; then
  echo "set A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION to arm the MDU gateway" >&2
  exit 77
fi

# These maxima admit the frozen training/sim2sim command envelope.  The exact
# WAIT->F->WAIT->B->WAIT dynamic gate reaches 1.3079 rad command-to-state
# error, so 1.50 rad catches outliers without clipping valid policy output.
# They are validator ceilings; the PC message still carries the exact Kp/Kd.
exec "${ROOT_DIR}/scripts/run_mdu_state_bridge.sh" \
  --command-enable \
  --command-watchdog-ms 100 \
  --max-state-age-ms 100 \
  --max-position-delta-rad 1.50 \
  --max-session-excursion-rad 3.0 \
  --max-abs-velocity 0 \
  --max-abs-effort 0 \
  --max-kp 250 \
  --max-kd 8
