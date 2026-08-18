#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TTS="${1:-0.25}"
shift || true

exec "${BUNDLE_DIR}/scripts/run_mujoco.sh" \
  --no-serve \
  --no-serve-time-to-strike "${TTS}" \
  --num-serves 1 \
  --hold-min-seconds 0 \
  --hold-max-seconds 0 \
  --stand-min-hold-seconds 0 \
  --diagnostics-out "/tmp/model50000_tts_${TTS}.json" \
  "$@"
