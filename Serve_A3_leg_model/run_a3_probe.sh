#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export A3_LATENCY_LOG="${A3_LATENCY_LOG:-${A3_PROBE_LATENCY_LOG:-verbose}}"
exec "${SCRIPT_DIR}/run_a3.sh" \
  --probe \
  --probe-source "${A3_PROBE_SOURCE:-both}" \
  --frame-log-interval "${A3_FRAME_LOG_INTERVAL:-50}" \
  "$@"
