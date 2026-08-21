#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_UPSTREAM_DIR="${SCRIPT_DIR}/hope_deploy"
WORKSPACE_UPSTREAM_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)/HOPE2026_serve_deploy"
UPSTREAM_DIR="${HOPE_DEPLOY_DIR:-${DEFAULT_UPSTREAM_DIR}}"

# This repository keeps the robot-facing deployment wrapper separate from the
# HOPE controller source. During development, use the sibling source checkout
# when the optional nested checkout has not been initialized.
if [[ ! -f "${UPSTREAM_DIR}/scripts/build_a3_deploy_pkg.sh" &&
      -f "${WORKSPACE_UPSTREAM_DIR}/scripts/build_a3_deploy_pkg.sh" ]]; then
  UPSTREAM_DIR="${WORKSPACE_UPSTREAM_DIR}"
fi

if [[ ! -f "${UPSTREAM_DIR}/scripts/build_a3_deploy_pkg.sh" ]]; then
  cat >&2 <<EOF
HOPE deploy source is unavailable.
Expected: ${DEFAULT_UPSTREAM_DIR}
Or set HOPE_DEPLOY_DIR to a HOPE2026_serve_deploy checkout.
EOF
  exit 66
fi

cd "${UPSTREAM_DIR}"
exec scripts/build_a3_deploy_pkg.sh "$@" --arch rockchip --disable-auto-rknn
