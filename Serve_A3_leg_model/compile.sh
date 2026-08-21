#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/hope_deploy"

if [[ ! -f "${UPSTREAM_DIR}/scripts/build_a3_deploy_pkg.sh" ]]; then
  cat >&2 <<EOF
hope_deploy submodule is not initialized: ${UPSTREAM_DIR}
Run: git submodule update --init --recursive
EOF
  exit 66
fi

cd "${UPSTREAM_DIR}"
exec scripts/build_a3_deploy_pkg.sh "$@" --arch rockchip --disable-auto-rknn
