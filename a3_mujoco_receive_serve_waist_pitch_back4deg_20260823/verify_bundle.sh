#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
sha256sum --check SHA256SUMS
"${PYTHON_BIN:-python3}" "${ROOT_DIR}/scripts/verify_waist_pitch_contract.py"
echo "[成功] 压缩包文件校验通过"
