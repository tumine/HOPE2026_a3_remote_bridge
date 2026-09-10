#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
sha256sum --check SHA256SUMS
echo "[成功] 压缩包文件校验通过"
