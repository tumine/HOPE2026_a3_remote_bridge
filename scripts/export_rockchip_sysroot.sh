#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/export_rockchip_sysroot.sh [--image IMAGE] [--output PATH]

Options:
  --image IMAGE    Rockchip target image to export from.
                   Default: registry.agibot.com/agibot-tech/rockchip:1.0
  --output PATH    Output tarball.
                   Default: thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz
  -h, --help       Show this help message.

The arm64 image is only exported; it is not executed, so qemu is unnecessary.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="registry.agibot.com/agibot-tech/rockchip:1.0"
OUTPUT="${REPO_ROOT}/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz"
PLATFORM="linux/arm64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
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

if [[ -z "${IMAGE}" || -z "${OUTPUT}" ]]; then
  echo "--image and --output cannot be empty" >&2
  exit 64
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 69
fi

mkdir -p "$(dirname "${OUTPUT}")"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/a3-rockchip-sysroot.XXXXXX")"
container_id=""

cleanup() {
  if [[ -n "${container_id}" ]]; then
    docker rm -f "${container_id}" >/dev/null 2>&1 || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker pull --platform "${PLATFORM}" "${IMAGE}"
fi

container_id="$(docker create --platform "${PLATFORM}" "${IMAGE}" /bin/true)"
mkdir -p "${tmp_dir}/sysroot"

echo "extracting target sysroot paths from ${IMAGE}"
docker export "${container_id}" | tar --no-same-owner -C "${tmp_dir}/sysroot" -xf - \
  opt/ros/jazzy \
  usr/include \
  usr/share/eigen3 \
  usr/lib/aarch64-linux-gnu

tmp_output="${OUTPUT}.tmp"
rm -f "${tmp_output}"
tar -C "${tmp_dir}/sysroot" -czf "${tmp_output}" \
  opt/ros/jazzy \
  usr/include \
  usr/share/eigen3 \
  usr/lib/aarch64-linux-gnu
mv -f "${tmp_output}" "${OUTPUT}"

(
  cd "$(dirname "${OUTPUT}")"
  sha256sum "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
)

echo "Rockchip sysroot bundle ready: ${OUTPUT}"
