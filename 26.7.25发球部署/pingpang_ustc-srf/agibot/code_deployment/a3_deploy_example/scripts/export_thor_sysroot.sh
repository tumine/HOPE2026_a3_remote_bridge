#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/export_thor_sysroot.sh [--image IMAGE] [--output PATH]

Options:
  --image IMAGE    Thor target image to export from.
                   Default: registry.agibot.com/agibot-tech/devops/thor:1.0
  --output PATH    Output tarball.
                   Default: a3_deploy_example/thirdparty/thor_sysroot/thor-1.0-aarch64-sysroot.tar.gz
  -h, --help       Show this help message.

The script creates a container from the arm64 Thor image and copies the target
sysroot pieces needed by Dockerfile.a3-thor-builder. It does not execute inside
the arm64 container, so qemu is not required.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GEAR_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="registry.agibot.com/agibot-tech/devops/thor:1.0"
OUTPUT="${GEAR_ROOT}/thirdparty/thor_sysroot/thor-1.0-aarch64-sysroot.tar.gz"
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

if [[ -z "${IMAGE}" ]]; then
  echo "--image cannot be empty" >&2
  exit 64
fi
if [[ -z "${OUTPUT}" ]]; then
  echo "--output cannot be empty" >&2
  exit 64
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 69
fi

mkdir -p "$(dirname "${OUTPUT}")"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/thor-sysroot-export.XXXXXX")"
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

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "$(dirname "${OUTPUT}")"
    sha256sum "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
  )
fi

echo "Thor sysroot bundle ready: ${OUTPUT}"
