#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/build_mdu_package.sh [--jobs N] [--sysroot PATH]

Build an aarch64 package for the Rockchip MDU from an x86_64 Docker host.

Options:
  --jobs N       Parallel build jobs. Default: nproc.
  --sysroot PATH Rockchip sysroot archive. Default:
                 thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz
                 Also searches ../rockchip_sysroot and ../a3_deploy_example.
  -h, --help     Show this help message.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
JOBS="$(nproc)"
SYSROOT="${REPO_ROOT}/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz"
WORKSPACE_SYSROOT="${REPO_ROOT}/../rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz"
LEGACY_SYSROOT="${REPO_ROOT}/../a3_deploy_example/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz"
SYSROOT_EXPLICIT=0
INSIDE_DOCKER=0
IMAGE="a3-mdu-rockchip-builder:1.0"
PROXY_ENV_NAMES=(
  http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --sysroot)
      SYSROOT="$(readlink -f "${2:-}")"
      SYSROOT_EXPLICIT=1
      shift 2
      ;;
    --inside-docker)
      INSIDE_DOCKER=1
      shift
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

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer; got '${JOBS}'" >&2
  exit 64
fi

append_proxy_build_args() {
  local -n output_args="$1"
  local name
  for name in "${PROXY_ENV_NAMES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      output_args+=(--build-arg "${name}=${!name}")
    fi
  done
}

append_proxy_run_args() {
  local -n output_args="$1"
  local name
  for name in "${PROXY_ENV_NAMES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      output_args+=(-e "${name}=${!name}")
    fi
  done
}

proxy_requires_host_network() {
  local name value
  for name in "${PROXY_ENV_NAMES[@]}"; do
    value="${!name:-}"
    if [[ "${value}" =~ (^|://)(localhost|127\.0\.0\.1)(:|/|$) ]]; then
      return 0
    fi
  done
  return 1
}

build_inside_docker() {
  local build_dir="${REPO_ROOT}/build/a3_mdu_rockchip"
  local package_dir="${REPO_ROOT}/dist/a3_mdu_state_bridge"

  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u

  cmake -S "${REPO_ROOT}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/cmake/toolchains/aarch64-linux-gnu.cmake" \
    -DGS_SKIP_ROSIDL_GENERATOR_PY=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_EXECUTABLE=/usr/bin/python3

  cmake --build "${build_dir}" \
    --target a3_mdu_state_bridge a3_hdu_command_dry_relay \
      a3_mdu_planner_receiver -j"${JOBS}"

  rm -rf "${package_dir}"
  mkdir -p "${package_dir}/dist" "${package_dir}/config" \
    "${package_dir}/models" "${package_dir}/scripts"
  cp -a "${build_dir}/dist/." "${package_dir}/dist/"
  cp -a "${build_dir}/models/." "${package_dir}/models/"
  cp -a "${REPO_ROOT}/config/a3_mdu_iceoryx.yaml" \
    "${REPO_ROOT}/config/fastrtps_mdu.xml" \
    "${REPO_ROOT}/config/fastrtps_mdu_planner.xml" \
    "${REPO_ROOT}/config/fastrtps_hdu_dual_nic.xml" \
    "${package_dir}/config/"
  cp -a "${REPO_ROOT}/scripts/run_mdu_state_bridge.sh" \
    "${REPO_ROOT}/scripts/run_mdu_rl_control.sh" \
    "${REPO_ROOT}/scripts/run_mdu_planner_receiver.sh" \
    "${REPO_ROOT}/scripts/mdu_rl_stack.sh" \
    "${REPO_ROOT}/scripts/run_hdu_command_dry_relay.sh" \
    "${package_dir}/scripts/"
  chmod +x "${package_dir}/scripts/run_mdu_state_bridge.sh" \
    "${package_dir}/scripts/run_mdu_rl_control.sh" \
    "${package_dir}/scripts/run_mdu_planner_receiver.sh" \
    "${package_dir}/scripts/mdu_rl_stack.sh" \
    "${package_dir}/scripts/run_hdu_command_dry_relay.sh"

  aarch64-linux-gnu-readelf -h "${package_dir}/dist/a3_mdu_state_bridge" \
    | grep -E 'Class:|Machine:'
  aarch64-linux-gnu-readelf -h "${package_dir}/dist/a3_hdu_command_dry_relay" \
    | grep -E 'Class:|Machine:'
  aarch64-linux-gnu-readelf -h "${package_dir}/dist/a3_mdu_planner_receiver" \
    | grep -E 'Class:|Machine:'
  aarch64-linux-gnu-readelf -h "${package_dir}/dist/libonnxruntime.so.1" \
    | grep -E 'Class:|Machine:'
  if ! aarch64-linux-gnu-readelf -d \
      "${package_dir}/dist/a3_mdu_planner_receiver" \
      | grep -F 'libonnxruntime.so.1' >/dev/null; then
    echo "packaged planner receiver is not linked to ONNX Runtime" >&2
    exit 1
  fi
  if [[ "$(sha256sum "${package_dir}/models/hope_pingpong.onnx" | cut -d' ' -f1)" \
      != "8f87d7d9fe6f47007b064cb65150e5fe82e98e944ddbfcb5653e6ed14e080057" ]]; then
    echo "packaged model_53000 SHA256 mismatch" >&2
    exit 1
  fi
  if ! aarch64-linux-gnu-nm -C "${package_dir}/dist/a3_mdu_state_bridge" \
      | grep -F ' T robot_io::CreateA3AimrtBackend()' >/dev/null; then
    echo "A3 backend strong factory symbol is missing from packaged executable" >&2
    exit 1
  fi
  if ! aarch64-linux-gnu-nm -C "${package_dir}/dist/a3_mdu_planner_receiver" \
      | grep -F ' T robot_io::CreateA3AimrtBackend()' >/dev/null; then
    echo "A3 backend is missing from packaged planner receiver" >&2
    exit 1
  fi
  echo "A3 backend factory: linked"
  echo "ONNX Runtime and model_53000: linked and verified"
  echo "MDU package ready: ${package_dir}"
}

if [[ "${INSIDE_DOCKER}" -eq 1 ]]; then
  build_inside_docker
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 69
fi
if [[ "${SYSROOT_EXPLICIT}" -eq 0 && ! -f "${SYSROOT}" && -f "${WORKSPACE_SYSROOT}" ]]; then
  SYSROOT="${WORKSPACE_SYSROOT}"
  echo "using Rockchip sysroot from workspace: ${SYSROOT}"
elif [[ "${SYSROOT_EXPLICIT}" -eq 0 && ! -f "${SYSROOT}" && -f "${LEGACY_SYSROOT}" ]]; then
  SYSROOT="${LEGACY_SYSROOT}"
  echo "using Rockchip sysroot exported by a3_deploy_example: ${SYSROOT}"
fi
if [[ ! -f "${SYSROOT}" ]]; then
  cat >&2 <<EOF
missing Rockchip sysroot bundle: ${SYSROOT}

Generate it first:
  scripts/export_rockchip_sysroot.sh
EOF
  exit 66
fi

context_dir="$(mktemp -d "${TMPDIR:-/tmp}/a3-mdu-builder.XXXXXX")"
docker_build_args=()
docker_run_args=()
cleanup() {
  rm -rf "${context_dir}"
}
trap cleanup EXIT

mkdir -p "${context_dir}/docker" "${context_dir}/thirdparty/rockchip_sysroot"
cp -a "${REPO_ROOT}/docker/Dockerfile.mdu-rockchip-builder" "${context_dir}/docker/"
if ! ln "${SYSROOT}" \
    "${context_dir}/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz" \
    2>/dev/null; then
  cp -a "${SYSROOT}" \
    "${context_dir}/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz"
fi

append_proxy_build_args docker_build_args
append_proxy_run_args docker_run_args
if proxy_requires_host_network; then
  docker_build_args+=(--network host)
  docker_run_args+=(--network host)
fi

docker build --platform linux/amd64 \
  "${docker_build_args[@]}" \
  -f "${context_dir}/docker/Dockerfile.mdu-rockchip-builder" \
  -t "${IMAGE}" \
  "${context_dir}"

docker run --rm --platform linux/amd64 \
  "${docker_run_args[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e HAS_ROS2=1 \
  -v "${REPO_ROOT}:/work" \
  -w /work \
  "${IMAGE}" \
  bash -lc "scripts/build_mdu_package.sh --inside-docker --jobs '${JOBS}'"
