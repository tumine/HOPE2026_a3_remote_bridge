#!/usr/bin/env bash
#
# Build the ROS 2 workspace while keeping generated rosidl paths ASCII-only.
# The source workspace may remain at its current (possibly non-ASCII) path.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  hope_ws/scripts/build_ros_ascii.sh [OUTPUT_ROOT] [COLCON_BUILD_ARGS...]

Examples:
  hope_ws/scripts/build_ros_ascii.sh
  hope_ws/scripts/build_ros_ascii.sh /tmp/hope_ws_ros2_1000 --packages-select hope_msgs

OUTPUT_ROOT defaults to /tmp/hope_ws_ros2_<uid>. The script only creates or
updates OUTPUT_ROOT/{build,install,log}; it never removes data and does not
source the resulting overlay into the caller's shell.
EOF
}

is_broad_output_root() {
  case "$1" in
    /|/tmp|/home|/data1|/usr|/opt|/var|/etc|/root) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_dir="$(cd -- "${script_dir}/.." && pwd -P)"

if (($# > 0)) && [[ "$1" != -* ]]; then
  output_root="$1"
  shift
else
  output_root="/tmp/hope_ws_ros2_${UID}"
fi

if [[ "${output_root}" != /* ]]; then
  echo "error: OUTPUT_ROOT must be an absolute path: ${output_root}" >&2
  exit 2
fi
if is_broad_output_root "${output_root}"; then
  echo "error: refusing broad OUTPUT_ROOT: ${output_root}" >&2
  exit 2
fi

mkdir -p -- "${output_root}"
output_root="$(cd -- "${output_root}" && pwd -P)"
if is_broad_output_root "${output_root}"; then
  echo "error: refusing resolved broad OUTPUT_ROOT: ${output_root}" >&2
  exit 2
fi

# rosidl_cmake in ROS 2 Humble mis-parses generated IDL tuples when their
# absolute build path contains non-ASCII bytes. Colons/whitespace are also
# unsafe because the tuple syntax itself uses ':' as a separator.
if [[ "${output_root}" == *[!A-Za-z0-9_./-]* ]]; then
  echo "error: resolved OUTPUT_ROOT must contain only A-Z, a-z, 0-9, _, -, ., /" >&2
  echo "       resolved path: ${output_root}" >&2
  exit 2
fi

ros_setup="/opt/ros/humble/setup.bash"
if [[ ! -r "${ros_setup}" ]]; then
  echo "error: ROS 2 Humble setup not found: ${ros_setup}" >&2
  exit 2
fi

# Several ROS generator entrypoints use '#!/usr/bin/env python3'. Put the ROS
# distribution's matching system Python ahead of an active conda environment.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
# ROS Humble's Python packages are tested with the Ubuntu system setuptools.
# Ignore incompatible user-site upgrades while generating rosidl bindings.
export PYTHONNOUSERSITE=1
# ROS setup hooks are not consistently nounset-safe.
set +u
source "${ros_setup}"
set -u

cd -- "${workspace_dir}"
colcon --log-base "${output_root}/log" build \
  --build-base "${output_root}/build" \
  --install-base "${output_root}/install" \
  "$@" \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

printf '\nBuild completed. Activate it in your current shell with:\n'
printf '  source %q\n' "${output_root}/install/setup.bash"
