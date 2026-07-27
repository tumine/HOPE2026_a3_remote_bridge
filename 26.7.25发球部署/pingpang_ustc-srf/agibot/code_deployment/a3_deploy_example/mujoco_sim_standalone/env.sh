#!/usr/bin/env bash
set -e

package_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source_setup() {
  local setup_file=$1
  local had_nounset=0
  case $- in
    *u*) had_nounset=1; set +u ;;
  esac
  # ROS 2 / ament setup files may read optional environment variables.
  # Keep the caller's nounset mode, but do not apply it inside those scripts.
  # shellcheck disable=SC1090
  source "$setup_file"
  if [[ $had_nounset -eq 1 ]]; then
    set -u
  fi
}

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source_setup /opt/ros/humble/setup.bash
fi

export LD_LIBRARY_PATH="$package_dir/bin:$package_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$package_dir/bin${PATH:+:$PATH}"

for pkg in aimrt_msgs joint_msgs ros2_plugin_proto; do
  if [[ -f "$package_dir/share/$pkg/local_setup.bash" ]]; then
    source_setup "$package_dir/share/$pkg/local_setup.bash"
  fi
done
