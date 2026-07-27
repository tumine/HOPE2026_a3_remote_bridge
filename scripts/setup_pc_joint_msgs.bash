#!/usr/bin/env bash

# Source this file; do not execute it.  It exposes the native joint_msgs
# Python package generated in this repository's build tree.

_A3_REMOTE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_A3_JOINT_BUILD="${_A3_REMOTE_ROOT}/build/thirdparty/joint_msgs"
_A3_JOINT_PY="${_A3_JOINT_BUILD}/rosidl_generator_py"
_A3_JOINT_PY_LIB="${_A3_JOINT_PY}/joint_msgs"

if [[ -z "${ROS_DISTRO:-}" && -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [[ ! -f "${_A3_JOINT_PY}/joint_msgs/msg/_joint_command.py" ]]; then
  echo "joint_msgs Python code is missing; build it first:" >&2
  echo "  source /opt/ros/humble/setup.bash" >&2
  echo "  cmake --build ${_A3_REMOTE_ROOT}/build --target joint_msgs__rosidl_typesupport_c__pyext -j2" >&2
  unset _A3_REMOTE_ROOT _A3_JOINT_BUILD _A3_JOINT_PY _A3_JOINT_PY_LIB
  return 66
fi

export PYTHONPATH="${_A3_JOINT_PY}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${_A3_JOINT_BUILD}:${_A3_JOINT_PY_LIB}:${LD_LIBRARY_PATH:-}"

unset _A3_REMOTE_ROOT _A3_JOINT_BUILD _A3_JOINT_PY _A3_JOINT_PY_LIB

