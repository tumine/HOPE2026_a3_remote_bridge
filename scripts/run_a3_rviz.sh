#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${A3_PC_ROS_SETUP:-/opt/ros/humble/setup.bash}"
MOCAP_SETUP="${A3_MOCAP_ROS_SETUP:-/home/bth/workspace/Mocap/install/setup.bash}"
MODEL_DIR="${A3_RVIZ_MODEL_DIR:-${ROOT_DIR}/26.7.25发球部署/pingpang_ustc-srf/agibot/URDF/A3T2.5-URDF-std-pingpang}"
SOURCE_URDF="${A3_RVIZ_URDF:-${MODEL_DIR}/urdf/URDF-JOINT-LINK.urdf}"
RVIZ_CONFIG="${A3_RVIZ_CONFIG:-${ROOT_DIR}/config/a3_robot.rviz}"
RL_CONFIG="${A3_RL_CONFIG:-${ROOT_DIR}/config/a3_rl_deploy.yaml}"
JOINT_TOPIC="${A3_RVIZ_JOINT_TOPIC:-/a3_internal/joint_states}"
FRAME_PREFIX="${A3_RVIZ_FRAME_PREFIX:-a3/}"
CHECK_ONLY=0
NO_RVIZ=0

usage() {
  cat <<'EOF'
Usage: run_a3_rviz.sh [--check] [--no-rviz]

  --check     validate dependencies, URDF and visualization config, then exit
  --no-rviz   publish live robot TF/description without opening the RViz window

Environment overrides:
  A3_RVIZ_JOINT_TOPIC  default /a3_internal/joint_states
  A3_RVIZ_MODEL_DIR    A3 description package containing urdf/ and meshes/
  A3_RVIZ_CONFIG       RViz config file
EOF
}

while (($#)); do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --no-rviz) NO_RVIZ=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

for path in "${ROS_SETUP}" "${MOCAP_SETUP}" "${SOURCE_URDF}" "${RVIZ_CONFIG}" "${RL_CONFIG}"; do
  if [[ ! -r "${path}" ]]; then
    echo "required RViz input not found: ${path}" >&2
    exit 66
  fi
done
if [[ "${FRAME_PREFIX}" != */ ]]; then
  echo "A3_RVIZ_FRAME_PREFIX must end with /" >&2
  exit 64
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${MOCAP_SETUP}"
set -u

for executable in ros2 rviz2 check_urdf; do
  if ! command -v "${executable}" >/dev/null 2>&1; then
    echo "required executable is missing: ${executable}" >&2
    exit 66
  fi
done
if ! ros2 pkg prefix robot_state_publisher >/dev/null 2>&1; then
  echo "required ROS 2 package is missing: robot_state_publisher" >&2
  exit 66
fi
RSP_EXECUTABLE="$(ros2 pkg prefix robot_state_publisher)/lib/robot_state_publisher/robot_state_publisher"
if [[ ! -x "${RSP_EXECUTABLE}" ]]; then
  echo "robot_state_publisher binary is missing: ${RSP_EXECUTABLE}" >&2
  exit 66
fi

export ROS_DOMAIN_ID="${A3_ROS_DOMAIN_ID:-232}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT_DIR}/config/fastrtps_pc_direct_mdu.xml"
export ROS_LOG_DIR="${A3_ROS_LOG_DIR:-${ROOT_DIR}/log/ros}"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
mkdir -p -- "${ROS_LOG_DIR}"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/a3_rviz.XXXXXX")"
PORTABLE_URDF="${TEMP_DIR}/a3_pingpong_rviz.urdf"
POSE_PID=""
RSP_PID=""
GUI_PID=""

cleanup() {
  local result=$?
  trap - EXIT INT TERM
  for pid in "${GUI_PID}" "${RSP_PID}" "${POSE_PID}"; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -INT "${pid}" 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    local any_alive=0
    for pid in "${GUI_PID}" "${RSP_PID}" "${POSE_PID}"; do
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    ((any_alive)) || break
    sleep 0.05
  done
  for pid in "${GUI_PID}" "${RSP_PID}" "${POSE_PID}"; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${GUI_PID}" "${RSP_PID}" "${POSE_PID}"; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    wait "${pid}" 2>/dev/null || true
  done
  rm -f -- "${PORTABLE_URDF}"
  rmdir -- "${TEMP_DIR}" 2>/dev/null || true
  exit "${result}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_child() {
  local pid="$1"
  # A blocking bash wait can defer a trapped signal when this launcher itself
  # is a background job.  Polling keeps Ctrl-C/TERM responsive so cleanup can
  # always stop the TF, robot-state and GUI children it owns.
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 0.2
  done
  wait "${pid}"
}

# The vendor description is a ROS 1 package, so resource_retriever cannot find
# its package:// URI through the ROS 2 ament index.  Materialize a temporary
# equivalent URDF with absolute file:// mesh URIs; the source asset is untouched.
python3 - "${SOURCE_URDF}" "${MODEL_DIR}" "${PORTABLE_URDF}" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

source, model_dir, output = map(Path, sys.argv[1:])
document = source.read_text(encoding="utf-8")
package_prefix = "package://0000014503_A3T2.5-URDF-std-pingpang-0409/"
replacement = model_dir.resolve().as_uri() + "/"
if package_prefix not in document:
    raise SystemExit(f"expected mesh package URI is absent from {source}")
root = ET.fromstring(document.replace(package_prefix, replacement))
for joint in root.findall("joint"):
    if joint.get("type") != "fixed":
        continue
    # The vendor ROS 1 export puts empty/invalid axis and limit elements on
    # fixed joints.  ROS 2 validates those unused elements, so omit them in
    # this temporary visualization-only copy.
    for tag in ("axis", "limit"):
        element = joint.find(tag)
        if element is not None:
            joint.remove(element)
ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
PY

check_urdf "${PORTABLE_URDF}" >/dev/null
python3 "${ROOT_DIR}/pc_tools/a3_rviz_pose_bridge.py" \
  --config "${RL_CONFIG}" \
  --child-frame "${FRAME_PREFIX}pelvis_link" \
  --check

if ((CHECK_ONLY)); then
  echo "RViz check OK: PPMocap scene + A3T2.5 ping-pong URDF, joints=${JOINT_TOPIC}, fixed=world"
  exit 0
fi

python3 "${ROOT_DIR}/pc_tools/a3_rviz_pose_bridge.py" \
  --config "${RL_CONFIG}" \
  --child-frame "${FRAME_PREFIX}pelvis_link" &
POSE_PID=$!

# The vendor URDF's inertial data on its root link is irrelevant to TF/RViz
# but KDL warns about it; retain actual errors while keeping startup readable.
"${RSP_EXECUTABLE}" "${PORTABLE_URDF}" \
  --ros-args \
  -r "joint_states:=${JOINT_TOPIC}" \
  -p "frame_prefix:=${FRAME_PREFIX}" \
  --log-level error &
RSP_PID=$!

sleep 0.8
if ! kill -0 "${POSE_PID}" 2>/dev/null || ! kill -0 "${RSP_PID}" 2>/dev/null; then
  echo "RViz TF/robot-state publisher failed during startup" >&2
  exit 70
fi

echo "A3 visualization ready:"
echo "  pose: /ppmocap/frame -> world -> hope_table -> ${FRAME_PREFIX}pelvis_link"
echo "  joints: ${JOINT_TOPIC} -> ${FRAME_PREFIX}*"
echo "  ground: hope_ground at table-surface z=-0.76m"
echo "  scene: /ppmocap/markers + measured/predicted ball paths + optional camera"

if ((NO_RVIZ)); then
  wait_for_child "${POSE_PID}"
else
  rviz2 -d "${RVIZ_CONFIG}" &
  GUI_PID=$!
  wait_for_child "${GUI_PID}"
fi
