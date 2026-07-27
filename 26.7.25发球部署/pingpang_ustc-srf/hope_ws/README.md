# HOPE ROS 2 workspace

The source workspace can stay in this repository even though its absolute path
contains non-ASCII characters. ROS 2 Humble's `rosidl_cmake` mis-parses generated
IDL tuples when the **build path** contains those characters, and ROS generator
entrypoints can also select an incompatible conda `python3` through `PATH`.

Use the wrapper below. It keeps `build`, `install`, and `log` under one
ASCII-only external directory, puts `/usr/bin/python3` first for the build, does
not delete existing data, and prints (but does not execute) the final `source`
command:

```bash
# Build the whole workspace. Default output: /tmp/hope_ws_ros2_<uid>
hope_ws/scripts/build_ros_ascii.sh

# Or choose a persistent ASCII-only output root / build selected packages.
hope_ws/scripts/build_ros_ascii.sh /data1/hope_ws_ros2 \
  --packages-up-to hope_bringup

# Activate the overlay after a successful build (use the path printed above).
source /data1/hope_ws_ros2/install/setup.bash
```

Only the generated output roots need an ASCII path; no source symlink or source
copy is required. The wrapper accepts normal `colcon build` options after the
optional output root, such as `--packages-select hope_msgs`.

A full workspace build also needs the vendored mocap client's system dependency:

```bash
sudo apt install ros-humble-vrpn
```

This dependency is not needed for a planner/message-only build such as
`--packages-up-to hope_planner`.

## Planner startup timing

The shipped planner admits a new `task_id` only when the predicted
`time_to_strike` is in the closed `[0.95, 1.0] s` window. This matches the
policy's 1.0 s training lead: predictions above the window wait for a later
mocap sample, while predictions already below it drop that physical ball
without consuming a task id. Once a task is active, lower-TTS revisions remain
enabled through contact.

`fake_ball_publisher` starts both default trajectories slightly above the
window (about 1.02--1.06 s before their strike), so
`use_fake_ball:=true` exercises the same admission path as a live rally.
