# Clean-room reference runner (`a3_deploy_onnx_ref_pingpong`)

A from-scratch Python implementation of the public HOPE deploy contract.
It exists to document the contract **executably** and to run the exported policy
against the shipped MuJoCo sim. It contains none of the vendor runner's source,
tuned constants, or gates.

## Install & run

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD"          # or use ../scripts/run_pingpong_sim.sh
python -m a3_deploy_onnx_ref_pingpong \
    --config ../config/hope_pingpong_runtime.yaml \
    --onnx /path/to/hope_pingpong.onnx \
    --view --realtime
```

Flags: `--backend {mujoco,aimrt}`, `--onnx`, `--model-xml`, `--view`, `--realtime`,
`--duration N` / `--max-ticks N`, `--idle` (no command feed, robot just holds).

## Module layout

| Module | Responsibility |
| --- | --- |
| `joint_order.py` | The 31-DOF Agibot A3 joint order (the single order used everywhere). |
| `quaternion.py` | `(w,x,y,z)` quaternion helpers (projected gravity, base forward). |
| `observation.py` | `build_observation(...) -> float32[111]` — the exact 111-D layout. |
| `action_adapter.py` | Shared ActionAdapter: `q_des = default_q + raw*scale`, then clamp. |
| `racket_command.py` | `RacketCommand` + command sources (queue seam, example feed). |
| `lifecycle.py` | `ready -> swing -> follow-through -> recovery` state machine. |
| `onnx_policy.py` | onnxruntime actor wrapper `obs[1,111] -> raw_action[1,31]`. |
| `sim_bridge.py` | `MujocoDirectBridge` (default) + `AimrtSimBridge` (seam). |
| `config.py` | Runtime config loader. |
| `runner.py` | The 50 Hz control loop. |
| `__main__.py` | CLI entrypoint. |

## Per-tick control loop (`runner.py`)

1. read robot state from the sim bridge;
2. poll the latest `RacketCommand`; emit the current swing goal;
3. assemble the 111-D observation (raw, no normalization);
4. run the ONNX actor → `raw_action[31]`;
5. zero the passive head columns (idx 3, 4) to form the **applied action** and feed
   that back as the next `last_action` (matching training's zeroed feedback);
6. map the applied action → 31 joint targets via the shared ActionAdapter (holding the
   passive neck at its default);
7. write the targets, step the sim, then advance the lifecycle clock once.

No gates, failure checks, rejections, reference playback, or state resets between
tasks — a single continuous 111-D path. `task_id`/`task_revision` semantics: a new
(strictly increasing) `task_id` engages exactly one swing and locks `swing_side`;
a higher `task_revision` refines the target/time-to-strike **before** contact only.

## Current reference clock

The shipped runtime clock matches the two current training defaults,
`ours_forehand_guarded_1p8s_wrist_x` and
`ours_backhand_guarded_1p8s`. Both have 91 samples at 50 Hz, strike at frame
50, and reference timestamps from 0 through 1.8 s. During the actor window,
`time_to_strike` is emitted from +1.0 s through 0 to -0.8 s. Because both
endpoints receive an action, that is 91 control ticks (1.82 s), not 90 ticks.

Training samples a 0--2 s pre-swing hold at each wrap. The planner-less
`ExampleCommandFeed` uses its 1.0 s mean: 50 held actions followed by 91 clip
actions, so its default period is 2.82 s. A real planner supplies the same
countdown using `RacketCommand`; transport and mailbox age are subtracted as
described below.

This default clock is not backward-compatible by assumption. A policy trained
on a legacy 361-frame reference must be evaluated with its original timing
metadata or explicit compatibility overrides, rather than being silently
paired with the current lifecycle.

## How it drives MuJoCo

`MujocoDirectBridge` (the default, fully runnable path) loads the same
`a3_pingpong` MJCF that the AimRT MuJoCo sim wraps and steps MuJoCo in-process:

- joint state (`q`, `qd`) is read from the mapped `qpos`/`qvel` addresses;
- base orientation comes from the pelvis free-joint quaternion and base angular
  velocity from the pelvis gyro sensor;
- joint-position targets are realized with an explicit PD law
  (`tau = kp*(q_des - q) - kd*qd`) written to the model's torque actuators — the
  same implicit-PD shape the AimRT backend uses — and clamped to each actuator's
  control range;
- one 50 Hz control tick advances 20 physics substeps at 1 kHz, recomputing the
  explicit torque PD each substep. Isaac's 200 Hz PD is solved implicitly inside
  PhysX; using that same 5 ms step with MuJoCo's explicit torque loop is unstable,
  so the finer MuJoCo step is the verified numerical equivalent rather than a
  falsely identical integrator setting;
- the MJCF's named 31-joint velocity-limit vector mirrors Isaac
  `velocity_limit_sim`; because MuJoCo has no native joint-speed constraint, the
  bridge projects robot `qvel` onto those bounds after every physics step without
  changing the actuator torque/control-range clipping;
- named effort ranges and armatures match every Isaac actuator group (including
  the ankle limits); passive joint damping/friction are zero so the explicit
  `kd` is not counted twice;
- robot-to-robot collision bits are disabled, matching Isaac
  `enabled_self_collisions=False`; robot-floor and ball-racket contacts remain
  enabled.

The runtime config names all 31 nominal Isaac simulation PD gains explicitly.
They are still **not** vendor real-robot deploy gains.

## Live planner input (`--planner`)

`RosRacketCommandSource` (`ros_command_source.py`) is the wired planner → runner
path: it subscribes the planner's `hope_msgs/RacketCommand` topic (default
`/racket/command`) on a background rclpy executor and feeds the 50 Hz loop through
the same `QueueRacketCommandSource` mailbox the other sources use:

```bash
# Build from the repo root; generated paths must be ASCII on ROS 2 Humble.
hope_ws/scripts/build_ros_ascii.sh /tmp/hope_ws_ros2_${UID} --packages-up-to hope_msgs
source /tmp/hope_ws_ros2_${UID}/install/setup.bash
python -m a3_deploy_onnx_ref_pingpong --planner --view --realtime
```

The planner message uses the canonical table-surface frame. For the standalone
robot MJCF, the runner applies the configured pure translation into its floor-based
MuJoCo world; velocity is unchanged because the axes are parallel.

The planner defines `time_to_strike` at the ball-capture/planning instant carried
by `RacketCommand.header.stamp`. The ROS bridge subtracts transport time in the
ROS clock domain, and its latest-value mailbox subtracts the remaining
callback-to-control-loop delay with a monotonic clock. Zero/invalid/future stamps
never increase TTS. Offline evaluators disable wall-clock mailbox ageing so
simulation-time tests stay deterministic.

The included `ExampleCommandFeed` (the default source) is a planner-less
demonstration feed so the sim is runnable without a planner; it is **not** part of
the deploy contract and is not a scripted swing. Its target/velocity examples are
the centres of the Isaac per-side training boxes; the swing trajectory is always
produced by the learned policy. `--idle` runs with no commands at all.

## Integration seams (explicitly not wired)

- **`AimrtSimBridge`** — driving the live AimRT MuJoCo sim *process* over its
  `/body_drive/*` channels. Wiring it needs the AimRT Python runtime plus the
  `joint_msgs` typesupport (a vendor build). It raises `NotImplementedError` with
  the exact channel/message mapping rather than faking state. Use
  `MujocoDirectBridge` to actually run.

## Notes

- Observation normalization is `none` (raw observation) by contract.
- `head_yaw` / `head_pitch` are passive at deploy (held at their default) but still
  occupy their action columns, so every vector stays length 31.
- The sample motion clips used in training are reference examples only, not
  performance-tuned; replace them with your own.
