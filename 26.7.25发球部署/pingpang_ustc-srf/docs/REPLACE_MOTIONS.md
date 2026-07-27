# Replacing the motion clips

Training imitates two motion clips. The current defaults use the equal-length,
91-frame (1.8 s timestamp span at 50 Hz) guarded-retime pair below. Their
immutable 361-frame predecessors remain in the same directory only as
legacy/provenance sources; they are not the runtime defaults. The separate
`hope_forehand.npz` and `hope_backhand.npz` files are only physically-neutral
placeholders that let loader and shape checks pass.

This page documents the file format so you can produce your own. There is deliberately **no** motion
validator, scorer, receipt, or qualification step — the loader reads the fixed format below, and
ordinary file/shape errors surface naturally.

## The two clips

```
hope_training/motions/preprocessed/ours_forehand_guarded_1p8s_wrist_x.npz   (clip 0, swing_side +1)
hope_training/motions/preprocessed/ours_forehand_guarded_1p8s_wrist_x.yaml
hope_training/motions/preprocessed/ours_backhand_guarded_1p8s.npz   (clip 1, swing_side -1)
hope_training/motions/preprocessed/ours_backhand_guarded_1p8s.yaml
```

Each clip should cover one full swing: **ready → strike → follow-through → recoverable end pose**.
Keep both files' `fps`, joint order, and tracked-body list identical to the schema below; the loader
validates that the arrays match.

## `.npz` schema

All arrays are `float32`, retargeted to the Agibot A3 and expressed in the shared world frame
(+x forward, +y left, +z up). `F` is the frame count.

| Key | Shape | Meaning |
|-----|-------|---------|
| `fps` | scalar | frames per second (e.g. 50) |
| `joint_pos` | `(F, 31)` | joint positions, in the [31-DOF joint order](POLICY_INTERFACE.md#joint-order) |
| `joint_vel` | `(F, 31)` | joint velocities, same order |
| `body_pos_w` | `(F, 14, 3)` | world positions of the 14 tracked bodies |
| `body_quat_w` | `(F, 14, 4)` | world orientations (quaternion, **wxyz**) |
| `body_lin_vel_w` | `(F, 14, 3)` | world linear velocities of the tracked bodies |
| `body_ang_vel_w` | `(F, 14, 3)` | world angular velocities of the tracked bodies |

The 14 tracked bodies are stored in this exact order (index 0 is the root, index 7 is the anchor the
imitation reward aligns to):

```
 0 pelvis_link         (root)        7 torso_Link          (anchor)
 1 left_hip_roll_Link                8 left_shoulder_roll_Link
 2 left_knee_Link                    9 left_elbow_Link
 3 left_ankle_roll_Link             10 left_wrist_yaw_Link
 4 right_hip_roll_Link              11 right_shoulder_roll_Link
 5 right_knee_Link                  12 right_elbow_Link
 6 right_ankle_roll_Link            13 right_wrist_yaw_Link
```

The loader raises a clear error if the tracked-body count does not match, so keep this list and its
order in sync with the YAML sidecar and the robot asset.

## `.yaml` sidecar

The sidecar describes the clip's phase structure, retiming provenance, and racket
convention (see `ours_forehand_guarded_1p8s_wrist_x.yaml` for a complete example):

- `name`, `swing_side` (`+1` forehand / `-1` backhand)
- `fps`, `frame_count`, `frame_time_s`, `duration_s`
- `strike_frame`, `strike_phase` (fraction of the clip at the strike)
- `ready_interval_frames`, `follow_through_end_frame`, `recover_end_frame`
- `joint_order` (the 31 joint names, in order)
- `tracked_bodies`, `anchor_body`, `root_body`
- `racket_link`, `racket_body`, `mount_offset_xyz` (wrist → racket-centre offset in the wrist frame),
  `blade_normal_axis`, `blade_normal_sign` (the public blade-face convention)

The shipped guarded clips crop zero-based source frames 72--320 (inclusive) and map them
to output frames 0--90 (91 frames). A shared, bounded inverse-arc-length
retime uses both clips' normalized joint-limit demand, joint arc, and
racket-centre arc to allocate samples. It compresses the complete outer
phases, including active wind-up before contact and follow-through/recovery
after contact, while limiting any local output step to at most eight source
intervals. The finite-difference-safe unit-speed guard maps
source 173--187 one-to-one to output 43--57. Its inner reward window maps
source 174--186 to output 44--56, so the complete ±0.12 s reward interval and
the velocity differences at both boundaries stay at original speed. The
semantic strike is source frame 180 / output frame 50, giving a 1.0 s lead
and 0.8 s follow interval (`strike_phase = 50 / 90`).

The reference timestamps cover 90 intervals, hence `duration_s: 1.8`. The
actor nevertheless consumes all 91 inclusive samples at 50 Hz: one swing
uses 91 control ticks, or 1.82 s of control-loop time, and observes
`time_to_strike` from +1.0 s through 0 to -0.8 s. The environment's 10 s
episode is deliberately not shortened; it contains repeated continuous
swings (with the configured inter-swing hold) without teleporting or resetting
the robot at clip boundaries.

## Producing your own clips

Record or synthesize a forehand and a backhand swing, retarget them to the A3's 31 DOF, resample to a
fixed `fps`, and write the arrays above into a `.npz` plus a matching `.yaml`. Point training at them
either by placing them at the default paths, or via the CLI:

```bash
python scripts/train.py task=HOPEPingPong \
    motion_file=/path/to/your_forehand.npz \
    motion_file_2=/path/to/your_backhand.npz
```

(or set `commands.motion.motion_file` in `cfg/task/HOPEPingPong.yaml`). The retargeting pipeline
itself is out of scope for this repository — any tool that emits the schema above works.
