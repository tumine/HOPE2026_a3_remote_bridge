# Frames

HOPE has two axis-aligned frames with one explicit translation between them.
Keeping this boundary visible is important: the planner/mocap arena is naturally
table-relative, while the default Isaac tracking task and MuJoCo controller use a
floor-based, station-relative world.

## Canonical table frame (`world`)

The mocap and planner use a right-handed ROS 2 REP-103 frame:

| Axis | Direction | Range over the table |
|------|-----------|----------------------|
| +x | forward, toward the opponent (P2) | `[0, 2.74]` m |
| +y | left, from the robot's (P1) perspective | table occupies `[-1.525, 0]` m |
| +z | up | `0` is the table surface |

- The origin is the near-side left corner of the table surface from P1's
  perspective.
- The floor is at `z = -0.76` m and the net plane is at `x = 1.37` m.
- The nominal robot station is `[-0.5, -0.7625, -0.76]`: 0.5 m behind the
  near edge, centred on the table, on the floor.
- Geometry and ball parameters come from
  [`configs/ball_physics.yaml`](../../configs/ball_physics.yaml). Named arena
  landmarks are published from
  [`hope_world_frame.yaml`](../../hope_ws/src/hope_bringup/config/hope_world_frame.yaml).

`RacketCommand` published by the ROS planner is in this canonical table frame.
Its `header.frame_id` is the configured planner world frame.

## Policy backend frame (Isaac tracking and MuJoCo)

The 111-D policy is trained in the default Isaac tracking task's local
floor-world frame:

- the fixed station is XY `[0, 0]`;
- floor height is `z = 0`;
- the table near edge is `x = 0.5`, its centreline is `y = 0`, and its surface
  is `z = 0.76`;
- the grounded default pelvis height is `z = 1.0664`.

The corrected MuJoCo model uses the same placement. For a backend whose fixed
station is `[s_x, s_y]`, convert a canonical planner position with

```text
p_backend = p_table + [s_x + 0.5, s_y + 0.7625, 0.76]
v_backend = v_table
```

The axes are identical, so velocity and orientation vectors are not rotated.
The reference runner's direct `--planner` path applies this translation in
`TranslatedRacketCommandSource`; the standalone MuJoCo evaluator constructs its
commands directly in backend coordinates. A real-robot integration must
calibrate and apply the equivalent table-to-station transform before building
the policy observation.

The actor then receives `racket_target_rel_base = p_backend - p_base`. It does
not receive a ball position. Default training intentionally evaluates the ball
analytically from the sampled interception state; the optional physical-table
Isaac task uses the canonical table frame but is not the default tracking task.

With multiple Isaac environments, each environment has its own translated
backend frame at `env_origin`; subtracting that origin gives the values above.

## Robot and mocap details

- The root body is `pelvis_link`.
- The racket is mounted on `right_wrist_yaw_Link`; when present, the dedicated
  racket body is `pingpang_red_Link`. See
  [`A3_ASSETS.md`](../../A3_ASSETS.md).
- Mocap ball positions are expressed in the canonical table frame. Robot base
  yaw comes from the robot IMU, not from the ball marker stream.
- The authoritative mocap topic/frame contract is
  [`mocap/README.md`](../../mocap/README.md). The older arena design document is
  background material; where it differs, the README and the transform above
  take precedence.
