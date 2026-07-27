# Planner interface

The planner turns a stream of motion-capture ball positions into a per-strike racket target
for the policy. It is a no-spin, continuous, fixed-station planner: it predicts where the
incoming ball crosses side-specific strike planes, chooses forehand or backhand, and publishes a
racket target position, velocity, time-to-strike, and swing side.

## Data flow

```
mocap ball positions (/poses)
  -> position/velocity estimate
  -> no-spin trajectory prediction
  -> side-specific strike planes + grounded training-box gate
  -> forehand/backhand split
  -> fixed opponent-half landing target
  -> RacketCommand (position, velocity, time_to_strike, swing_side)
```

- Every incoming mocap sample feeds the estimator; the (more expensive) trajectory solve runs
  at **at most 50 Hz**.
- A capture gap longer than `new_ball_gap_s` (default `0.25 s`) clears the
  polynomial fit and releases the previous task's side lock, so a new physical
  ball cannot inherit stale samples or a stale forehand/backhand choice.
- The first timestamped sample seeds position; a second sample at a different timestamp
  enables the velocity estimate; after that the planner publishes directly.
- The ball model is no-spin: state is `[x, y, z, vx, vy, vz]` with gravity, measured drag, and
  measured table/paddle restitution from `configs/ball_physics.yaml`. There is no spin
  estimation or Magnus force. The strike planes are fixed per side at the centre
  of the policy's grounded training x ranges.

## Continuous rallies

Each incoming ball opens a new **task**:

- `task_id` — a new unique id per incoming ball.
- `task_revision` — increments monotonically as the pre-strike trajectory estimate is refined
  for the *same* ball.
- `swing_side` — chosen once when the task opens and held constant for that task.

After a strike the planner opens the next task for the next ball. The robot is never reset
between tasks. All four adjacent side transitions (FH→FH, FH→BH, BH→FH, BH→BH) occur naturally
across a rally.

## Forehand / backhand selection

The planner predicts one crossing at the centre of each side's trained x range. A
candidate is publishable only if its predicted `x/y/z` lies inside that side's
grounded Isaac target box and the planned racket `vx/vy/vz` lies inside that
side's training velocity envelope. It then compares lateral (`y`) position to
`swing_side_split_y` (with optional small hysteresis):

```
crossing_y <  swing_side_split_y  -> FOREHAND (+1)
crossing_y >= swing_side_split_y  -> BACKHAND (-1)
```

A ball arriving **below** the split (toward the paddle side, `-y`) is taken forehand; a ball
at or above the split is taken backhand. Boundary cases, with `split = swing_side_split_y`
and hysteresis `h` (`prev` = the previous task's side):

| `crossing_y`            | `prev`     | selected side |
|-------------------------|------------|---------------|
| `< split`               | none       | FOREHAND      |
| `= split` or `> split`  | none       | BACKHAND      |
| `<= split + h`          | FOREHAND   | FOREHAND (sticky) |
| `> split + h`           | FOREHAND   | BACKHAND      |
| `>= split - h`          | BACKHAND   | BACKHAND (sticky) |
| `< split - h`           | BACKHAND   | FOREHAND      |

This convention is implemented in `hope_planner/side_selection.py` and pinned by
`hope_planner/test/test_side_selection.py`. There is no higher-level shot selection, side
optimization, or opponent adaptation. `swing_side` is a formal field of the message — it is
no longer inferred downstream from the target's Y sign.

The default boxes are expressed in the canonical table-surface frame and map to
the current Isaac station-relative/floor-world target boxes:

| side | planner-frame x | planner-frame y | planner-frame z | floor-world/station box |
|---|---:|---:|---:|---|
| forehand | `[-0.32,-0.10]` | `[-1.5225,-1.2825]` | `[0.24,0.45]` | `[0.18,0.40] × [-0.76,-0.52] × [1.00,1.21]` |
| backhand | `[-0.05,0.25]` | `[-1.0625,-0.6625]` | `[0.08,0.34]` | `[0.45,0.75] × [-0.30,0.10] × [0.84,1.10]` |

The transform is the fixed translation `[+0.5,+0.7625,+0.76]` for the
documented station. Out-of-box physical predictions are explicitly withheld;
they are never clipped into the policy's training distribution.

The side-aware racket-velocity admission envelopes (world axes; translation
does not affect vectors) are:

| side | `vx` | `vy` | `vz` |
|---|---:|---:|---:|
| forehand | `[1.75,3.40]` | `[0.25,1.10]` | `[0.35,1.45]` |
| backhand | `[1.05,3.00]` | `[-0.25,0.50]` | `[0.45,1.45]` |

Both the position and velocity checks are closed intervals. Non-finite or
out-of-envelope values suppress publication and are logged; the planner never
clips them. Pure contract tests compare these YAML values to the current Isaac
`racket_pos_range_per_clip` and `racket_vel_range_per_clip` declarations so
configuration drift fails CI.

These are conservative outward bounds over the legal
`position × incoming-ball-velocity` domains, not percentile estimates. They
include at least `0.02 m/s` numerical margin around the observed union of the
Torch training planner (`dt=0.01`) and high-resolution NumPy live planner
(`dt=0.001`).

## `RacketCommand.msg`

Published on the racket-command topic (default `/racket/command`), consumed by the runner:

```
std_msgs/Header header
uint64 task_id
uint32 task_revision
int8 FOREHAND=1
int8 BACKHAND=-1
int8 swing_side
geometry_msgs/Point position          # target racket position, world frame, m
geometry_msgs/Vector3 velocity        # target racket velocity, world frame, m/s
float64 time_to_strike                # seconds from header.stamp until the strike
```

The message carries only what the policy needs. There is intentionally no `valid`/`reason`/
failure flag, no outgoing-ball prediction, no net/bounce prediction, no confidence, and no
diagnostics. The planner does not maintain a readiness or failure state — if the incoming
data is insufficient it simply has not published yet.

`header.stamp` is the ball-capture/planning reference time used to compute the
prediction; it is deliberately passed through from the input sample.
`time_to_strike` is therefore not a duration from subscriber receipt. A live
runner must subtract both ROS transport age and any local queue age before the
value reaches observation index 109. Invalid/zero/future stamps must not add
time; the runner falls back to queue-only ageing in those cases. Offline
simulation uses its deterministic simulation clock rather than wall time.

## Configuration

`hope_ws/src/hope_planner/config/hope_planner.yaml` holds the public parameters: the per-side
position/velocity command envelopes, `swing_side_split_y` (and hysteresis), the fixed opponent-half landing
target, the prediction horizon, and the solve rate. Ball physics is read from the shared
`configs/ball_physics.yaml`.
