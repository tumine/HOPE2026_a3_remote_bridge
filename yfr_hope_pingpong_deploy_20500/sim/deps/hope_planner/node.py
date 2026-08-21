"""ROS 2 node for the HOPE no-spin racket planner.

Subscribes to the mocap ball stream (``geometry_msgs/PoseArray`` on the
configured poses topic, ball at ``ball_pose_index``), estimates the ball
position/velocity, predicts the no-spin trajectory to side-specific strike
planes, gates crossings against the grounded Isaac target boxes, and publishes
the typed ``hope_msgs/RacketCommand``.

Lifecycle: each new incoming ball gets a new ``task_id``; pre-strike updates
keep that id and increase ``task_revision``; ``swing_side`` is chosen once per
task and locked within it. The first sample seeds position and subsequent
samples enable the velocity fit, after which commands are published directly —
there is no readiness/validity/failure state.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from hope_msgs.msg import RacketCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .constants import PlannerConfig, load_ball_physics, load_paddle_params, load_table_params
from .planner import HOPEPlanner
from .side_selection import select_swing_side
from .strike_selection import StrikeRegion, select_strike_candidate, validate_side_regions
from .task_timing import NewTaskTTSGate, TaskTimingDecision

_TASK_ID_WRAP = 1 << 64
_REVISION_WRAP = 1 << 32


class HOPEPlannerNode(Node):
    """ROS 2 wrapper around :class:`HOPEPlanner`."""

    def __init__(self):
        super().__init__("hope_planner")

        # --- Topics / frames ---
        self.declare_parameter("poses_topic", "/poses")
        self.declare_parameter("command_topic", "/racket/command")
        self.declare_parameter("frame_id", "world")
        # Which slot in the PoseArray is the ball (PoseArray carries no names).
        self.declare_parameter("ball_pose_index", 0)

        # --- Planner geometry / tuning ---
        self.declare_parameter("x_hit", 0.0)              # fixed strike-plane x (m)
        self.declare_parameter("use_side_aware_strike_regions", True)
        # Canonical table-frame boxes corresponding exactly to the current grounded
        # Isaac station-relative target boxes after adding [-0.5, -0.7625, -0.76].
        self.declare_parameter("forehand_strike_x_range", [-0.32, -0.10])
        self.declare_parameter("forehand_strike_y_range", [-1.5225, -1.2825])
        self.declare_parameter("forehand_strike_z_range", [0.24, 0.45])
        self.declare_parameter("forehand_velocity_x_range", [1.75, 3.40])
        self.declare_parameter("forehand_velocity_y_range", [0.25, 1.10])
        self.declare_parameter("forehand_velocity_z_range", [0.35, 1.45])
        self.declare_parameter("backhand_strike_x_range", [-0.05, 0.25])
        self.declare_parameter("backhand_strike_y_range", [-1.0625, -0.6625])
        self.declare_parameter("backhand_strike_z_range", [0.08, 0.34])
        self.declare_parameter("backhand_velocity_x_range", [1.05, 3.00])
        self.declare_parameter("backhand_velocity_y_range", [-0.25, 0.50])
        self.declare_parameter("backhand_velocity_z_range", [0.45, 1.45])
        self.declare_parameter("swing_side_split_y", -1.1725)   # mid-gap between trained boxes
        self.declare_parameter("swing_side_hysteresis_y", 0.0)  # optional band around the split (m)
        self.declare_parameter("target_land_x", 2.055)   # fixed landing target x (m)
        self.declare_parameter("target_land_y", -0.7625)  # fixed landing target y (m)
        self.declare_parameter("delta_t_flight", 0.5)     # desired post-strike flight time (s)
        self.declare_parameter("max_predict_time", 2.0)   # prediction horizon (s)
        # Push every mocap sample into the estimator; run the predict+plan solve
        # at most every solve_period_s (<= 50 Hz). 0.0 = solve on every sample.
        self.declare_parameter("solve_period_s", 0.02)
        # A longer capture gap separates physical balls and invalidates the old
        # polynomial fit/task lock. Zero disables gap-based task separation.
        self.declare_parameter("new_ball_gap_s", 0.25)
        # A new policy task must begin at the same 1.0 s lead used in training.
        # Once active, revisions remain admissible below this startup window.
        self.declare_parameter("new_task_tts_min_s", 0.25)
        self.declare_parameter("new_task_tts_max_s", 1.0)
        # Table's +y edge in the play frame (table occupies y in [y_max - width, y_max]).
        self.declare_parameter("table_y_max", 0.0)
        # Optional explicit path to configs/ball_physics.yaml ("" = auto-discover).
        self.declare_parameter("ball_physics_path", "")

        self._ball_index = int(self.get_parameter("ball_pose_index").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._split_y = float(self.get_parameter("swing_side_split_y").value)
        self._hysteresis_y = max(0.0, float(self.get_parameter("swing_side_hysteresis_y").value))
        self._solve_period = float(self.get_parameter("solve_period_s").value)
        self._new_ball_gap = max(
            0.0, float(self.get_parameter("new_ball_gap_s").value)
        )
        self._new_task_tts_gate = NewTaskTTSGate(
            float(self.get_parameter("new_task_tts_min_s").value),
            float(self.get_parameter("new_task_tts_max_s").value),
        )
        self._side_aware = bool(
            self.get_parameter("use_side_aware_strike_regions").value
        )
        self._strike_regions = {
            RacketCommand.FOREHAND: StrikeRegion(
                RacketCommand.FOREHAND,
                tuple(self.get_parameter("forehand_strike_x_range").value),
                tuple(self.get_parameter("forehand_strike_y_range").value),
                tuple(self.get_parameter("forehand_strike_z_range").value),
                (
                    tuple(self.get_parameter("forehand_velocity_x_range").value),
                    tuple(self.get_parameter("forehand_velocity_y_range").value),
                    tuple(self.get_parameter("forehand_velocity_z_range").value),
                ),
            ),
            RacketCommand.BACKHAND: StrikeRegion(
                RacketCommand.BACKHAND,
                tuple(self.get_parameter("backhand_strike_x_range").value),
                tuple(self.get_parameter("backhand_strike_y_range").value),
                tuple(self.get_parameter("backhand_strike_z_range").value),
                (
                    tuple(self.get_parameter("backhand_velocity_x_range").value),
                    tuple(self.get_parameter("backhand_velocity_y_range").value),
                    tuple(self.get_parameter("backhand_velocity_z_range").value),
                ),
            ),
        }
        if self._side_aware:
            validate_side_regions(
                self._strike_regions, self._split_y, self._hysteresis_y
            )

        physics_path = str(self.get_parameter("ball_physics_path").value) or None
        physics = load_ball_physics(physics_path)
        paddle = load_paddle_params(physics_path)
        table = load_table_params(physics_path, y_max=float(self.get_parameter("table_y_max").value))
        config = PlannerConfig(
            x_hit=float(self.get_parameter("x_hit").value),
            # Landing target z = ball radius: the outgoing arc is solved for the ball
            # CENTROID reaching table contact, matching the bounce-plane convention.
            target_land=np.array([
                float(self.get_parameter("target_land_x").value),
                float(self.get_parameter("target_land_y").value),
                physics.radius,
            ]),
            delta_t_flight=float(self.get_parameter("delta_t_flight").value),
            max_predict_time=float(self.get_parameter("max_predict_time").value),
            C_r=paddle["C_r"],
            paddle_a_t=paddle["paddle_a_t"],
            paddle_b_t=paddle["paddle_b_t"],
            paddle_mu=paddle["paddle_mu"],
        )
        self.planner = HOPEPlanner(physics=physics, config=config, table=table)

        # --- Task lifecycle state ---
        self._task_id = 0
        self._task_revision = 0
        self._task_active = False
        self._locked_side = RacketCommand.FOREHAND
        self._prev_side = 0            # side of the previous task (for hysteresis); 0 = none
        self._last_solve_t = None
        self._last_sample_t = None

        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PoseArray, str(self.get_parameter("poses_topic").value), self._poses_cb, mocap_qos)
        self.cmd_pub = self.create_publisher(
            RacketCommand, str(self.get_parameter("command_topic").value), command_qos)

        strike_geometry = (
            "strike_planes="
            f"FH:{self._strike_regions[RacketCommand.FOREHAND].x_hit:.3f},"
            f"BH:{self._strike_regions[RacketCommand.BACKHAND].x_hit:.3f} m"
            if self._side_aware
            else f"x_hit={config.x_hit:.3f} m"
        )
        self.get_logger().info(
            f"HOPE planner started: {strike_geometry}, "
            f"landing={config.target_land[:2]}, split_y={self._split_y:.3f} m, "
            f"solve_period={self._solve_period:.3f} s, ball_pose_index={self._ball_index}")

    def _select_side(self, intercept_y: float) -> int:
        """Binary forehand/backhand split on the predicted lateral y (optional hysteresis).

        y below the split -> FOREHAND, at/above -> BACKHAND (the convention in
        docs/PLANNER_INTERFACE.md). Delegates to the pure
        :func:`~hope_planner.side_selection.select_swing_side`, whose boundary
        behaviour is pinned by test/test_side_selection.py. The ROS message
        constants match the pure module's (+1 / -1) by definition of the msg.
        """
        return select_swing_side(
            float(intercept_y), self._split_y, self._hysteresis_y, self._prev_side
        )

    def _poses_cb(self, msg: PoseArray) -> None:
        if len(msg.poses) <= self._ball_index:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = msg.poses[self._ball_index]
        p_ball = np.array([pose.position.x, pose.position.y, pose.position.z])

        if self._last_sample_t is not None:
            sample_dt = t - self._last_sample_t
            gap = self._new_ball_gap > 0.0 and sample_dt > self._new_ball_gap
            if sample_dt < 0.0 or gap:
                self.planner.estimator.reset()
                self._task_active = False
                self._new_task_tts_gate.reset_ball()
                self._last_solve_t = None
        self._last_sample_t = t

        # Feed every sample to the estimator, but rate-limit the solve.
        if (self._solve_period > 0.0 and self._last_solve_t is not None
                and 0.0 <= (t - self._last_solve_t) < self._solve_period):
            self.planner.estimator.push(t, p_ball)
            return
        self._last_solve_t = t

        # A degenerate mocap frame must degrade to "no command", never kill the node.
        try:
            timing_decision = None
            if self._side_aware:
                command_rejection = ""
                candidates = self.planner.update_candidates(
                    t,
                    p_ball,
                    {
                        side: region.x_hit
                        for side, region in self._strike_regions.items()
                    },
                )
                selected = select_strike_candidate(
                    candidates,
                    self._strike_regions,
                    split_y=self._split_y,
                    hysteresis_y=self._hysteresis_y,
                    previous_side=self._prev_side,
                    locked_side=self._locked_side if self._task_active else 0,
                )
                if selected:
                    candidate_tts = self.planner.candidate_time_to_strike(
                        selected[1]
                    )
                    timing_decision = self._new_task_tts_gate.decide(
                        candidate_tts, task_active=self._task_active
                    )
                    if timing_decision is TaskTimingDecision.ACCEPT:
                        candidate_cmd = self.planner.preview_strike(selected[1])
                        region = self._strike_regions[selected[0]]
                        if region.contains_command(
                            candidate_cmd.p_intercept, candidate_cmd.v_racket
                        ):
                            cmd = self.planner.commit_strike(
                                selected[1], candidate_cmd
                            )
                            selected_side = selected[0]
                        else:
                            cmd = None
                            selected_side = 0
                            command_rejection = (
                                f"{'FH' if selected[0] == RacketCommand.FOREHAND else 'BH'}: "
                                f"{region.rejection_reason(candidate_cmd.p_intercept, candidate_cmd.v_racket)}"
                            )
                    else:
                        cmd = None
                        selected_side = 0
                else:
                    cmd = None
                    selected_side = 0
            else:
                command_rejection = ""
                candidates = self.planner.update_candidates(
                    t, p_ball, {0: self.planner.config.x_hit}
                )
                strike = candidates.get(0)
                if strike is None:
                    cmd = None
                    selected_side = 0
                else:
                    candidate_tts = self.planner.candidate_time_to_strike(strike)
                    timing_decision = self._new_task_tts_gate.decide(
                        candidate_tts, task_active=self._task_active
                    )
                    if timing_decision is TaskTimingDecision.ACCEPT:
                        candidate_cmd = self.planner.preview_strike(strike)
                        cmd = self.planner.commit_strike(strike, candidate_cmd)
                        selected_side = self._select_side(
                            float(cmd.p_intercept[1])
                        )
                    else:
                        cmd = None
                        selected_side = 0
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().warning(
                f"planner solve skipped ({type(exc).__name__}: {exc}); check the mocap feed "
                "(units, frame, outliers)", throttle_duration_sec=2.0)
            return

        if cmd is None:
            # Ball struck / moving away -> end the active task; next incoming
            # ball starts a fresh task_id.
            if self.planner.ball_incoming is False:
                self._task_active = False
                self._new_task_tts_gate.reset_ball()
            elif timing_decision is TaskTimingDecision.WAIT:
                # This ball is early enough; a later estimator sample can enter
                # the startup window without consuming task_id or revision.
                return
            elif timing_decision is TaskTimingDecision.DROP:
                self.get_logger().warning(
                    "incoming ball reached the planner too late for a new task; "
                    "dropping this ball without consuming task_id",
                    throttle_duration_sec=1.0,
                )
            elif (
                self._side_aware
                and candidates
                and (
                    not self._task_active
                    or self._locked_side in candidates
                )
            ):
                position_details = "; ".join(
                    f"{'FH' if side == RacketCommand.FOREHAND else 'BH'}: "
                    f"{self._strike_regions[side].rejection_reason(strike.p_ball)}"
                    for side, strike in candidates.items()
                )
                details = command_rejection or position_details
                if self._task_active:
                    label = "FH" if self._locked_side == RacketCommand.FOREHAND else "BH"
                    summary = f"locked {label} task has no admitted same-side revision"
                else:
                    summary = "predicted command is outside the grounded Isaac target envelopes"
                self.get_logger().warning(
                    f"{summary}; command not published ({details})",
                    throttle_duration_sec=1.0,
                )
            return

        if not self._task_active:
            self._task_id = (self._task_id + 1) % _TASK_ID_WRAP
            self._task_revision = 0
            self._locked_side = selected_side
            self._prev_side = self._locked_side
            self._task_active = True
        else:
            self._task_revision = (self._task_revision + 1) % _REVISION_WRAP

        self._publish(cmd, msg.header)

    def _publish(self, cmd, header) -> None:
        out = RacketCommand()
        out.header = header
        out.header.frame_id = self._frame_id
        out.task_id = self._task_id
        out.task_revision = self._task_revision
        out.swing_side = self._locked_side
        out.position.x = float(cmd.p_intercept[0])
        out.position.y = float(cmd.p_intercept[1])
        out.position.z = float(cmd.p_intercept[2])
        out.velocity.x = float(cmd.v_racket[0])
        out.velocity.y = float(cmd.v_racket[1])
        out.velocity.z = float(cmd.v_racket[2])
        tts = self.planner.time_to_strike
        out.time_to_strike = float(tts) if tts is not None else 0.0
        self.cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HOPEPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
