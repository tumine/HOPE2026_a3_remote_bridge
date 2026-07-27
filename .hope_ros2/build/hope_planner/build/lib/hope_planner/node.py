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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .constants import PlannerConfig, load_ball_physics, load_paddle_params, load_table_params
from .incoming_cycle import IncomingCycleDetector
from .planner import HOPEPlanner
from .side_selection import select_swing_side
from .strike_selection import (
    StrikeRegion,
    incoming_plane_crossing,
    select_strike_candidate,
    validate_side_regions,
)
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
        # A marker can stay visible while a human retrieves and re-serves the
        # ball. Re-arm only after stable non-incoming motion followed by stable
        # incoming motion in front of both strike planes. The separated speed
        # thresholds form a dead band that rejects estimator sign chatter.
        self.declare_parameter("new_ball_incoming_vx_threshold_mps", -0.50)
        self.declare_parameter("new_ball_nonincoming_vx_threshold_mps", -0.10)
        self.declare_parameter("new_ball_incoming_confirm_samples", 5)
        self.declare_parameter("new_ball_nonincoming_confirm_samples", 10)
        self.declare_parameter("new_ball_rearm_min_x", 0.30)
        # Optional live commissioning margins around the frozen training
        # envelopes. The shipped YAML keeps these at zero; run_hope_planner.sh
        # supplies deliberately bounded real-robot defaults.
        self.declare_parameter("strike_y_margin_m", 0.0)
        self.declare_parameter("strike_z_margin_m", 0.0)
        self.declare_parameter("racket_velocity_margin_mps", 0.0)
        # A new policy task must begin at the same 1.0 s lead used in training.
        # Once active, revisions remain admissible below this startup window.
        self.declare_parameter("new_task_tts_min_s", 0.95)
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
        self._incoming_cycle = IncomingCycleDetector(
            incoming_vx_threshold=float(
                self.get_parameter("new_ball_incoming_vx_threshold_mps").value
            ),
            nonincoming_vx_threshold=float(
                self.get_parameter("new_ball_nonincoming_vx_threshold_mps").value
            ),
            incoming_confirm_samples=int(
                self.get_parameter("new_ball_incoming_confirm_samples").value
            ),
            nonincoming_confirm_samples=int(
                self.get_parameter("new_ball_nonincoming_confirm_samples").value
            ),
            min_rearm_x=float(
                self.get_parameter("new_ball_rearm_min_x").value
            ),
        )
        strike_y_margin = float(self.get_parameter("strike_y_margin_m").value)
        strike_z_margin = float(self.get_parameter("strike_z_margin_m").value)
        self._racket_velocity_margin = float(
            self.get_parameter("racket_velocity_margin_mps").value
        )
        live_margins = (
            strike_y_margin,
            strike_z_margin,
            self._racket_velocity_margin,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in live_margins):
            raise ValueError("live strike/velocity margins must be finite and non-negative")
        self._strike_position_margin = np.array(
            [0.0, strike_y_margin, strike_z_margin], dtype=np.float64
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
        self._diagnostic_ball_id = 1
        self._last_observed_ball = None
        self._reported_plane_crossings = set()
        self._reported_timing_decisions = set()
        self._plane_decisions = {
            RacketCommand.FOREHAND: "WAIT: estimator needs at least 6 samples",
            RacketCommand.BACKHAND: "WAIT: estimator needs at least 6 samples",
        }

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
            f"new_task_tts=[{self._new_task_tts_gate.min_s:.3f},"
            f"{self._new_task_tts_gate.max_s:.3f}] s, "
            f"ball_rearm=(vx_in<={self._incoming_cycle.incoming_vx_threshold:.2f},"
            f" vx_out>={self._incoming_cycle.nonincoming_vx_threshold:.2f} m/s, "
            f"confirm={self._incoming_cycle.incoming_confirm_samples}/"
            f"{self._incoming_cycle.nonincoming_confirm_samples}, "
            f"x>={self._incoming_cycle.min_rearm_x:.2f} m), "
            f"live_margins=(y={self._strike_position_margin[1]:.3f}m, "
            f"z={self._strike_position_margin[2]:.3f}m, "
            f"racket_v={self._racket_velocity_margin:.3f}m/s), "
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

    @staticmethod
    def _side_label(side: int) -> str:
        return "FOREHAND" if side == RacketCommand.FOREHAND else "BACKHAND"

    def _reset_plane_diagnostics(self) -> None:
        self._diagnostic_ball_id += 1
        self._last_observed_ball = None
        self._reported_plane_crossings.clear()
        self._reported_timing_decisions.clear()
        self._plane_decisions = {
            RacketCommand.FOREHAND: "WAIT: estimator needs at least 6 samples",
            RacketCommand.BACKHAND: "WAIT: estimator needs at least 6 samples",
        }

    def _report_measured_plane_crossings(self, p_ball: np.ndarray) -> None:
        previous = self._last_observed_ball
        self._last_observed_ball = np.asarray(p_ball, dtype=np.float64).copy()
        if previous is None:
            return
        for side, region in self._strike_regions.items():
            if side in self._reported_plane_crossings:
                continue
            crossing = incoming_plane_crossing(previous, p_ball, region.x_hit)
            if crossing is None:
                continue
            self._reported_plane_crossings.add(side)
            self.get_logger().warning(
                "STRIKE_PLANE_CROSS ball=%d plane=%s x=%.3f measured_y=%.4f "
                "measured_z=%.4f result=%s"
                % (
                    self._diagnostic_ball_id,
                    self._side_label(side),
                    region.x_hit,
                    crossing[1],
                    crossing[2],
                    self._plane_decisions[side],
                )
            )

    def _update_candidate_diagnostics(self, candidates) -> None:
        for side, region in self._strike_regions.items():
            # Once a task is accepted, later estimator jitter must not rewrite
            # the eventual physical-plane report into a misleading rejection.
            if (
                self._task_active
                and side == self._locked_side
                and self._plane_decisions[side].startswith("ACCEPT")
            ):
                continue
            strike = candidates.get(side)
            if strike is None:
                if self.planner.ball_incoming is None:
                    reason = "WAIT: estimator needs at least 6 samples"
                elif self.planner.ball_incoming is False:
                    reason = "REJECT: estimated ball velocity is not incoming (vx >= 0)"
                else:
                    reason = (
                        "REJECT: no usable future crossing within %.2fs "
                        "(plane not reached or dead-ball crossing)"
                        % self.planner.config.max_predict_time
                    )
                self._plane_decisions[side] = reason
                continue
            if not region.contains(
                strike.p_ball, margin=self._strike_position_margin
            ):
                self._plane_decisions[side] = (
                    "REJECT: predicted position "
                    + region.rejection_reason(
                        strike.p_ball,
                        position_margin=self._strike_position_margin,
                    )
                )
                continue
            classified = self._select_side(float(strike.p_ball[1]))
            if classified != side:
                self._plane_decisions[side] = (
                    "REJECT: predicted y=%.4f classified as %s"
                    % (strike.p_ball[1], self._side_label(classified))
                )
                continue
            if self._task_active and self._locked_side != side:
                self._plane_decisions[side] = (
                    "REJECT: active task is locked to %s"
                    % self._side_label(self._locked_side)
                )
                continue
            self._plane_decisions[side] = (
                "WAIT: position is inside %s box; timing/velocity gates pending"
                % self._side_label(side)
            )

    def _update_incoming_cycle(self) -> None:
        """Re-arm diagnostics after a debounced physical incoming cycle."""

        position = self.planner.estimated_ball_position
        velocity = self.planner.estimated_ball_velocity
        if position is None or velocity is None:
            return
        if self._incoming_cycle.update(float(position[0]), float(velocity[0])):
            self._reset_plane_diagnostics()
            self.get_logger().warning(
                "BALL_REARM ball=%d reason=confirmed incoming cycle "
                "x=%.3fm vx=%.3fm/s (hysteresis+consecutive-sample gate)"
                % (
                    self._diagnostic_ball_id,
                    position[0],
                    velocity[0],
                )
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
                self._incoming_cycle.reset()
                self._reset_plane_diagnostics()
        self._last_sample_t = t
        self._report_measured_plane_crossings(p_ball)

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
                self._update_incoming_cycle()
                self._update_candidate_diagnostics(candidates)
                selected = select_strike_candidate(
                    candidates,
                    self._strike_regions,
                    split_y=self._split_y,
                    hysteresis_y=self._hysteresis_y,
                    previous_side=self._prev_side,
                    locked_side=self._locked_side if self._task_active else 0,
                    position_margin=self._strike_position_margin,
                )
                if selected:
                    candidate_tts = self.planner.candidate_time_to_strike(
                        selected[1]
                    )
                    was_dropped = self._new_task_tts_gate.current_ball_dropped
                    timing_decision = self._new_task_tts_gate.decide(
                        candidate_tts, task_active=self._task_active
                    )
                    timing_key = timing_decision.value
                    should_report_timing = (
                        not self._task_active
                        and timing_key not in self._reported_timing_decisions
                        and not (
                            timing_decision is TaskTimingDecision.DROP
                            and was_dropped
                        )
                    )
                    if should_report_timing:
                        self._reported_timing_decisions.add(timing_key)
                        estimated_position = self.planner.estimated_ball_position
                        estimated_velocity = self.planner.estimated_ball_velocity
                        state_text = (
                            " est_x=%.4fm est_vx=%.4fm/s"
                            % (estimated_position[0], estimated_velocity[0])
                            if estimated_position is not None
                            and estimated_velocity is not None
                            else ""
                        )
                        self.get_logger().warning(
                            "TTS_GATE ball=%d side=%s decision=%s tts=%.4fs "
                            "window=[%.4f, %.4f]s%s"
                            % (
                                self._diagnostic_ball_id,
                                self._side_label(selected[0]),
                                timing_decision.value.upper(),
                                candidate_tts,
                                self._new_task_tts_gate.min_s,
                                self._new_task_tts_gate.max_s,
                                state_text,
                            )
                        )
                    if timing_decision is TaskTimingDecision.ACCEPT:
                        candidate_cmd = self.planner.preview_strike(selected[1])
                        region = self._strike_regions[selected[0]]
                        if region.contains_command(
                            candidate_cmd.p_intercept,
                            candidate_cmd.v_racket,
                            position_margin=self._strike_position_margin,
                            velocity_margin=self._racket_velocity_margin,
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
                                f"{region.rejection_reason(candidate_cmd.p_intercept, candidate_cmd.v_racket, position_margin=self._strike_position_margin, velocity_margin=self._racket_velocity_margin)}"
                            )
                            self._plane_decisions[selected[0]] = (
                                "REJECT: " + command_rejection
                            )
                    elif timing_decision is TaskTimingDecision.WAIT:
                        cmd = None
                        selected_side = 0
                        self._plane_decisions[selected[0]] = (
                            "WAIT: time_to_strike=%.4fs is earlier than admission "
                            "window [%.4f, %.4f]s"
                            % (
                                candidate_tts,
                                self._new_task_tts_gate.min_s,
                                self._new_task_tts_gate.max_s,
                            )
                        )
                    else:
                        cmd = None
                        selected_side = 0
                        self._plane_decisions[selected[0]] = (
                            "REJECT: time_to_strike=%.4fs is too late for admission "
                            "window [%.4f, %.4f]s; physical ball dropped"
                            % (
                                candidate_tts,
                                self._new_task_tts_gate.min_s,
                                self._new_task_tts_gate.max_s,
                            )
                        )
                else:
                    cmd = None
                    selected_side = 0
            else:
                command_rejection = ""
                candidates = self.planner.update_candidates(
                    t, p_ball, {0: self.planner.config.x_hit}
                )
                self._update_incoming_cycle()
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
                    f"{self._strike_regions[side].rejection_reason(strike.p_ball, position_margin=self._strike_position_margin)}"
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

        new_task = not self._task_active
        if new_task:
            self._task_id = (self._task_id + 1) % _TASK_ID_WRAP
            self._task_revision = 0
            self._locked_side = selected_side
            self._prev_side = self._locked_side
            self._task_active = True
        else:
            self._task_revision = (self._task_revision + 1) % _REVISION_WRAP

        self._plane_decisions[selected_side] = (
            "ACCEPT: task_id=%d revision=%d side=%s tts=%.4fs "
            "target=[%.4f,%.4f,%.4f] racket_v=[%.4f,%.4f,%.4f]"
            % (
                self._task_id,
                self._task_revision,
                self._side_label(selected_side),
                self.planner.time_to_strike,
                cmd.p_intercept[0],
                cmd.p_intercept[1],
                cmd.p_intercept[2],
                cmd.v_racket[0],
                cmd.v_racket[1],
                cmd.v_racket[2],
            )
        )

        if new_task:
            self.get_logger().warning(
                "PLANNER_TASK_PUBLISHED ball=%d task=%d side=%s tts=%.4fs "
                "target=[%.4f,%.4f,%.4f] racket_v=[%.4f,%.4f,%.4f]"
                % (
                    self._diagnostic_ball_id,
                    self._task_id,
                    self._side_label(selected_side),
                    self.planner.time_to_strike,
                    cmd.p_intercept[0],
                    cmd.p_intercept[1],
                    cmd.p_intercept[2],
                    cmd.v_racket[0],
                    cmd.v_racket[1],
                    cmd.v_racket[2],
                )
            )

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
