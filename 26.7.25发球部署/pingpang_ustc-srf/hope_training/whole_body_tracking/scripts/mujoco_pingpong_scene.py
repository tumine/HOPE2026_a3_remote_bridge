# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""Real-physics MuJoCo ping-pong scene for the sim-to-sim success_rate evaluation.

This module assembles a single MuJoCo model that contains

  * the shipped ``a3_pingpong`` robot (loaded verbatim from its MJCF -- no copy of
    the robot model lives here), and
  * a static **table top**, **net** and **floor** contact surface, plus a dynamic
    **ball** (free joint, sphere), all sized/placed and given contact restitution +
    friction from ``configs/ball_physics.yaml``.

The point of the scene is that ``success_rate`` can be measured from an ACTUAL
simulated ball that really bounces off the racket, table and net -- not from an
analytic predicted-landing rollout. The robot's racket link already carries a
collision geom in the shipped MJCF (``right_racket_collision``), so no extra racket
contact geom has to be added; the ball collides with it directly.

Frames
------
The MuJoCo world frame is the robot frame: the robot's own floor is at ``z = 0`` and
the robot stands on it. The **table frame** used by the success metric places the
table playing surface at ``z = 0`` with its origin at the near-side left corner of
the table (``+x`` toward the opponent, ``+y`` left). The table centreline can be
aligned with the robot's fixed startup station ``table_center_y``. The two frames
differ by a pure translation
``offset = (near_edge_x, table_center_y + width/2, table_height)`` so that

    table_frame_position = mujoco_world_position - offset

Velocities are identical in both frames (pure translation). All success checks
(net crossing, opponent-half first bounce) are evaluated in the table frame; the
policy-facing quantities (robot state, racket target) stay in the MuJoCo world frame
exactly as the reference deploy runner expects.

Restitution
-----------
MuJoCo has no direct restitution parameter; a bouncy contact is a lightly-damped
soft constraint. Each ball<->surface contact is added as an explicit ``<pair>`` whose
``solref`` damping ratio is derived from the configured normal restitution ``e`` via
the linear spring-damper log-decrement ``zeta = -ln(e) / sqrt(pi^2 + ln(e)^2)``.
Near-elastic surfaces use a slightly softer time constant for numerical stability.
This remains an approximation for the net/floor. For the table and racket, native
contacts prove that a collision occurred, then the evaluator applies the shared
fitted no-spin normal/tangential impulse. This keeps the real MuJoCo contact geometry
while avoiding a second, engine-specific Coulomb response.

Robot dynamics
--------------
The robot uses the same 50 Hz action period, named PD gains, effort ranges,
armatures and passive-joint parameters as the Isaac training task. Isaac's 200 Hz
PD drive is solved implicitly inside PhysX; MuJoCo's reference path realizes the
same equation as an explicit torque loop, for which a verified 1 ms step is needed
for stability. The finer step is also needed because the 40 mm ball can traverse
roughly one diameter in 5 ms, which misses racket contacts and shifts landing
events. MuJoCo has no native joint-speed constraint corresponding to PhysX
``velocity_limit_sim``; the MJCF therefore stores those 31 canonical limits as
``isaac_joint_velocity_limits`` and this scene projects robot qvel onto them after
every contact step. Ball qvel is never affected by this projection.
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

# Physics substep applied-drag uses the ball's world-frame linear velocity. For a
# MuJoCo free joint the first three qvel entries are exactly that world velocity.


def _dampratio_from_restitution(e: float) -> float:
    """Linear spring-damper damping ratio that yields normal restitution ``e``."""
    e = float(min(max(e, 1e-3), 0.999))
    le = math.log(e)
    return -le / math.sqrt(math.pi * math.pi + le * le)


@dataclass
class StepResult:
    """Physical events observed during one 50 Hz control step (sub-step resolution).

    Everything here is a raw physical observation of the real simulated ball; the
    success DEFINITION (after-contact gating, net clearance threshold, opponent-half
    test) is applied by the evaluator, not here.
    """

    ball_racket_contact: bool = False
    # Each net-plane (x = net_x) crossing during the step: (z_table_at_crossing, x_sign)
    # where x_sign = +1 if the ball moved in +x (outgoing toward the opponent).
    net_crossings: list = field(default_factory=list)
    # Each table-surface plane (z_table = ball_radius) crossing: (x_table, y_table, z_sign)
    # where z_sign = -1 for a downward crossing (a bounce onto the surface).
    surface_crossings: list = field(default_factory=list)
    # Actual ball<->table contacts that begin during this control step, in the
    # canonical table frame. Unlike a plane crossing, this proves collision.
    table_contacts: list = field(default_factory=list)


@dataclass
class RobotObsState:
    """Proprioceptive state consumed by the reference 111-D observation builder.

    Attribute names match the reference ``RobotState`` so it can be passed straight
    into ``build_observation`` without conversion.
    """

    base_pos_w: np.ndarray
    base_quat_w: np.ndarray
    base_ang_vel_b: np.ndarray
    q: np.ndarray
    qd: np.ndarray


class PingPongRealPhysicsScene:
    """Robot + table + net + floor + dynamic ball, stepped with real MuJoCo physics."""

    def __init__(
        self,
        robot_xml_path: str,
        ball_cfg: dict,
        joint_names,
        control_dt: float = 0.02,
        near_edge_x: float = 0.30,
        table_center_y: float = 0.0,
        launch_viewer: bool = False,
        paddle_contact_model=None,
        paddle_physics=None,
        reset_joint_pos=None,
        joint_velocity_limits=None,
        contact_physics_dt: float = 0.001,
        video_path: str | None = None,
        video_width: int = 1280,
        video_height: int = 720,
        video_fps: float | None = None,
    ) -> None:
        import mujoco  # lazy import so the module imports without MuJoCo present

        self._mj = mujoco
        self.control_dt = float(control_dt)
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)
        self._reset_joint_pos = (
            None
            if reset_joint_pos is None
            else np.asarray(reset_joint_pos, dtype=np.float64)
            .reshape(self.num_joints)
            .copy()
        )

        # --- geometry from the shared ball-physics config -----------------------
        g = float(ball_cfg.get("gravity", 9.81))
        table = ball_cfg.get("table", {})
        net = ball_cfg.get("net", {})
        ball = ball_cfg.get("ball", {})
        drag = ball_cfg.get("drag", {})
        self.length = float(table.get("length", 2.74))
        self.width = float(table.get("width", 1.525))
        self.table_height = float(table.get("height", 0.76))
        self.table_thickness = float(table.get("thickness", 0.05))
        self.net_height = float(net.get("height", 0.1525))
        self.net_x_table = float(net.get("x_position", self.length / 2.0))
        self.net_overhang = float(net.get("overhang", 0.15))
        self.net_thickness = float(net.get("thickness", 0.01))
        self.ball_radius = float(ball.get("radius", 0.020))
        self.ball_mass = float(ball.get("mass", 0.0027))
        self.drag_k = float(drag.get("k", 0.1261))
        self.velocity_clip = float(drag.get("velocity_clip", 50.0))
        self.gravity = g
        table_contact = ball_cfg.get("contact", {}).get("table", {})
        self._table_restitution = float(table_contact.get("restitution", 0.9215))
        self._table_tangential_damping = float(
            table_contact.get("tangential_damping", 0.369)
        )
        self._table_tangential_cap = float(
            table_contact.get("tangential_cap", 2.0)
        )
        # MuJoCo's native Coulomb contact does not reproduce the fitted analytic
        # tangential damping (sharing the YAML friction value is insufficient).
        # The evaluator may therefore supply the exact shared no-spin paddle
        # impulse. MuJoCo still determines whether/where contact occurs; only the
        # first post-contact ball velocity is replaced by the calibrated model.
        self._paddle_contact_model = paddle_contact_model
        self._paddle_physics = paddle_physics
        # Training uses local +Y for forehand and -Y for backhand. This task
        # selection intentionally survives reset_stand().
        self._paddle_face_sign = 1.0
        self._paddle_contact_latched = False
        self._table_contact_active = False
        self.last_paddle_contact = None
        self.last_table_contact = None

        self.near_edge_x = float(near_edge_x)
        self.table_center_y = float(table_center_y)
        # mujoco_world -> table_frame is a pure translation by this offset.
        self.offset = np.array(
            [self.near_edge_x, self.table_center_y + self.width / 2.0, self.table_height],
            dtype=np.float64,
        )
        self.net_x_mujoco = self.near_edge_x + self.net_x_table

        # --- build the combined model via the MuJoCo spec API -------------------
        self._build_model(mujoco, robot_xml_path, ball_cfg)
        contact_physics_dt = float(contact_physics_dt)
        if not np.isfinite(contact_physics_dt) or contact_physics_dt <= 0.0:
            raise ValueError(
                f"contact_physics_dt must be finite and > 0, got {contact_physics_dt}"
            )
        # Keep the evaluator's contact/explicit-PD step explicit even when a user
        # supplies a custom robot MJCF; see the module-level rationale.
        self.model.opt.timestep = contact_physics_dt

        # --- resolve addresses -------------------------------------------------
        m = self.model
        self._base_qadr = int(m.jnt_qposadr[self._joint_id("pelvis_free_joint")])
        self._base_vadr = int(m.jnt_dofadr[self._joint_id("pelvis_free_joint")])
        self._ball_qadr = int(m.jnt_qposadr[self._joint_id("ball_free_joint")])
        self._ball_vadr = int(m.jnt_dofadr[self._joint_id("ball_free_joint")])
        self._ball_bid = self._body_id("ball")
        self._racket_sid = self._site_id("right_racket")
        self._ball_gid = self._geom_id("ball_geom")
        self._racket_gid = self._geom_id("right_racket_collision")
        self._table_gid = self._geom_id("table_geom")
        self._gyro_adr = self._sensor_adr("pelvis_imu_gyro")

        # Controlled-joint qpos/qvel addresses + driving actuator indices.
        self._q_adr = np.zeros(self.num_joints, dtype=int)
        self._v_adr = np.zeros(self.num_joints, dtype=int)
        self._act_idx = np.full(self.num_joints, -1, dtype=int)
        trn_joint = m.actuator_trnid[:, 0]
        for i, name in enumerate(self.joint_names):
            jid = self._joint_id(name)
            self._q_adr[i] = int(m.jnt_qposadr[jid])
            self._v_adr[i] = int(m.jnt_dofadr[jid])
            matches = np.where(trn_joint == jid)[0]
            if matches.size == 0:
                raise ValueError(f"no actuator drives joint '{name}'")
            self._act_idx[i] = int(matches[0])
        self._ctrl_lo = m.actuator_ctrlrange[self._act_idx, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[self._act_idx, 1].copy()
        self._ctrl_limited = m.actuator_ctrllimited[self._act_idx].astype(bool)

        self._substeps = _integer_substeps(self.control_dt, m.opt.timestep)
        self._joint_velocity_limits = (
            _model_velocity_limits(self._mj, m, self.num_joints)
            if joint_velocity_limits is None
            else _validate_velocity_limits(joint_velocity_limits, self.num_joints)
        )

        self._q_des = np.zeros(self.num_joints)
        self._kp = np.zeros(self.num_joints)
        self._kd = np.zeros(self.num_joints)

        self._viewer = None
        if launch_viewer:
            from mujoco import viewer as mj_viewer

            self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.cam.lookat[:] = [1.2, self.table_center_y, 0.8]
            self._viewer.cam.distance = 4.5
            self._viewer.cam.azimuth = 135.0
            self._viewer.cam.elevation = -20.0

        self._video_renderer = None
        self._video_process = None
        self.video_frame_count = 0
        if video_path is not None:
            self._start_video_recording(
                pathlib.Path(video_path),
                width=int(video_width),
                height=int(video_height),
                fps=float(video_fps or (1.0 / self.control_dt)),
            )

    # -- model construction -----------------------------------------------------
    def _build_model(self, mujoco, robot_xml_path, ball_cfg) -> None:
        spec = mujoco.MjSpec.from_file(str(robot_xml_path))
        wb = spec.worldbody

        # Collision bitmasks: the ball collides with the racket (already in the
        # robot model) + our table/net, but our table/net do NOT collide with the
        # robot (so the static surfaces never shove the robot around).
        BALL_CT, BALL_CA = 8, 15   # ball
        SURF_CT, SURF_CA = 8, 8    # table + net (share the ball's bit only)

        x0, w, h, th = self.near_edge_x, self.width, self.table_height, self.table_thickness
        L = self.length

        # Table top slab: top face at table_height (z=0 in the table frame).
        tb = wb.add_body(
            name="table_top", pos=[x0 + L / 2.0, self.table_center_y, h - th / 2.0]
        )
        tb.add_geom(
            name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[L / 2.0, w / 2.0, th / 2.0], contype=SURF_CT, conaffinity=SURF_CA,
            rgba=[0.10, 0.35, 0.55, 1.0],
        )
        # Net slab at the net plane, spanning the table width + overhang each side.
        nb = wb.add_body(
            name="net_body", pos=[self.net_x_mujoco, self.table_center_y, h + self.net_height / 2.0]
        )
        nb.add_geom(
            name="net_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.net_thickness / 2.0, w / 2.0 + self.net_overhang, self.net_height / 2.0],
            contype=SURF_CT, conaffinity=SURF_CA, rgba=[0.9, 0.9, 0.9, 0.35],
        )
        # Dynamic ball (free joint).
        bb = wb.add_body(name="ball", pos=[x0 + L / 2.0, self.table_center_y, h + 0.5])
        bb.add_freejoint(name="ball_free_joint")
        bb.add_geom(
            name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.ball_radius, 0.0, 0.0], mass=self.ball_mass,
            contype=BALL_CT, conaffinity=BALL_CA, rgba=[1.0, 0.55, 0.0, 1.0],
        )

        # Contact pairs: restitution + friction from the config (see module docstring).
        contact = ball_cfg.get("contact", {})

        def _friction(surface_key: str, default: float) -> float:
            return float(contact.get(surface_key, {}).get("dynamic_friction", default))

        def _restitution(surface_key: str, default: float) -> float:
            return float(contact.get(surface_key, {}).get("restitution", default))

        def _add_pair(g1: str, g2: str, e: float, fr: float) -> None:
            pair = spec.add_pair(geomname1=g1, geomname2=g2)
            timeconst = 0.03 if e > 0.8 else 0.02   # softer for near-elastic stability
            pair.solref = [timeconst, _dampratio_from_restitution(e)]
            pair.friction = [fr, fr, 0.005, 1e-4, 1e-4]
            pair.condim = 3

        _add_pair("ball_geom", "table_geom", _restitution("table", 0.9215), _friction("table", 0.40))
        _add_pair("ball_geom", "net_geom", _restitution("net", 0.10), _friction("net", 0.50))
        _add_pair("ball_geom", "right_racket_collision", _restitution("paddle", 0.654), _friction("paddle", 0.60))
        _add_pair("ball_geom", "floor", _restitution("floor", 0.40), _friction("floor", 0.80))

        # Extend the model's stand keyframe with a resting ball pose so the combined
        # nq matches (the ball's 7 qpos are overwritten per serve anyway).
        if spec.keys:
            key = spec.keys[0]
            key.qpos = list(np.array(key.qpos, dtype=np.float64)) + [
                x0 + L / 2.0, self.table_center_y, h + 0.5, 1.0, 0.0, 0.0, 0.0
            ]

        self.model = spec.compile()
        self.data = self._mj.MjData(self.model)
        # Honour the configured gravity magnitude (drag is applied on top via xfrc).
        self.model.opt.gravity[:] = [0.0, 0.0, -self.gravity]

    # -- id / address helpers ---------------------------------------------------
    def _joint_id(self, name: str) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' not found")
        return jid

    def _body_id(self, name: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SITE, name)

    def _geom_id(self, name: str) -> int:
        gid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"geom '{name}' not found")
        return gid

    def _sensor_adr(self, name: str) -> int:
        sid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else -1

    # -- frame transform --------------------------------------------------------
    def to_table(self, pos_w) -> np.ndarray:
        """MuJoCo world position -> table-frame position (pure translation)."""
        return np.asarray(pos_w, dtype=np.float64) - self.offset

    # -- reset / serve ----------------------------------------------------------
    def reset_stand(self) -> None:
        """Reset the robot to its grounded stand keyframe; park the ball far away."""
        m, d = self.model, self.data
        if m.nkey > 0:
            self._mj.mj_resetDataKeyframe(m, d, 0)
        else:
            self._mj.mj_resetData(m, d)
        if self._reset_joint_pos is not None:
            # Keep the shared ActionAdapter default authoritative for custom/older
            # MJCF keyframes as well as the shipped aligned keyframe.
            d.qpos[self._q_adr] = self._reset_joint_pos
        d.xfrc_applied[:] = 0.0
        # Park the ball out of play until a serve is set.
        d.qpos[self._ball_qadr:self._ball_qadr + 3] = [
            self.near_edge_x + self.length / 2.0,
            self.table_center_y,
            self.table_height + 1.0,
        ]
        d.qpos[self._ball_qadr + 3:self._ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_vadr:self._ball_vadr + 6] = 0.0
        self._paddle_contact_latched = False
        self._table_contact_active = False
        self.last_paddle_contact = None
        self.last_table_contact = None
        self._mj.mj_forward(m, d)

    def set_ball(self, pos_w, vel_w) -> None:
        """Place the ball at ``pos_w`` (MuJoCo world) with world linear velocity ``vel_w``."""
        d = self.data
        d.qpos[self._ball_qadr:self._ball_qadr + 3] = np.asarray(pos_w, dtype=np.float64)
        d.qpos[self._ball_qadr + 3:self._ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_vadr:self._ball_vadr + 3] = np.asarray(vel_w, dtype=np.float64)
        d.qvel[self._ball_vadr + 3:self._ball_vadr + 6] = 0.0
        self._paddle_contact_latched = False
        self._table_contact_active = False
        self.last_paddle_contact = None
        self.last_table_contact = None
        self._mj.mj_forward(self.model, self.data)

    def set_paddle_face_sign(self, sign: float) -> None:
        """Select the striking face (+1 forehand, -1 backhand).

        ``reset_stand`` deliberately does not touch this value, so an
        independent-trial reset cannot silently turn a backhand into forehand.
        """

        sign = float(sign)
        if not np.isfinite(sign) or sign not in (-1.0, 1.0):
            raise ValueError(f"paddle face sign must be exactly +1 or -1, got {sign}")
        self._paddle_face_sign = sign

    # -- state readout ----------------------------------------------------------
    def read_robot_state(self) -> RobotObsState:
        d = self.data
        base_pos = d.qpos[self._base_qadr:self._base_qadr + 3].copy()
        base_quat = d.qpos[self._base_qadr + 3:self._base_qadr + 7].copy()  # (w,x,y,z)
        if self._gyro_adr >= 0:
            base_ang_vel = d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        else:
            base_ang_vel = d.qvel[self._base_vadr + 3:self._base_vadr + 6].copy()
        return RobotObsState(
            base_pos_w=base_pos,
            base_quat_w=base_quat,
            base_ang_vel_b=base_ang_vel,
            q=d.qpos[self._q_adr].copy(),
            qd=d.qvel[self._v_adr].copy(),
        )

    def ball_state(self):
        """Return (pos_w, vel_w) of the ball in the MuJoCo world frame."""
        d = self.data
        pos = d.qpos[self._ball_qadr:self._ball_qadr + 3].copy()
        vel = d.qvel[self._ball_vadr:self._ball_vadr + 3].copy()
        return pos, vel

    def racket_site_state(self):
        """Return (pos_w, vel_w) of the racket site in the MuJoCo world frame."""
        d = self.data
        pos = d.site_xpos[self._racket_sid].copy()
        res = np.zeros(6)
        self._mj.mj_objectVelocity(self.model, d, self._mj.mjtObj.mjOBJ_SITE, self._racket_sid, res, 0)
        return pos, res[3:6].copy()

    def base_pos_w(self) -> np.ndarray:
        return self.data.qpos[self._base_qadr:self._base_qadr + 3].copy()

    def base_fallen(self, min_height: float = 0.4) -> bool:
        """Crude fall check: pelvis dropped well below the stand height."""
        return bool(self.data.qpos[self._base_qadr + 2] < min_height)

    # -- control + stepping -----------------------------------------------------
    def write_targets(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._q_des = np.asarray(q_des, dtype=np.float64).reshape(self.num_joints)
        self._kp = np.asarray(kp, dtype=np.float64).reshape(self.num_joints)
        self._kd = np.asarray(kd, dtype=np.float64).reshape(self.num_joints)

    def _apply_pd(self) -> None:
        d = self.data
        q = d.qpos[self._q_adr]
        qd = d.qvel[self._v_adr]
        tau = self._kp * (self._q_des - q) - self._kd * qd
        tau = np.where(self._ctrl_limited, np.clip(tau, self._ctrl_lo, self._ctrl_hi), tau)
        d.ctrl[self._act_idx] = tau

    def _enforce_joint_velocity_limits(self) -> bool:
        """Project robot qd onto Isaac's simulation bounds; leave ball qvel alone."""

        qd = self.data.qvel[self._v_adr]
        clipped = np.clip(qd, -self._joint_velocity_limits, self._joint_velocity_limits)
        if np.array_equal(qd, clipped):
            return False
        self.data.qvel[self._v_adr] = clipped
        return True

    def _apply_ball_drag(self) -> None:
        """No-spin aerodynamic drag as a Cartesian force F = -m*k*|v|*v (world frame)."""
        d = self.data
        v = d.qvel[self._ball_vadr:self._ball_vadr + 3]
        speed = float(np.linalg.norm(v))
        speed = min(speed, self.velocity_clip)
        d.xfrc_applied[self._ball_bid, :3] = -self.ball_mass * self.drag_k * speed * v

    def _apply_calibrated_table_bounce(self, incoming_velocity: np.ndarray) -> None:
        """Replace native Coulomb response with the shared fitted no-spin bounce."""

        incoming = np.asarray(incoming_velocity, dtype=np.float64)
        if incoming[2] >= 0.0:
            return
        horizontal = incoming[:2]
        horizontal_speed = float(np.linalg.norm(horizontal))
        removed_speed = min(
            self._table_tangential_damping * horizontal_speed,
            self._table_tangential_cap
            * (1.0 + self._table_restitution)
            * abs(float(incoming[2])),
        )
        horizontal_scale = (
            max(0.0, 1.0 - removed_speed / horizontal_speed)
            if horizontal_speed > 1.0e-12
            else 0.0
        )
        outgoing = np.array(
            [
                horizontal[0] * horizontal_scale,
                horizontal[1] * horizontal_scale,
                -self._table_restitution * incoming[2],
            ],
            dtype=np.float64,
        )
        self.data.qvel[self._ball_vadr:self._ball_vadr + 3] = outgoing
        # Move the centre just outside the contact surface so MuJoCo cannot add a
        # second solver-dependent impulse on the next sub-step.
        self.data.qpos[self._ball_qadr + 2] = (
            self.table_height + self.ball_radius + 1.0e-4
        )
        self.last_table_contact = {
            "incoming_ball_velocity": incoming.copy(),
            "outgoing_ball_velocity": outgoing.copy(),
            "position_table": self.to_table(
                self.data.qpos[self._ball_qadr:self._ball_qadr + 3]
            ).copy(),
        }
        self._mj.mj_forward(self.model, self.data)

    def _start_video_recording(
        self, path: pathlib.Path, *, width: int, height: int, fps: float
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0.0:
            raise ValueError("video width, height and fps must be positive")
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), height
        )
        self._video_renderer = self._mj.Renderer(
            self.model, height=height, width=width
        )
        self._video_camera = self._mj.MjvCamera()
        self._video_camera.lookat[:] = [
            self.near_edge_x + 0.5 * self.length,
            self.table_center_y - 0.15,
            0.85,
        ]
        self._video_camera.distance = 4.6
        self._video_camera.azimuth = 135.0
        self._video_camera.elevation = -20.0
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self._video_process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def _record_video_frame(self) -> None:
        if self._video_renderer is None or self._video_process is None:
            return
        self._video_renderer.update_scene(self.data, camera=self._video_camera)
        frame = np.asarray(self._video_renderer.render(), dtype=np.uint8)
        if self._video_process.stdin is None:
            raise RuntimeError("video encoder stdin is unavailable")
        try:
            self._video_process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while encoding MuJoCo video") from exc
        self.video_frame_count += 1

    def step(self) -> StepResult:
        """Advance one 50 Hz control tick; report physical ball events sub-step wise."""
        wall_step_started = time.perf_counter()
        mj = self._mj
        m, d = self.model, self.data
        result = StepResult()
        surface_table = self.ball_radius

        for _ in range(self._substeps):
            self._apply_pd()
            self._apply_ball_drag()
            p_before = d.qpos[self._ball_qadr:self._ball_qadr + 3].copy()
            ball_vel_before = d.qvel[self._ball_vadr:self._ball_vadr + 3].copy()
            racket_pos_before, racket_vel_before = self.racket_site_state()
            racket_normal_before = (
                d.site_xmat[self._racket_sid].reshape(3, 3)[:, 1].copy()
                * self._paddle_face_sign
            )
            mj.mj_step(m, d)
            p_integrated = d.qpos[
                self._ball_qadr:self._ball_qadr + 3
            ].copy()
            if self._enforce_joint_velocity_limits():
                # Refresh site/body velocities and sensors after the qvel projection.
                mj.mj_forward(m, d)

            # real ball<->racket contact this sub-step
            paddle_contact = None
            table_contact = None
            for c in range(d.ncon):
                con = d.contact[c]
                if {con.geom1, con.geom2} == {self._ball_gid, self._racket_gid}:
                    paddle_contact = con
                    result.ball_racket_contact = True
                elif {con.geom1, con.geom2} == {self._ball_gid, self._table_gid}:
                    table_contact = con

            if table_contact is not None and not self._table_contact_active:
                result.table_contacts.append(
                    (
                        float(p_integrated[0] - self.offset[0]),
                        float(p_integrated[1] - self.offset[1]),
                    )
                )
                self._apply_calibrated_table_bounce(ball_vel_before)
                self._table_contact_active = True
            elif table_contact is None:
                self._table_contact_active = False

            if (
                paddle_contact is not None
                and not self._paddle_contact_latched
                and self._paddle_contact_model is not None
            ):
                outgoing = self._paddle_contact_model(
                    ball_vel_before,
                    racket_vel_before,
                    racket_normal_before,
                    self._paddle_physics,
                )
                d.qvel[self._ball_vadr:self._ball_vadr + 3] = np.asarray(
                    outgoing, dtype=np.float64
                )
                # Remove the solver's tiny first-substep penetration so its soft
                # constraint cannot add a second, model-dependent impulse.
                ball_pos = d.qpos[self._ball_qadr:self._ball_qadr + 3].copy()
                contact_pos = np.asarray(paddle_contact.pos, dtype=np.float64).copy()
                separation = ball_pos - contact_pos
                separation_norm = float(np.linalg.norm(separation))
                if separation_norm < 1.0e-9:
                    separation = racket_normal_before
                    separation_norm = float(np.linalg.norm(separation))
                separation /= separation_norm
                d.qpos[self._ball_qadr:self._ball_qadr + 3] = (
                    contact_pos + separation * (self.ball_radius + 3.0e-3)
                )
                self.last_paddle_contact = {
                    "incoming_ball_velocity": ball_vel_before.copy(),
                    "racket_velocity": racket_vel_before.copy(),
                    "racket_normal": racket_normal_before.copy(),
                    "outgoing_ball_velocity": np.asarray(outgoing, dtype=np.float64).copy(),
                    "resolved_ball_position": d.qpos[
                        self._ball_qadr:self._ball_qadr + 3
                    ].copy(),
                }
                self._paddle_contact_latched = True
                mj.mj_forward(m, d)

            # net-plane (x = net_x) crossing, evaluated in the table frame
            xb = p_before[0] - self.offset[0]
            xa = p_integrated[0] - self.offset[0]
            if (xb < self.net_x_table <= xa) or (xa < self.net_x_table <= xb):
                dx = xa - xb
                frac = (self.net_x_table - xb) / dx if abs(dx) > 1e-12 else 0.5
                z_cross = (
                    p_before[2]
                    + frac * (p_integrated[2] - p_before[2])
                ) - self.offset[2]
                result.net_crossings.append((float(z_cross), 1.0 if xa > xb else -1.0))

            # table-surface plane (z_table = ball_radius) crossing
            zb = p_before[2] - self.offset[2]
            za = p_integrated[2] - self.offset[2]
            if (zb > surface_table >= za) or (za > surface_table >= zb):
                dz = za - zb
                frac = (surface_table - zb) / dz if abs(dz) > 1e-12 else 0.5
                x_cross = (
                    p_before[0]
                    + frac * (p_integrated[0] - p_before[0])
                ) - self.offset[0]
                y_cross = (
                    p_before[1]
                    + frac * (p_integrated[1] - p_before[1])
                ) - self.offset[1]
                result.surface_crossings.append((float(x_cross), float(y_cross), -1.0 if za < zb else 1.0))

        # clear the applied drag so it never leaks onto other bodies/steps
        d.xfrc_applied[self._ball_bid, :3] = 0.0

        if self._viewer is not None:
            # MuJoCo's passive viewer does not rate-limit simulation. Keep visual
            # playback near the policy's 50 Hz control rate so swings and serves
            # are observable instead of finishing as fast as the CPU can step.
            remaining = self.control_dt - (time.perf_counter() - wall_step_started)
            if remaining > 0.0:
                time.sleep(remaining)
            self._viewer.sync()
        self._record_video_frame()
        return result

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._video_renderer is not None:
            self._video_renderer.close()
            self._video_renderer = None
        if self._video_process is not None:
            if self._video_process.stdin is not None:
                self._video_process.stdin.close()
            return_code = self._video_process.wait()
            self._video_process = None
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")


def _integer_substeps(control_dt: float, physics_dt: float) -> int:
    """Return exact MuJoCo decimation instead of silently rounding the ratio."""

    control_dt = float(control_dt)
    physics_dt = float(physics_dt)
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError(f"control_dt must be finite and > 0, got {control_dt}")
    if not np.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError(f"MuJoCo timestep must be finite and > 0, got {physics_dt}")
    ratio = control_dt / physics_dt
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "control_dt must be an integer multiple of the MuJoCo timestep; "
            f"got control_dt={control_dt}, timestep={physics_dt}, ratio={ratio}"
        )
    return substeps


def _validate_velocity_limits(values, expected_size: int) -> np.ndarray:
    limits = np.asarray(values, dtype=np.float64).reshape(-1)
    if limits.shape != (expected_size,):
        raise ValueError(
            f"joint velocity limits must have shape ({expected_size},), got {limits.shape}"
        )
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("joint velocity limits must all be finite and > 0")
    return limits.copy()


def _model_velocity_limits(mujoco, model, expected_size: int) -> np.ndarray:
    """Load canonical Isaac ``velocity_limit_sim`` values embedded in the MJCF."""

    name = "isaac_joint_velocity_limits"
    numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id < 0:
        raise ValueError(
            f"MJCF is missing custom numeric '{name}'. Use the aligned shipped model "
            "or pass joint_velocity_limits explicitly."
        )
    start = int(model.numeric_adr[numeric_id])
    size = int(model.numeric_size[numeric_id])
    return _validate_velocity_limits(
        model.numeric_data[start:start + size], expected_size
    )
