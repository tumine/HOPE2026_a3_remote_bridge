# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Simulation bridges: how the reference runner reads state and writes targets.

Two clean mechanisms are provided behind one ``SimBridge`` interface:

* ``MujocoDirectBridge`` -- the default, fully runnable path. It loads the SAME
  ``a3_pingpong`` MJCF that the shipped AimRT MuJoCo sim wraps and steps MuJoCo
  in-process. Joint-position targets are realized with an explicit PD law
  (the model uses torque ``motor`` actuators, exactly like the AimRT backend's
  implicit-PD path), so no AimRT/iceoryx/ROS2 stack is required to see the policy
  drive the robot. Requires ``pip install mujoco``.

* ``AimrtSimBridge`` -- an explicit, documented integration seam for driving the
  live AimRT MuJoCo sim process over its ``/body_drive/*`` channels. Wiring it
  needs the AimRT Python bindings plus the ``joint_msgs`` typesupport (a heavy
  vendor build), so it is intentionally left un-wired rather than faked.

The PD gains used by ``MujocoDirectBridge`` are the named nominal Isaac simulation
gains supplied by the runtime config. The shipped MJCF also carries Isaac's
per-joint ``velocity_limit_sim`` values as a named custom numeric. MuJoCo has no
native joint-speed constraint, so the bridge projects only the simulated joint
velocities back onto those bounds after every physics step; actuator torque remains
subject to the ordinary motor/control ranges. These are simulation values, not
vendor real-robot deploy gains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .joint_order import JOINT_NAMES, NUM_JOINTS
from .observation import RobotState


class SimBridge(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def read_state(self) -> RobotState: ...

    @abstractmethod
    def write_targets(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None: ...

    @abstractmethod
    def step(self) -> None:
        """Advance one 50 Hz control tick (several physics substeps)."""

    def sync_viewer(self) -> None:  # optional
        ...

    def is_viewer_running(self) -> bool:
        return True

    def close(self) -> None:  # optional
        ...


class MujocoDirectBridge(SimBridge):
    def __init__(
        self,
        model_xml_path: str | Path,
        control_dt: float = 0.02,
        launch_viewer: bool = False,
        reset_joint_pos: np.ndarray | None = None,
        joint_velocity_limits: np.ndarray | None = None,
    ) -> None:
        import mujoco  # lazy import

        self._mj = mujoco
        model_xml_path = str(model_xml_path)
        if not Path(model_xml_path).is_file():
            raise FileNotFoundError(f"MJCF model not found: {model_xml_path}")
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)

        self._substeps = _integer_substeps(self.control_dt, self.model.opt.timestep)

        # Free (pelvis) joint address block.
        free_jid = self._joint_id("pelvis_free_joint", required=False)
        if free_jid < 0:
            # Fall back to the first free joint in the model.
            free_jid = int(np.argmax(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self._base_qadr = int(self.model.jnt_qposadr[free_jid])
        self._base_vadr = int(self.model.jnt_dofadr[free_jid])

        # Per-controlled-joint qpos/qvel addresses and driving actuator indices,
        # resolved by joint identity (robust to actuator naming).
        self._q_adr = np.zeros(NUM_JOINTS, dtype=int)
        self._v_adr = np.zeros(NUM_JOINTS, dtype=int)
        self._act_idx = np.full(NUM_JOINTS, -1, dtype=int)
        trn_joint = self.model.actuator_trnid[:, 0]
        for i, name in enumerate(JOINT_NAMES):
            jid = self._joint_id(name, required=True)
            self._q_adr[i] = int(self.model.jnt_qposadr[jid])
            self._v_adr[i] = int(self.model.jnt_dofadr[jid])
            matches = np.where(trn_joint == jid)[0]
            if matches.size == 0:
                raise ValueError(f"no actuator drives joint '{name}'")
            self._act_idx[i] = int(matches[0])

        self._ctrl_lo = self.model.actuator_ctrlrange[self._act_idx, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[self._act_idx, 1].copy()
        self._ctrl_limited = self.model.actuator_ctrllimited[self._act_idx].astype(bool)
        self._joint_velocity_limits = (
            _model_velocity_limits(self._mj, self.model, NUM_JOINTS)
            if joint_velocity_limits is None
            else _validate_velocity_limits(joint_velocity_limits, NUM_JOINTS)
        )
        self._reset_joint_pos = (
            None
            if reset_joint_pos is None
            else np.asarray(reset_joint_pos, dtype=np.float64).reshape(NUM_JOINTS).copy()
        )

        # Pelvis gyro sensor address (base angular velocity, body frame).
        self._gyro_adr = self._sensor_adr("pelvis_imu_gyro", dim=3)

        # Latched targets.
        self._q_des = np.zeros(NUM_JOINTS)
        self._kp = np.zeros(NUM_JOINTS)
        self._kd = np.zeros(NUM_JOINTS)

        self._viewer = None
        if launch_viewer:
            from mujoco import viewer as mj_viewer

            self._viewer = mj_viewer.launch_passive(self.model, self.data)

        self.reset()

    # -- model lookups ------------------------------------------------------
    def _joint_id(self, name: str, required: bool) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 and required:
            raise ValueError(f"joint '{name}' not found in the MJCF")
        return jid

    def _sensor_adr(self, name: str, dim: int) -> int:
        sid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return -1
        return int(self.model.sensor_adr[sid])

    # -- SimBridge ----------------------------------------------------------
    def reset(self) -> None:
        if self.model.nkey > 0:
            # Keyframe 0 is the grounded shared ActionAdapter default pose.
            self._mj.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            self._mj.mj_resetData(self.model, self.data)
        if self._reset_joint_pos is not None:
            # Runtime config remains authoritative even when --model-xml points at
            # an older/custom keyframe.
            self.data.qpos[self._q_adr] = self._reset_joint_pos
        self._mj.mj_forward(self.model, self.data)

    def read_state(self) -> RobotState:
        d = self.data
        base_pos = d.qpos[self._base_qadr:self._base_qadr + 3].copy()
        base_quat = d.qpos[self._base_qadr + 3:self._base_qadr + 7].copy()  # (w,x,y,z)
        if self._gyro_adr >= 0:
            base_ang_vel = d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        else:
            base_ang_vel = d.qvel[self._base_vadr + 3:self._base_vadr + 6].copy()
        q = d.qpos[self._q_adr].copy()
        qd = d.qvel[self._v_adr].copy()
        return RobotState(
            base_pos_w=base_pos,
            base_quat_w=base_quat,
            base_ang_vel_b=base_ang_vel,
            q=q,
            qd=qd,
        )

    def write_targets(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._q_des = np.asarray(q_des, dtype=np.float64).reshape(NUM_JOINTS)
        self._kp = np.asarray(kp, dtype=np.float64).reshape(NUM_JOINTS)
        self._kd = np.asarray(kd, dtype=np.float64).reshape(NUM_JOINTS)

    def _apply_pd(self) -> None:
        d = self.data
        q = d.qpos[self._q_adr]
        qd = d.qvel[self._v_adr]
        tau = self._kp * (self._q_des - q) - self._kd * qd
        tau = np.where(
            self._ctrl_limited, np.clip(tau, self._ctrl_lo, self._ctrl_hi), tau
        )
        d.ctrl[self._act_idx] = tau

    def _enforce_joint_velocity_limits(self) -> bool:
        """Project qd onto Isaac's simulation bounds without altering motor control."""

        qd = self.data.qvel[self._v_adr]
        clipped = np.clip(qd, -self._joint_velocity_limits, self._joint_velocity_limits)
        if np.array_equal(qd, clipped):
            return False
        self.data.qvel[self._v_adr] = clipped
        return True

    def step(self) -> None:
        for _ in range(self._substeps):
            self._apply_pd()             # PD recomputed each physics substep
            self._mj.mj_step(self.model, self.data)
            if self._enforce_joint_velocity_limits():
                # qvel was projected after integration; refresh body velocities,
                # gyro sensors and other derived state before the next substep/read.
                self._mj.mj_forward(self.model, self.data)

    def sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def is_viewer_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class AimrtSimBridge(SimBridge):
    """Integration seam for the live AimRT MuJoCo sim process (NOT wired here).

    To drive the running AimRT ``MujocoSimModule`` over the ``/body_drive/*``
    contract (see ``A3_MuJoCo_Sim`` README), a real implementation would:

      * publish ``joint_msgs/msg/JointCommand`` (per-joint position + stiffness +
        damping + effort) to ``/body_drive/{waist,neck,arm,leg}_joint_command``,
        scattering the 31 targets into the four joint groups by name;
      * subscribe ``joint_msgs/msg/JointState`` from
        ``/body_drive/{waist,neck,arm,leg}_joint_state`` for ``q`` / ``qd``;
      * subscribe ``sensor_msgs/msg/Imu`` from ``/body_drive/pelvis_imu/data`` for
        base orientation and angular velocity.

    That requires the AimRT Python runtime, the iceoryx transport, and the
    ``joint_msgs`` rosidl/protobuf typesupport built for the target -- a vendor
    toolchain outside this reference. The seam is left explicit on purpose rather
    than returning fabricated state.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "AimrtSimBridge is a documented integration seam. Use MujocoDirectBridge "
            "(the default, in-process MuJoCo path) to run this reference, or wire the "
            "/body_drive/* AimRT channels with your vendor AimRT + joint_msgs build. "
            "See the class docstring and A3_MuJoCo_Sim/aimrt_mujoco_sim/README.md."
        )

    def reset(self) -> None:  # pragma: no cover - seam
        raise NotImplementedError

    def read_state(self) -> RobotState:  # pragma: no cover - seam
        raise NotImplementedError

    def write_targets(self, q_des, kp, kd) -> None:  # pragma: no cover - seam
        raise NotImplementedError

    def step(self) -> None:  # pragma: no cover - seam
        raise NotImplementedError


def _integer_substeps(control_dt: float, physics_dt: float) -> int:
    """Return an exact integer decimation, rejecting silent timing drift."""

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
