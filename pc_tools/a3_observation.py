#!/usr/bin/env python3

"""A3 policy observation construction matching a3_deploy_onnx_ref.

This module is deliberately independent of ROS 2 and inference runtimes.  It
turns one synchronized 31-DOF joint/pelvis-IMU frame into the 930-float
proprioception history used by the reference policy.  A future policy runner
can prepend its own 640-float command/tokenizer input to form obs_dict[1570].
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional

import numpy as np


HISTORY_LENGTH = 10
POLICY_DOF = 29
PROPRIO_PER_STEP = 93
PROPRIO_SIZE = 930
TOKENIZER_SIZE = 640
OBS_DICT_SIZE = 1570

# MDU bridge output order.  This is the A3 31-DOF MuJoCo/SDK layout.
A3_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# 31-DOF SDK -> 29-DOF MuJoCo policy view: skip head/neck slots 3 and 4.
POLICY_FROM_SDK = np.asarray(
    [0, 1, 2, *range(5, 12), *range(12, 19), *range(19, 25), *range(25, 31)],
    dtype=np.int64,
)

# Gather map copied from a3_policy_parameters.hpp:
#   isaaclab[i] = mujoco[A3_MUJOCO_TO_ISAACLAB[i]]
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        17, 23, 0, 18, 24, 1, 19, 25, 2, 20,
        26, 3, 10, 21, 27, 4, 11, 22, 28, 5,
        12, 6, 13, 7, 14, 8, 15, 9, 16,
    ],
    dtype=np.int64,
)

ISAACLAB_TO_MUJOCO = np.asarray(
    [
        2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
        12, 16, 20, 22, 24, 26, 28,
        0, 3, 6, 9, 13, 17,
        1, 4, 7, 10, 14, 18,
    ],
    dtype=np.int64,
)

# 29-DOF MuJoCo policy-view default pose from a3_policy_parameters.hpp.
DEFAULT_ANGLES_MUJOCO = np.asarray(
    [
        0.0, 0.0, 0.0,
        0.3, 0.12, 0.0, 0.8, 0.0, 0.0, 0.0,
        0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0,
        -0.1311, 0.0056, -0.0348, 0.2468, -0.1204, -0.0078,
        -0.1311, -0.0056, 0.0348, 0.2468, -0.1204, 0.0078,
    ],
    dtype=np.float64,
)


def reorder_joint_values(
    names: Iterable[str], values: Iterable[float]
) -> np.ndarray:
    """Return values in the canonical 31-DOF SDK order."""

    names_tuple = tuple(names)
    values_array = np.asarray(tuple(values), dtype=np.float64)
    if len(names_tuple) != len(A3_JOINT_NAMES) or values_array.shape != (31,):
        raise ValueError("joint message must contain exactly 31 names and values")
    if len(set(names_tuple)) != 31:
        raise ValueError("joint names must be unique")
    index = {name: i for i, name in enumerate(names_tuple)}
    missing = [name for name in A3_JOINT_NAMES if name not in index]
    if missing:
        raise ValueError("missing A3 joints: " + ", ".join(missing))
    ordered = values_array[[index[name] for name in A3_JOINT_NAMES]]
    if not np.isfinite(ordered).all():
        raise ValueError("joint values contain NaN or infinity")
    return ordered


def projected_gravity_xyzw(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Match GetGravityOrientation_d() from the reference deployment."""

    q = np.asarray(tuple(quaternion_xyzw), dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("IMU quaternion must contain four finite values")
    qx, qy, qz, qw = q
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float64,
    )


class A3ObservationBuilder:
    """Construct the reference policy's term-first 10-frame observation."""

    def __init__(self) -> None:
        self._history: Deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self._last_action_mujoco = np.zeros(POLICY_DOF, dtype=np.float64)

    @property
    def ticks_buffered(self) -> int:
        return min(len(self._history), HISTORY_LENGTH)

    def reset(self) -> None:
        self._history.clear()
        self._last_action_mujoco.fill(0.0)

    def remember_action(self, raw_action_isaaclab: Iterable[float]) -> None:
        action = np.asarray(tuple(raw_action_isaaclab), dtype=np.float32)
        if action.shape != (POLICY_DOF,) or not np.isfinite(action).all():
            raise ValueError("raw policy action must contain 29 finite values")
        self._last_action_mujoco = action[ISAACLAB_TO_MUJOCO].astype(
            np.float64, copy=True
        )

    def push(
        self,
        q_sdk: Iterable[float],
        dq_sdk: Iterable[float],
        pelvis_quaternion_xyzw: Iterable[float],
        pelvis_angular_velocity: Iterable[float],
    ) -> None:
        q31 = np.asarray(tuple(q_sdk), dtype=np.float64)
        dq31 = np.asarray(tuple(dq_sdk), dtype=np.float64)
        gyro = np.asarray(tuple(pelvis_angular_velocity), dtype=np.float64)
        if q31.shape != (31,) or dq31.shape != (31,):
            raise ValueError("q_sdk and dq_sdk must each contain 31 values")
        if gyro.shape != (3,):
            raise ValueError("pelvis angular velocity must contain three values")
        if not (np.isfinite(q31).all() and np.isfinite(dq31).all()):
            raise ValueError("joint state contains NaN or infinity")
        if not np.isfinite(gyro).all():
            raise ValueError("pelvis angular velocity contains NaN or infinity")

        q_mujoco = q31[POLICY_FROM_SDK]
        dq_mujoco = dq31[POLICY_FROM_SDK]
        gravity = projected_gravity_xyzw(pelvis_quaternion_xyzw)

        frame = np.zeros(PROPRIO_PER_STEP, dtype=np.float32)
        frame[0:3] = gravity.astype(np.float32)
        frame[3:6] = gyro.astype(np.float32)

        # The reference implementation casts both operands to float32 before
        # subtracting the default pose in IsaacLab order.
        q_isaac = q_mujoco[MUJOCO_TO_ISAACLAB].astype(np.float32)
        default_isaac = DEFAULT_ANGLES_MUJOCO[
            MUJOCO_TO_ISAACLAB
        ].astype(np.float32)
        frame[6:35] = q_isaac - default_isaac
        frame[35:64] = dq_mujoco[MUJOCO_TO_ISAACLAB].astype(np.float32)
        frame[64:93] = self._last_action_mujoco[
            MUJOCO_TO_ISAACLAB
        ].astype(np.float32)

        if not self._history:
            for _ in range(HISTORY_LENGTH):
                self._history.append(frame.copy())
        else:
            self._history.append(frame)

    def build_proprio_history(self) -> np.ndarray:
        if not self._history:
            return np.zeros(PROPRIO_SIZE, dtype=np.float32)
        history = np.stack(tuple(self._history), axis=0)
        if history.shape != (HISTORY_LENGTH, PROPRIO_PER_STEP):
            raise RuntimeError(f"unexpected history shape: {history.shape}")
        # Reference layout is TERM-FIRST, each term oldest -> newest.
        return np.concatenate(
            (
                history[:, 3:6].reshape(-1),
                history[:, 6:35].reshape(-1),
                history[:, 35:64].reshape(-1),
                history[:, 64:93].reshape(-1),
                history[:, 0:3].reshape(-1),
            )
        ).astype(np.float32, copy=False)

    def build_obs_dict(
        self, tokenizer_640: Optional[Iterable[float]] = None
    ) -> np.ndarray:
        if tokenizer_640 is None:
            tokenizer = np.zeros(TOKENIZER_SIZE, dtype=np.float32)
        else:
            tokenizer = np.asarray(tuple(tokenizer_640), dtype=np.float32)
            if tokenizer.shape != (TOKENIZER_SIZE,):
                raise ValueError("tokenizer input must contain 640 values")
            if not np.isfinite(tokenizer).all():
                raise ValueError("tokenizer input contains NaN or infinity")
        return np.concatenate((tokenizer, self.build_proprio_history())).astype(
            np.float32, copy=False
        )

