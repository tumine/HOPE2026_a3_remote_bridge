#!/usr/bin/env python3

"""Pure-numpy PPMocap to HOPE canonical-table frame conversion.

The PPMocap driver publishes ball, table and robot poses in one synchronized
``MocapFrame``.  HOPE's planner and policy use a different, explicit frame:

* origin: tabletop near-left corner;
* +X: from the robot toward the opponent;
* +Y: the robot's left;
* +Z: upward, with the tabletop at Z=0.

This module has no ROS dependency so calibration and pose composition can be
unit-tested without a running robot or DDS graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _vec(value: Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def quaternion_xyzw_to_matrix(value: Iterable[float]) -> np.ndarray:
    """Return a 3x3 active rotation matrix from a ROS XYZW quaternion."""

    x, y, z, w = _vec(value, 4, "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Return a normalized Hamilton WXYZ quaternion from a rotation matrix."""

    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation matrix must be finite 3x3")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
             (r[1, 0] - r[0, 1]) / s)
        )
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
            result = np.asarray(
                ((r[2, 1] - r[1, 2]) / s, 0.25 * s,
                 (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
            )
        elif index == 1:
            s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
            result = np.asarray(
                ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                 0.25 * s, (r[1, 2] + r[2, 1]) / s)
            )
        else:
            s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0
            result = np.asarray(
                ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                 (r[1, 2] + r[2, 1]) / s, 0.25 * s)
            )
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-9:
        raise ValueError("rotation matrix produced a zero quaternion")
    result /= norm
    return result if result[0] >= 0.0 else -result


def rpy_matrix(value: Iterable[float]) -> np.ndarray:
    """ROS fixed-axis roll/pitch/yaw rotation (Rz @ Ry @ Rx)."""

    roll, pitch, yaw = _vec(value, 3, "RPY")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MarkerToPelvis:
    """Fixed pose of the pelvis origin expressed in the BotA3 marker frame."""

    translation_m: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_values(
        cls, translation_m: Iterable[float], rpy_rad: Iterable[float]
    ) -> "MarkerToPelvis":
        return cls(_vec(translation_m, 3, "marker_to_pelvis translation"), rpy_matrix(rpy_rad))


@dataclass(frozen=True)
class CanonicalTableFrame:
    """Locked transform from PPMocap world to the HOPE table-surface frame."""

    origin_world: np.ndarray
    rotation_world_from_table: np.ndarray

    def point_from_world(self, point_world: Iterable[float]) -> np.ndarray:
        point = _vec(point_world, 3, "world point")
        return self.rotation_world_from_table.T @ (point - self.origin_world)

    def vector_from_world(self, vector_world: Iterable[float]) -> np.ndarray:
        return self.rotation_world_from_table.T @ _vec(vector_world, 3, "world vector")

    def pelvis_from_marker(
        self,
        marker_position_world: Iterable[float],
        marker_quaternion_xyzw: Iterable[float],
        marker_to_pelvis: MarkerToPelvis,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = _vec(marker_position_world, 3, "marker position")
        rotation_world_from_marker = quaternion_xyzw_to_matrix(marker_quaternion_xyzw)
        pelvis_world = position + rotation_world_from_marker @ marker_to_pelvis.translation_m
        rotation_world_from_pelvis = rotation_world_from_marker @ marker_to_pelvis.rotation
        pelvis_table = self.point_from_world(pelvis_world)
        rotation_table_from_pelvis = (
            self.rotation_world_from_table.T @ rotation_world_from_pelvis
        )
        return pelvis_table, matrix_to_quaternion_wxyz(rotation_table_from_pelvis)


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    total = np.sum(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(total)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class CanonicalTableCalibrator:
    """Lock a stable table pose and orient its long axis using the robot side."""

    def __init__(
        self,
        sample_count: int,
        table_length_m: float,
        table_width_m: float,
        surface_down_offset_m: float,
        max_translation_jitter_m: float,
        max_rotation_jitter_deg: float,
    ) -> None:
        if sample_count < 2:
            raise ValueError("mocap calibration sample_count must be >= 2")
        if min(table_length_m, table_width_m) <= 0.0:
            raise ValueError("table dimensions must be positive")
        self.sample_count = int(sample_count)
        self.table_length_m = float(table_length_m)
        self.table_width_m = float(table_width_m)
        self.surface_down_offset_m = float(surface_down_offset_m)
        self.max_translation_jitter_m = float(max_translation_jitter_m)
        self.max_rotation_jitter_deg = float(max_rotation_jitter_deg)
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []
        self.frame: CanonicalTableFrame | None = None

    def reset(self) -> None:
        self._samples.clear()
        self.frame = None

    @property
    def samples(self) -> int:
        return len(self._samples)

    def add(
        self,
        table_position_world: Iterable[float],
        table_quaternion_xyzw: Iterable[float],
        robot_position_world: Iterable[float],
        long_axis_is_x: bool,
    ) -> CanonicalTableFrame | None:
        if self.frame is not None:
            return self.frame
        position = _vec(table_position_world, 3, "table position")
        rotation = quaternion_xyzw_to_matrix(table_quaternion_xyzw)
        robot = _vec(robot_position_world, 3, "robot position")
        self._samples.append((position, rotation, robot, bool(long_axis_is_x)))
        if len(self._samples) > self.sample_count:
            self._samples.pop(0)
        if len(self._samples) < self.sample_count:
            return None

        center = np.median(np.stack([sample[0] for sample in self._samples]), axis=0)
        rotation = _average_rotation([sample[1] for sample in self._samples])
        robot = np.median(np.stack([sample[2] for sample in self._samples]), axis=0)
        translation_jitter = max(
            float(np.linalg.norm(sample[0] - center)) for sample in self._samples
        )
        rotation_jitter = max(
            _rotation_error_deg(rotation, sample[1]) for sample in self._samples
        )
        if (
            translation_jitter > self.max_translation_jitter_m
            or rotation_jitter > self.max_rotation_jitter_deg
        ):
            return None

        long_axis_is_x = sum(sample[3] for sample in self._samples) > self.sample_count / 2
        z_axis = rotation[:, 2].copy()
        if z_axis[2] < 0.0:
            z_axis *= -1.0
        z_axis /= np.linalg.norm(z_axis)
        x_axis = rotation[:, 0 if long_axis_is_x else 1].copy()
        x_axis -= z_axis * float(np.dot(x_axis, z_axis))
        x_axis /= np.linalg.norm(x_axis)
        # The robot is behind the near edge, hence on the negative canonical X side.
        if float(np.dot(robot - center, x_axis)) > 0.0:
            x_axis *= -1.0
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        world_from_table = np.column_stack((x_axis, y_axis, z_axis))
        # canonical [0,0,0] is the near-left tabletop corner.  Table centre is
        # [L/2, -W/2, +surface_down_offset] in this convention.
        origin = (
            center
            - 0.5 * self.table_length_m * x_axis
            + 0.5 * self.table_width_m * y_axis
            - self.surface_down_offset_m * z_axis
        )
        self.frame = CanonicalTableFrame(origin, world_from_table)
        return self.frame
