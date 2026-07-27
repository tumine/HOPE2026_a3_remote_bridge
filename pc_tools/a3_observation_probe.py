#!/usr/bin/env python3

"""Read remote A3 state and validate reference-policy observations.

No publisher is created by this program.  It cannot send robot commands.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import math
import time
from typing import Dict, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState

from a3_observation import (
    A3ObservationBuilder,
    OBS_DICT_SIZE,
    PROPRIO_SIZE,
    reorder_joint_values,
)


MAX_PENDING_FRAMES = 64
MAX_STATE_AGE_SEC = 0.100


def stamp_ns(msg: object) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class A3ObservationProbe(Node):
    def __init__(self, topic_prefix: str, duration_sec: float) -> None:
        super().__init__("a3_pc_observation_probe")
        self._topic_prefix = topic_prefix.rstrip("/")
        self._duration_sec = duration_sec
        self._builder = A3ObservationBuilder()
        self._joints: "OrderedDict[int, JointState]" = OrderedDict()
        self._pelvis: "OrderedDict[int, Imu]" = OrderedDict()
        self._latest_frame: Tuple[np.ndarray, np.ndarray, Imu] | None = None
        self._latest_frame_received = 0.0
        self._last_consumed_stamp = -1
        self._matched = 0
        self._policy_ticks = 0
        self._invalid = 0
        self._stale = 0
        self._measurement_start: float | None = None
        self._measurement_end: float | None = None
        self.done = False

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            JointState,
            self._topic_prefix + "/joint_states",
            self._on_joints,
            qos,
        )
        self.create_subscription(
            Imu,
            self._topic_prefix + "/pelvis_imu",
            self._on_pelvis,
            qos,
        )
        # Torso IMU is intentionally not subscribed here: the reference
        # observation uses only the primary/pelvis IMU.
        self.create_timer(0.02, self._policy_tick)
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "observation probe ready: prefix=%s reliability=reliable "
            "policy=50Hz duration=%.1fs command_publishers=0"
            % (self._topic_prefix, self._duration_sec)
        )

    @staticmethod
    def _trim(values: Dict[int, object]) -> None:
        while len(values) > MAX_PENDING_FRAMES:
            values.pop(next(iter(values)))

    def _on_joints(self, msg: JointState) -> None:
        self._joints[stamp_ns(msg)] = msg
        self._trim(self._joints)
        self._match(stamp_ns(msg))

    def _on_pelvis(self, msg: Imu) -> None:
        self._pelvis[stamp_ns(msg)] = msg
        self._trim(self._pelvis)
        self._match(stamp_ns(msg))

    def _match(self, stamp: int) -> None:
        joints = self._joints.get(stamp)
        pelvis = self._pelvis.get(stamp)
        if joints is None or pelvis is None:
            return
        self._joints.pop(stamp, None)
        self._pelvis.pop(stamp, None)
        try:
            q = reorder_joint_values(joints.name, joints.position)
            dq = reorder_joint_values(joints.name, joints.velocity)
            imu_values = (
                pelvis.orientation.x,
                pelvis.orientation.y,
                pelvis.orientation.z,
                pelvis.orientation.w,
                pelvis.angular_velocity.x,
                pelvis.angular_velocity.y,
                pelvis.angular_velocity.z,
            )
            if not all(math.isfinite(value) for value in imu_values):
                raise ValueError("pelvis IMU contains NaN or infinity")
        except ValueError as exc:
            self._invalid += 1
            self.get_logger().warning(str(exc))
            return
        self._latest_frame = (q, dq, pelvis)
        self._latest_frame_received = time.monotonic()
        self._last_consumed_stamp = stamp
        self._matched += 1
        if self._measurement_start is None:
            self._measurement_start = self._latest_frame_received
        elif self._latest_frame_received - self._measurement_start >= self._duration_sec:
            self._measurement_end = self._latest_frame_received
            self.done = True

    def _policy_tick(self) -> None:
        if self._latest_frame is None:
            return
        age = time.monotonic() - self._latest_frame_received
        if age > MAX_STATE_AGE_SEC:
            self._stale += 1
            return
        q, dq, pelvis = self._latest_frame
        try:
            self._builder.push(
                q,
                dq,
                (
                    pelvis.orientation.x,
                    pelvis.orientation.y,
                    pelvis.orientation.z,
                    pelvis.orientation.w,
                ),
                (
                    pelvis.angular_velocity.x,
                    pelvis.angular_velocity.y,
                    pelvis.angular_velocity.z,
                ),
            )
            proprio = self._builder.build_proprio_history()
            obs = self._builder.build_obs_dict()
            if (
                proprio.shape != (PROPRIO_SIZE,)
                or obs.shape != (OBS_DICT_SIZE,)
                or not np.isfinite(obs).all()
            ):
                raise ValueError("constructed observation is invalid")
            self._policy_ticks += 1
        except (RuntimeError, ValueError) as exc:
            self._invalid += 1
            self.get_logger().warning(str(exc))

    def _report(self) -> None:
        age_ms = (
            -1.0
            if self._latest_frame is None
            else 1000.0 * (time.monotonic() - self._latest_frame_received)
        )
        self.get_logger().info(
            "matched=%d policy_ticks=%d buffered=%d obs=%d age=%.1fms "
            "stale=%d invalid=%d command_publishers=0"
            % (
                self._matched,
                self._policy_ticks,
                self._builder.ticks_buffered,
                OBS_DICT_SIZE,
                age_ms,
                self._stale,
                self._invalid,
            )
        )

    def report_final(self) -> None:
        if self._measurement_start is None:
            self.get_logger().error("FINAL OBSERVATION: no synchronized state received")
            return
        end = self._measurement_end or time.monotonic()
        elapsed = max(end - self._measurement_start, 1e-9)
        self.get_logger().info(
            "FINAL OBSERVATION: duration=%.3fs matched=%d(%.3fHz) "
            "policy_ticks=%d(%.3fHz) buffered=%d proprio=%d obs=%d "
            "stale=%d invalid=%d command_publishers=0"
            % (
                elapsed,
                self._matched,
                self._matched / elapsed,
                self._policy_ticks,
                self._policy_ticks / elapsed,
                self._builder.ticks_buffered,
                PROPRIO_SIZE,
                OBS_DICT_SIZE,
                self._stale,
                self._invalid,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the reference A3 policy observation over ROS 2"
    )
    parser.add_argument("--topic-prefix", default="/a3_internal")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    args = parser.parse_args()
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS topic prefix")
    if args.duration_sec <= 0.0:
        parser.error("duration-sec must be positive")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = A3ObservationProbe(args.topic_prefix, args.duration_sec)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.report_final()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
