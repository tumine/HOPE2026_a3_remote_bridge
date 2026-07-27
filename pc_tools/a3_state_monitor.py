#!/usr/bin/env python3

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState


class StateMonitor(Node):
    def __init__(self, topic_prefix: str, duration_sec: float) -> None:
        super().__init__("a3_pc_state_monitor")
        self._topic_prefix = topic_prefix.rstrip("/")
        self._duration_sec = duration_sec
        self._counts = {"joints": 0, "pelvis": 0, "torso": 0}
        self._totals = {"joints": 0, "pelvis": 0, "torso": 0}
        self._invalid = 0
        self._last_report = time.monotonic()
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
            Imu, self._topic_prefix + "/pelvis_imu", self._on_pelvis, qos
        )
        self.create_subscription(
            Imu, self._topic_prefix + "/torso_imu", self._on_torso, qos
        )
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "state monitor ready: prefix=%s reliability=reliable duration=%.1fs"
            % (self._topic_prefix, self._duration_sec)
        )

    def _on_joints(self, msg: JointState) -> None:
        now = time.monotonic()
        if self._measurement_start is None:
            self._measurement_start = now
        self._counts["joints"] += 1
        self._totals["joints"] += 1
        values = list(msg.position) + list(msg.velocity) + list(msg.effort)
        if (
            len(msg.name) != 31
            or len(msg.position) != 31
            or len(msg.velocity) != 31
            or len(msg.effort) != 31
            or not all(math.isfinite(value) for value in values)
        ):
            self._invalid += 1
        if now - self._measurement_start >= self._duration_sec:
            self._measurement_end = now
            self.done = True

    def _on_pelvis(self, _: Imu) -> None:
        self._counts["pelvis"] += 1
        self._totals["pelvis"] += 1

    def _on_torso(self, _: Imu) -> None:
        self._counts["torso"] += 1
        self._totals["torso"] += 1

    def _report(self) -> None:
        now = time.monotonic()
        elapsed = max(now - self._last_report, 1e-9)
        self.get_logger().info(
            "rates: joints=%.1fHz pelvis=%.1fHz torso=%.1fHz invalid=%d"
            % (
                self._counts["joints"] / elapsed,
                self._counts["pelvis"] / elapsed,
                self._counts["torso"] / elapsed,
                self._invalid,
            )
        )
        self._counts = {"joints": 0, "pelvis": 0, "torso": 0}
        self._last_report = now

    def report_final(self) -> None:
        if self._measurement_start is None:
            self.get_logger().error("FINAL STATE: no joint state received")
            return
        end = self._measurement_end or time.monotonic()
        elapsed = max(end - self._measurement_start, 1e-9)
        self.get_logger().info(
            "FINAL STATE: duration=%.3fs joints=%d(%.3fHz) "
            "pelvis=%d(%.3fHz) torso=%d(%.3fHz) invalid=%d"
            % (
                elapsed,
                self._totals["joints"],
                self._totals["joints"] / elapsed,
                self._totals["pelvis"],
                self._totals["pelvis"] / elapsed,
                self._totals["torso"],
                self._totals["torso"] / elapsed,
                self._invalid,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor direct A3 state topics")
    parser.add_argument("--topic-prefix", default="/a3_remote")
    parser.add_argument("--duration-sec", type=float, default=10.0)
    args = parser.parse_args()
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS topic prefix")
    if args.duration_sec <= 0.0:
        parser.error("duration-sec must be positive")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = StateMonitor(args.topic_prefix, args.duration_sec)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.report_final()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
