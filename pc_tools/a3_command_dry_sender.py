#!/usr/bin/env python3

"""Send zero-gain dry-run commands and wait for MDU validation ACKs."""

from __future__ import annotations

import argparse
import os
import time

import rclpy
from joint_msgs.msg import JointCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt32

from a3_command_prepare import build_zero_gain_hold, validate_prepared_command


QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=64,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class DryCommandSender(Node):
    def __init__(
        self, hz: float, count: int, warmup_sec: float, topic_prefix: str
    ) -> None:
        super().__init__("a3_pc_command_dry_sender")
        self._topic_prefix = topic_prefix.rstrip("/")
        self._count = count
        self._sequence = 0
        self._acks = set()
        self._latest_state: JointState | None = None
        self._latest_state_received = 0.0
        self._start_after = time.monotonic() + warmup_sec
        self._last_send_time = 0.0
        self._session = "a3_remote_dry/%d-%d" % (os.getpid(), time.time_ns())
        self.done = False

        self._publisher = self.create_publisher(
            JointCommand, self._topic_prefix + "/joint_command_dry_run", QOS
        )
        self.create_subscription(
            JointState, self._topic_prefix + "/joint_states", self._on_state, QOS
        )
        self.create_subscription(
            UInt32, self._topic_prefix + "/joint_command_ack", self._on_ack, QOS
        )
        self.create_timer(1.0 / hz, self._send)
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "dry sender ready: session=%s prefix=%s warmup=%.1fs count=%d; "
            "all gains=0"
            % (self._session, self._topic_prefix, warmup_sec, count)
        )

    def _on_state(self, message: JointState) -> None:
        self._latest_state = message
        self._latest_state_received = time.monotonic()

    def _on_ack(self, message: UInt32) -> None:
        if int(message.data) < self._count:
            self._acks.add(int(message.data))

    def _send(self) -> None:
        now = time.monotonic()
        if now < self._start_after or self._sequence >= self._count:
            return
        if self._latest_state is None or now - self._latest_state_received > 0.2:
            return

        command = build_zero_gain_hold(self._latest_state, self._sequence)
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self._session
        validate_prepared_command(command)
        self._publisher.publish(command)
        self._last_send_time = now
        self._sequence += 1

    def _report(self) -> None:
        self.get_logger().info(
            "dry command progress: sent=%d acked=%d pending=%d gains=0"
            % (self._sequence, len(self._acks), self._sequence - len(self._acks))
        )
        if self._sequence >= self._count and (
            len(self._acks) >= self._count
            or time.monotonic() - self._last_send_time >= 2.0
        ):
            self.get_logger().info(
                "FINAL DRY COMMAND: sent=%d acked=%d lost_or_rejected=%d "
                "all_gains=0"
                % (self._count, len(self._acks), self._count - len(self._acks))
            )
            self.done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send zero-gain commands to the MDU dry-run validator"
    )
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--topic-prefix", default="/a3_remote")
    args = parser.parse_args()
    if args.hz <= 0.0 or args.count <= 0 or args.warmup_sec < 0.0:
        parser.error("hz/count must be positive and warmup-sec non-negative")
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS topic prefix")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = DryCommandSender(
        args.hz, args.count, args.warmup_sec, args.topic_prefix
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
