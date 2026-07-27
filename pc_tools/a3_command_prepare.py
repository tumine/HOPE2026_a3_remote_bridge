#!/usr/bin/env python3

"""Prepare one full A3 JointCommand from live state without publishing it."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from joint_msgs.msg import Command, JointCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from a3_observation import A3_JOINT_NAMES, reorder_joint_values


QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=64,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def build_zero_gain_hold(state: JointState, sequence: int = 0) -> JointCommand:
    """Build a 31-DOF hold-position packet with no active control effort."""

    q = reorder_joint_values(state.name, state.position)
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must fit uint32")

    message = JointCommand()
    message.header.stamp = state.header.stamp
    message.header.frame_id = "a3_remote_dry_prepare"
    message.joints = []
    for index, name in enumerate(A3_JOINT_NAMES):
        row = Command()
        row.name = name
        row.sequence = sequence
        row.position = float(q[index])
        row.velocity = 0.0
        row.effort = 0.0
        row.stiffness = 0.0
        row.damping = 0.0
        message.joints.append(row)
    return message


def validate_prepared_command(message: JointCommand) -> None:
    if len(message.joints) != 31:
        raise ValueError("prepared command must contain 31 joints")
    if tuple(row.name for row in message.joints) != A3_JOINT_NAMES:
        raise ValueError("prepared command joint order is not canonical")
    sequences = {row.sequence for row in message.joints}
    if len(sequences) != 1:
        raise ValueError("all command rows must carry the same sequence")
    for row in message.joints:
        values = (
            row.position,
            row.velocity,
            row.effort,
            row.stiffness,
            row.damping,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite command value for {row.name}")
        if row.velocity != 0.0 or row.effort != 0.0:
            raise ValueError("dry command velocity and effort must be zero")
        if row.stiffness != 0.0 or row.damping != 0.0:
            raise ValueError("dry command gains must be zero")


class CommandPrepareNode(Node):
    def __init__(self, timeout_sec: float, topic_prefix: str) -> None:
        super().__init__("a3_pc_command_prepare")
        self._topic_prefix = topic_prefix.rstrip("/")
        self.done = False
        self.success = False
        self._deadline = time.monotonic() + timeout_sec
        self.create_subscription(
            JointState,
            self._topic_prefix + "/joint_states",
            self._on_state,
            QOS,
        )
        self.create_timer(0.1, self._check_timeout)
        self.get_logger().info(
            "waiting for %s/joint_states; command publishers=0"
            % self._topic_prefix
        )

    def _on_state(self, state: JointState) -> None:
        if self.done:
            return
        try:
            command = build_zero_gain_hold(state)
            validate_prepared_command(command)
        except ValueError as exc:
            self.get_logger().warning("state rejected: %s" % exc)
            return

        positions = [row.position for row in command.joints]
        self.get_logger().info(
            "PREPARED ONLY: joints=%d sequence=%d q_min=%.6f q_max=%.6f "
            "velocity=0 effort=0 kp=0 kd=0 publishers=0"
            % (
                len(command.joints),
                command.joints[0].sequence,
                min(positions),
                max(positions),
            )
        )
        self.get_logger().info(
            "joint order: first=%s neck=%s/%s last=%s"
            % (
                command.joints[0].name,
                command.joints[3].name,
                command.joints[4].name,
                command.joints[-1].name,
            )
        )
        self.success = True
        self.done = True

    def _check_timeout(self) -> None:
        if not self.done and time.monotonic() >= self._deadline:
            self.get_logger().error("timed out waiting for a valid joint state")
            self.done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, validate, and print one A3 command without publishing"
    )
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--topic-prefix", default="/a3_internal")
    args = parser.parse_args()
    if args.timeout_sec <= 0.0:
        parser.error("timeout-sec must be positive")
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS topic prefix")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CommandPrepareNode(args.timeout_sec, args.topic_prefix)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        success = node.success
        node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
