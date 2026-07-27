#!/usr/bin/env python3

"""Send explicitly armed A3 control commands to the guarded MDU gateway.

This program can cause physical robot motion.  It refuses to publish unless
the acknowledgement flag is present.  The default mode sends only zero-gain
commands; joint-sine is intended solely for a secured/suspended robot.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import rclpy
from joint_msgs.msg import Command, JointCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt32

from a3_observation import A3_JOINT_NAMES, reorder_joint_values


QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=64,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def build_command(
    state: JointState,
    sequence: int,
    session: str,
    selected_joint: str | None,
    selected_target: float | None,
    kp: float,
    kd: float,
) -> JointCommand:
    q = reorder_joint_values(state.name, state.position)
    message = JointCommand()
    message.header.stamp = state.header.stamp
    message.header.frame_id = session
    for index, name in enumerate(A3_JOINT_NAMES):
        row = Command()
        row.name = name
        row.sequence = sequence
        row.position = float(q[index])
        row.velocity = 0.0
        row.effort = 0.0
        row.stiffness = 0.0
        row.damping = 0.0
        if name == selected_joint:
            assert selected_target is not None
            row.position = float(selected_target)
            row.stiffness = kp
            row.damping = kd
        message.joints.append(row)
    return message


class ControlSender(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("a3_pc_control_sender")
        self._args = args
        self._prefix = args.topic_prefix.rstrip("/")
        self._session = "a3_remote_control/%d-%d" % (os.getpid(), time.time_ns())
        self._latest_state: JointState | None = None
        self._latest_state_at = 0.0
        self._initial_q: float | None = None
        self._measured_q: float | None = None
        self._measured_min: float | None = None
        self._measured_max: float | None = None
        self._measured_velocity_min: float | None = None
        self._measured_velocity_max: float | None = None
        self._measured_effort_min: float | None = None
        self._measured_effort_max: float | None = None
        self._sequence = 0
        self._acks: set[int] = set()
        self._start_at = time.monotonic() + args.warmup_sec
        self._last_send_at = 0.0
        self._finished_sending = False
        self.done = False

        self._publisher = self.create_publisher(
            JointCommand, self._prefix + "/joint_command_control", QOS
        )
        self.create_subscription(
            JointState, self._prefix + "/joint_states", self._on_state, QOS
        )
        self.create_subscription(
            UInt32,
            self._prefix + "/joint_command_control_ack",
            self._on_ack,
            QOS,
        )
        self.create_timer(1.0 / args.hz, self._send)
        self.create_timer(1.0, self._report)
        self.get_logger().warning(
            "ACTUATION sender armed: mode=%s session=%s prefix=%s hz=%.1f "
            "duration=%.2fs joint=%s offset=%.6frad kp=%.3f kd=%.3f"
            % (
                args.mode,
                self._session,
                self._prefix,
                args.hz,
                args.duration_sec,
                args.joint or "none",
                args.offset_rad,
                args.kp,
                args.kd,
            )
        )

    def _on_state(self, message: JointState) -> None:
        self._latest_state = message
        self._latest_state_at = time.monotonic()
        if self._args.joint and self._initial_q is None:
            q = reorder_joint_values(message.name, message.position)
            self._initial_q = float(q[A3_JOINT_NAMES.index(self._args.joint)])
        if self._args.joint:
            joint_index = A3_JOINT_NAMES.index(self._args.joint)
            q = reorder_joint_values(message.name, message.position)
            measured = float(q[joint_index])
            self._measured_q = measured
            if time.monotonic() >= self._start_at:
                self._measured_min = (
                    measured
                    if self._measured_min is None
                    else min(self._measured_min, measured)
                )
                self._measured_max = (
                    measured
                    if self._measured_max is None
                    else max(self._measured_max, measured)
                )
                if len(message.velocity) == len(message.name):
                    velocity = float(
                        reorder_joint_values(message.name, message.velocity)[joint_index]
                    )
                    self._measured_velocity_min = (
                        velocity
                        if self._measured_velocity_min is None
                        else min(self._measured_velocity_min, velocity)
                    )
                    self._measured_velocity_max = (
                        velocity
                        if self._measured_velocity_max is None
                        else max(self._measured_velocity_max, velocity)
                    )
                if len(message.effort) == len(message.name):
                    effort = float(
                        reorder_joint_values(message.name, message.effort)[joint_index]
                    )
                    self._measured_effort_min = (
                        effort
                        if self._measured_effort_min is None
                        else min(self._measured_effort_min, effort)
                    )
                    self._measured_effort_max = (
                        effort
                        if self._measured_effort_max is None
                        else max(self._measured_effort_max, effort)
                    )

    def _on_ack(self, message: UInt32) -> None:
        sequence = int(message.data)
        if sequence < self._sequence:
            self._acks.add(sequence)

    def _send(self) -> None:
        now = time.monotonic()
        if self._finished_sending or now < self._start_at:
            return
        if self._latest_state is None or now - self._latest_state_at > 0.1:
            return

        elapsed = now - self._start_at
        if elapsed >= self._args.duration_sec:
            self._finished_sending = True
            self._last_send_at = now
            return

        target = None
        if self._args.mode == "joint-sine":
            if self._initial_q is None:
                return
            phase = 2.0 * math.pi * elapsed / self._args.duration_sec
            # Starts and ends at the measured initial pose with zero slope.
            target = self._initial_q + 0.5 * self._args.offset_rad * (
                1.0 - math.cos(phase)
            )

        command = build_command(
            self._latest_state,
            self._sequence,
            self._session,
            self._args.joint if self._args.mode == "joint-sine" else None,
            target,
            self._args.kp if self._args.mode == "joint-sine" else 0.0,
            self._args.kd if self._args.mode == "joint-sine" else 0.0,
        )
        command.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(command)
        self._last_send_at = now
        self._sequence += 1

    def _report(self) -> None:
        self.get_logger().info(
            "control progress: sent=%d acked=%d pending=%d"
            % (self._sequence, len(self._acks), self._sequence - len(self._acks))
        )
        if self._finished_sending and (
            len(self._acks) >= self._sequence
            or time.monotonic() - self._last_send_at >= 2.0
        ):
            self.get_logger().info(
                "FINAL CONTROL: sent=%d acked=%d lost_or_rejected=%d mode=%s"
                % (
                    self._sequence,
                    len(self._acks),
                    self._sequence - len(self._acks),
                    self._args.mode,
                )
            )
            if self._args.joint and self._initial_q is not None:
                self.get_logger().info(
                    "FINAL JOINT RESPONSE: joint=%s initial=%.9f min=%.9f "
                    "max=%.9f final=%.9f span=%.9frad"
                    % (
                        self._args.joint,
                        self._initial_q,
                        self._measured_min
                        if self._measured_min is not None
                        else float("nan"),
                        self._measured_max
                        if self._measured_max is not None
                        else float("nan"),
                        self._measured_q
                        if self._measured_q is not None
                        else float("nan"),
                        (
                            self._measured_max - self._measured_min
                            if self._measured_min is not None
                            and self._measured_max is not None
                            else float("nan")
                        ),
                    )
                )
                self.get_logger().info(
                    "FINAL JOINT LOAD: joint=%s velocity_min=%.9f velocity_max=%.9f "
                    "effort_min=%.9f effort_max=%.9f"
                    % (
                        self._args.joint,
                        self._measured_velocity_min
                        if self._measured_velocity_min is not None
                        else float("nan"),
                        self._measured_velocity_max
                        if self._measured_velocity_max is not None
                        else float("nan"),
                        self._measured_effort_min
                        if self._measured_effort_min is not None
                        else float("nan"),
                        self._measured_effort_max
                        if self._measured_effort_max is not None
                        else float("nan"),
                    )
                )
            self.done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send commands to an explicitly armed A3 MDU gateway"
    )
    parser.add_argument("--topic-prefix", default="/a3_internal")
    parser.add_argument("--mode", choices=("zero-gain", "joint-sine"), default="zero-gain")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--joint", choices=A3_JOINT_NAMES)
    parser.add_argument("--offset-rad", type=float, default=0.005)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=0.2)
    parser.add_argument(
        "--i-understand-this-can-move-the-robot",
        action="store_true",
        help="required acknowledgement before any publisher is created",
    )
    args = parser.parse_args()
    if not args.i_understand_this_can_move_the_robot:
        parser.error("--i-understand-this-can-move-the-robot is required")
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS topic prefix")
    if not math.isfinite(args.hz) or args.hz <= 0.0 or args.hz > 100.0:
        parser.error("hz must be in (0, 100]")
    if args.duration_sec <= 0.0 or args.warmup_sec < 0.0:
        parser.error("duration-sec must be positive and warmup-sec non-negative")
    for name in ("offset_rad", "kp", "kd"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{name} must be finite and non-negative")
    if args.mode == "joint-sine" and args.joint is None:
        parser.error("--joint is required for joint-sine mode")
    if args.mode == "zero-gain" and (args.joint is not None or args.kp != 2.0 or args.kd != 0.2):
        parser.error("joint/kp/kd options are not used in zero-gain mode")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ControlSender(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
