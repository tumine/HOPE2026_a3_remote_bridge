#!/usr/bin/env python3

"""Send local ROS planner inputs to the MDU using the 96-byte A3PP UDP protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import secrets
import socket
import struct
import time
import zlib

import rclpy
from geometry_msgs.msg import PoseStamped
from hope_msgs.msg import RacketCommand
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


MAGIC = 0x50503341
VERSION = 1
POSE_VALID = 1 << 0
COMMAND_VALID = 1 << 1
PACKET_SIZE = 96
PAYLOAD_FORMAT = struct.Struct("<IBBHIII7fQIb3xi6f")
CRC_FORMAT = struct.Struct("<I")
assert PAYLOAD_FORMAT.size == 92
assert PAYLOAD_FORMAT.size + CRC_FORMAT.size == PACKET_SIZE


@dataclass
class PoseSample:
    sequence: int
    received_at: float
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass
class CommandSample:
    deadline: float
    task_id: int
    revision: int
    swing_side: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]


def _stamp_ns(message: object) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


class PlannerUdpSender(Node):
    def __init__(
        self,
        source_address: str,
        destination_address: str,
        port: int,
        rate_hz: float,
        pose_timeout_s: float,
        expected_frame: str,
        status_period_s: float,
    ) -> None:
        super().__init__("a3_planner_udp_sender")
        self.destination = (destination_address, port)
        self.rate_hz = rate_hz
        self.pose_timeout_s = pose_timeout_s
        self.expected_frame = expected_frame
        self.session_id = secrets.randbits(32)
        self.packet_sequence = 0
        self.pose_sequence = 0
        self.pose: PoseSample | None = None
        self.command: CommandSample | None = None
        self.sent_packets = 0
        self.sent_bytes = 0
        self.invalid_poses = 0
        self.invalid_commands = 0
        self.send_errors = 0
        self.last_pose_received_at: float | None = None
        self.max_pose_gap_s = 0.0
        self.last_send_at: float | None = None
        self.max_send_gap_s = 0.0

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
        self.socket.bind((source_address, 0))

        pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseStamped, "/a3_mocap/pelvis_pose", self._on_pose, pose_qos
        )
        self.create_subscription(
            RacketCommand, "/racket/command", self._on_command, command_qos
        )
        self.create_timer(1.0 / rate_hz, self._send)
        self.create_timer(status_period_s, self._log_status)
        self.get_logger().warning(
            "A3PP UDP sender started: source=%s destination=%s:%d rate=%.1fHz "
            "packet=%dB estimated_wire=%.3fMbps"
            % (
                source_address,
                destination_address,
                port,
                rate_hz,
                PACKET_SIZE,
                (PACKET_SIZE + 28 + 38) * rate_hz * 8.0 / 1.0e6,
            )
        )

    def destroy_node(self) -> bool:
        self.socket.close()
        return super().destroy_node()

    def _on_pose(self, message: PoseStamped) -> None:
        if message.header.frame_id != self.expected_frame:
            self.invalid_poses += 1
            return
        position = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            float(message.pose.position.z),
        )
        quaternion = (
            float(message.pose.orientation.w),
            float(message.pose.orientation.x),
            float(message.pose.orientation.y),
            float(message.pose.orientation.z),
        )
        if not _finite(position + quaternion) or sum(x * x for x in quaternion) < 1e-12:
            self.invalid_poses += 1
            return
        received_at = time.monotonic()
        if self.last_pose_received_at is not None:
            self.max_pose_gap_s = max(
                self.max_pose_gap_s, received_at - self.last_pose_received_at
            )
        self.last_pose_received_at = received_at
        self.pose_sequence = (self.pose_sequence + 1) & 0xFFFFFFFF
        self.pose = PoseSample(
            self.pose_sequence, received_at, position, quaternion
        )

    def _on_command(self, message: RacketCommand) -> None:
        if message.header.frame_id != self.expected_frame:
            self.invalid_commands += 1
            return
        position = (
            float(message.position.x),
            float(message.position.y),
            float(message.position.z),
        )
        velocity = (
            float(message.velocity.x),
            float(message.velocity.y),
            float(message.velocity.z),
        )
        tts = float(message.time_to_strike)
        side = int(message.swing_side)
        if not _finite(position + velocity + (tts,)) or tts < 0.0 or side not in (-1, 1):
            self.invalid_commands += 1
            return
        source_stamp_ns = _stamp_ns(message.header.stamp)
        now_ros_ns = self.get_clock().now().nanoseconds
        if source_stamp_ns > 0 and now_ros_ns >= source_stamp_ns:
            tts -= (now_ros_ns - source_stamp_ns) * 1.0e-9
        if tts <= 0.0:
            self.invalid_commands += 1
            return
        self.command = CommandSample(
            deadline=time.monotonic() + tts,
            task_id=int(message.task_id),
            revision=int(message.task_revision),
            swing_side=side,
            position=position,
            velocity=velocity,
        )

    def _send(self) -> None:
        now = time.monotonic()
        if self.last_send_at is not None:
            self.max_send_gap_s = max(self.max_send_gap_s, now - self.last_send_at)
        self.last_send_at = now
        flags = 0
        pose_sequence = self.pose_sequence
        pose_position = (0.0, 0.0, 0.0)
        pose_quaternion = (1.0, 0.0, 0.0, 0.0)
        if self.pose is not None and now - self.pose.received_at <= self.pose_timeout_s:
            flags |= POSE_VALID
            pose_sequence = self.pose.sequence
            pose_position = self.pose.position
            pose_quaternion = self.pose.quaternion_wxyz

        task_id = 0
        revision = 0
        swing_side = 0
        tts_us = 0
        target_position = (0.0, 0.0, 0.0)
        target_velocity = (0.0, 0.0, 0.0)
        if self.command is not None:
            remaining_us = int((self.command.deadline - now) * 1.0e6)
            if remaining_us > 0:
                flags |= COMMAND_VALID
                task_id = self.command.task_id
                revision = self.command.revision
                swing_side = self.command.swing_side
                tts_us = min(remaining_us, 0x7FFFFFFF)
                target_position = self.command.position
                target_velocity = self.command.velocity

        payload = PAYLOAD_FORMAT.pack(
            MAGIC,
            VERSION,
            flags,
            PACKET_SIZE,
            self.session_id,
            self.packet_sequence,
            pose_sequence,
            *pose_position,
            *pose_quaternion,
            task_id,
            revision,
            swing_side,
            tts_us,
            *target_position,
            *target_velocity,
        )
        packet = payload + CRC_FORMAT.pack(zlib.crc32(payload) & 0xFFFFFFFF)
        try:
            sent = self.socket.sendto(packet, self.destination)
            if sent != PACKET_SIZE:
                raise OSError(f"short UDP send: {sent}/{PACKET_SIZE}")
            self.sent_packets += 1
            self.sent_bytes += sent
            self.packet_sequence = (self.packet_sequence + 1) & 0xFFFFFFFF
        except OSError as error:
            self.send_errors += 1
            if self.send_errors <= 5:
                self.get_logger().error(f"UDP send failed: {error}")

    def _log_status(self) -> None:
        now = time.monotonic()
        pose_age_ms = (
            (now - self.pose.received_at) * 1000.0 if self.pose else -1.0
        )
        pose_valid = (
            self.pose is not None
            and pose_age_ms <= self.pose_timeout_s * 1000.0
        )
        self.get_logger().info(
            "udp packets=%d payload_bytes=%d pose_seq=%d pose_valid=%s "
            "pose_age_ms=%.1f max_pose_gap_ms=%.1f max_send_gap_ms=%.1f "
            "command=%s invalid=(pose:%d,command:%d) send_errors=%d"
            % (
                self.sent_packets,
                self.sent_bytes,
                self.pose_sequence,
                "yes" if pose_valid else "no",
                pose_age_ms,
                self.max_pose_gap_s * 1000.0,
                self.max_send_gap_s * 1000.0,
                "valid"
                if self.command and self.command.deadline > now
                else "none",
                self.invalid_poses,
                self.invalid_commands,
                self.send_errors,
            )
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-address", default="192.168.1.11")
    parser.add_argument("--destination-address", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=15001)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--pose-timeout-ms", type=float, default=100.0)
    parser.add_argument("--expected-frame", default="hope_table")
    parser.add_argument("--status-period-s", type=float, default=5.0)
    arguments = parser.parse_args()
    if not (1 <= arguments.port <= 65535):
        parser.error("--port must be in [1, 65535]")
    if not math.isfinite(arguments.rate_hz) or arguments.rate_hz <= 0.0:
        parser.error("--rate-hz must be finite and positive")
    if arguments.rate_hz > 100.0:
        parser.error("--rate-hz above 100 would violate the 0.2 Mbps budget")
    if not math.isfinite(arguments.pose_timeout_ms) or arguments.pose_timeout_ms <= 0.0:
        parser.error("--pose-timeout-ms must be finite and positive")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    rclpy.init()
    node: PlannerUdpSender | None = None
    try:
        node = PlannerUdpSender(
            arguments.source_address,
            arguments.destination_address,
            arguments.port,
            arguments.rate_hz,
            arguments.pose_timeout_ms * 1.0e-3,
            arguments.expected_frame,
            arguments.status_period_s,
        )
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
