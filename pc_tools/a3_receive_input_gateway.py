#!/usr/bin/env python3

"""Feed the local HOPE planner and publish canonical pelvis pose to the MDU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from ppmocap_msgs.msg import MocapFrame
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from a3_mocap_frame import CanonicalTableCalibrator, MarkerToPelvis


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config/a3_rl_deploy.yaml"


def _point(value: object) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _quaternion_xyzw(value: object) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z, value.w), dtype=np.float64)


@dataclass(frozen=True)
class GatewayConfig:
    input_topic: str
    planner_input_topic: str
    base_pose_topic: str
    canonical_frame: str
    calibration_samples: int
    table_length_m: float
    table_width_m: float
    surface_down_offset_m: float
    calibration_translation_jitter_m: float
    calibration_rotation_jitter_deg: float
    marker_to_pelvis: MarkerToPelvis
    expected_station_xy: np.ndarray
    station_tolerance_m: float
    base_height_range_m: tuple[float, float]

    @classmethod
    def load(cls, path: Path) -> "GatewayConfig":
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("mocap"), dict):
            raise ValueError(f"missing mocap mapping in {path}")
        mocap = document["mocap"]
        table = mocap["table"]
        marker = mocap["robot_marker_to_pelvis"]
        validation = mocap["validation"]
        expected_station = np.asarray(
            validation["expected_station_xy"], dtype=np.float64
        )
        height = tuple(float(value) for value in validation["base_height_range_m"])
        if expected_station.shape != (2,) or not np.isfinite(expected_station).all():
            raise ValueError("expected_station_xy must contain two finite values")
        if len(height) != 2 or not all(math.isfinite(value) for value in height):
            raise ValueError("base_height_range_m must contain two finite values")
        if height[0] >= height[1]:
            raise ValueError("base_height_range_m lower bound must be below upper bound")
        result = cls(
            input_topic=str(mocap["input_topic"]),
            planner_input_topic=str(mocap["planner_input_topic"]),
            base_pose_topic=str(mocap.get("base_pose_topic", "/a3_mocap/pelvis_pose")),
            canonical_frame=str(mocap["canonical_frame"]),
            calibration_samples=int(mocap["calibration_samples"]),
            table_length_m=float(table["length_m"]),
            table_width_m=float(table["width_m"]),
            surface_down_offset_m=float(table["surface_down_offset_m"]),
            calibration_translation_jitter_m=float(
                mocap["calibration_translation_jitter_m"]
            ),
            calibration_rotation_jitter_deg=float(
                mocap["calibration_rotation_jitter_deg"]
            ),
            marker_to_pelvis=MarkerToPelvis.from_values(
                marker["translation_m"], marker["rpy_rad"]
            ),
            expected_station_xy=expected_station,
            station_tolerance_m=float(validation["station_tolerance_m"]),
            base_height_range_m=(height[0], height[1]),
        )
        for topic in (result.input_topic, result.planner_input_topic, result.base_pose_topic):
            if not topic.startswith("/"):
                raise ValueError(f"ROS topic must be absolute: {topic}")
        if not result.canonical_frame:
            raise ValueError("canonical_frame cannot be empty")
        return result


class ReceiveInputGateway(Node):
    def __init__(self, settings: GatewayConfig, status_period_s: float) -> None:
        super().__init__("a3_receive_input_gateway")
        self.settings = settings
        self.calibrator = CanonicalTableCalibrator(
            settings.calibration_samples,
            settings.table_length_m,
            settings.table_width_m,
            settings.surface_down_offset_m,
            settings.calibration_translation_jitter_m,
            settings.calibration_rotation_jitter_deg,
        )
        self.frames = 0
        self.ball_frames = 0
        self.base_poses = 0
        self.invalid_frames = 0
        self.last_frame_at: float | None = None
        self.last_ball_visible: bool | None = None
        self.last_base_position: np.ndarray | None = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        planner_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            MocapFrame, settings.input_topic, self._on_mocap, sensor_qos
        )
        self.ball_publisher = self.create_publisher(
            PoseArray, settings.planner_input_topic, planner_qos
        )
        self.base_pose_publisher = self.create_publisher(
            PoseStamped, settings.base_pose_topic, pose_qos
        )
        self.create_timer(status_period_s, self._log_status)
        self.get_logger().warning(
            "receive gateway started: mocap=%s planner_input=%s base_pose=%s; "
            "inference=off joint_commands=off"
            % (
                settings.input_topic,
                settings.planner_input_topic,
                settings.base_pose_topic,
            )
        )

    def _on_mocap(self, message: MocapFrame) -> None:
        self.frames += 1
        self.last_frame_at = time.monotonic()
        visible = bool(message.ball.visible)
        if visible != self.last_ball_visible:
            self.last_ball_visible = visible
            self.get_logger().info("ball tracking: %s" % ("visible" if visible else "lost"))

        try:
            table_position = _point(message.table.pose.position)
            table_quaternion = _quaternion_xyzw(message.table.pose.orientation)
            robot_position = _point(message.robot.pose.position)
            robot_quaternion = _quaternion_xyzw(message.robot.pose.orientation)
            vectors = (
                table_position,
                table_quaternion,
                robot_position,
                robot_quaternion,
            )
            if not all(np.isfinite(value).all() for value in vectors):
                raise ValueError("PPMocap table/robot pose contains NaN or infinity")

            if self.calibrator.frame is None:
                if not message.table.tracked or not message.robot.tracked:
                    return
                frame = self.calibrator.add(
                    table_position,
                    table_quaternion,
                    robot_position,
                    bool(message.table.long_axis_is_x),
                )
                if frame is not None:
                    self.get_logger().warning(
                        "PPMocap calibrated: origin=near-left tabletop, "
                        "+X=robot-to-opponent, +Y=robot-left, +Z=up"
                    )

            frame = self.calibrator.frame
            if frame is None:
                return

            if message.robot.tracked:
                base_position, base_quaternion_wxyz = frame.pelvis_from_marker(
                    robot_position,
                    robot_quaternion,
                    self.settings.marker_to_pelvis,
                )
                output = PoseStamped()
                output.header = message.header
                output.header.frame_id = self.settings.canonical_frame
                output.pose.position.x = float(base_position[0])
                output.pose.position.y = float(base_position[1])
                output.pose.position.z = float(base_position[2])
                output.pose.orientation.w = float(base_quaternion_wxyz[0])
                output.pose.orientation.x = float(base_quaternion_wxyz[1])
                output.pose.orientation.y = float(base_quaternion_wxyz[2])
                output.pose.orientation.z = float(base_quaternion_wxyz[3])
                self.base_pose_publisher.publish(output)
                self.base_poses += 1
                self.last_base_position = base_position

            if visible:
                ball_position = frame.point_from_world(_point(message.ball.position))
                output = PoseArray()
                output.header = message.header
                output.header.frame_id = self.settings.canonical_frame
                pose = Pose()
                pose.position.x = float(ball_position[0])
                pose.position.y = float(ball_position[1])
                pose.position.z = float(ball_position[2])
                pose.orientation.w = 1.0
                output.poses.append(pose)
                self.ball_publisher.publish(output)
                self.ball_frames += 1
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.invalid_frames += 1
            if self.invalid_frames <= 5:
                self.get_logger().error(f"invalid PPMocap frame: {error}")

    def _log_status(self) -> None:
        frame_age_ms = (
            -1.0
            if self.last_frame_at is None
            else max(0.0, time.monotonic() - self.last_frame_at) * 1000.0
        )
        station_ok = False
        if self.last_base_position is not None:
            station_error = float(
                np.linalg.norm(
                    self.last_base_position[:2] - self.settings.expected_station_xy
                )
            )
            height_ok = (
                self.settings.base_height_range_m[0]
                <= float(self.last_base_position[2])
                <= self.settings.base_height_range_m[1]
            )
            station_ok = station_error <= self.settings.station_tolerance_m and height_ok
        self.get_logger().info(
            "gateway calibrated=%s frame_age=%.1fms frames=%d ball=%d "
            "base_poses=%d invalid=%d station_ok=%s"
            % (
                "yes" if self.calibrator.frame is not None else "no",
                frame_age_ms,
                self.frames,
                self.ball_frames,
                self.base_poses,
                self.invalid_frames,
                "yes" if station_ok else "no",
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPMocap gateway for local HOPE planning and MDU policy input"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--status-period-s", type=float, default=1.0)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.status_period_s) or arguments.status_period_s <= 0.0:
        parser.error("--status-period-s must be finite and positive")
    return arguments


def main() -> int:
    arguments = parse_args()
    settings = GatewayConfig.load(arguments.config)
    rclpy.init()
    node = ReceiveInputGateway(settings, arguments.status_period_s)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
