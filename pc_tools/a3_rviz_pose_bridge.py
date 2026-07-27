#!/usr/bin/env python3

"""Publish the live A3 pelvis pose in the policy's canonical table frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import TransformStamped
from ppmocap_msgs.msg import MocapFrame
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from a3_mocap_frame import (
    CanonicalTableCalibrator,
    MarkerToPelvis,
    matrix_to_quaternion_wxyz,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "config" / "a3_rl_deploy.yaml"


def _point(message: object) -> np.ndarray:
    return np.asarray((message.x, message.y, message.z), dtype=np.float64)


def _quaternion_xyzw(message: object) -> np.ndarray:
    return np.asarray(
        (message.x, message.y, message.z, message.w), dtype=np.float64
    )


class RvizSettings:
    def __init__(self, path: Path) -> None:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        mocap = document["mocap"]
        table = mocap["table"]
        marker = mocap["robot_marker_to_pelvis"]
        visualization = document.get("visualization", {})

        self.input_topic = str(mocap["input_topic"])
        self.canonical_frame = str(mocap["canonical_frame"])
        self.mocap_world_frame = str(
            visualization.get("mocap_world_frame", "world")
        )
        self.ground_frame = str(visualization.get("ground_frame", "hope_ground"))
        self.ground_z_m = float(visualization.get("ground_z_m", -0.76))
        self.marker_to_pelvis = MarkerToPelvis.from_values(
            marker["translation_m"], marker["rpy_rad"]
        )
        self.calibrator = CanonicalTableCalibrator(
            int(mocap["calibration_samples"]),
            float(table["length_m"]),
            float(table["width_m"]),
            float(table["surface_down_offset_m"]),
            float(mocap["calibration_translation_jitter_m"]),
            float(mocap["calibration_rotation_jitter_deg"]),
        )
        if not self.input_topic.startswith("/"):
            raise ValueError("mocap input topic must be absolute")
        if not self.canonical_frame or self.canonical_frame.startswith("/"):
            raise ValueError("canonical frame must be a non-empty TF frame without leading /")
        if not self.mocap_world_frame or self.mocap_world_frame.startswith("/"):
            raise ValueError("mocap world frame must be non-empty and have no leading /")
        if not self.ground_frame or self.ground_frame.startswith("/"):
            raise ValueError("ground frame must be a non-empty TF frame without leading /")
        if not np.isfinite(self.ground_z_m):
            raise ValueError("ground_z_m must be finite")


class A3RvizPoseBridge(Node):
    def __init__(self, settings: RvizSettings, child_frame: str) -> None:
        super().__init__("a3_rviz_pose_bridge")
        self.settings = settings
        self.child_frame = child_frame
        self.dynamic_tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._calibrated_announced = False
        self._tracked = False
        self._invalid = 0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            MocapFrame, settings.input_topic, self._on_mocap, sensor_qos
        )
        self.get_logger().info(
            "RViz pose bridge waiting for %s; %s -> %s -> %s, ground z=%.3fm"
            % (
                settings.input_topic,
                settings.mocap_world_frame,
                settings.canonical_frame,
                child_frame,
                settings.ground_z_m,
            )
        )

    def _publish_scene_transforms(self, frame: object) -> None:
        stamp = self.get_clock().now().to_msg()

        table = TransformStamped()
        table.header.stamp = stamp
        table.header.frame_id = self.settings.mocap_world_frame
        table.child_frame_id = self.settings.canonical_frame
        table.transform.translation.x = float(frame.origin_world[0])
        table.transform.translation.y = float(frame.origin_world[1])
        table.transform.translation.z = float(frame.origin_world[2])
        table_quaternion = matrix_to_quaternion_wxyz(
            frame.rotation_world_from_table
        )
        table.transform.rotation.w = float(table_quaternion[0])
        table.transform.rotation.x = float(table_quaternion[1])
        table.transform.rotation.y = float(table_quaternion[2])
        table.transform.rotation.z = float(table_quaternion[3])

        ground = TransformStamped()
        ground.header.stamp = stamp
        ground.header.frame_id = self.settings.canonical_frame
        ground.child_frame_id = self.settings.ground_frame
        ground.transform.translation.z = self.settings.ground_z_m
        ground.transform.rotation.w = 1.0
        self.static_tf.sendTransform((table, ground))

    def _on_mocap(self, message: MocapFrame) -> None:
        try:
            table_position = _point(message.table.pose.position)
            table_quaternion = _quaternion_xyzw(message.table.pose.orientation)
            robot_position = _point(message.robot.pose.position)
            robot_quaternion = _quaternion_xyzw(message.robot.pose.orientation)
            values = (
                table_position,
                table_quaternion,
                robot_position,
                robot_quaternion,
            )
            if not all(np.isfinite(value).all() for value in values):
                raise ValueError("PPMocap table/robot pose contains NaN or infinity")

            frame = self.settings.calibrator.frame
            if frame is None:
                if not message.table.tracked or not message.robot.tracked:
                    return
                frame = self.settings.calibrator.add(
                    table_position,
                    table_quaternion,
                    robot_position,
                    bool(message.table.long_axis_is_x),
                )
                if frame is None:
                    return
                if not self._calibrated_announced:
                    self._calibrated_announced = True
                    self._publish_scene_transforms(frame)
                    self.get_logger().info(
                        "RVIZ CALIBRATED: world -> hope_table uses the same locked transform as the policy"
                    )

            if not message.robot.tracked:
                if self._tracked:
                    self.get_logger().warning(
                        "BotA3 tracking lost; freezing the last RViz pelvis pose"
                    )
                self._tracked = False
                return

            pelvis_position, pelvis_quaternion_wxyz = frame.pelvis_from_marker(
                robot_position,
                robot_quaternion,
                self.settings.marker_to_pelvis,
            )
            transform = TransformStamped()
            transform.header = message.header
            transform.header.frame_id = self.settings.canonical_frame
            transform.child_frame_id = self.child_frame
            transform.transform.translation.x = float(pelvis_position[0])
            transform.transform.translation.y = float(pelvis_position[1])
            transform.transform.translation.z = float(pelvis_position[2])
            transform.transform.rotation.w = float(pelvis_quaternion_wxyz[0])
            transform.transform.rotation.x = float(pelvis_quaternion_wxyz[1])
            transform.transform.rotation.y = float(pelvis_quaternion_wxyz[2])
            transform.transform.rotation.z = float(pelvis_quaternion_wxyz[3])
            self.dynamic_tf.sendTransform(transform)
            if not self._tracked:
                self.get_logger().info(
                    "BotA3 tracked: pelvis=[%.3f %.3f %.3f] in %s"
                    % (
                        pelvis_position[0],
                        pelvis_position[1],
                        pelvis_position[2],
                        self.settings.canonical_frame,
                    )
                )
            self._tracked = True
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._invalid += 1
            if self._invalid <= 5:
                self.get_logger().error(f"invalid PPMocap frame: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish hope_table -> A3 pelvis TF for RViz"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--child-frame", default="a3/pelvis_link")
    parser.add_argument(
        "--check", action="store_true", help="validate configuration and exit"
    )
    args = parser.parse_args()
    if not args.child_frame or args.child_frame.startswith("/"):
        parser.error("child-frame must be a non-empty TF frame without leading /")
    return args


def main() -> int:
    args = parse_args()
    try:
        settings = RvizSettings(args.config)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"invalid RViz bridge configuration: {exc}", file=sys.stderr)
        return 66
    if args.check:
        print(
            "RViz bridge config OK: input=%s fixed=%s table=%s child=%s ground=%s@z=%.3f"
            % (
                settings.input_topic,
                settings.mocap_world_frame,
                settings.canonical_frame,
                args.child_frame,
                settings.ground_frame,
                settings.ground_z_m,
            )
        )
        return 0

    rclpy.init()
    node = A3RvizPoseBridge(settings, args.child_frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        # rclpy's installed SIGTERM handler can invalidate the context while
        # spin() is rebuilding its wait set.  Treat that shutdown race as a
        # normal exit, while still surfacing genuine runtime failures.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
