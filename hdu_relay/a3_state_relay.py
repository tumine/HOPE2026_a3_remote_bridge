#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class A3StateRelay(Node):
    def __init__(self) -> None:
        super().__init__("a3_hdu_state_relay")
        qos = sensor_qos()
        self._counts = {"joints": 0, "pelvis": 0, "torso": 0}

        self._joints_pub = self.create_publisher(
            JointState, "/a3_remote/joint_states", qos
        )
        self._pelvis_pub = self.create_publisher(
            Imu, "/a3_remote/pelvis_imu", qos
        )
        self._torso_pub = self.create_publisher(
            Imu, "/a3_remote/torso_imu", qos
        )

        self.create_subscription(
            JointState,
            "/a3_internal/joint_states",
            self._relay_joints,
            qos,
        )
        self.create_subscription(
            Imu, "/a3_internal/pelvis_imu", self._relay_pelvis, qos
        )
        self.create_subscription(
            Imu, "/a3_internal/torso_imu", self._relay_torso, qos
        )
        self.create_timer(1.0, self._report)

    def _relay_joints(self, msg: JointState) -> None:
        self._counts["joints"] += 1
        self._joints_pub.publish(msg)

    def _relay_pelvis(self, msg: Imu) -> None:
        self._counts["pelvis"] += 1
        self._pelvis_pub.publish(msg)

    def _relay_torso(self, msg: Imu) -> None:
        self._counts["torso"] += 1
        self._torso_pub.publish(msg)

    def _report(self) -> None:
        self.get_logger().info(
            "relay counts: joints=%d pelvis=%d torso=%d"
            % (
                self._counts["joints"],
                self._counts["pelvis"],
                self._counts["torso"],
            )
        )


def main() -> None:
    rclpy.init()
    node = A3StateRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

