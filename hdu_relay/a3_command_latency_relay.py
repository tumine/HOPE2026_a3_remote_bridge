#!/usr/bin/env python3

"""Relay non-actuating command-shaped latency packets across both HDU NICs."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray


QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class CommandLatencyRelay(Node):
    def __init__(self) -> None:
        super().__init__("a3_hdu_command_latency_relay")
        self._forwarded = 0
        self._returned = 0
        self._mdu_pub = self.create_publisher(
            Float64MultiArray, "/a3_test/mdu_ping", QOS
        )
        self._pc_pub = self.create_publisher(
            Float64MultiArray, "/a3_test/pc_pong", QOS
        )
        self.create_subscription(
            Float64MultiArray, "/a3_test/pc_ping", self._to_mdu, QOS
        )
        self.create_subscription(
            Float64MultiArray, "/a3_test/mdu_pong", self._to_pc, QOS
        )
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "latency relay ready; test topics only, no /body_drive command publisher"
        )

    def _to_mdu(self, msg: Float64MultiArray) -> None:
        self._forwarded += 1
        self._mdu_pub.publish(msg)

    def _to_pc(self, msg: Float64MultiArray) -> None:
        self._returned += 1
        self._pc_pub.publish(msg)

    def _report(self) -> None:
        self.get_logger().info(
            "latency relay counts: pc_to_mdu=%d mdu_to_pc=%d"
            % (self._forwarded, self._returned)
        )


def main() -> None:
    rclpy.init()
    node = CommandLatencyRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
