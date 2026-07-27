#!/usr/bin/env python3

"""Echo non-actuating latency packets on MDU ROS 2 transport.

The program has no iceoryx backend and cannot publish body-drive commands.
"""

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


class CommandLatencyEcho(Node):
    def __init__(self) -> None:
        super().__init__("a3_mdu_command_latency_echo")
        self._count = 0
        self._publisher = self.create_publisher(
            Float64MultiArray, "/a3_test/mdu_pong", QOS
        )
        self.create_subscription(
            Float64MultiArray, "/a3_test/mdu_ping", self._echo, QOS
        )
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "latency echo ready; ROS 2 test topics only, iceoryx commands disabled"
        )

    def _echo(self, msg: Float64MultiArray) -> None:
        self._count += 1
        self._publisher.publish(msg)

    def _report(self) -> None:
        self.get_logger().info("latency echo count: %d" % self._count)


def main() -> None:
    rclpy.init()
    node = CommandLatencyEcho()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
