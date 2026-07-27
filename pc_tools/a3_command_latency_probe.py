#!/usr/bin/env python3

"""Measure PC-HDU-MDU-HDU-PC RTT with command-sized, non-actuating packets."""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List

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


def percentile(sorted_values: List[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    index = min(len(sorted_values) - 1, int(math.ceil(fraction * len(sorted_values))) - 1)
    return sorted_values[max(index, 0)]


class CommandLatencyProbe(Node):
    def __init__(
        self,
        hz: float,
        count: int,
        payload_doubles: int,
        wait_sec: float,
        warmup_sec: float,
    ) -> None:
        super().__init__("a3_pc_command_latency_probe")
        self._count = count
        self._payload_doubles = payload_doubles
        self._wait_sec = wait_sec
        self._next_sequence = 0
        self._sent_at: Dict[int, int] = {}
        self._latencies_ms: List[float] = []
        self._received_sequences = set()
        self._duplicates = 0
        self._malformed = 0
        self._last_send_time = 0.0
        self._start_after = time.monotonic() + warmup_sec
        self.done = False

        self._publisher = self.create_publisher(
            Float64MultiArray, "/a3_test/pc_ping", QOS
        )
        self.create_subscription(
            Float64MultiArray, "/a3_test/pc_pong", self._on_pong, QOS
        )
        self._send_timer = self.create_timer(1.0 / hz, self._send)
        self._report_timer = self.create_timer(1.0, self._progress)
        self.get_logger().info(
            "waiting %.1fs for DDS discovery, then starting non-actuating RTT probe: "
            "%.1fHz, count=%d, payload=%d doubles (%d bytes), qos_depth=1"
            % (warmup_sec, hz, count, payload_doubles, payload_doubles * 8)
        )

    def _send(self) -> None:
        if time.monotonic() < self._start_after:
            return
        if self._next_sequence >= self._count:
            self._send_timer.cancel()
            if self._last_send_time == 0.0:
                self._last_send_time = time.monotonic()
            return
        sequence = self._next_sequence
        msg = Float64MultiArray()
        msg.data = [0.0] * self._payload_doubles
        msg.data[0] = float(sequence)
        self._sent_at[sequence] = time.perf_counter_ns()
        self._publisher.publish(msg)
        self._next_sequence += 1
        if self._next_sequence == self._count:
            self._last_send_time = time.monotonic()

    def _on_pong(self, msg: Float64MultiArray) -> None:
        now_ns = time.perf_counter_ns()
        if len(msg.data) != self._payload_doubles or not msg.data:
            self._malformed += 1
            return
        raw_sequence = msg.data[0]
        sequence = int(raw_sequence)
        if float(sequence) != raw_sequence or sequence not in self._sent_at:
            self._malformed += 1
            return
        if sequence in self._received_sequences:
            self._duplicates += 1
            return
        self._received_sequences.add(sequence)
        self._latencies_ms.append((now_ns - self._sent_at[sequence]) / 1.0e6)

    def _progress(self) -> None:
        sent = self._next_sequence
        received = len(self._received_sequences)
        self.get_logger().info(
            "progress: sent=%d received=%d pending=%d malformed=%d duplicates=%d"
            % (sent, received, sent - received, self._malformed, self._duplicates)
        )
        if sent >= self._count and (
            received >= self._count
            or time.monotonic() - self._last_send_time >= self._wait_sec
        ):
            self._print_final()
            self.done = True

    def _print_final(self) -> None:
        values = sorted(self._latencies_ms)
        received = len(values)
        lost = self._count - received
        loss_percent = 100.0 * lost / self._count
        mean = sum(values) / received if received else math.nan
        self.get_logger().info(
            "FINAL RTT: sent=%d received=%d lost=%d loss=%.2f%% "
            "min=%.3fms mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms max=%.3fms "
            "malformed=%d duplicates=%d command_publishers=0"
            % (
                self._count,
                received,
                lost,
                loss_percent,
                values[0] if values else math.nan,
                mean,
                percentile(values, 0.50),
                percentile(values, 0.95),
                percentile(values, 0.99),
                values[-1] if values else math.nan,
                self._malformed,
                self._duplicates,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure safe command-shaped RTT through PC/HDU/MDU"
    )
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--payload-doubles", type=int, default=156)
    parser.add_argument("--wait-sec", type=float, default=2.0)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    args = parser.parse_args()
    if (
        args.hz <= 0.0
        or args.count <= 0
        or args.payload_doubles < 1
        or args.wait_sec < 0.0
        or args.warmup_sec < 0.0
    ):
        parser.error(
            "hz/count must be positive; payload-doubles must be >= 1; "
            "wait/warmup must be non-negative"
        )
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CommandLatencyProbe(
        args.hz,
        args.count,
        args.payload_doubles,
        args.wait_sec,
        args.warmup_sec,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
