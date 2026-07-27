"""Planner -> runner control-path tests (finding: the documented path must be implemented).

Two layers:

  * ``racket_command_from_msg`` field-mapping tests run everywhere (the conversion is
    duck-typed, so a stand-in message object exercises it without ROS);
  * a live ROS 2 round-trip publishes a ``hope_msgs/RacketCommand`` and asserts the
    runner-side ``RosRacketCommandSource`` receives it. This layer is skipped
    automatically when ``rclpy`` / ``hope_msgs`` are not available (e.g. outside a
    sourced ROS environment).

Run:  python tests/test_planner_runner_bridge.py   (or pytest)
"""

from __future__ import annotations

import ast
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_MUJOCO_EVALUATOR = os.path.join(
    _REPO, "hope_training", "whole_body_tracking", "scripts", "mujoco_eval_onnx.py"
)
sys.path.insert(0, _REFERENCE_DIR)

from a3_deploy_onnx_ref_pingpong.racket_command import (  # noqa: E402
    BACKHAND,
    FOREHAND,
    QueueRacketCommandSource,
)
from a3_deploy_onnx_ref_pingpong.ros_command_source import (  # noqa: E402
    racket_command_from_msg,
)


def _stub_msg(**overrides):
    fields = dict(
        task_id=7,
        task_revision=3,
        swing_side=BACKHAND,
        position=SimpleNamespace(x=0.5, y=-0.3, z=0.9),
        velocity=SimpleNamespace(x=1.5, y=0.8, z=0.4),
        time_to_strike=0.42,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _header(sec: int, nanosec: int = 0):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def test_msg_conversion_field_mapping():
    cmd = racket_command_from_msg(_stub_msg())
    assert cmd.task_id == 7
    assert cmd.task_revision == 3
    assert cmd.swing_side == BACKHAND
    np.testing.assert_allclose(cmd.position, [0.5, -0.3, 0.9])
    np.testing.assert_allclose(cmd.velocity, [1.5, 0.8, 0.4])
    assert cmd.time_to_strike == 0.42


def test_msg_conversion_normalizes_side():
    # The dataclass normalizes any non-negative side to FOREHAND, negative to BACKHAND.
    assert racket_command_from_msg(_stub_msg(swing_side=1)).swing_side == FOREHAND
    assert racket_command_from_msg(_stub_msg(swing_side=-1)).swing_side == BACKHAND


def test_msg_conversion_subtracts_transport_age_from_valid_header_stamp():
    msg = _stub_msg(header=_header(100, 200_000_000))
    cmd = racket_command_from_msg(msg, receipt_time_s=100.35)
    assert np.isclose(cmd.time_to_strike, 0.27)


def test_msg_conversion_never_adds_time_for_invalid_or_future_stamps():
    cases = (
        _stub_msg(),                              # header absent
        _stub_msg(header=_header(0, 0)),         # ROS zero stamp
        _stub_msg(header=_header(101, 0)),       # future / clock skew
        _stub_msg(header=_header(100, -1)),      # malformed nanoseconds
        _stub_msg(header=SimpleNamespace()),     # malformed header
    )
    for msg in cases:
        cmd = racket_command_from_msg(msg, receipt_time_s=100.5)
        assert cmd.time_to_strike == 0.42

    for bad_receipt in (None, float("nan"), float("inf"), "not-a-clock"):
        cmd = racket_command_from_msg(
            _stub_msg(header=_header(100)), receipt_time_s=bad_receipt
        )
        assert cmd.time_to_strike == 0.42


def test_non_ros_mailbox_subtracts_queue_age_without_mutating_command():
    now = [5.0]
    queue = QueueRacketCommandSource(monotonic_clock=lambda: now[0])
    original = racket_command_from_msg(_stub_msg(time_to_strike=0.40))
    queue.submit(original)

    now[0] = 5.07
    aged = queue.poll()
    assert np.isclose(aged.time_to_strike, 0.33)
    assert original.time_to_strike == 0.40

    # A monotonic-clock rollback is invalid and must not make TTS larger.
    now[0] = 4.0
    rolled_back = queue.poll()
    assert rolled_back.time_to_strike == 0.40


def test_offline_mailbox_can_disable_wall_clock_ageing():
    now = [1.0]
    queue = QueueRacketCommandSource(
        monotonic_clock=lambda: now[0],
        age_queued_commands=False,
    )
    queue.submit(racket_command_from_msg(_stub_msg(time_to_strike=0.88)))
    now[0] = 99.0
    assert queue.poll().time_to_strike == 0.88


def test_mujoco_evaluator_explicitly_disables_wall_clock_queue_ageing():
    with open(_MUJOCO_EVALUATOR, encoding="utf-8") as stream:
        tree = ast.parse(stream.read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QueueRacketCommandSource"
    ]
    assert len(calls) == 2
    for call in calls:
        keywords = {item.arg: item.value for item in call.keywords}
        value = keywords.get("age_queued_commands")
        assert isinstance(value, ast.Constant) and value.value is False


def test_ros_transport_and_runner_queue_delays_compose():
    msg = _stub_msg(
        time_to_strike=0.50,
        header=_header(20, 100_000_000),
    )
    at_callback = racket_command_from_msg(msg, receipt_time_s=20.18)
    assert np.isclose(at_callback.time_to_strike, 0.42)

    monotonic_now = [7.0]
    queue = QueueRacketCommandSource(monotonic_clock=lambda: monotonic_now[0])
    queue.submit(at_callback)
    monotonic_now[0] = 7.06
    at_observation = queue.poll()
    assert np.isclose(at_observation.time_to_strike, 0.36)


def _ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
        from hope_msgs.msg import RacketCommand  # noqa: F401
    except ImportError:
        return False
    return True


def test_ros_round_trip_planner_command_reaches_runner_source():
    """Publish on /racket/command; the runner's source must poll the same command."""
    if not _ros_available():
        import pytest

        pytest.skip("rclpy / hope_msgs not available (needs a sourced ROS 2 + hope_ws overlay)")

    import rclpy
    from hope_msgs.msg import RacketCommand as RacketCommandMsg
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    from a3_deploy_onnx_ref_pingpong.ros_command_source import RosRacketCommandSource

    topic = "/racket/command"
    source = RosRacketCommandSource(topic=topic, node_name="bridge_test_sink")

    ctx = rclpy.Context()
    rclpy.init(context=ctx, args=None)
    pub_node = rclpy.create_node("bridge_test_pub", context=ctx)
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    pub = pub_node.create_publisher(RacketCommandMsg, topic, qos)

    msg = RacketCommandMsg()
    msg.task_id = 11
    msg.task_revision = 2
    msg.swing_side = RacketCommandMsg.FOREHAND
    msg.position.x, msg.position.y, msg.position.z = 0.55, -0.35, 0.85
    msg.velocity.x, msg.velocity.y, msg.velocity.z = 1.5, 1.3, 0.6
    msg.time_to_strike = 0.8

    try:
        received = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            msg.header.stamp = pub_node.get_clock().now().to_msg()
            pub.publish(msg)
            time.sleep(0.05)
            received = source.poll()
            if received is not None:
                break
        assert received is not None, "planner command never reached the runner source"
        assert received.task_id == 11
        assert received.task_revision == 2
        assert received.swing_side == FOREHAND
        np.testing.assert_allclose(received.position, [0.55, -0.35, 0.85])
        np.testing.assert_allclose(received.velocity, [1.5, 1.3, 0.6])
        assert 0.0 < received.time_to_strike < 0.8
    finally:
        pub_node.destroy_node()
        rclpy.shutdown(context=ctx)
        source.close()


def main() -> int:
    tests = [
        ("test_msg_conversion_field_mapping", test_msg_conversion_field_mapping),
        ("test_msg_conversion_normalizes_side", test_msg_conversion_normalizes_side),
        (
            "test_msg_conversion_subtracts_transport_age_from_valid_header_stamp",
            test_msg_conversion_subtracts_transport_age_from_valid_header_stamp,
        ),
        (
            "test_msg_conversion_never_adds_time_for_invalid_or_future_stamps",
            test_msg_conversion_never_adds_time_for_invalid_or_future_stamps,
        ),
        (
            "test_non_ros_mailbox_subtracts_queue_age_without_mutating_command",
            test_non_ros_mailbox_subtracts_queue_age_without_mutating_command,
        ),
        (
            "test_offline_mailbox_can_disable_wall_clock_ageing",
            test_offline_mailbox_can_disable_wall_clock_ageing,
        ),
        (
            "test_mujoco_evaluator_explicitly_disables_wall_clock_queue_ageing",
            test_mujoco_evaluator_explicitly_disables_wall_clock_queue_ageing,
        ),
        (
            "test_ros_transport_and_runner_queue_delays_compose",
            test_ros_transport_and_runner_queue_delays_compose,
        ),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[ok] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
    if _ros_available():
        try:
            test_ros_round_trip_planner_command_reaches_runner_source()
            print("[ok] test_ros_round_trip_planner_command_reaches_runner_source")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] test_ros_round_trip: {e}")
    else:
        print("[skip] ROS round-trip (rclpy / hope_msgs not available)")
    print(f"\nplanner-runner bridge tests: {'FAILED' if failed else 'passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
