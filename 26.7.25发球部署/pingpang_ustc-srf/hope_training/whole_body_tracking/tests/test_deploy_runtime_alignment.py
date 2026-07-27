"""Pure-Python regression tests for Isaac -> reference/MuJoCo runtime alignment."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_RUNTIME_YAML = os.path.join(
    _REPO, "a3_deploy", "a3_deploy_example", "config", "hope_pingpong_runtime.yaml"
)
_BUNDLE_RUNTIME_YAML = os.path.join(
    _REPO, "a3_deploy", "a3_pingpong_deploy_bundle", "config", "runtime.yaml"
)
_MJCF_PATHS = (
    os.path.join(
        _REPO,
        "a3_deploy",
        "A3_MuJoCo_Sim",
        "aimrt_mujoco_sim",
        "src",
        "models",
        "bin",
        "cfg",
        "model",
        "a3_pingpong",
        "a3_pingpong.xml",
    ),
    os.path.join(
        _REPO,
        "agibot",
        "A3_MuJoCo_Sim",
        "aimrt_mujoco_sim",
        "src",
        "models",
        "bin",
        "cfg",
        "model",
        "a3_pingpong",
        "a3_pingpong.xml",
    ),
)

sys.path.insert(0, _REFERENCE_DIR)

from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig  # noqa: E402
from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES  # noqa: E402
from a3_deploy_onnx_ref_pingpong.lifecycle import (  # noqa: E402
    LifecycleConfig,
    Phase,
    SwingLifecycle,
)
from a3_deploy_onnx_ref_pingpong.observation import RobotState  # noqa: E402
from a3_deploy_onnx_ref_pingpong.quaternion import base_forward_xy  # noqa: E402
from a3_deploy_onnx_ref_pingpong.racket_command import (  # noqa: E402
    BACKHAND,
    FOREHAND,
    ExampleCommandFeed,
    QueueRacketCommandSource,
    RacketCommand,
    TranslatedRacketCommandSource,
)


def _standing_state() -> RobotState:
    return RobotState(
        base_pos_w=np.array([0.0, 0.0, 1.0664]),
        base_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel_b=np.zeros(3),
        q=np.zeros(31),
        qd=np.zeros(31),
    )


def _command(
    *, task_id: int = 1, revision: int = 0, tts: float = 0.08
) -> RacketCommand:
    return RacketCommand(
        task_id=task_id,
        task_revision=revision,
        swing_side=FOREHAND,
        position=np.array([0.285, -0.675, 1.1325]),
        velocity=np.array([1.645, 0.410, 0.975]),
        time_to_strike=tts,
    )


def test_lifecycle_emits_planner_tts_before_advancing_transition():
    lifecycle = SwingLifecycle(LifecycleConfig(dt=0.02))
    state = _standing_state()

    first = lifecycle.update(_command(tts=0.80), state)
    assert first.time_to_strike == 0.80
    lifecycle.advance()
    second = lifecycle.update(_command(tts=0.80), state)
    assert np.isclose(second.time_to_strike, 0.78)

    # A fresh planner revision also belongs to the current pre-physics state; it
    # must not be decremented until the caller completes that transition.
    revised = lifecycle.update(_command(revision=1, tts=0.75), state)
    assert revised.time_to_strike == 0.75
    lifecycle.advance()
    assert np.isclose(lifecycle.update(_command(revision=1, tts=0.75), state).time_to_strike, 0.73)


def test_base_forward_normalization_matches_isaac_epsilon_semantics():
    forward = base_forward_xy(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(forward, [1.0 / (1.0 + 1.0e-6), 0.0])


def test_shipped_runtime_ready_world_target_stays_fixed_and_returns_to_forehand():
    cfg = RuntimeConfig.load(_RUNTIME_YAML)
    lifecycle = SwingLifecycle(cfg.lifecycle)
    state = _standing_state()
    expected_rel = np.array(
        [0.3201315701007843, -0.6952511668205261, -0.057795558124780655]
    )
    expected_vel = np.array(
        [2.3803303241729736, 0.6271703243255615, 1.0374835729599]
    )

    ready = lifecycle.update(None, state)
    first_world_target = ready.pos_w.copy()
    np.testing.assert_allclose(ready.pos_w - state.base_pos_w, expected_rel)
    np.testing.assert_allclose(ready.vel_w, expected_vel)
    assert ready.time_to_strike == 1.0
    assert ready.swing_side == FOREHAND

    moved_state = _standing_state()
    moved_state.base_pos_w = np.array([0.7, -0.3, 1.02])
    moved_ready = lifecycle.update(None, moved_state)
    np.testing.assert_allclose(moved_ready.pos_w, first_world_target)
    np.testing.assert_allclose(
        moved_ready.pos_w - moved_state.base_pos_w,
        expected_rel + state.base_pos_w - moved_state.base_pos_w,
    )
    np.testing.assert_allclose(moved_ready.vel_w, expected_vel)

    # A completed backhand must return to the same canonical forehand READY input,
    # rather than mirroring the old example reach using the previous swing side.
    backhand = _command(tts=0.0)
    backhand.swing_side = BACKHAND
    lifecycle.update(backhand, state)
    for _ in range(100):
        lifecycle.advance()
        if lifecycle.phase == Phase.READY:
            break
        lifecycle.update(None, state)
    assert lifecycle.phase == Phase.READY
    after_backhand = lifecycle.update(None, state)
    np.testing.assert_allclose(after_backhand.pos_w - state.base_pos_w, expected_rel)
    np.testing.assert_allclose(after_backhand.vel_w, expected_vel)
    assert after_backhand.swing_side == FOREHAND


def test_current_bundle_runtime_uses_the_validated_ready_anchor():
    cfg = RuntimeConfig.load(_BUNDLE_RUNTIME_YAML)
    expected_rel = [0.3201315701007843, -0.6952511668205261, -0.057795558124780655]
    expected_vel = [2.3803303241729736, 0.6271703243255615, 1.0374835729599]
    np.testing.assert_allclose(cfg.lifecycle.ready_target_rel_base_w, expected_rel)
    np.testing.assert_allclose(cfg.lifecycle.ready_target_vel_w, expected_vel)
    assert cfg.lifecycle.ready_time_to_strike == 1.0
    assert cfg.lifecycle.ready_swing_side == FOREHAND
    assert cfg.lifecycle.follow_through_s == 0.8
    assert cfg.lifecycle.recovery_s == 0.0


def test_one_no_serve_command_reaches_follow_through_recovery_and_ready():
    lifecycle = SwingLifecycle(
        LifecycleConfig(dt=0.02, follow_through_s=0.04, recovery_s=0.04)
    )
    state = _standing_state()
    cmd = _command(tts=0.04)
    phases = []

    for _ in range(12):
        lifecycle.update(cmd, state)
        phases.append(lifecycle.phase)
        lifecycle.advance()
        if lifecycle.phase == Phase.READY:
            break

    assert Phase.SWING in phases
    assert Phase.FOLLOW_THROUGH in phases
    assert Phase.RECOVERY in phases
    assert lifecycle.phase == Phase.READY


def test_default_lifecycle_spans_the_full_91_frame_reference_clock():
    lifecycle = SwingLifecycle()
    state = _standing_state()
    cmd = _command(tts=1.0)
    observed = []

    for _ in range(91):
        observed.append(lifecycle.update(cmd, state).time_to_strike)
        lifecycle.advance()

    assert len(observed) == 91
    assert observed[0] == 1.0
    assert np.isclose(observed[50], 0.0, atol=1.0e-12)
    assert np.isclose(observed[90], -0.8, atol=1.0e-12)
    assert lifecycle.phase == Phase.READY


def test_lifecycle_permanently_rejects_expired_new_task():
    lifecycle = SwingLifecycle()
    state = _standing_state()

    stale = lifecycle.update(_command(tts=-0.02), state)
    assert lifecycle.phase == Phase.READY
    assert stale.time_to_strike == lifecycle.cfg.ready_time_to_strike

    # A later revision cannot resurrect the consumed stale task id.
    lifecycle.update(_command(revision=1, tts=1.0), state)
    assert lifecycle.phase == Phase.READY

    fresh = lifecycle.update(_command(task_id=2, tts=1.0), state)
    assert lifecycle.phase == Phase.SWING
    assert fresh.time_to_strike == 1.0


def test_runtime_pd_gains_match_isaac_nominal_per_joint_values():
    cfg = RuntimeConfig.load(_RUNTIME_YAML)
    expected = {
        "waist_yaw_joint": (85.0, 3.0),
        "waist_roll_joint": (50.0, 2.0),
        "waist_pitch_joint": (50.0, 2.0),
        "head_yaw_joint": (40.0, 2.0),
        "head_pitch_joint": (40.0, 2.0),
        "left_shoulder_pitch_joint": (40.0, 3.0),
        "left_shoulder_roll_joint": (40.0, 3.0),
        "left_shoulder_yaw_joint": (30.0, 2.0),
        "left_elbow_joint": (30.0, 2.0),
        "left_wrist_roll_joint": (30.0, 2.0),
        "left_wrist_pitch_joint": (20.0, 2.0),
        "left_wrist_yaw_joint": (20.0, 2.0),
        "right_shoulder_pitch_joint": (40.0, 3.0),
        "right_shoulder_roll_joint": (40.0, 3.0),
        "right_shoulder_yaw_joint": (30.0, 2.0),
        "right_elbow_joint": (30.0, 2.0),
        "right_wrist_roll_joint": (30.0, 2.0),
        "right_wrist_pitch_joint": (20.0, 2.0),
        "right_wrist_yaw_joint": (20.0, 2.0),
        "left_hip_pitch_joint": (80.0, 3.0),
        "left_hip_roll_joint": (120.0, 4.0),
        "left_hip_yaw_joint": (80.0, 3.0),
        "left_knee_joint": (250.0, 8.0),
        "left_ankle_pitch_joint": (50.0, 2.0),
        "left_ankle_roll_joint": (50.0, 2.0),
        "right_hip_pitch_joint": (80.0, 3.0),
        "right_hip_roll_joint": (120.0, 4.0),
        "right_hip_yaw_joint": (80.0, 3.0),
        "right_knee_joint": (250.0, 8.0),
        "right_ankle_pitch_joint": (50.0, 2.0),
        "right_ankle_roll_joint": (50.0, 2.0),
    }
    np.testing.assert_array_equal(cfg.sim_kp, [expected[name][0] for name in JOINT_NAMES])
    np.testing.assert_array_equal(cfg.sim_kd, [expected[name][1] for name in JOINT_NAMES])
    assert cfg.lifecycle.follow_through_s == 0.8
    assert cfg.lifecycle.recovery_s == 0.0
    assert cfg.lifecycle.ready_time_to_strike == 1.0


def test_all_shipped_mjcf_stand_keyframes_use_action_adapter_default_q():
    cfg = RuntimeConfig.load(_RUNTIME_YAML)
    for path in _MJCF_PATHS:
        root = ET.parse(path)
        key = root.find("./keyframe/key[@name='stand']")
        assert key is not None, path
        qpos = np.fromstring(key.attrib["qpos"], sep=" ")
        assert qpos.shape == (7 + len(JOINT_NAMES),), path
        np.testing.assert_array_equal(qpos[7:], cfg.action_adapter.default_q)
        np.testing.assert_array_equal(qpos[:2], [0.0, 0.0])
        np.testing.assert_array_equal(qpos[3:7], [1.0, 0.0, 0.0, 0.0])
        for name in JOINT_NAMES:
            joint = root.find(f".//joint[@name='{name}']")
            assert joint is not None, f"{path}: missing {name}"
            assert float(joint.attrib.get("damping", 0.0)) == 0.0, f"{path}: {name}"
            assert float(joint.attrib.get("frictionloss", 0.0)) == 0.0, f"{path}: {name}"


def test_table_frame_source_applies_translation_to_position_only():
    cfg = RuntimeConfig.load(_RUNTIME_YAML)
    station_xy = np.array([0.2, -0.1])
    translation = cfg.mujoco_table_frame.translation(station_xy)
    np.testing.assert_allclose(translation, [0.7, 0.6625, 0.76])

    # This test isolates the frame transform; queue-age accounting is covered by
    # test_planner_runner_bridge.py with an injected monotonic clock.
    queue = QueueRacketCommandSource(monotonic_clock=lambda: 10.0)
    original = RacketCommand(
        task_id=7,
        task_revision=3,
        swing_side=BACKHAND,
        position=np.array([0.2, -0.8, 0.3]),
        velocity=np.array([-2.0, 0.1, -1.0]),
        time_to_strike=0.6,
    )
    queue.submit(original)
    translated = TranslatedRacketCommandSource(queue, translation).poll()
    np.testing.assert_allclose(translated.position, original.position + translation)
    np.testing.assert_array_equal(translated.velocity, original.velocity)
    assert translated.task_id == 7 and translated.task_revision == 3
    assert translated.swing_side == BACKHAND and translated.time_to_strike == 0.6


def test_default_example_feed_stays_inside_training_boxes():
    station = np.array([0.2, -0.1])
    feed = ExampleCommandFeed(dt=0.02, period_s=0.04, station_xy=tuple(station))
    forehand = feed.poll()
    feed.poll()
    backhand = feed.poll()

    assert forehand.swing_side == FOREHAND
    assert backhand.swing_side == BACKHAND
    np.testing.assert_array_less([0.18, -0.76, 1.00], forehand.position - [*station, 0.0])
    np.testing.assert_array_less(forehand.position - [*station, 0.0], [0.40, -0.52, 1.21])
    np.testing.assert_array_less([0.45, -0.30, 0.84], backhand.position - [*station, 0.0])
    np.testing.assert_array_less(backhand.position - [*station, 0.0], [0.75, 0.10, 1.10])
    np.testing.assert_array_less([1.75, 0.25, 0.35], forehand.velocity)
    np.testing.assert_array_less(forehand.velocity, [3.40, 1.10, 1.45])
    np.testing.assert_array_less([1.05, -0.25, 0.45], backhand.velocity)
    np.testing.assert_array_less(backhand.velocity, [3.00, 0.50, 1.45])

    defaults = ExampleCommandFeed()
    assert defaults.lead_time_s == 1.0
    assert defaults.hold_time_s == 1.0
    assert np.isclose(defaults.period_s, defaults.hold_time_s + 91 * defaults.dt)


def test_default_example_feed_reproduces_training_pre_swing_hold_and_clock():
    feed = ExampleCommandFeed()
    lifecycle = SwingLifecycle()
    state = _standing_state()
    task_ids = []
    observed_tts = []

    # 50 held actions followed by all 91 reference-frame actions.
    for _ in range(141):
        cmd = feed.poll()
        task_ids.append(cmd.task_id)
        observed_tts.append(lifecycle.update(cmd, state).time_to_strike)
        lifecycle.advance()

    assert set(task_ids) == {1}
    np.testing.assert_allclose(observed_tts[:51], 1.0, atol=1.0e-12)
    assert np.isclose(observed_tts[100], 0.0, atol=1.0e-12)
    assert np.isclose(observed_tts[140], -0.8, atol=1.0e-12)
    assert lifecycle.phase == Phase.READY

    next_cmd = feed.poll()
    assert next_cmd.task_id == 2
    assert lifecycle.update(next_cmd, state).time_to_strike == 1.0
    assert lifecycle.phase == Phase.SWING
