"""Pure regressions for Isaac-training -> MuJoCo robot-dynamics parity.

Physical ball/contact differences are intentionally outside this file.  These
checks pin the robot action/control contract: exact timing, joint position and
effort limits, armature, zero passive damping/friction, and the explicit
MuJoCo emulation of Isaac ``velocity_limit_sim``.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_SCRIPTS_DIR = os.path.join(_ROOT, "scripts")
_RUNTIME_YAML = os.path.join(
    _REPO, "a3_deploy", "a3_deploy_example", "config", "hope_pingpong_runtime.yaml"
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
sys.path.insert(0, _SCRIPTS_DIR)

from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig  # noqa: E402
from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES  # noqa: E402
from a3_deploy_onnx_ref_pingpong.sim_bridge import (  # noqa: E402
    MujocoDirectBridge,
    _integer_substeps,
    _validate_velocity_limits,
)
from mujoco_pingpong_scene import PingPongRealPhysicsScene  # noqa: E402


def _symmetric(names, value):
    return {name: float(value) for name in names}


_EFFORT = {
    "waist_yaw_joint": 220.0,
    "waist_roll_joint": 46.0,
    "waist_pitch_joint": 118.0,
    **_symmetric(("head_yaw_joint", "head_pitch_joint"), 6.0),
    **_symmetric(
        (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
        ),
        60.0,
    ),
    **_symmetric(
        (
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ),
        24.0,
    ),
    **_symmetric(
        (
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
        6.0,
    ),
    **_symmetric(
        (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        ),
        220.0,
    ),
    **_symmetric(("left_knee_joint", "right_knee_joint"), 320.0),
    **_symmetric(("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 118.0),
    **_symmetric(("left_ankle_roll_joint", "right_ankle_roll_joint"), 55.0),
}

_VELOCITY = {
    "waist_yaw_joint": 12.0,
    "waist_roll_joint": 22.7,
    "waist_pitch_joint": 9.2,
    **_symmetric(("head_yaw_joint", "head_pitch_joint"), 12.7),
    **_symmetric(
        (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
        ),
        13.6,
    ),
    **_symmetric(
        (
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ),
        15.7,
    ),
    **_symmetric(
        (
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
        12.7,
    ),
    **_symmetric(
        (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        ),
        12.0,
    ),
    **_symmetric(("left_knee_joint", "right_knee_joint"), 14.6),
    **_symmetric(("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 10.8),
    **_symmetric(("left_ankle_roll_joint", "right_ankle_roll_joint"), 19.3),
}

_ARMATURE = {
    "waist_yaw_joint": 0.066,
    "waist_roll_joint": 0.015,
    "waist_pitch_joint": 0.088,
    **_symmetric(("head_yaw_joint", "head_pitch_joint"), 0.0008),
    **_symmetric(
        (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
        ),
        0.012,
    ),
    **_symmetric(
        (
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ),
        0.005,
    ),
    **_symmetric(
        (
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
        0.0008,
    ),
    **_symmetric(
        (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        ),
        0.066,
    ),
    **_symmetric(("left_knee_joint", "right_knee_joint"), 0.120),
    **_symmetric(("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 0.064),
    **_symmetric(("left_ankle_roll_joint", "right_ankle_roll_joint"), 0.020),
}


def test_shipped_mjcf_robot_dynamics_match_isaac_nominal():
    runtime = RuntimeConfig.load(_RUNTIME_YAML)
    expected_velocity = np.array([_VELOCITY[name] for name in JOINT_NAMES])

    for path in _MJCF_PATHS:
        root = ET.parse(path).getroot()
        assert float(root.find("option").attrib["timestep"]) == 0.001
        assert _integer_substeps(runtime.control_dt, 0.001) == 20

        numeric = root.find("./custom/numeric[@name='isaac_joint_velocity_limits']")
        assert numeric is not None, path
        np.testing.assert_array_equal(
            np.fromstring(numeric.attrib["data"], sep=" "), expected_velocity
        )
        collision_default = root.find("./default/default[@class='collision']/geom")
        floor = root.find(".//geom[@name='floor']")
        assert collision_default is not None
        assert floor is not None
        assert collision_default.attrib["contype"] == "1"
        assert collision_default.attrib["conaffinity"] == "6"
        assert floor.attrib["conaffinity"] == "7"
        assert float(collision_default.attrib["friction"].split()[0]) == 1.0
        assert float(floor.attrib["friction"].split()[0]) == 1.0

        for index, name in enumerate(JOINT_NAMES):
            joint = root.find(f".//joint[@name='{name}']")
            motor = root.find(f".//motor[@joint='{name}']")
            assert joint is not None, f"{path}: missing joint {name}"
            assert motor is not None, f"{path}: missing motor {name}"

            np.testing.assert_array_equal(
                np.fromstring(joint.attrib["range"], sep=" "),
                [runtime.action_adapter.clamp_lower[index], runtime.action_adapter.clamp_upper[index]],
            )
            np.testing.assert_array_equal(
                np.fromstring(joint.attrib["actuatorfrcrange"], sep=" "),
                [-_EFFORT[name], _EFFORT[name]],
            )
            np.testing.assert_array_equal(
                np.fromstring(motor.attrib["ctrlrange"], sep=" "),
                [-_EFFORT[name], _EFFORT[name]],
            )
            assert float(joint.attrib["armature"]) == _ARMATURE[name]
            assert float(joint.attrib.get("damping", 0.0)) == 0.0
            assert float(joint.attrib.get("frictionloss", 0.0)) == 0.0


def test_duplicate_shipped_mjcf_files_cannot_drift():
    with open(_MJCF_PATHS[0], "rb") as first, open(_MJCF_PATHS[1], "rb") as second:
        assert first.read() == second.read()


def test_timing_ratio_validation_rejects_silent_rounding():
    assert _integer_substeps(0.02, 0.001) == 20
    with pytest.raises(ValueError, match="integer multiple"):
        _integer_substeps(0.02, 0.003)
    with pytest.raises(ValueError, match="finite and > 0"):
        _integer_substeps(0.02, 0.0)


def test_velocity_limit_validation_is_strict():
    expected = np.array([_VELOCITY[name] for name in JOINT_NAMES])
    np.testing.assert_array_equal(_validate_velocity_limits(expected, 31), expected)
    with pytest.raises(ValueError, match="shape"):
        _validate_velocity_limits(expected[:-1], 31)
    bad = expected.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite and > 0"):
        _validate_velocity_limits(bad, 31)


def test_paddle_face_sign_is_strict_and_persists_as_scene_task_state():
    scene = PingPongRealPhysicsScene.__new__(PingPongRealPhysicsScene)
    scene._paddle_face_sign = 1.0
    scene.set_paddle_face_sign(-1)
    assert scene._paddle_face_sign == -1.0
    with pytest.raises(ValueError, match=r"exactly \+1 or -1"):
        scene.set_paddle_face_sign(0.0)
    with pytest.raises(ValueError, match=r"exactly \+1 or -1"):
        scene.set_paddle_face_sign(np.nan)


@pytest.mark.parametrize(
    "owner_type",
    (MujocoDirectBridge, PingPongRealPhysicsScene),
)
def test_qvel_projection_caps_only_controlled_joints_and_not_actuator_ctrl(owner_type):
    owner = owner_type.__new__(owner_type)
    owner._v_adr = np.array([1, 3, 5], dtype=int)
    owner._joint_velocity_limits = np.array([1.0, 2.0, 3.0])
    owner.data = SimpleNamespace(
        qvel=np.array([99.0, 1.5, 98.0, -2.5, 97.0, 2.0, 96.0]),
        ctrl=np.array([7.0, -8.0]),
    )
    ctrl_before = owner.data.ctrl.copy()

    assert owner._enforce_joint_velocity_limits()
    np.testing.assert_array_equal(owner.data.qvel, [99.0, 1.0, 98.0, -2.0, 97.0, 2.0, 96.0])
    np.testing.assert_array_equal(owner.data.ctrl, ctrl_before)
    assert not owner._enforce_joint_velocity_limits()
