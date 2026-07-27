"""Pure tensor tests for the HOPE lower-body/base stability rewards.

The reward module is loaded with tiny package stubs so these tests do not launch Isaac Sim.  Fake
assets and sensors still expose the same tensor shapes/selectors used by Isaac Lab managers.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
REWARD_FILE = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py"
)


def _load_reward_module():
    stub_names = (
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
        "whole_body_tracking.tasks.tracking.mdp.rewards",
        "whole_body_tracking.tasks.tracking.mdp.hope_commands",
    )
    saved = {name: sys.modules.get(name) for name in stub_names}
    try:
        packages = {}
        for name in stub_names[:4]:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
            packages[name] = package

        imitation = types.ModuleType(stub_names[4])
        hope_commands = types.ModuleType(stub_names[5])
        hope_commands.RacketTargetCommand = object
        sys.modules[stub_names[4]] = imitation
        sys.modules[stub_names[5]] = hope_commands
        packages[stub_names[3]].rewards = imitation

        spec = importlib.util.spec_from_file_location("hope_stability_rewards_test_module", REWARD_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


rewards = _load_reward_module()


class _Scene:
    def __init__(self, robot, sensor):
        self._entities = {"robot": robot}
        self.sensors = {"contact_forces": sensor}

    def __getitem__(self, name):
        return self._entities[name]


def _cfg(name: str, body_ids=slice(None)):
    return SimpleNamespace(name=name, body_ids=body_ids)


def _make_env(num_envs: int = 3):
    dtype = torch.float64
    asset_data = SimpleNamespace(
        root_pos_w=torch.zeros(num_envs, 3, dtype=dtype),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype).repeat(num_envs, 1),
        root_lin_vel_w=torch.zeros(num_envs, 3, dtype=dtype),
        body_link_pos_w=torch.zeros(num_envs, 4, 3, dtype=dtype),
        body_link_quat_w=torch.zeros(num_envs, 4, 4, dtype=dtype),
        body_link_lin_vel_w=torch.zeros(num_envs, 4, 3, dtype=dtype),
    )
    asset_data.body_link_quat_w[..., 0] = 1.0
    sensor_data = SimpleNamespace(net_forces_w=torch.zeros(num_envs, 4, 3, dtype=dtype))
    robot = SimpleNamespace(data=asset_data)
    sensor = SimpleNamespace(data=sensor_data)
    return SimpleNamespace(scene=_Scene(robot, sensor)), asset_data, sensor_data


def test_base_height_and_planar_velocity_terms_have_expected_scale():
    env, data, _ = _make_env()
    data.root_pos_w[:, 2] = torch.tensor([0.9, 1.0, 1.2], dtype=torch.float64)
    data.root_lin_vel_w[:, :2] = torch.tensor(
        [[3.0, 4.0], [0.0, 0.0], [-1.0, 2.0]], dtype=torch.float64
    )
    robot_cfg = _cfg("robot")

    torch.testing.assert_close(
        rewards.base_height_tracking_exp(env, 1.0, 0.1, robot_cfg),
        torch.exp(torch.tensor([-1.0, 0.0, -4.0], dtype=torch.float64)),
    )
    torch.testing.assert_close(
        rewards.base_planar_velocity_l2(env, robot_cfg),
        torch.tensor([25.0, 0.0, 5.0], dtype=torch.float64),
    )


def test_both_feet_contact_never_awards_a_single_planted_foot():
    env, _, sensor = _make_env()
    # Sensor feet are deliberately non-contiguous to exercise resolved body ids.
    sensor.net_forces_w[0, 0, 2] = 11.0
    sensor.net_forces_w[0, 2, 2] = 12.0
    sensor.net_forces_w[1, 0, 2] = 11.0
    sensor.net_forces_w[1, 2, 2] = 9.0

    actual = rewards.both_feet_contact(
        env, _cfg("contact_forces", [0, 2]), force_threshold=10.0
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="exactly 2"):
        rewards.both_feet_contact(env, _cfg("contact_forces", 0))


def test_foot_slip_is_contact_gated_and_selector_order_independent_of_body_index():
    env, data, sensor = _make_env()
    # Asset feet [1, 3] correspond in order to sensor feet [0, 2].
    data.body_link_lin_vel_w[:, 1, 0] = torch.tensor([2.0, 3.0, 8.0])
    data.body_link_lin_vel_w[:, 3, 1] = torch.tensor([4.0, 5.0, 9.0])
    sensor.net_forces_w[0, [0, 2], 2] = 20.0
    sensor.net_forces_w[1, 0, 2] = 20.0

    actual = rewards.foot_slip_l2(
        env,
        _cfg("robot", [1, 3]),
        _cfg("contact_forces", [0, 2]),
        force_threshold=10.0,
    )
    # mean([2**2, 4**2]), mean([3**2, 0]), mean([0, 0])
    torch.testing.assert_close(actual, torch.tensor([10.0, 4.5, 0.0], dtype=torch.float64))


def test_foot_flat_orientation_distinguishes_flat_tilted_and_inverted():
    env, data, _ = _make_env()
    root_half = 2.0**-0.5
    # Env 1: one foot is tilted 90 degrees around X. Env 2: both are inverted.
    data.body_link_quat_w[1, 1] = torch.tensor(
        [root_half, root_half, 0.0, 0.0], dtype=torch.float64
    )
    data.body_link_quat_w[2, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    data.body_link_quat_w[2, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    actual = rewards.foot_flat_orientation(env, _cfg("robot", [1, 3]))
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64))


def test_feet_stance_tracks_width_and_yaw_rotated_support_midpoint():
    env, data, _ = _make_env(num_envs=2)
    data.body_link_pos_w[0, 1, :2] = torch.tensor([0.0, -0.1])
    data.body_link_pos_w[0, 3, :2] = torch.tensor([0.0, 0.1])

    data.root_pos_w[1, :2] = torch.tensor([1.0, 1.0])
    half = 2.0**-0.5
    data.root_quat_w[1] = torch.tensor([half, 0.0, 0.0, half])
    # Robot-frame +0.1 X rotates to world-frame +0.1 Y at this yaw.
    data.body_link_pos_w[1, 1, :2] = torch.tensor([1.0, 0.9])
    data.body_link_pos_w[1, 3, :2] = torch.tensor([1.0, 1.3])

    actual = rewards.feet_stance_l2(
        env,
        _cfg("robot", [1, 3]),
        target_width=0.2,
        midpoint_weight=2.0,
        target_midpoint_xy_b=(0.1, 0.0),
    )
    torch.testing.assert_close(actual, torch.tensor([0.02, 0.04], dtype=torch.float64))


def test_invalid_scales_and_misaligned_foot_selectors_fail_fast():
    env, _, _ = _make_env()
    with pytest.raises(ValueError, match="std must be positive"):
        rewards.base_height_tracking_exp(env, 1.0, 0.0, _cfg("robot"))
    with pytest.raises(ValueError, match="same number"):
        rewards.foot_slip_l2(
            env,
            _cfg("robot", [1, 3]),
            _cfg("contact_forces", [0]),
        )
    with pytest.raises(ValueError, match="exactly 2"):
        rewards.feet_stance_l2(env, _cfg("robot", [1, 2, 3]), target_width=0.2)
