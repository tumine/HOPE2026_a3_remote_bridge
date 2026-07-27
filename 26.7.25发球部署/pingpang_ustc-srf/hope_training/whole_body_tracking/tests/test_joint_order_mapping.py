"""Pure tests for canonical policy columns versus PhysX articulation order."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UTILS = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "utils")


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


joint_config = _load_by_path(
    "hope_joint_order_mapping", os.path.join(_UTILS, "action_adapter_config.py")
)


# Actual Isaac Sim 5.1 enumeration produced by the bundled A3 URDF. It is a
# kinematic-tree order and intentionally differs from the deploy contract.
PHYSX_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def test_physx_order_round_trips_to_canonical_by_name():
    canonical = joint_config.load_joint_order()
    indices = joint_config.resolve_canonical_joint_indices(PHYSX_ORDER, canonical)
    assert tuple(PHYSX_ORDER[index] for index in indices) == canonical
    assert indices != tuple(range(31))


def test_identity_order_is_supported():
    canonical = joint_config.load_joint_order()
    assert joint_config.resolve_canonical_joint_indices(canonical, canonical) == tuple(range(31))


def test_missing_or_unexpected_joint_is_rejected():
    canonical = joint_config.load_joint_order()
    bad = list(PHYSX_ORDER)
    bad[-1] = "not_an_a3_joint"
    with pytest.raises(ValueError, match="joint sets differ"):
        joint_config.resolve_canonical_joint_indices(bad, canonical)


def test_duplicate_joint_is_rejected():
    canonical = joint_config.load_joint_order()
    bad = list(PHYSX_ORDER)
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="must be unique"):
        joint_config.resolve_canonical_joint_indices(bad, canonical)


def _fake_live_env(*, observation_ids=None, action_ids=None, motion_ids=None):
    canonical = joint_config.load_joint_order()
    expected_ids = joint_config.resolve_canonical_joint_indices(PHYSX_ORDER, canonical)
    observation_ids = list(expected_ids if observation_ids is None else observation_ids)
    action_ids = list(expected_ids if action_ids is None else action_ids)
    motion_ids = list(expected_ids if motion_ids is None else motion_ids)

    asset_cfg = lambda: SimpleNamespace(joint_names=list(canonical), joint_ids=list(observation_ids))
    term_cfgs = [
        SimpleNamespace(params={"asset_cfg": asset_cfg()}),
        SimpleNamespace(params={"asset_cfg": asset_cfg()}),
    ]
    observation_manager = SimpleNamespace(
        active_terms={"policy": ["joint_pos", "joint_vel"], "critic": ["joint_pos", "joint_vel"]},
        _group_obs_term_cfgs={"policy": term_cfgs, "critic": term_cfgs},
    )
    action_term = SimpleNamespace(_joint_names=list(canonical), _joint_ids=list(action_ids))
    motion_term = SimpleNamespace(joint_indexes=list(motion_ids))
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_names=list(PHYSX_ORDER)))},
        action_manager=SimpleNamespace(get_term=lambda _name: action_term),
        observation_manager=observation_manager,
        command_manager=SimpleNamespace(get_term=lambda _name: motion_term),
    )


def test_live_contract_accepts_consistent_name_mapping():
    expected = joint_config.resolve_canonical_joint_indices(PHYSX_ORDER)
    assert joint_config.validate_live_joint_order_contract(_fake_live_env()) == expected


def test_live_contract_rejects_raw_physx_action_columns():
    with pytest.raises(ValueError, match="action joint columns are not canonical"):
        joint_config.validate_live_joint_order_contract(_fake_live_env(action_ids=range(31)))


def test_live_contract_rejects_raw_physx_observation_columns():
    with pytest.raises(ValueError, match="policy.joint_pos is not canonical"):
        joint_config.validate_live_joint_order_contract(_fake_live_env(observation_ids=range(31)))


def test_live_contract_rejects_raw_physx_motion_columns():
    with pytest.raises(ValueError, match="motion reference columns are not canonical"):
        joint_config.validate_live_joint_order_contract(_fake_live_env(motion_ids=range(31)))
