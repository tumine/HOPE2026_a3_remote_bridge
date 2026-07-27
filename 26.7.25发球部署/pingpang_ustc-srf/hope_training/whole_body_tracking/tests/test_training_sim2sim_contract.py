"""Cross-stack regressions for the Isaac-training -> planner -> MuJoCo contract."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[3]
ENV_CFG = (
    REPO
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
HOPE_COMMANDS = (
    REPO
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)
ROBOT_CFG = (
    REPO
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
EVALUATOR = REPO / "hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py"
TASK_YAML = REPO / "hope_training/whole_body_tracking/cfg/task/HOPEPingPong.yaml"
BASE_ENV_YAML = REPO / "hope_training/whole_body_tracking/cfg/base/env_base.yaml"
PLANNER_YAML = REPO / "hope_ws/src/hope_planner/config/hope_planner.yaml"
JOINT_ORDER_YAML = REPO / "hope_training/config/joint_order_agibot_a3.yaml"
RUNTIME_YAML = REPO / "a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml"
REFERENCE_DIR = REPO / "a3_deploy/a3_deploy_example/reference"
MJCF_PATHS = (
    REPO
    / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml",
    REPO
    / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml",
)
MOTION_PATHS = (
    REPO
    / "hope_training/motions/preprocessed/"
    "ours_forehand_guarded_1p8s_wrist_x.npz",
    REPO / "hope_training/motions/preprocessed/ours_backhand_guarded_1p8s.npz",
)


def _load_file_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _call_keyword(path: pathlib.Path, call_name: str, keyword: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if func_name != call_name:
            continue
        for item in node.keywords:
            if item.arg == keyword:
                return ast.literal_eval(item.value)
    raise AssertionError(f"{path}: missing {call_name}(..., {keyword}=...)")


def _assigned_literal(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: missing assignment {name}")


def _class_assigned_literal(path: pathlib.Path, class_name: str, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, (ast.Assign, ast.AnnAssign)):
                targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                if any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in targets
                ):
                    return ast.literal_eval(item.value)
    raise AssertionError(f"{path}: missing {class_name}.{name}")


def _env_contract():
    return {
        "ground_offsets": _call_keyword(
            ENV_CFG, "MotionCommandCfg", "ground_height_offset_per_clip"
        ),
        "position_boxes": _call_keyword(
            ENV_CFG, "RacketTargetCommandCfg", "racket_pos_range_per_clip"
        ),
        "velocity_boxes": _call_keyword(
            ENV_CFG, "RacketTargetCommandCfg", "racket_vel_range_per_clip"
        ),
    }


def test_default_episode_contains_repeated_complete_strikes():
    """The 10 s horizon contains repeated complete guarded swings."""

    task = yaml.safe_load(TASK_YAML.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_ENV_YAML.read_text(encoding="utf-8"))
    configured_motions = (
        REPO / task["motion_file"],
        REPO / task["motion_file_2"],
    )
    assert configured_motions == MOTION_PATHS
    assert _call_keyword(
        ENV_CFG, "RacketTargetCommandCfg", "strike_phase_per_clip"
    ) == (0.5555555555555556, 0.5555555555555556)
    for motion_path in MOTION_PATHS:
        metadata = yaml.safe_load(
            motion_path.with_suffix(".yaml").read_text(encoding="utf-8")
        )
        with np.load(motion_path) as motion:
            assert motion["joint_pos"].shape[0] == 91
            assert float(motion["fps"]) == 50.0
        assert metadata["frame_count"] == 91
        assert metadata["strike_frame"] == 50
        assert metadata["duration_s"] == 1.8

    task_horizon = float(task["env"]["episode_length_s"])
    assert task_horizon == float(base["env"]["episode_length_s"]) == 10.0

    # 91 actor actions are applied (both +1.0 and -0.8 endpoints included).
    clip_seconds = 91 / 50.0
    max_hold_seconds = 2.0
    assert task_horizon >= 2 * (clip_seconds + max_hold_seconds)
    assert "self.episode_length_s = 10.0" in ENV_CFG.read_text(encoding="utf-8")


def test_motion_grounding_and_joint_limit_contract_match_training_and_preview():
    offsets = _env_contract()["ground_offsets"]
    assert offsets == (
        _assigned_literal(ROBOT_CFG, "A3_FOREHAND_MOTION_GROUND_OFFSET"),
        _assigned_literal(ROBOT_CFG, "A3_BACKHAND_MOTION_GROUND_OFFSET"),
    )
    assert _call_keyword(
        ROBOT_CFG, "ArticulationCfg", "soft_joint_pos_limit_factor"
    ) == 1.0


def test_racket_fk_uses_link_origin_velocity_and_exact_vendor_mount():
    """Racket point velocity must not mix a link-origin pose with COM velocity."""

    tree = ast.parse(HOPE_COMMANDS.read_text(encoding="utf-8"))
    compute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_compute_racket_state"
    )
    attributes = {
        node.attr for node in ast.walk(compute) if isinstance(node, ast.Attribute)
    }
    assert "body_link_lin_vel_w" in attributes
    assert "body_lin_vel_w" not in attributes

    exact_mount = (0.21021, 0.032078, 0.032036)
    assert _assigned_literal(ROBOT_CFG, "A3_MOUNT_OFFSET") == exact_mount
    assert (
        _class_assigned_literal(
            HOPE_COMMANDS, "RacketTargetCommandCfg", "mount_offset"
        )
        == exact_mount
    )

    for mjcf_path in MJCF_PATHS:
        root = ET.parse(mjcf_path).getroot()
        racket_site = root.find(".//site[@name='right_racket']")
        assert racket_site is not None
        site_pos = tuple(float(value) for value in racket_site.attrib["pos"].split())
        assert site_pos == exact_mount


def test_planner_and_evaluator_boxes_equal_grounded_training_contract(monkeypatch):
    contract = _env_contract()
    position_boxes = np.asarray(contract["position_boxes"], dtype=np.float64)
    velocity_boxes = np.asarray(contract["velocity_boxes"], dtype=np.float64)

    planner = yaml.safe_load(PLANNER_YAML.read_text(encoding="utf-8"))
    params = planner["hope_planner"]["ros__parameters"]
    translation = np.array([0.5, 0.7625, 0.76], dtype=np.float64)
    for index, prefix in enumerate(("forehand", "backhand")):
        canonical_pos = np.asarray(
            [
                params[f"{prefix}_strike_x_range"],
                params[f"{prefix}_strike_y_range"],
                params[f"{prefix}_strike_z_range"],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            canonical_pos + translation[:, None], position_boxes[index], atol=1.0e-12
        )
        planner_velocity = np.asarray(
            [
                params[f"{prefix}_velocity_x_range"],
                params[f"{prefix}_velocity_y_range"],
                params[f"{prefix}_velocity_z_range"],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(planner_velocity, velocity_boxes[index], atol=1.0e-12)

    evaluator = _load_file_module("hope_mujoco_eval_contract", EVALUATOR)
    monkeypatch.setattr(sys, "argv", ["mujoco_eval_onnx.py"])
    args = evaluator.parse_args()
    eval_boxes = np.asarray(
        [
            [
                [args.forehand_strike_forward_min, args.forehand_strike_forward_max],
                [args.forehand_strike_lateral_min, args.forehand_strike_lateral_max],
                [args.forehand_strike_height_min, args.forehand_strike_height_max],
            ],
            [
                [args.backhand_strike_forward_min, args.backhand_strike_forward_max],
                [args.backhand_strike_lateral_min, args.backhand_strike_lateral_max],
                [args.backhand_strike_height_min, args.backhand_strike_height_max],
            ],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(eval_boxes, position_boxes, atol=1.0e-12)
    assert args.policy_lead_time == 50 / 50.0 == 1.0
    assert args.training_aligned_preroll is True

    sys.path.insert(0, str(REFERENCE_DIR))
    from a3_deploy_onnx_ref_pingpong.racket_command import ExampleCommandFeed

    feed = ExampleCommandFeed()
    expected_midpoints = position_boxes.mean(axis=2)
    np.testing.assert_allclose(feed.forehand_pos, expected_midpoints[0])
    np.testing.assert_allclose(feed.backhand_pos, expected_midpoints[1])


def test_legacy_contract_restores_its_361_frame_clock(tmp_path):
    """A pre-retime ONNX is evaluated with its own motion clock, not the new default."""

    contract_path = tmp_path / "legacy.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "motion": {
                    "forehand": (
                        "hope_training/motions/preprocessed/"
                        "ours_forehand_cropped_smooth15.npz"
                    ),
                    "backhand": (
                        "hope_training/motions/preprocessed/ours_backhand_cropped.npz"
                    ),
                    "strike_phase_per_clip": [0.5, 0.5],
                },
                "racket_position_box": {
                    side: {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
                    for side in ("forehand", "backhand")
                },
                "racket_velocity_box": {
                    side: {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
                    for side in ("forehand", "backhand")
                },
            }
        ),
        encoding="utf-8",
    )
    evaluator = _load_file_module("hope_mujoco_eval_legacy_clock", EVALUATOR)
    _, _, metadata = evaluator._load_legacy_training_contract(
        contract_path, repo_root=REPO, control_dt=0.02
    )

    assert metadata["frame_count"] == 361
    assert metadata["strike_frame"] == 180
    assert metadata["policy_lead_time_s"] == 3.6
    assert metadata["follow_through_s"] == 3.6
    assert len(metadata["sha256"]) == 64
    assert set(metadata["motion_artifacts"]) == {"forehand", "backhand"}
    for artifact in metadata["motion_artifacts"].values():
        assert pathlib.Path(artifact["path"]).is_file()
        assert len(artifact["sha256"]) == 64


def test_legacy_contract_rejects_shared_box_overrides():
    evaluator = _load_file_module("hope_mujoco_eval_legacy_override", EVALUATOR)
    args = SimpleNamespace(
        strike_forward_min=0.2,
        strike_forward_max=0.4,
        strike_lateral_offset=None,
        strike_lateral_jitter=None,
    )
    with pytest.raises(ValueError, match="frozen evaluation profile"):
        evaluator._reject_legacy_shared_overrides(args)


def test_evaluator_hold_sampling_matches_isaac_integer_clamp():
    evaluator = _load_file_module("hope_mujoco_eval_hold_sampling", EVALUATOR)

    class FixedRng:
        def __init__(self, value):
            self.value = value

        def integers(self, low, high):
            assert (low, high) == (0, 101)
            return self.value

    kwargs = {
        "hold_min_seconds": 0.0,
        "hold_max_seconds": 2.0,
        "stand_min_hold_seconds": 0.5,
        "dt": 0.02,
    }
    # Isaac randint(0, 101), followed by clamp(min=25) for true stand starts.
    assert evaluator._sample_training_hold_ticks(
        FixedRng(0), stand_start=True, **kwargs
    ) == 25
    assert evaluator._sample_training_hold_ticks(
        FixedRng(25), stand_start=True, **kwargs
    ) == 25
    assert evaluator._sample_training_hold_ticks(
        FixedRng(100), stand_start=True, **kwargs
    ) == 100
    assert evaluator._sample_training_hold_ticks(
        FixedRng(0), stand_start=False, **kwargs
    ) == 0


def _minimum_foot_mesh_z(model, data) -> float:
    import mujoco

    minimum = float("inf")
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) == 0:
            continue
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        if body_name not in ("left_ankle_roll_Link", "right_ankle_roll_Link"):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        vertices_w = data.geom_xpos[geom_id] + vertices @ rotation.T
        minimum = min(minimum, float(vertices_w[:, 2].min()))
    assert np.isfinite(minimum)
    return minimum


def test_default_pose_and_every_motion_frame_are_grounded():
    """Recompute grounding from the collision meshes and all 182 default frames."""

    mujoco = pytest.importorskip("mujoco")
    assert MJCF_PATHS[0].read_bytes() == MJCF_PATHS[1].read_bytes()
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATHS[0]))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    stand_clearance = _minimum_foot_mesh_z(model, data)
    assert 0.0 <= stand_clearance <= 0.002
    assert np.isclose(data.qpos[2], _assigned_literal(ROBOT_CFG, "A3_STAND_PELVIS_HEIGHT"))

    joint_names = yaml.safe_load(JOINT_ORDER_YAML.read_text(encoding="utf-8"))[
        "joint_order"
    ]
    offsets = _env_contract()["ground_offsets"]
    for motion_path, offset in zip(MOTION_PATHS, offsets, strict=True):
        with np.load(motion_path) as motion:
            minimum = float("inf")
            for frame in range(motion["joint_pos"].shape[0]):
                mujoco.mj_resetData(model, data)
                data.qpos[:3] = motion["body_pos_w"][frame, 0]
                data.qpos[3:7] = motion["body_quat_w"][frame, 0]
                for column, joint_name in enumerate(joint_names):
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    qpos_address = int(model.jnt_qposadr[joint_id])
                    data.qpos[qpos_address] = motion["joint_pos"][frame, column]
                mujoco.mj_forward(model, data)
                minimum = min(minimum, _minimum_foot_mesh_z(model, data))

        assert minimum < -0.07
        clearance = minimum + float(offset)
        assert 0.000999 <= clearance <= 0.01
