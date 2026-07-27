"""Static contract checks for the stability-focused HOPE environment configuration.

These tests intentionally avoid importing Isaac Sim.  They pin the user-visible
training defaults and, most importantly, prove that the actor observation layout
was not changed while reset, action, curriculum, reward, and termination behavior
was updated.
"""

from __future__ import annotations

import ast
import pathlib

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_env_cfg.py"
)
ROBOT_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/robots"
    / "agibot_a3.py"
)


def _tree() -> ast.Module:
    return ast.parse(ENV_CFG.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _nested_class(parent: ast.ClassDef, name: str) -> ast.ClassDef:
    return next(node for node in parent.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment(class_node: ast.ClassDef, name: str) -> ast.AST:
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise AssertionError(f"{class_node.name}.{name} is missing")


def _call_keyword(call: ast.Call, keyword: str):
    for item in call.keywords:
        if item.arg == keyword:
            return ast.literal_eval(item.value)
    raise AssertionError(f"call is missing {keyword}=...")


def _call_keyword_node(call: ast.Call, keyword: str) -> ast.AST:
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    raise AssertionError(f"call is missing {keyword}=...")


def _mdp_function(call: ast.Call) -> str:
    func_keyword = next(item.value for item in call.keywords if item.arg == "func")
    assert isinstance(func_keyword, ast.Attribute)
    assert isinstance(func_keyword.value, ast.Name) and func_keyword.value.id == "mdp"
    return func_keyword.attr


def test_reset_and_target_curriculum_defaults_are_enabled():
    commands = _class(_tree(), "CommandsCfg")
    motion = _assignment(commands, "motion")
    target = _assignment(commands, "racket_target")
    assert isinstance(motion, ast.Call) and isinstance(target, ast.Call)

    assert _call_keyword(motion, "stand_start_prob") == 0.80
    assert _call_keyword(motion, "rsi_stable_frame_only") is True
    assert _call_keyword(motion, "rsi_phase_range") == (0.0, 0.9)
    assert _call_keyword(motion, "rsi_settle_steps_range") == (10, 20)
    assert _call_keyword(motion, "joint_position_range") == (-0.03, 0.03)

    assert _call_keyword(target, "target_curriculum_duration_steps") == 200_000
    assert _call_keyword(target, "position_curriculum_initial_fraction") == 0.35
    assert _call_keyword(target, "incoming_velocity_curriculum_initial_fraction") == 0.40


def test_stability_rewards_and_earlier_fall_boundaries_are_wired():
    tree = _tree()
    rewards = _class(tree, "RewardsCfg")
    expected = {
        "termination_penalty": "is_terminated",
        "base_height": "base_height_tracking_exp",
        "base_planar_velocity": "base_planar_velocity_l2",
        "both_feet_contact": "both_feet_contact",
        "foot_slip": "foot_slip_l2",
        "foot_flat": "foot_flat_orientation",
        "feet_stance": "feet_stance_l2",
        "action_magnitude": "action_l2",
        "leg_joint_acceleration": "joint_acc_l2",
    }
    for term_name, function_name in expected.items():
        value = _assignment(rewards, term_name)
        assert isinstance(value, ast.Call)
        assert _mdp_function(value) == function_name

    terminations = _class(tree, "TerminationsCfg")
    tilted = _assignment(terminations, "base_tilted")
    too_low = _assignment(terminations, "base_too_low")
    assert isinstance(tilted, ast.Call) and isinstance(too_low, ast.Call)
    assert _call_keyword(tilted, "params") == {"threshold": 0.65}
    assert _call_keyword(too_low, "params") == {"min_height": 0.75}


def test_motion_reward_imitates_pelvis_waist_legs_and_arms():
    tree = _tree()
    commands = _class(tree, "CommandsCfg")
    motion = _assignment(commands, "motion")
    rewards = _class(tree, "RewardsCfg")
    imitation = _assignment(rewards, "imitation")
    assert isinstance(motion, ast.Call) and isinstance(imitation, ast.Call)

    motion_bodies = _call_keyword_node(motion, "body_names")
    assert isinstance(motion_bodies, ast.Name) and motion_bodies.id == "A3_TRACKED_BODIES"

    params = _call_keyword_node(imitation, "params")
    assert isinstance(params, ast.Dict)
    body_names = next(
        value
        for key, value in zip(params.keys, params.values)
        if isinstance(key, ast.Constant) and key.value == "body_names"
    )
    assert isinstance(body_names, ast.Name) and body_names.id == "A3_TRACKED_BODIES"

    robot_tree = ast.parse(ROBOT_CFG.read_text(encoding="utf-8"))
    tracked_node = next(
        node.value
        for node in robot_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "A3_TRACKED_BODIES" for target in node.targets)
    )
    tracked = set(ast.literal_eval(tracked_node))
    assert {
        "pelvis_link",
        "torso_Link",
        "left_hip_roll_Link",
        "left_knee_Link",
        "left_ankle_roll_Link",
        "right_hip_roll_Link",
        "right_knee_Link",
        "right_ankle_roll_Link",
    } <= tracked


def test_actor_observation_terms_remain_the_111d_contract():
    observations = _class(_tree(), "ObservationsCfg")
    policy = _nested_class(observations, "PolicyCfg")
    term_names = [
        target.id
        for node in policy.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert term_names == [
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
        "projected_gravity",
        "base_forward_xy",
        "fixed_station_error_xy",
        "racket_target_rel_base",
        "racket_target_vel_w",
        "time_to_strike",
        "swing_side",
    ]


def test_env_wires_the_shared_action_transform_clip_and_scale():
    source = ENV_CFG.read_text(encoding="utf-8")
    assert (
        "self.actions.joint_pos.raw_action_transform = adapter.raw_action_transform"
        in source
    )
    assert "self.actions.joint_pos.action_clip = adapter.action_clip" in source
    assert "self.actions.joint_pos.scale = adapter.action_scale_by_name()" in source


def test_initial_ppo_exploration_matches_the_bounded_action_domain():
    algo = yaml.safe_load((ROOT / "cfg/algo/ppo.yaml").read_text(encoding="utf-8"))
    assert algo["policy"]["init_noise_std"] == 0.6
