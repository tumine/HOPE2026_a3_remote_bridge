"""Regressions for legal one-bounce MuJoCo incoming trajectories."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[3]
EVALUATOR = REPO / "hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py"
ENV_CFG = (
    REPO
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "hope_mujoco_eval_legal_incoming", EVALUATOR
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _physics(evaluator):
    metric = evaluator._load_success_metric(REPO)
    cfg = metric.load_ball_physics_config()
    return (
        metric.BallPhysics.from_config(cfg),
        evaluator._TableBounceModel.from_config(cfg),
    )


def _training_incoming_boxes():
    tree = ast.parse(ENV_CFG.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name != "RacketTargetCommandCfg":
            continue
        for keyword in node.keywords:
            if keyword.arg == "incoming_ball_vel_range_per_clip":
                return np.asarray(ast.literal_eval(keyword.value), dtype=np.float64)
    raise AssertionError("training incoming-ball velocity boxes not found")


def test_solver_clears_net_and_bounces_once_on_robot_half():
    evaluator = _load_evaluator()
    physics, table_bounce = _physics(evaluator)
    origin_table = np.array([2.15, -1.39, 0.25], dtype=np.float64)
    target_table = np.array([-0.22, -1.44, 0.37], dtype=np.float64)

    velocity, flight = evaluator._solve_bounced_serve_velocity(
        origin_table,
        target_table,
        0.9,
        physics,
        table_bounce,
    )

    assert np.all(np.isfinite(velocity))
    assert evaluator._legal_single_bounce(
        flight, target_table, table_bounce
    )
    bounce = flight.bounces[0]
    assert 0.0 < bounce.position_table[0] < table_bounce.net_x
    assert -table_bounce.width < bounce.position_table[1] < 0.0
    assert bounce.incoming_velocity[2] < 0.0 < bounce.outgoing_velocity[2]
    assert flight.incoming_net_crossings_z[0] > table_bounce.net_clear_z
    np.testing.assert_allclose(
        flight.position_table, target_table, rtol=0.0, atol=2.0e-4
    )


def test_sampler_keeps_lateral_edge_cases_on_table(monkeypatch):
    evaluator = _load_evaluator()
    physics, table_bounce = _physics(evaluator)

    class FakeScene:
        near_edge_x = 0.5
        table_height = 0.76
        offset = np.array([0.5, 0.7625, 0.76], dtype=np.float64)

        def to_table(self, position):
            return np.asarray(position, dtype=np.float64) - self.offset

    monkeypatch.setattr(sys, "argv", ["mujoco_eval_onnx.py"])
    args = evaluator.parse_args()
    args.control_dt_hint = 0.02
    rng = np.random.default_rng(20260723)
    incoming_boxes = _training_incoming_boxes()
    for index in range(12):
        clip_index = index % 2
        result = evaluator._sample_serve(
            rng,
            1 if clip_index == 0 else -1,
            FakeScene(),
            physics,
            table_bounce,
            args,
            np.zeros(2, dtype=np.float64),
        )
        origin_table = FakeScene().to_table(result[1])
        bounce = np.asarray(result[-1]["position_table"], dtype=np.float64)
        assert -table_bounce.width < origin_table[1] < 0.0
        assert 0.0 < bounce[0] < table_bounce.net_x
        assert -table_bounce.width < bounce[1] < 0.0
        assert result[-1]["incoming_net_crossing_z"] > table_bounce.net_clear_z
        arrival = np.asarray(result[4], dtype=np.float64)
        assert np.all(arrival >= incoming_boxes[clip_index, :, 0])
        assert np.all(arrival <= incoming_boxes[clip_index, :, 1])
