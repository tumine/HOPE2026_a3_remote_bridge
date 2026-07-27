"""Parity/regression tests for planner, Torch training physics, and NumPy scoring."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from itertools import product

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[3]
UTILS = (
    ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils"
)
PLANNER_PACKAGE = ROOT / "hope_ws/src/hope_planner"
sys.path.insert(0, str(PLANNER_PACKAGE))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


torch_physics = _load("return_physics_torch_test", UTILS / "return_physics_torch.py")
success_metric = _load("success_metric_parity_test", UTILS / "success_metric.py")

from hope_planner.ball_contact import predict_paddle_contact as planner_contact  # noqa: E402
from hope_planner.ball_trajectory_predictor import StrikeTarget  # noqa: E402
from hope_planner.constants import (  # noqa: E402
    PlannerConfig,
    load_ball_physics,
    load_paddle_params,
    load_table_params,
)
from hope_planner.racket_target_planner import RacketTargetPlanner  # noqa: E402


CFG = success_metric.load_ball_physics_config()
PARAMS = torch_physics.ReturnPhysicsParams.from_mapping(CFG)
PLANNER_CFG = PlannerConfig(**load_paddle_params())
PLANNER_PHYSICS = load_ball_physics()
PLANNER = RacketTargetPlanner(
    PLANNER_PHYSICS, PLANNER_CFG, load_table_params()
)

COMMAND_DOMAINS = (
    (
        ((-0.32, -0.10), (-1.5225, -1.2825), (0.24, 0.45)),
        ((-2.40, -0.90), (-0.35, 0.35), (-2.20, 1.00)),
        ((1.75, 3.40), (0.25, 1.10), (0.35, 1.45)),
    ),
    (
        ((-0.05, 0.25), (-1.0625, -0.6625), (0.08, 0.34)),
        ((-2.30, -0.80), (-0.35, 0.35), (-2.80, 1.00)),
        ((1.05, 3.00), (-0.25, 0.50), (0.45, 1.45)),
    ),
)


def test_shared_config_values_reach_all_models():
    paddle = success_metric.PaddlePhysics.from_config(CFG)
    assert PARAMS.drag_k == PLANNER_PHYSICS.k
    assert PARAMS.ball_radius == PLANNER_PHYSICS.radius
    assert PARAMS.paddle_restitution == PLANNER_CFG.C_r == paddle.restitution
    assert PARAMS.paddle_tangential_damping == PLANNER_CFG.paddle_a_t == paddle.tangential_damping
    assert PARAMS.paddle_tangential_cap == PLANNER_CFG.paddle_mu == paddle.tangential_cap


def test_torch_contact_matches_planner_random_batch():
    rng = np.random.default_rng(20260723)
    count = 256
    incoming = rng.normal(size=(count, 3)) * [0.5, 0.2, 0.5] + [-2.1, 0.0, -4.0]
    racket = rng.normal(size=(count, 3)) * [0.4, 0.2, 0.3] + [1.2, 0.1, 0.9]
    normal = rng.normal(size=(count, 3))
    expected = np.stack(
        [
            planner_contact(
                incoming[i], racket[i], normal[i], PLANNER_PHYSICS, PLANNER_CFG
            )
            for i in range(count)
        ]
    )
    actual = torch_physics.predict_paddle_contact(
        torch.tensor(incoming, dtype=torch.float64),
        torch.tensor(racket, dtype=torch.float64),
        torch.tensor(normal, dtype=torch.float64),
        PARAMS,
    ).numpy()
    assert np.max(np.abs(actual - expected)) < 1.0e-6


def test_torch_target_planner_matches_numpy_planner():
    rng = np.random.default_rng(17)
    positions = []
    incoming = []
    expected_velocity = []
    expected_normal = []
    for _ in range(8):
        position = np.array(
            [
                rng.uniform(-0.32, 0.25),
                rng.uniform(-1.5225, -0.6625),
                rng.uniform(0.08, 0.45),
            ]
        )
        velocity = np.array(
            [
                rng.uniform(-2.4, -0.8),
                rng.uniform(-0.35, 0.35),
                rng.uniform(-2.8, 1.0),
            ]
        )
        command = PLANNER.plan(
            StrikeTarget(
                p_ball=position,
                v_ball=velocity,
                t_strike=0.5,
                num_bounces=0,
                valid=True,
            )
        )
        positions.append(position)
        incoming.append(velocity)
        expected_velocity.append(command.v_racket)
        expected_normal.append(command.n_racket)

    actual_velocity, actual_normal, _ = torch_physics.plan_racket_command(
        torch.tensor(np.asarray(positions), dtype=torch.float64),
        torch.tensor(np.asarray(incoming), dtype=torch.float64),
        torch.tensor(PLANNER_CFG.target_land, dtype=torch.float64).expand(8, 3),
        PLANNER_CFG.delta_t_flight,
        PARAMS,
        integration_dt=PLANNER_CFG.dt_integrate,
    )
    assert np.max(np.abs(actual_velocity.numpy() - expected_velocity)) < 1.0e-5
    assert np.max(np.abs(actual_normal.numpy() - expected_normal)) < 1.0e-6


def test_authoritative_velocity_envelopes_cover_training_and_live_domains():
    """All domain corners plus deterministic interior samples must be admitted."""

    corner_unit = torch.tensor(
        list(product((0.0, 1.0), repeat=6)), dtype=torch.float32
    )
    for side_index, (position_box, incoming_box, velocity_box) in enumerate(
        COMMAND_DOMAINS
    ):
        generator = torch.Generator().manual_seed(20260723 + side_index)
        unit = torch.cat(
            (
                torch.rand(2048, 6, generator=generator),
                corner_unit,
                torch.full((1, 6), 0.5),
            )
        )
        source_box = position_box + incoming_box
        source_lo = torch.tensor([axis[0] for axis in source_box])
        source_hi = torch.tensor([axis[1] for axis in source_box])
        samples = source_lo + (source_hi - source_lo) * unit
        positions = samples[:, :3]
        incoming = samples[:, 3:]
        landing = torch.tensor(PLANNER_CFG.target_land).expand_as(positions)
        training_velocity, _, _ = torch_physics.plan_racket_command(
            positions,
            incoming,
            landing,
            PLANNER_CFG.delta_t_flight,
            PARAMS,
            integration_dt=0.01,
        )
        admitted_lo = torch.tensor([axis[0] for axis in velocity_box])
        admitted_hi = torch.tensor([axis[1] for axis in velocity_box])
        assert torch.all(training_velocity >= admitted_lo)
        assert torch.all(training_velocity <= admitted_hi)

        # The high-resolution live planner has slightly different numerical
        # extrema; explicitly cover every source-domain corner as well.
        for row in samples[-65:-1].numpy():
            command = PLANNER.plan(
                StrikeTarget(
                    p_ball=row[:3],
                    v_ball=row[3:],
                    t_strike=0.5,
                    num_bounces=0,
                    valid=True,
                )
            )
            assert np.all(command.v_racket >= admitted_lo.numpy())
            assert np.all(command.v_racket <= admitted_hi.numpy())


def _sample_box(count: int, ranges: tuple[tuple[float, float], ...]) -> torch.Tensor:
    return torch.stack([torch.empty(count).uniform_(*axis) for axis in ranges], dim=1)


def test_planned_training_targets_are_physically_reachable():
    torch.manual_seed(31)
    cases = [
        (
            ((0.18, 0.40), (-0.76, -0.52), (1.00, 1.21)),
            ((-2.40, -0.90), (-0.35, 0.35), (-2.20, 1.00)),
        ),
        (
            ((0.45, 0.75), (-0.30, 0.10), (0.84, 1.10)),
            ((-2.30, -0.80), (-0.35, 0.35), (-2.80, 1.00)),
        ),
    ]
    for position_box, incoming_box in cases:
        count = 512
        position = _sample_box(count, position_box)
        # Tracking floor frame -> canonical table frame.
        position[:, 0] -= 0.5
        position[:, 1] -= 0.5 * PARAMS.table_width
        position[:, 2] -= 0.76
        incoming = _sample_box(count, incoming_box)
        landing = torch.tensor([2.055, -0.7625, 0.020]).expand(count, 3)
        racket_velocity, racket_normal, _ = torch_physics.plan_racket_command(
            position, incoming, landing, 0.5, PARAMS, integration_dt=0.01
        )
        outgoing = torch_physics.predict_paddle_contact(
            incoming, racket_velocity, racket_normal, PARAMS
        )
        outcome = torch_physics.integrate_return(
            position, outgoing, PARAMS, dt=0.01, max_time=2.0
        )
        assert torch.all(torch.isfinite(racket_velocity))
        assert torch.all(racket_normal[:, 0] > 0.0)
        assert torch.all(outcome.net_clear & outcome.on_opponent)
        landing_error = torch.linalg.vector_norm(
            outcome.landing_xy - landing[:, :2], dim=-1
        )
        assert torch.max(landing_error) < 1.0e-4


def test_training_rollout_matches_high_resolution_numpy_on_planned_returns():
    torch.manual_seed(47)
    count = 32
    position = _sample_box(
        count, ((0.18, 0.40), (-0.76, -0.52), (1.00, 1.21))
    )
    position[:, 0] -= 0.5
    position[:, 1] -= 0.5 * PARAMS.table_width
    position[:, 2] -= 0.76
    incoming = _sample_box(
        count, ((-2.40, -0.90), (-0.35, 0.35), (-2.20, 1.00))
    )
    landing = torch.tensor([2.055, -0.7625, 0.020]).expand(count, 3)
    racket_velocity, racket_normal, _ = torch_physics.plan_racket_command(
        position, incoming, landing, 0.5, PARAMS, integration_dt=0.01
    )
    outgoing = torch_physics.predict_paddle_contact(
        incoming, racket_velocity, racket_normal, PARAMS
    )
    coarse = torch_physics.integrate_return(
        position, outgoing, PARAMS, dt=0.01, max_time=2.0
    )

    physics = success_metric.BallPhysics.from_config(CFG)
    table = success_metric.TableGeometry.from_config(CFG)
    for index in range(count):
        fine = success_metric.integrate_outgoing_ball(
            position[index].numpy(),
            outgoing[index].numpy(),
            physics,
            table,
            dt=0.002,
            max_time=2.0,
        )
        assert fine.net_clear == bool(coarse.net_clear[index])
        assert fine.on_opponent == bool(coarse.on_opponent[index])
        assert fine.landing_xy is not None
        error = np.linalg.norm(
            np.asarray(fine.landing_xy) - coarse.landing_xy[index].numpy()
        )
        assert error < 0.012
