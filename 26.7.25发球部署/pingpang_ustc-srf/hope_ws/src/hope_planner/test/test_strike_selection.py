"""Pure tests for side-aware, training-distribution-safe strike selection."""

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from hope_planner.ball_trajectory_predictor import StrikeTarget
from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.planner import HOPEPlanner
from hope_planner.racket_target_planner import RacketTargetPlanner
from hope_planner.side_selection import BACKHAND, FOREHAND
from hope_planner.strike_selection import (
    StrikeRegion,
    incoming_plane_crossing,
    select_strike_candidate,
    validate_side_regions,
)
from hope_planner.task_timing import NewTaskTTSGate, TaskTimingDecision


SPLIT = -1.1725
HYSTERESIS = 0.04
REGIONS = {
    FOREHAND: StrikeRegion(
        FOREHAND,
        (-0.32, -0.10),
        (-1.5225, -1.2825),
        (0.24, 0.45),
        ((1.75, 3.40), (0.25, 1.10), (0.35, 1.45)),
    ),
    BACKHAND: StrikeRegion(
        BACKHAND,
        (-0.05, 0.25),
        (-1.0625, -0.6625),
        (0.08, 0.34),
        ((1.05, 3.00), (-0.25, 0.50), (0.45, 1.45)),
    ),
}


def _strike(position, t=1.0):
    return StrikeTarget(
        p_ball=np.asarray(position, dtype=float),
        v_ball=np.array([-2.0, 0.0, -1.0]),
        t_strike=t,
        num_bounces=1,
        valid=True,
    )


def test_plane_midpoints_transform_into_grounded_training_x_boxes():
    # canonical table -> floor/station translation is [+0.5, +0.7625, +0.76]
    assert REGIONS[FOREHAND].x_hit + 0.5 == pytest.approx(0.29)
    assert REGIONS[BACKHAND].x_hit + 0.5 == pytest.approx(0.60)


def test_incoming_plane_crossing_interpolates_measured_y_and_z():
    crossing = incoming_plane_crossing(
        np.array([0.14, -0.80, 0.30]),
        np.array([0.06, -0.90, 0.20]),
        0.10,
    )
    assert crossing is not None
    assert np.allclose(crossing, [0.10, -0.85, 0.25])


@pytest.mark.parametrize(
    "previous,current",
    [
        ([0.06, 0.0, 0.2], [0.14, 0.0, 0.2]),  # outgoing
        ([0.14, 0.0, 0.2], [0.12, 0.0, 0.2]),  # has not crossed
        ([0.06, 0.0, 0.2], [0.04, 0.0, 0.2]),  # already past
    ],
)
def test_incoming_plane_crossing_ignores_non_crossing_segments(previous, current):
    assert (
        incoming_plane_crossing(
            np.asarray(previous, dtype=float),
            np.asarray(current, dtype=float),
            0.10,
        )
        is None
    )


@pytest.mark.parametrize(
    "side,canonical,expected_floor",
    [
        (FOREHAND, (-0.21, -1.4025, 0.345), (0.29, -0.64, 1.105)),
        (BACKHAND, (0.10, -0.8625, 0.21), (0.60, -0.10, 0.97)),
    ],
)
def test_in_box_candidates_publish_without_clipping(side, canonical, expected_floor):
    candidate = _strike(canonical)
    selected = select_strike_candidate(
        {side: candidate},
        REGIONS,
        split_y=SPLIT,
        hysteresis_y=HYSTERESIS,
    )
    assert selected == (side, candidate)
    assert np.allclose(np.asarray(canonical) + [0.5, 0.7625, 0.76], expected_floor)


@pytest.mark.parametrize(
    "side,position,axis",
    [
        (FOREHAND, (-0.21, -1.53, 0.345), "y"),
        (FOREHAND, (-0.21, -1.4025, 0.46), "z"),
        (BACKHAND, (0.10, -0.65, 0.21), "y"),
        (BACKHAND, (0.10, -0.8625, 0.35), "z"),
    ],
)
def test_real_prediction_outside_y_or_z_is_rejected_not_clipped(side, position, axis):
    candidate = _strike(position)
    assert (
        select_strike_candidate(
            {side: candidate},
            REGIONS,
            split_y=SPLIT,
            hysteresis_y=HYSTERESIS,
        )
        is None
    )
    assert f"{axis}=" in REGIONS[side].rejection_reason(candidate.p_ball)
    assert np.array_equal(candidate.p_ball, np.asarray(position))


@pytest.mark.parametrize("side", [FOREHAND, BACKHAND])
@pytest.mark.parametrize("axis", range(3))
def test_velocity_outside_training_envelope_is_rejected_not_clipped(side, axis):
    region = REGIONS[side]
    position = np.array(
        [region.x_hit, sum(region.y) / 2.0, sum(region.z) / 2.0]
    )
    velocity = np.array([sum(limits) / 2.0 for limits in region.velocity])
    assert region.contains_command(position, velocity)

    velocity[axis] = region.velocity[axis][1] + 1.0e-3
    original = velocity.copy()
    assert not region.contains_command(position, velocity)
    assert f"v{'xyz'[axis]}=" in region.rejection_reason(position, velocity)
    assert np.array_equal(velocity, original)


def test_bounded_live_margins_admit_near_misses_without_changing_targets():
    region = REGIONS[BACKHAND]
    position = np.array([region.x_hit, -1.10, 0.37])
    velocity = np.array([0.95, 0.0, 0.44])

    assert not region.contains_command(position, velocity)
    assert region.contains_command(
        position,
        velocity,
        position_margin=(0.0, 0.07, 0.04),
        velocity_margin=0.15,
    )
    selected = select_strike_candidate(
        {BACKHAND: _strike(position)},
        REGIONS,
        split_y=SPLIT,
        hysteresis_y=HYSTERESIS,
        position_margin=(0.0, 0.07, 0.04),
    )
    assert selected is not None
    assert np.array_equal(selected[1].p_ball, position)


def test_live_margins_still_reject_large_mocap_outliers():
    region = REGIONS[BACKHAND]
    position = np.array([region.x_hit, -3.0, 0.80])
    velocity = np.array([1.5, 0.0, 0.8])
    margins = (0.0, 0.07, 0.04)

    assert not region.contains_command(
        position,
        velocity,
        position_margin=margins,
        velocity_margin=0.15,
    )
    reason = region.rejection_reason(
        position,
        velocity,
        position_margin=margins,
        velocity_margin=0.15,
    )
    assert "y=-3.0000 outside [-1.1325, -0.5925]" in reason
    assert "z=0.8000 outside [0.0400, 0.3800]" in reason


def test_locked_task_never_flips_to_other_valid_side():
    backhand = _strike((0.10, -0.8575, 0.1967))
    assert (
        select_strike_candidate(
            {BACKHAND: backhand},
            REGIONS,
            split_y=SPLIT,
            hysteresis_y=HYSTERESIS,
            locked_side=FOREHAND,
        )
        is None
    )


def test_previous_side_breaks_rare_two_candidate_tie():
    forehand = _strike((-0.215, -1.4375, 0.3725), t=1.2)
    backhand = _strike((0.10, -0.8575, 0.1967), t=1.0)
    selected = select_strike_candidate(
        {FOREHAND: forehand, BACKHAND: backhand},
        REGIONS,
        split_y=SPLIT,
        hysteresis_y=HYSTERESIS,
        previous_side=FOREHAND,
    )
    assert selected == (FOREHAND, forehand)


def test_configuration_rejects_split_or_hysteresis_overlapping_training_boxes():
    validate_side_regions(REGIONS, SPLIT, HYSTERESIS)
    with pytest.raises(ValueError, match="forehand y range"):
        validate_side_regions(REGIONS, -1.40, HYSTERESIS)
    with pytest.raises(ValueError, match="y range"):
        validate_side_regions(REGIONS, SPLIT, 0.30)


def test_shipped_yaml_is_pinned_to_grounded_training_regions():
    config_path = Path(__file__).parents[1] / "config" / "hope_planner.yaml"
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "hope_planner"
    ]["ros__parameters"]
    assert params["use_side_aware_strike_regions"] is True
    assert params["swing_side_split_y"] == pytest.approx(SPLIT)
    assert params["strike_y_margin_m"] == pytest.approx(0.0)
    assert params["strike_z_margin_m"] == pytest.approx(0.0)
    assert params["racket_velocity_margin_mps"] == pytest.approx(0.0)
    for name, side in (("forehand", FOREHAND), ("backhand", BACKHAND)):
        assert tuple(params[f"{name}_strike_x_range"]) == REGIONS[side].x
        assert tuple(params[f"{name}_strike_y_range"]) == REGIONS[side].y
        assert tuple(params[f"{name}_strike_z_range"]) == REGIONS[side].z
        configured_velocity = tuple(
            tuple(params[f"{name}_velocity_{axis}_range"]) for axis in "xyz"
        )
        assert configured_velocity == REGIONS[side].velocity


def _training_command_keyword(name):
    repo_root = Path(__file__).parents[4]
    cfg_path = (
        repo_root
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CommandsCfg":
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "racket_target"
                        for target in statement.targets
                    )
                    and isinstance(statement.value, ast.Call)
                ):
                    for keyword in statement.value.keywords:
                        if keyword.arg == name:
                            return ast.literal_eval(keyword.value)
    raise AssertionError(f"training command keyword {name!r} not found")


def test_planner_yaml_matches_current_isaac_command_contract():
    """Prevent planner admission ranges silently drifting from training."""

    training_positions = _training_command_keyword("racket_pos_range_per_clip")
    training_velocities = _training_command_keyword("racket_vel_range_per_clip")
    translation = np.array([0.5, 0.7625, 0.76])
    for index, side in enumerate((FOREHAND, BACKHAND)):
        region = REGIONS[side]
        canonical_position = np.asarray(training_positions[index]) - translation[:, None]
        assert np.allclose(
            canonical_position,
            np.asarray((region.x, region.y, region.z)),
            atol=5.0e-5,
        )
        assert np.allclose(training_velocities[index], region.velocity)


@pytest.mark.parametrize(
    "side,serve",
    [
        (
            FOREHAND,
            (2.15, -1.4025, 0.25, -3.2534613, 0.0, 2.6822721),
        ),
        (
            BACKHAND,
            (2.15, -0.8625, 0.25, -2.8232580, 0.0, 2.3846943),
        ),
    ],
)
def test_fake_bringup_serves_enter_timing_and_strike_admission(side, serve):
    """The smoke feed must enter the TTS gate with a physically valid command."""

    physics = BallPhysics()
    config = PlannerConfig(max_predict_time=2.0)
    predictor = BallTrajectoryPredictor(
        physics, config, TableParams()
    )
    command = RacketTargetPlanner(
        physics, config, TableParams()
    )
    planner = HOPEPlanner(physics=physics, config=config, table=TableParams())
    gate = NewTaskTTSGate(0.95, 1.0)
    p_ball = np.asarray(serve[:3], dtype=float)
    v_ball = np.asarray(serve[3:], dtype=float)
    t = 0.0
    dt = 1.0 / 300.0

    initial = predictor.predict(
        p_ball, v_ball, t, x_hit=REGIONS[side].x_hit
    )
    assert initial.valid
    assert 1.0 < initial.t_strike - t < 1.1

    admitted = None
    decisions = []
    for sample_index in range(90):
        if sample_index % 6 == 0:  # shipped solve_period_s = 0.02 at 300 Hz
            candidates = planner.update_candidates(
                t,
                p_ball,
                {candidate_side: region.x_hit
                 for candidate_side, region in REGIONS.items()},
            )
        else:
            planner.estimator.push(t, p_ball)
            candidates = {}

        selected = select_strike_candidate(
            candidates,
            REGIONS,
            split_y=SPLIT,
            hysteresis_y=HYSTERESIS,
        )
        if selected is not None:
            tts = planner.candidate_time_to_strike(selected[1])
            decision = gate.decide(tts, task_active=False)
            decisions.append(decision)
            if decision is TaskTimingDecision.ACCEPT:
                candidate_command = command.plan(selected[1])
                if REGIONS[side].contains_command(
                    candidate_command.p_intercept, candidate_command.v_racket
                ):
                    admitted = (selected[1], candidate_command, tts)
                    break

        # Match fake_ball_publisher's shared quadratic-drag integration.
        acceleration = (
            -physics.k * np.linalg.norm(v_ball) * v_ball + physics.g
        )
        v_ball = v_ball + acceleration * dt
        p_ball = p_ball + v_ball * dt
        t += dt
        if (
            p_ball[2] <= physics.radius
            and v_ball[2] < 0.0
            and 0.0 <= p_ball[0] <= TableParams().length
        ):
            p_ball[2] = physics.radius
            v_ball[0] *= physics.C_h
            v_ball[1] *= physics.C_h
            v_ball[2] = -physics.C_v * v_ball[2]

    assert admitted is not None
    assert all(
        decision is TaskTimingDecision.WAIT for decision in decisions[:-1]
    )
    assert decisions[-1] is TaskTimingDecision.ACCEPT
    strike, candidate_command, tts = admitted
    assert 0.95 <= tts <= 1.0
    assert select_strike_candidate(
        {side: strike},
        REGIONS,
        split_y=SPLIT,
        hysteresis_y=HYSTERESIS,
    ) == (side, strike)
    assert REGIONS[side].contains_command(
        candidate_command.p_intercept, candidate_command.v_racket
    )
