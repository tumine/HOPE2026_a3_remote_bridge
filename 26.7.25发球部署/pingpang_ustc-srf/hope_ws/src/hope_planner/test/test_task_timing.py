"""Pure tests for new-task time-to-strike admission."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from hope_planner.ball_trajectory_predictor import StrikeTarget
from hope_planner.planner import HOPEPlanner
from hope_planner.task_timing import (
    NewTaskTTSGate,
    TaskTimingDecision,
    validate_new_task_tts_window,
)


def test_window_validation_is_strict():
    assert validate_new_task_tts_window(0.95, 1.0) == (0.95, 1.0)
    assert validate_new_task_tts_window(1.0, 1.0) == (1.0, 1.0)
    for minimum, maximum in (
        (-0.01, 1.0),
        (1.01, 1.0),
        (float("nan"), 1.0),
        (0.95, float("inf")),
    ):
        with pytest.raises(ValueError):
            validate_new_task_tts_window(minimum, maximum)


def test_new_task_waits_enters_window_then_latches_late_ball():
    gate = NewTaskTTSGate(0.95, 1.0)

    assert gate.decide(1.10, task_active=False) is TaskTimingDecision.WAIT
    assert gate.decide(1.0001, task_active=False) is TaskTimingDecision.WAIT
    assert gate.decide(1.0, task_active=False) is TaskTimingDecision.ACCEPT
    assert gate.decide(0.95, task_active=False) is TaskTimingDecision.ACCEPT
    assert gate.decide(0.9499, task_active=False) is TaskTimingDecision.DROP
    assert gate.current_ball_dropped

    # Estimator jitter cannot resurrect the same expired physical ball.
    assert gate.decide(0.98, task_active=False) is TaskTimingDecision.DROP
    gate.reset_ball()
    assert not gate.current_ball_dropped
    assert gate.decide(0.98, task_active=False) is TaskTimingDecision.ACCEPT


def test_active_task_revisions_bypass_startup_window():
    gate = NewTaskTTSGate(0.95, 1.0)
    assert gate.decide(0.30, task_active=True) is TaskTimingDecision.ACCEPT
    assert gate.decide(-0.02, task_active=True) is TaskTimingDecision.ACCEPT

    # Even a latched inactive-ball drop never blocks an already active task.
    assert gate.decide(0.10, task_active=False) is TaskTimingDecision.DROP
    assert gate.decide(0.05, task_active=True) is TaskTimingDecision.ACCEPT


def test_candidate_time_to_strike_is_public_and_does_not_commit():
    planner = HOPEPlanner()
    standalone = StrikeTarget(
        p_ball=np.zeros(3),
        v_ball=np.zeros(3),
        t_strike=1.0,
        num_bounces=0,
        valid=True,
    )
    with pytest.raises(RuntimeError, match="successful state estimate"):
        planner.candidate_time_to_strike(standalone)

    dt = 1.0 / 300.0
    candidates = {}
    for index in range(8):
        t = index * dt
        candidates = planner.update_candidates(
            t,
            np.array([1.2 - 3.0 * t, -0.7625, 1.0]),
            {0: 0.0},
        )
    strike = candidates[0]
    candidate_tts = planner.candidate_time_to_strike(strike)
    assert candidate_tts == pytest.approx(strike.t_strike - 7 * dt)
    assert planner.strike_target is None
    assert planner.racket_command is None


def test_shipped_yaml_pins_training_lead_window():
    config_path = Path(__file__).parents[1] / "config" / "hope_planner.yaml"
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "hope_planner"
    ]["ros__parameters"]
    assert params["new_task_tts_min_s"] == pytest.approx(0.20)
    assert params["new_task_tts_max_s"] == pytest.approx(1.0)
