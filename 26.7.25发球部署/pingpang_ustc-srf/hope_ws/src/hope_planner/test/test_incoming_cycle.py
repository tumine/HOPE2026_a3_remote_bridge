"""Tests for debounced physical-ball cycle detection."""

import pytest

from hope_planner.incoming_cycle import IncomingCycleDetector


def _feed(detector, count, *, x, vx):
    return [detector.update(x, vx) for _ in range(count)]


def test_stationary_sign_noise_never_rearms():
    detector = IncomingCycleDetector()
    for index in range(200):
        vx = -0.08 if index % 2 else 0.08
        assert not detector.update(1.0, vx)


def test_first_incoming_uses_existing_ball_then_next_cycle_rearms_once():
    detector = IncomingCycleDetector()

    # Stable staging position arms the lifecycle, but the first throw remains
    # diagnostic ball 1, which already exists at startup.
    assert not any(_feed(detector, 10, x=1.5, vx=0.0))
    assert not any(_feed(detector, 5, x=1.4, vx=-2.0))

    # Brief direction glitches during flight cannot re-arm it.
    assert not any(_feed(detector, 3, x=1.0, vx=0.1))
    assert not any(_feed(detector, 5, x=0.9, vx=-2.0))
    assert not detector.update(0.20, -2.0)

    # A stable retrieval/staging interval arms exactly one later throw.
    assert not any(_feed(detector, 10, x=1.5, vx=0.0))
    events = _feed(detector, 12, x=1.4, vx=-2.0)
    assert events.count(True) == 1
    assert events[4]


def test_rearm_requires_ball_to_be_in_front_of_strike_planes():
    detector = IncomingCycleDetector()
    _feed(detector, 5, x=1.0, vx=-2.0)
    detector.update(0.20, -2.0)
    _feed(detector, 10, x=1.0, vx=0.0)

    assert not any(_feed(detector, 20, x=0.20, vx=-2.0))
    assert not any(_feed(detector, 4, x=0.31, vx=-2.0))
    assert detector.update(0.31, -2.0)


def test_hysteresis_dead_band_does_not_change_cycle_state():
    detector = IncomingCycleDetector()
    _feed(detector, 5, x=1.0, vx=-2.0)

    # -0.3 m/s is between the incoming and non-incoming thresholds.
    assert not any(_feed(detector, 50, x=1.0, vx=-0.3))
    assert not any(_feed(detector, 5, x=1.0, vx=-2.0))


def test_fit_transition_cannot_rearm_twice_before_ball_advances():
    detector = IncomingCycleDetector()
    _feed(detector, 10, x=1.5, vx=0.0)
    _feed(detector, 5, x=1.4, vx=-0.6)

    # A polynomial-fit transition can temporarily look non-incoming and then
    # strongly incoming again. Until the ball has advanced past x=0.30 m this
    # remains one physical incoming cycle.
    assert not any(_feed(detector, 20, x=1.35, vx=0.2))
    assert not any(_feed(detector, 20, x=1.20, vx=-2.4))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"incoming_vx_threshold": -0.1, "nonincoming_vx_threshold": -0.1},
        {"incoming_confirm_samples": 0},
        {"nonincoming_confirm_samples": 0},
        {"min_rearm_x": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        IncomingCycleDetector(**kwargs)
