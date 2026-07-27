"""Unit tests for the shared no-spin success metric (pure NumPy; no Isaac / torch needed).

Run:  python tests/test_success_metric.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_UTILS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "source", "whole_body_tracking", "whole_body_tracking", "utils",
)


def _load(name: str, filename: str):
    path = os.path.join(_UTILS, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered to resolve annotations
    spec.loader.exec_module(mod)
    return mod


sm = _load("success_metric", "success_metric.py")


def _phys_table():
    return sm.BallPhysics(), sm.PaddlePhysics(), sm.TableGeometry()


def _successful_strike():
    target = np.array([0.10, -0.76, 0.35])
    incoming = np.array([-2.0, 0.0, -1.0])
    racket_vel = np.array([1.5, 0.0, 1.0])
    racket_normal = np.array([np.cos(np.deg2rad(30.0)), 0.0, np.sin(np.deg2rad(30.0))])
    return target, incoming, racket_vel, racket_normal


def test_geometry_opponent_half():
    _, _, table = _phys_table()
    # Opponent half is beyond the configured net plane up to the far edge, within table width.
    assert table.net_x == table.length / 2.0
    assert table.on_opponent_half(table.net_x + 0.3, -table.width / 2.0)
    assert not table.on_opponent_half(table.net_x - 0.3, -table.width / 2.0)  # our half
    assert not table.on_opponent_half(table.length + 0.5, -table.width / 2.0)  # long


def test_geometry_reads_noncentral_net_position():
    table = sm.TableGeometry.from_config(
        {
            "table": {"length": 2.8, "width": 1.5},
            "net": {"height": 0.16, "x_position": 1.25},
        }
    )
    assert table.net_x == 1.25
    assert table.on_opponent_half(1.26, -0.75)
    assert not table.on_opponent_half(1.24, -0.75)


def test_successful_return():
    phys, paddle, table = _phys_table()
    target, incoming, racket_vel, racket_normal = _successful_strike()
    achieved = target.copy()  # perfect contact
    out = sm.evaluate_return(
        target, achieved, incoming, racket_vel, racket_normal, phys, table, paddle
    )
    assert out.contacted and out.net_clear and out.on_opponent and out.success
    assert out.outgoing_velocity is not None


def test_perfect_position_has_nonzero_relative_approach():
    """Regression: target == racket position must not collapse the contact direction to zero."""
    phys, paddle, table = _phys_table()
    target, incoming, racket_vel, racket_normal = _successful_strike()
    out = sm.evaluate_return(
        target,
        target.copy(),
        incoming,
        racket_vel,
        racket_normal,
        phys,
        table,
        paddle,
    )
    assert out.contacted
    assert out.relative_normal_speed > 0.3


def test_missed_contact_is_failure():
    phys, paddle, table = _phys_table()
    target, incoming, racket_vel, racket_normal = _successful_strike()
    achieved = target + np.array([0.3, 0.0, 0.0])  # 30 cm away -> no contact (radius 0.10)
    out = sm.evaluate_return(
        target, achieved, incoming, racket_vel, racket_normal, phys, table, paddle
    )
    assert not out.contacted and not out.success


def test_short_ball_fails_net_and_bounds():
    phys, paddle, table = _phys_table()
    target = np.array([0.10, -0.76, 0.20])
    incoming = np.array([-0.4, 0.0, -0.1])
    racket_vel = np.array([0.2, 0.0, 0.1])
    racket_normal = np.array([1.0, 0.0, 0.0])
    out = sm.evaluate_return(
        target, target.copy(), incoming, racket_vel, racket_normal, phys, table, paddle
    )
    assert out.contacted and not out.net_clear and not out.on_opponent and not out.success


def test_success_rate_accumulation():
    phys, paddle, table = _phys_table()
    acc = sm.SuccessRate()
    target, incoming, racket_vel, racket_normal = _successful_strike()
    good = sm.evaluate_return(
        target, target.copy(), incoming, racket_vel, racket_normal, phys, table, paddle
    )
    bad = sm.evaluate_return(
        target,
        target + np.array([0.3, 0.0, 0.0]),
        incoming,
        racket_vel,
        racket_normal,
        phys,
        table,
        paddle,
    )
    acc.add(good)
    acc.add(bad)
    assert acc.attempts == 2 and acc.successes == 1
    assert abs(acc.value - 0.5) < 1e-9
    assert acc.as_dict() == {"success_rate": 0.5}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} success-metric tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
