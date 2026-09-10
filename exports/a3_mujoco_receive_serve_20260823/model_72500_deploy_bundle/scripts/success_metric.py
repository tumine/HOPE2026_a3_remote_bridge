# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""Shared no-spin return-success metric — the single public evaluation metric.

``success_rate`` is the ONLY shipped metric (see docs and the interface spec). A ball counts once it
enters a strike task (the denominator). A return is a success when ALL of the following hold:

1. **contact** — the racket actually reaches the interception point, and
2. **net crossing** — the outgoing ball crosses the net plane clearing the net top, and
3. **opponent-half first bounce** — the outgoing ball's first bounce lands on the opponent half,
   inside the table bounds.

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

The outgoing ball is rolled out with the shipped **no-spin** physics (quadratic drag + gravity, no
Magnus / spin) from ``configs/ball_physics.yaml`` — the same file the training scene and the planner
read. This module is pure NumPy (no torch / Isaac imports) so it can be used both by the in-sim
evaluator and by the MuJoCo sim-to-sim evaluator.

World frame (shared with the table geometry): origin at the near-side left corner of the table
surface; +x toward the opponent, +y left, +z up with z = 0 at the surface. The robot stands on the
near (x < net_x) half; the opponent half is ``net_x < x <= length``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Environment override + repo-relative location of the shared physics config.
CONFIG_ENV_VAR = "HOPE_BALL_PHYSICS_CONFIG"
CONFIG_REL_PATH = os.path.join("configs", "ball_physics.yaml")

# No-spin subset of the shipped config; also documents the keys this module reads. The shipped
# configs/ball_physics.yaml overrides these when present (single source of truth).
_DEFAULTS = {
    "ball": {"radius": 0.020},
    "drag": {"k": 0.1261, "velocity_clip": 50.0},
    "gravity": 9.81,
    "table": {"length": 2.74, "width": 1.525, "height": 0.76},
    "net": {"height": 0.1525, "x_position": 1.37},
    "contact": {
        "paddle": {
            "restitution": 0.654,
            "tangential_damping": 0.52,
            "tangential_cap": 0.5,
        }
    },
}


def _find_config_path() -> str | None:
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return env_path if os.path.isfile(env_path) else None
    here = os.path.abspath(os.path.dirname(__file__))
    prev = None
    while here != prev:
        candidate = os.path.join(here, CONFIG_REL_PATH)
        if os.path.isfile(candidate):
            return candidate
        prev, here = here, os.path.dirname(here)
    return None


def load_ball_physics_config() -> dict:
    """Return ``configs/ball_physics.yaml`` merged over the no-spin defaults (defaults if absent)."""
    import copy

    cfg = copy.deepcopy(_DEFAULTS)
    path = _find_config_path()
    if path is None:
        return cfg
    try:
        import yaml
    except ImportError:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


@dataclass
class BallPhysics:
    """No-spin ball flight parameters (quadratic drag + gravity)."""

    gravity: float = 9.81         # m/s^2 (acts along -z)
    drag_k: float = 0.1261        # 1/m, a_drag = -k * |v| * v
    velocity_clip: float = 50.0   # m/s, numeric clip on |v| in the drag term
    ball_radius: float = 0.020    # m

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "BallPhysics":
        cfg = cfg if cfg is not None else load_ball_physics_config()
        return cls(
            gravity=float(cfg.get("gravity", 9.81)),
            drag_k=float(cfg.get("drag", {}).get("k", 0.1261)),
            velocity_clip=float(cfg.get("drag", {}).get("velocity_clip", 50.0)),
            ball_radius=float(cfg.get("ball", {}).get("radius", 0.020)),
        )

    def acceleration(self, v: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(v))
        speed = min(speed, self.velocity_clip)
        a = -self.drag_k * speed * v
        a = a + np.array([0.0, 0.0, -self.gravity])
        return a


@dataclass
class PaddlePhysics:
    """No-spin ball-paddle impulse parameters."""

    restitution: float = 0.654
    tangential_damping: float = 0.52
    tangential_angle_coupling: float = 0.0
    tangential_cap: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "PaddlePhysics":
        cfg = cfg if cfg is not None else load_ball_physics_config()
        paddle = cfg.get("contact", {}).get("paddle", {})
        return cls(
            restitution=float(paddle.get("restitution", 0.654)),
            tangential_damping=float(paddle.get("tangential_damping", 0.52)),
            tangential_angle_coupling=float(paddle.get("b_t", 0.0)),
            tangential_cap=float(paddle.get("tangential_cap", 0.5)),
        )


@dataclass
class TableGeometry:
    """Table / net bounds in the world frame (opponent half = ``net_x < x <= length``)."""

    length: float = 2.74
    width: float = 1.525
    net_height: float = 0.1525
    net_x_position: float = 1.37

    @property
    def net_x(self) -> float:
        return self.net_x_position

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "TableGeometry":
        cfg = cfg if cfg is not None else load_ball_physics_config()
        return cls(
            length=float(cfg.get("table", {}).get("length", 2.74)),
            width=float(cfg.get("table", {}).get("width", 1.525)),
            net_height=float(cfg.get("net", {}).get("height", 0.1525)),
            net_x_position=float(
                cfg.get("net", {}).get(
                    "x_position", 0.5 * float(cfg.get("table", {}).get("length", 2.74))
                )
            ),
        )

    def on_opponent_half(self, x: float, y: float) -> bool:
        return (self.net_x < x <= self.length) and (-self.width <= y <= 0.0)


@dataclass
class ReturnOutcome:
    contacted: bool = False
    net_crossed: bool = False
    net_clear: bool = False        # crossed the net plane above the net top
    on_opponent: bool = False      # first bounce on the opponent half, in bounds
    success: bool = False          # contacted AND net_clear AND on_opponent
    landing_xy: tuple[float, float] | None = None
    relative_normal_speed: float = 0.0
    outgoing_velocity: tuple[float, float, float] | None = None


def orient_normal(
    racket_normal_w: np.ndarray,
    incoming_ball_vel_w: np.ndarray,
    racket_vel_w: np.ndarray,
) -> np.ndarray:
    """Normalize and orient a two-sided racket normal toward the incoming-ball side."""

    normal = np.asarray(racket_normal_w, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-9:
        raise ValueError("racket_normal_w cannot be zero")
    normal = normal / norm
    relative = np.asarray(incoming_ball_vel_w, dtype=float) - np.asarray(racket_vel_w, dtype=float)
    if float(np.dot(relative, normal)) > 0.0:
        normal = -normal
    return normal


def relative_normal_speed(
    incoming_ball_vel_w: np.ndarray,
    racket_vel_w: np.ndarray,
    racket_normal_w: np.ndarray,
) -> float:
    """Unsigned relative closing speed normal to the racket plane."""

    normal = np.asarray(racket_normal_w, dtype=float)
    normal /= np.linalg.norm(normal) + 1.0e-9
    relative = np.asarray(incoming_ball_vel_w, dtype=float) - np.asarray(racket_vel_w, dtype=float)
    return abs(float(np.dot(relative, normal)))


def predict_paddle_contact(
    incoming_ball_vel_w: np.ndarray,
    racket_vel_w: np.ndarray,
    racket_normal_w: np.ndarray,
    paddle: PaddlePhysics,
) -> np.ndarray:
    """Outgoing ball velocity from the shared planner paddle-impulse model."""

    incoming = np.asarray(incoming_ball_vel_w, dtype=float)
    racket = np.asarray(racket_vel_w, dtype=float)
    normal = orient_normal(racket_normal_w, incoming, racket)
    relative = incoming - racket
    normal_signed = float(np.dot(relative, normal))
    tangent = relative - normal_signed * normal
    tangent_mag = float(np.linalg.norm(tangent))
    normal_abs = abs(normal_signed)
    relative_mag = float(np.linalg.norm(relative))

    cos_theta = normal_abs / (relative_mag + 1.0e-9)
    raw = (
        paddle.tangential_damping + paddle.tangential_angle_coupling * cos_theta
    ) * tangent_mag
    cap = paddle.tangential_cap * (1.0 + paddle.restitution) * normal_abs
    tangent_impulse = min(max(raw, 0.0), cap)
    if tangent_mag > 1.0e-9:
        delta_tangent = -tangent_impulse * tangent / (tangent_mag + 1.0e-9)
    else:
        delta_tangent = np.zeros(3)
    delta_normal = -(1.0 + paddle.restitution) * normal_signed * normal
    return incoming + delta_normal + delta_tangent


def integrate_outgoing_ball(
    start_pos_w: np.ndarray,
    ball_vel_w: np.ndarray,
    physics: BallPhysics,
    table: TableGeometry,
    dt: float = 0.002,
    max_time: float = 3.0,
) -> ReturnOutcome:
    """Roll out the outgoing ball (no-spin) and report net crossing + first-bounce location.

    ``contacted`` is not set here (the caller applies the contact gate). The ball starts at
    ``start_pos_w`` with velocity ``ball_vel_w``; a bounce is the first downward crossing of
    ``z = ball_radius`` (ball resting on the surface).
    """
    out = ReturnOutcome()
    p = np.asarray(start_pos_w, dtype=float).copy()
    v = np.asarray(ball_vel_w, dtype=float).copy()
    surface_z = physics.ball_radius
    n_steps = int(max_time / dt)
    for _ in range(n_steps):
        a = physics.acceleration(v)
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt
        # Net plane crossing (x increasing through net_x).
        if p[0] < table.net_x <= p_new[0]:
            dx = p_new[0] - p[0]
            frac = (table.net_x - p[0]) / dx if abs(dx) > 1e-12 else 0.5
            net_z = p[2] + frac * (p_new[2] - p[2])
            out.net_crossed = True
            out.net_clear = net_z > (table.net_height + physics.ball_radius)
        # First downward bounce on the table surface plane.
        if p[2] > surface_z >= p_new[2] and v[2] < 0.0:
            dz = p_new[2] - p[2]
            frac = (surface_z - p[2]) / dz if abs(dz) > 1e-12 else 0.5
            land_x = p[0] + frac * (p_new[0] - p[0])
            land_y = p[1] + frac * (p_new[1] - p[1])
            out.landing_xy = (float(land_x), float(land_y))
            out.on_opponent = table.on_opponent_half(land_x, land_y)
            return out
        p, v = p_new, v_new
    return out


def evaluate_return(
    target_pos_w: np.ndarray,
    achieved_racket_pos_w: np.ndarray,
    incoming_ball_vel_w: np.ndarray,
    racket_vel_w: np.ndarray,
    racket_normal_w: np.ndarray,
    physics: BallPhysics,
    table: TableGeometry,
    paddle: PaddlePhysics | None = None,
    contact_radius: float = 0.095,
    min_relative_normal_speed: float = 0.30,
) -> ReturnOutcome:
    """Score one strike attempt.

    * ``target_pos_w`` — the racket target (interception) position (world, m).
    * ``achieved_racket_pos_w`` — the racket's actual position at the strike frame (world, m).
    * ``incoming_ball_vel_w`` — virtual ball velocity at the interception point (world, m/s).
    * ``racket_vel_w`` / ``racket_normal_w`` — achieved racket contact state (world).

    Contact holds when the racket reaches the interception point and has non-trivial
    relative normal closing speed.  The outgoing ball is predicted by the same
    restitution/tangential impulse model used by the planner, then rolled out with
    shared drag and gravity.  The ball starts at the interception point, not at the
    slightly offset achieved racket centre.
    """
    target_pos_w = np.asarray(target_pos_w, dtype=float)
    achieved_racket_pos_w = np.asarray(achieved_racket_pos_w, dtype=float)
    incoming_ball_vel_w = np.asarray(incoming_ball_vel_w, dtype=float)
    racket_vel_w = np.asarray(racket_vel_w, dtype=float)
    racket_normal_w = np.asarray(racket_normal_w, dtype=float)
    paddle = paddle if paddle is not None else PaddlePhysics.from_config()

    closing_speed = relative_normal_speed(incoming_ball_vel_w, racket_vel_w, racket_normal_w)
    contacted = bool(
        np.linalg.norm(achieved_racket_pos_w - target_pos_w) <= contact_radius
        and closing_speed >= min_relative_normal_speed
    )
    if not contacted:
        return ReturnOutcome(contacted=False, relative_normal_speed=closing_speed)

    outgoing = predict_paddle_contact(
        incoming_ball_vel_w, racket_vel_w, racket_normal_w, paddle
    )
    out = integrate_outgoing_ball(target_pos_w, outgoing, physics, table)
    out.contacted = True
    out.success = bool(out.contacted and out.net_clear and out.on_opponent)
    out.relative_normal_speed = closing_speed
    out.outgoing_velocity = tuple(float(x) for x in outgoing)
    return out


class SuccessRate:
    """Accumulates ``success_rate`` over strike attempts (forehand, backhand, and rallies merged)."""

    def __init__(self) -> None:
        self.attempts = 0
        self.successes = 0

    def add(self, outcome: ReturnOutcome) -> None:
        self.attempts += 1
        if outcome.success:
            self.successes += 1

    def add_bool(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1

    @property
    def value(self) -> float:
        return (self.successes / self.attempts) if self.attempts else 0.0

    def as_dict(self) -> dict:
        """The machine-readable result: only ``success_rate``."""
        return {"success_rate": self.value}
