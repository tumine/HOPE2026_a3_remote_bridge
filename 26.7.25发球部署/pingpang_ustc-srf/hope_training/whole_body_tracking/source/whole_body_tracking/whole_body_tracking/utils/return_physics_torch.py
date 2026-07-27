"""Batched Torch implementation of the shared no-spin return physics.

This module deliberately has no Isaac Lab imports.  It mirrors the pure NumPy
planner/contact model while keeping command sampling and one-shot return
evaluation on the training device:

* quadratic-drag + gravity free flight;
* planner-compatible desired outgoing velocity and racket command;
* paddle normal restitution plus capped tangential damping;
* net clearance and first-bounce success checks in the canonical table frame.

Table-frame convention: the playing surface is ``z = 0``, the near edge is
``x = 0``, the net is at ``x = net_x``, and the width is ``y in [-width, 0]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


_EPS = 1.0e-9


@dataclass(frozen=True)
class ReturnPhysicsParams:
    """No-spin flight, paddle-contact, and table parameters."""

    gravity: float = 9.81
    drag_k: float = 0.1261
    velocity_clip: float = 50.0
    ball_radius: float = 0.020
    paddle_restitution: float = 0.654
    paddle_tangential_damping: float = 0.52
    paddle_tangential_angle_coupling: float = 0.0
    paddle_tangential_cap: float = 0.5
    table_length: float = 2.74
    table_width: float = 1.525
    net_x: float = 1.37
    net_height: float = 0.1525

    @classmethod
    def from_mapping(cls, cfg: dict) -> "ReturnPhysicsParams":
        ball = cfg.get("ball", {})
        drag = cfg.get("drag", {})
        contact = cfg.get("contact", {})
        paddle = contact.get("paddle", {}) if isinstance(contact, dict) else {}
        table = cfg.get("table", {})
        net = cfg.get("net", {})
        return cls(
            gravity=float(cfg.get("gravity", 9.81)),
            drag_k=float(drag.get("k", 0.1261)),
            velocity_clip=float(drag.get("velocity_clip", 50.0)),
            ball_radius=float(ball.get("radius", 0.020)),
            paddle_restitution=float(paddle.get("restitution", 0.654)),
            paddle_tangential_damping=float(paddle.get("tangential_damping", 0.52)),
            paddle_tangential_angle_coupling=float(paddle.get("b_t", 0.0)),
            paddle_tangential_cap=float(paddle.get("tangential_cap", 0.5)),
            table_length=float(table.get("length", 2.74)),
            table_width=float(table.get("width", 1.525)),
            net_x=float(net.get("x_position", 0.5 * float(table.get("length", 2.74)))),
            net_height=float(net.get("height", 0.1525)),
        )


@dataclass
class TorchReturnOutcome:
    """Batched flight outcome tensors."""

    net_crossed: torch.Tensor
    net_clear: torch.Tensor
    on_opponent: torch.Tensor
    landing_xy: torch.Tensor


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(_EPS)


def _opponent_facing_normal(candidate: torch.Tensor) -> torch.Tensor:
    """Planner-compatible unit normal with a positive +x component."""

    fallback = torch.zeros_like(candidate)
    fallback[..., 0] = 1.0
    norm = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
    normal = torch.where(norm > 1.0e-9, candidate / norm.clamp_min(_EPS), fallback)
    normal = torch.where(normal[..., :1] < 0.0, -normal, normal)
    sideways = normal[..., :1] <= 1.0e-6
    return torch.where(sideways, _unit(normal + fallback), normal)


def relative_normal_speed(
    incoming_ball_vel: torch.Tensor,
    racket_vel: torch.Tensor,
    racket_normal: torch.Tensor,
) -> torch.Tensor:
    """Unsigned closing speed normal to a two-sided racket face.

    A racket normal has an arbitrary sign because either face can strike the
    ball.  At the virtual interception point, the physically relevant quantity
    is therefore ``abs((v_ball - v_racket) dot n)``.  Unlike the old
    racket-to-target direction gate, this remains well-defined at zero position
    error.
    """

    normal = _unit(racket_normal)
    relative = incoming_ball_vel - racket_vel
    return torch.abs(torch.sum(relative * normal, dim=-1))


def orient_normal(
    racket_normal: torch.Tensor,
    incoming_ball_vel: torch.Tensor,
    racket_vel: torch.Tensor,
) -> torch.Tensor:
    """Match the planner's sign convention for the paddle normal."""

    normal = _unit(racket_normal)
    relative = incoming_ball_vel - racket_vel
    flip = torch.sum(relative * normal, dim=-1, keepdim=True) > 0.0
    return torch.where(flip, -normal, normal)


def predict_paddle_contact(
    incoming_ball_vel: torch.Tensor,
    racket_vel: torch.Tensor,
    racket_normal: torch.Tensor,
    params: ReturnPhysicsParams,
) -> torch.Tensor:
    """Predict outgoing ball velocity with the planner's no-spin impulse model."""

    normal = orient_normal(racket_normal, incoming_ball_vel, racket_vel)
    relative = incoming_ball_vel - racket_vel
    normal_signed = torch.sum(relative * normal, dim=-1, keepdim=True)
    tangent = relative - normal_signed * normal
    tangent_mag = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    normal_abs = torch.abs(normal_signed)
    relative_mag = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)

    cos_theta = normal_abs / relative_mag.clamp_min(_EPS)
    raw = (
        params.paddle_tangential_damping
        + params.paddle_tangential_angle_coupling * cos_theta
    ) * tangent_mag
    cap = params.paddle_tangential_cap * (1.0 + params.paddle_restitution) * normal_abs
    impulse_tangent = torch.minimum(torch.clamp_min(raw, 0.0), cap)
    tangent_direction = tangent / tangent_mag.clamp_min(_EPS)
    delta_tangent = -impulse_tangent * tangent_direction
    delta_normal = -(1.0 + params.paddle_restitution) * normal_signed * normal
    return incoming_ball_vel + delta_normal + delta_tangent


def flight_acceleration(velocity: torch.Tensor, params: ReturnPhysicsParams) -> torch.Tensor:
    """Quadratic drag plus gravity."""

    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    speed = torch.clamp(speed, max=params.velocity_clip)
    acceleration = -params.drag_k * speed * velocity
    gravity = torch.zeros_like(acceleration)
    gravity[..., 2] = -params.gravity
    return acceleration + gravity


def integrate_free_flight(
    start_pos: torch.Tensor,
    initial_vel: torch.Tensor,
    duration: float,
    params: ReturnPhysicsParams,
    dt: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a fixed duration with the same constant-acceleration step as the planner."""

    pos = start_pos.clone()
    vel = initial_vel.clone()
    remaining = float(duration)
    while remaining > 1.0e-12:
        step = min(float(dt), remaining)
        acceleration = flight_acceleration(vel, params)
        pos = pos + vel * step + 0.5 * acceleration * step * step
        vel = vel + acceleration * step
        remaining -= step
    return pos, vel


def solve_outgoing_velocity(
    strike_pos: torch.Tensor,
    landing_pos: torch.Tensor,
    flight_time: float,
    params: ReturnPhysicsParams,
    integration_dt: float = 0.01,
    iterations: int = 8,
) -> torch.Tensor:
    """Batched finite-difference Newton solve matching the planner's drag shooting method."""

    if flight_time <= 0.0:
        raise ValueError("flight_time must be positive")
    if strike_pos.shape != landing_pos.shape or strike_pos.shape[-1] != 3:
        raise ValueError("strike_pos and landing_pos must both have shape (..., 3)")

    acceleration = torch.zeros_like(strike_pos)
    acceleration[..., 2] = -params.gravity
    velocity = (landing_pos - strike_pos) / flight_time - 0.5 * acceleration * flight_time

    for _ in range(int(iterations)):
        epsilon = 1.0e-4 * torch.maximum(torch.ones_like(velocity), torch.abs(velocity))
        variants = velocity.unsqueeze(-2).expand(*velocity.shape[:-1], 4, 3).clone()
        for axis in range(3):
            variants[..., axis + 1, axis] += epsilon[..., axis]
        repeated_start = strike_pos.unsqueeze(-2).expand_as(variants)
        reached_variants, _ = integrate_free_flight(
            repeated_start.reshape(-1, 3),
            variants.reshape(-1, 3),
            flight_time,
            params,
            dt=integration_dt,
        )
        reached_variants = reached_variants.reshape(*velocity.shape[:-1], 4, 3)
        reached = reached_variants[..., 0, :]
        residual = reached - landing_pos

        columns = []
        for axis in range(3):
            columns.append(
                (reached_variants[..., axis + 1, :] - reached)
                / epsilon[..., axis].unsqueeze(-1)
            )
        jacobian = torch.stack(columns, dim=-1)
        try:
            step = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            step = torch.linalg.lstsq(jacobian, residual.unsqueeze(-1)).solution.squeeze(-1)
        step_norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        step = step * torch.clamp(5.0 / step_norm.clamp_min(_EPS), max=1.0)
        velocity = velocity - step
    return velocity


def plan_racket_command(
    strike_pos: torch.Tensor,
    incoming_ball_vel: torch.Tensor,
    landing_pos: torch.Tensor,
    flight_time: float,
    params: ReturnPhysicsParams,
    integration_dt: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Planner-compatible target racket velocity, face normal, and desired ball velocity."""

    outgoing = solve_outgoing_velocity(
        strike_pos,
        landing_pos,
        flight_time,
        params,
        integration_dt=integration_dt,
    )
    delta = outgoing - incoming_ball_vel
    normal = _opponent_facing_normal(delta)

    outgoing_n = torch.sum(outgoing * normal, dim=-1, keepdim=True)
    incoming_n = torch.sum(incoming_ball_vel * normal, dim=-1, keepdim=True)
    racket_n = (
        outgoing_n + params.paddle_restitution * incoming_n
    ) / (1.0 + params.paddle_restitution)
    racket_vel = racket_n * normal

    # Refine the normal-only initial guess against the complete contact model.
    # Direction(racket_vel) remains the face normal so the deploy message does
    # not need a separate orientation field.
    for _ in range(12):
        normal = _opponent_facing_normal(racket_vel)
        epsilon = 1.0e-4 * torch.maximum(torch.ones_like(racket_vel), torch.abs(racket_vel))
        variants = racket_vel.unsqueeze(-2).expand(*racket_vel.shape[:-1], 4, 3).clone()
        for axis in range(3):
            variants[..., axis + 1, axis] += epsilon[..., axis]
        variant_normals = _opponent_facing_normal(variants)
        variant_incoming = incoming_ball_vel.unsqueeze(-2).expand_as(variants)
        predicted_variants = predict_paddle_contact(
            variant_incoming.reshape(-1, 3),
            variants.reshape(-1, 3),
            variant_normals.reshape(-1, 3),
            params,
        ).reshape(*racket_vel.shape[:-1], 4, 3)
        predicted = predicted_variants[..., 0, :]
        residual = predicted - outgoing
        columns = []
        for axis in range(3):
            columns.append(
                (predicted_variants[..., axis + 1, :] - predicted)
                / epsilon[..., axis].unsqueeze(-1)
            )
        jacobian = torch.stack(columns, dim=-1)
        try:
            step = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            step = torch.linalg.lstsq(
                jacobian, residual.unsqueeze(-1)
            ).solution.squeeze(-1)
        step_norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        step = step * torch.clamp(1.0 / step_norm.clamp_min(_EPS), max=1.0)
        candidate = racket_vel - step
        racket_vel = torch.where(
            torch.isfinite(candidate).all(dim=-1, keepdim=True), candidate, racket_vel
        )
    normal = _opponent_facing_normal(racket_vel)
    return racket_vel, normal, outgoing


def integrate_return(
    start_pos: torch.Tensor,
    outgoing_vel: torch.Tensor,
    params: ReturnPhysicsParams,
    dt: float = 0.01,
    max_time: float = 3.0,
) -> TorchReturnOutcome:
    """Roll out to the first table-height crossing and evaluate return geometry."""

    if start_pos.ndim != 2 or start_pos.shape[-1] != 3:
        raise ValueError("start_pos and outgoing_vel must have shape (N, 3)")
    pos = start_pos.clone()
    vel = outgoing_vel.clone()
    count = start_pos.shape[0]
    device = start_pos.device
    net_crossed = torch.zeros(count, dtype=torch.bool, device=device)
    net_clear = torch.zeros_like(net_crossed)
    bounced = torch.zeros_like(net_crossed)
    landing_xy = torch.full((count, 2), float("nan"), dtype=start_pos.dtype, device=device)
    surface_z = params.ball_radius
    net_top = params.net_height + params.ball_radius

    steps = int(math.ceil(float(max_time) / float(dt)))
    for step_index in range(steps):
        step = min(float(dt), float(max_time) - step_index * float(dt))
        acceleration = flight_acceleration(vel, params)
        pos_new = pos + vel * step + 0.5 * acceleration * step * step
        vel_new = vel + acceleration * step
        active = ~bounced

        crosses_net = (
            active
            & (~net_crossed)
            & (pos[:, 0] < params.net_x)
            & (pos_new[:, 0] >= params.net_x)
        )
        dx = pos_new[:, 0] - pos[:, 0]
        net_fraction = (params.net_x - pos[:, 0]) / torch.where(
            torch.abs(dx) > 1.0e-12, dx, torch.ones_like(dx)
        )
        net_z = pos[:, 2] + net_fraction * (pos_new[:, 2] - pos[:, 2])
        net_crossed = net_crossed | crosses_net
        net_clear = net_clear | (crosses_net & (net_z > net_top))

        crosses_surface = (
            active
            & (pos[:, 2] > surface_z)
            & (pos_new[:, 2] <= surface_z)
            & (vel[:, 2] < 0.0)
        )
        dz = pos_new[:, 2] - pos[:, 2]
        bounce_fraction = (surface_z - pos[:, 2]) / torch.where(
            torch.abs(dz) > 1.0e-12, dz, torch.ones_like(dz)
        )
        land_x = pos[:, 0] + bounce_fraction * (pos_new[:, 0] - pos[:, 0])
        land_y = pos[:, 1] + bounce_fraction * (pos_new[:, 1] - pos[:, 1])
        candidate_xy = torch.stack((land_x, land_y), dim=-1)
        landing_xy = torch.where(crosses_surface.unsqueeze(-1), candidate_xy, landing_xy)
        bounced = bounced | crosses_surface

        pos = torch.where(bounced.unsqueeze(-1), pos, pos_new)
        vel = torch.where(bounced.unsqueeze(-1), vel, vel_new)

    on_opponent = (
        bounced
        & (landing_xy[:, 0] > params.net_x)
        & (landing_xy[:, 0] <= params.table_length)
        & (landing_xy[:, 1] >= -params.table_width)
        & (landing_xy[:, 1] <= 0.0)
    )
    return TorchReturnOutcome(
        net_crossed=net_crossed,
        net_clear=net_clear,
        on_opponent=on_opponent,
        landing_xy=landing_xy,
    )
