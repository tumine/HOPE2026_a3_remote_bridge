"""HOPE racket-task and lower-body stability reward terms.

The task-specific functions here cover whole-body imitation, racket tracking, return outcome,
in-place recovery, and always-on base/foot stability. Generic termination, root-velocity,
action, acceleration, and joint-limit penalties reuse ``isaaclab.envs.mdp`` terms directly.
Weights and kernel widths live in the environment config.

The racket position/velocity/blade terms are active only in a short window around the strike; the
contact/net/bounce terms fire once at the exact strike frame; imitation is active during the swing
(not the frozen pre-swing hold); recovery is active through the follow-through and the hold.

The stability helpers at the bottom of this module are deliberately independent of the racket
command.  They can therefore be used during every phase of an episode.  In particular,
``both_feet_contact`` is binary: one planted foot never receives half credit, while
``foot_slip_l2`` is contact-gated so an airborne swing foot is not penalized for moving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from whole_body_tracking.tasks.tracking.mdp import rewards as _imitation
from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


# --- (2) forehand/backhand sample imitation ------------------------------------------------- #
def sample_imitation(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_pos: float = 0.3,
    std_ori: float = 0.4,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Track the selected bodies from the imitated clip during the swing.

    Combines the anchor-relative body position and orientation tracking kernels and gates them to the
    active swing (zero during the frozen pre-swing hold). ``body_names`` selects the tracked subset
    (the HOPE env passes pelvis, waist-connected torso, both legs, and both arms)."""
    rp = _imitation.motion_relative_body_position_error_exp(env, command_name, std_pos, body_names)
    ro = _imitation.motion_relative_body_orientation_error_exp(env, command_name, std_ori, body_names)
    motion = env.command_manager.get_term(command_name)
    return (0.5 * rp + 0.5 * ro) * (~motion.in_hold).float()


# --- (3,4,5) racket goal tracking, active in the strike window ------------------------------ #
def racket_position(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the racket center against the target's swing-through point near the strike."""
    cmd = _cmd(env, command_name)
    target_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_now), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_window.float()


def racket_velocity(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the racket linear velocity against the desired velocity near the strike."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_window.float()


def racket_blade_direction(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Align the racket face normal with the desired blade direction near the strike (``std`` in rad)."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    return torch.exp(-(angle**2) / std**2) * cmd.strike_window.float()


# --- (6,7,8) no-spin return outcome, one-shot at the strike --------------------------------- #
def ball_contact(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """+1 on the strike frame when the racket actually contacts the target ball (near + approaching)."""
    return _cmd(env, command_name).ball_contact.float()


def ball_net_cross(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """+1 on the strike frame when the (no-spin) outgoing ball clears the net."""
    return _cmd(env, command_name).ball_net_cross.float()


def ball_opponent_bounce(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """+1 on the strike frame when the outgoing ball's first bounce lands on the opponent half."""
    return _cmd(env, command_name).ball_on_opponent.float()


# --- (9) in-place follow-through / recovery ------------------------------------------------- #
def follow_through_recovery(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.5, station_std: float = 0.3
) -> torch.Tensor:
    """Reward settling calmly AT the fixed station through the follow-through and the pre-swing hold.

    ``exp(-(|v_base_xy|/std)^2) * exp(-(station_err/station_std)^2) * feet_contact_frac`` active in the
    follow-through ((~pre_strike) & (~strike_window)) and during the hold. This is in-place recentring
    and balance only — it never rewards walking, footstep planning, or leaving the station.
    """
    cmd = _cmd(env, command_name)
    v_xy = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    calm = torch.exp(-torch.square(v_xy / std))
    station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.fixed_station_w, dim=-1)
    at_station = torch.exp(-torch.square(station_err / station_std))
    in_hold = cmd._motion().in_hold
    gate = ((~cmd.pre_strike) & (~cmd.strike_window)) | in_hold
    return calm * at_station * cmd.feet_contact_frac * gate.float()


# --- always-on lower-body / base stability ------------------------------------------------- #
def _normalized_entity_ids(entity_ids: Any) -> Any:
    """Keep entity selection rank-stable when a resolved selector contains one integer id."""
    if isinstance(entity_ids, int):
        return [entity_ids]
    if torch.is_tensor(entity_ids) and entity_ids.ndim == 0:
        return [int(entity_ids.item())]
    return entity_ids


def _selected_bodies(value: torch.Tensor, entity_ids: Any) -> torch.Tensor:
    """Select body columns from an ``[N, B, ...]`` tensor without squeezing ``B``."""
    return value[:, _normalized_entity_ids(entity_ids), ...]


def _require_body_count(value: torch.Tensor, expected: int | None, label: str) -> None:
    count = value.shape[1]
    if expected is not None and count != expected:
        raise ValueError(f"{label} requires exactly {expected} resolved bodies, got {count}")
    if count == 0:
        raise ValueError(f"{label} requires at least one resolved body")


def _foot_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Return latest-step contact flags for the bodies selected on a contact sensor."""
    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be non-negative, got {force_threshold}")
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = _selected_bodies(sensor.data.net_forces_w, sensor_cfg.body_ids)
    _require_body_count(forces, None, "foot contact")
    return torch.linalg.vector_norm(forces, dim=-1) > force_threshold


def _quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by Isaac's ``(w, x, y, z)`` quaternion convention."""
    quat_vector = quat[..., 1:]
    twice_cross = 2.0 * torch.cross(quat_vector, vector, dim=-1)
    return vector + quat[..., :1] * twice_cross + torch.cross(quat_vector, twice_cross, dim=-1)


def base_height_tracking_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward root height near ``target_height`` with a unit-height exponential kernel."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")
    asset = env.scene[asset_cfg.name]
    height_error = asset.data.root_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(height_error / std))


def base_planar_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize world-frame root translation in the ground plane."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1)


def both_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Return one only when both selected feet exceed ``force_threshold``.

    ``sensor_cfg`` must resolve exactly two foot bodies.  Failing fast here prevents an accidental
    full-body selector from silently turning this into an "all robot bodies in contact" reward.
    """
    contacts = _foot_contacts(env, sensor_cfg, force_threshold)
    _require_body_count(contacts, 2, "both_feet_contact")
    return torch.all(contacts, dim=-1).float()


def foot_slip_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize planar foot-link speed only while that foot is loaded.

    The body selectors on ``asset_cfg`` and ``sensor_cfg`` must resolve the same feet in the same
    order.  The per-foot values are averaged, keeping the reward scale invariant to selector size.
    """
    asset = env.scene[asset_cfg.name]
    velocity = _selected_bodies(asset.data.body_link_lin_vel_w, asset_cfg.body_ids)
    contacts = _foot_contacts(env, sensor_cfg, force_threshold)
    _require_body_count(velocity, None, "foot_slip_l2")
    if velocity.shape[1] != contacts.shape[1]:
        raise ValueError(
            "foot_slip_l2 asset/sensor selectors must resolve the same number of bodies, "
            f"got {velocity.shape[1]} and {contacts.shape[1]}"
        )
    planar_speed_sq = torch.sum(torch.square(velocity[..., :2]), dim=-1)
    return torch.mean(planar_speed_sq * contacts.to(planar_speed_sq.dtype), dim=-1)


def foot_flat_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sole_normal_axis: int = 2,
    sole_normal_sign: float = 1.0,
) -> torch.Tensor:
    """Penalize foot-sole tilt as mean ``1 - dot(normal, world_up)``.

    This form is zero for a flat sole, approximately ``angle**2 / 2`` near upright, and still
    penalizes an inverted foot (unlike a projected-XY-only orientation penalty).  The local sole
    normal axis/sign are configurable because imported foot frames are not universally ``+Z``.
    """
    if sole_normal_axis not in (0, 1, 2):
        raise ValueError(f"sole_normal_axis must be 0, 1, or 2, got {sole_normal_axis}")
    if sole_normal_sign == 0.0:
        raise ValueError("sole_normal_sign must be non-zero")
    asset = env.scene[asset_cfg.name]
    quat = _selected_bodies(asset.data.body_link_quat_w, asset_cfg.body_ids)
    _require_body_count(quat, None, "foot_flat_orientation")
    local_normal = torch.zeros_like(quat[..., :3])
    local_normal[..., sole_normal_axis] = 1.0 if sole_normal_sign > 0.0 else -1.0
    normal_w = _quat_apply_wxyz(quat, local_normal)
    normal_z = normal_w[..., 2].clamp(-1.0, 1.0)
    return torch.mean(1.0 - normal_z, dim=-1)


def feet_stance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float,
    midpoint_weight: float = 1.0,
    target_midpoint_xy_b: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Penalize foot separation error and an off-centre pelvis-to-feet midpoint.

    The two-foot width is the yaw-invariant XY distance.  ``target_midpoint_xy_b`` allows the
    desired support midpoint to have a fixed robot-frame offset from the root while remaining
    correct under arbitrary world yaw.
    """
    if target_width < 0.0:
        raise ValueError(f"target_width must be non-negative, got {target_width}")
    if midpoint_weight < 0.0:
        raise ValueError(f"midpoint_weight must be non-negative, got {midpoint_weight}")
    if len(target_midpoint_xy_b) != 2:
        raise ValueError("target_midpoint_xy_b must contain exactly two values")

    asset = env.scene[asset_cfg.name]
    feet_pos = _selected_bodies(asset.data.body_link_pos_w, asset_cfg.body_ids)
    _require_body_count(feet_pos, 2, "feet_stance_l2")

    separation = torch.linalg.vector_norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    width_error_sq = torch.square(separation - target_width)

    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    local_offset = root_pos.new_tensor(
        (target_midpoint_xy_b[0], target_midpoint_xy_b[1], 0.0)
    ).expand(root_pos.shape[0], -1)
    desired_midpoint = root_pos[:, :2] + _quat_apply_wxyz(root_quat, local_offset)[:, :2]
    actual_midpoint = torch.mean(feet_pos[..., :2], dim=1)
    midpoint_error_sq = torch.sum(torch.square(actual_midpoint - desired_midpoint), dim=-1)
    return width_error_sq + midpoint_weight * midpoint_error_sq
