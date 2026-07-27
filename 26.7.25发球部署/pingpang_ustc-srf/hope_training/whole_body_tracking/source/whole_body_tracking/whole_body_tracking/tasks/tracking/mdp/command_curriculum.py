"""Pure helpers for motion-reset filtering and command curricula.

This module intentionally has no Isaac Sim imports.  Keeping the numerical pieces
here makes their boundary behaviour deterministic and cheap to unit test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def linear_curriculum_progress(step: int, start_step: int, duration_steps: int) -> float:
    """Return linear curriculum progress in ``[0, 1]``.

    A non-positive duration disables the curriculum and therefore exposes the
    full task immediately.
    """

    if duration_steps <= 0:
        return 1.0
    return min(max((int(step) - int(start_step)) / float(duration_steps), 0.0), 1.0)


def curriculum_fraction(initial_fraction: float, progress: float) -> float:
    """Interpolate a box-width fraction from its initial value to full width."""

    initial = min(max(float(initial_fraction), 0.0), 1.0)
    phase = min(max(float(progress), 0.0), 1.0)
    return initial + (1.0 - initial) * phase


def contract_box_about_center(box: torch.Tensor, fraction: float) -> torch.Tensor:
    """Contract ``[..., axis, (lo, hi)]`` ranges about their midpoint."""

    if box.ndim < 2 or box.shape[-1] != 2:
        raise ValueError(f"Expected a [..., 2] range tensor, got shape {tuple(box.shape)}")
    frac = min(max(float(fraction), 0.0), 1.0)
    center = 0.5 * (box[..., 0] + box[..., 1])
    half_width = 0.5 * (box[..., 1] - box[..., 0]) * frac
    return torch.stack((center - half_width, center + half_width), dim=-1)


def stable_rsi_frame_indices(
    *,
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    seg_start: torch.Tensor,
    seg_len: torch.Tensor,
    foot_body_indices: Sequence[int],
    root_body_index: int,
    phase_range: tuple[float, float],
    max_foot_height_delta: float,
    max_foot_speed: float,
    max_root_lin_speed: float,
    max_root_ang_speed: float,
) -> tuple[torch.Tensor, ...]:
    """Select low-motion, approximately dual-support RSI frames per clip.

    Reference files do not contain contact impulses.  Equal foot-link heights
    plus low foot/root speeds are a conservative kinematic proxy for two-foot
    support.  The caller decides whether an empty result is an error or should
    fall back to the segment's first frame.
    """

    if len(foot_body_indices) != 2:
        raise ValueError("Stable RSI filtering requires exactly two foot body indices")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError("body_pos_w must have shape [T, B, 3]")
    if body_lin_vel_w.shape != body_pos_w.shape or body_ang_vel_w.shape != body_pos_w.shape:
        raise ValueError("body velocity tensors must match body_pos_w")
    if seg_start.ndim != 1 or seg_len.shape != seg_start.shape:
        raise ValueError("seg_start and seg_len must be equal-length 1-D tensors")

    phase_lo, phase_hi = (float(phase_range[0]), float(phase_range[1]))
    if not (0.0 <= phase_lo <= phase_hi <= 1.0):
        raise ValueError(f"rsi_phase_range must satisfy 0 <= lo <= hi <= 1, got {phase_range}")
    thresholds = (
        max_foot_height_delta,
        max_foot_speed,
        max_root_lin_speed,
        max_root_ang_speed,
    )
    if any(float(value) < 0.0 for value in thresholds):
        raise ValueError("Stable RSI thresholds must be non-negative")

    left_idx, right_idx = (int(foot_body_indices[0]), int(foot_body_indices[1]))
    root_idx = int(root_body_index)
    result: list[torch.Tensor] = []
    for start_tensor, length_tensor in zip(seg_start, seg_len, strict=True):
        start = int(start_tensor.item())
        length = int(length_tensor.item())
        if length <= 0:
            result.append(torch.empty(0, dtype=torch.long, device=body_pos_w.device))
            continue

        last_offset = length - 1
        first = start + int(math.ceil(phase_lo * last_offset))
        last = start + int(math.floor(phase_hi * last_offset))
        frames = torch.arange(first, last + 1, dtype=torch.long, device=body_pos_w.device)
        if frames.numel() == 0:
            result.append(frames)
            continue

        foot_height_delta = torch.abs(
            body_pos_w[frames, left_idx, 2] - body_pos_w[frames, right_idx, 2]
        )
        foot_speed = torch.linalg.vector_norm(
            body_lin_vel_w[frames][:, (left_idx, right_idx), :], dim=-1
        ).amax(dim=-1)
        root_lin_speed = torch.linalg.vector_norm(body_lin_vel_w[frames, root_idx], dim=-1)
        root_ang_speed = torch.linalg.vector_norm(body_ang_vel_w[frames, root_idx], dim=-1)
        stable = (
            (foot_height_delta <= float(max_foot_height_delta))
            & (foot_speed <= float(max_foot_speed))
            & (root_lin_speed <= float(max_root_lin_speed))
            & (root_ang_speed <= float(max_root_ang_speed))
        )
        result.append(frames[stable])
    return tuple(result)
