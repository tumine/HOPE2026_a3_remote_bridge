"""Dependency-light strike timing helpers shared by the HOPE command and tests."""

from __future__ import annotations

import torch


def compute_strike_timing(
    *,
    seg_start: torch.Tensor,
    seg_len: torch.Tensor,
    strike_phase: torch.Tensor,
    time_steps: torch.Tensor,
    step_dt: float,
    strike_window_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(strike_step, tts, pre_strike, in_window)`` for each environment.

    ``time_steps`` is an absolute index into the concatenated motion library. The
    strike phase is local to each segment and is rounded to the nearest stored frame.
    """
    strike_step = seg_start + (strike_phase * (seg_len - 1).float()).round().long()
    time_to_strike = (strike_step - time_steps).float() * float(step_dt)
    pre_strike = time_to_strike > 0.0
    strike_window = time_to_strike.abs() <= float(strike_window_s)
    return strike_step, time_to_strike, pre_strike, strike_window


def consume_exact_strike(
    time_to_strike: torch.Tensor,
    *,
    step_dt: float,
    armed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a one-shot exact-strike mask and the updated armed latch.

    The half-step tolerance assigns the discrete event to the closest control
    sample. ``armed`` makes repeated evaluation of that sample idempotent.
    """
    exact = (time_to_strike.abs() <= (0.5 * float(step_dt) + 1.0e-6)) & armed
    return exact, armed & ~exact
