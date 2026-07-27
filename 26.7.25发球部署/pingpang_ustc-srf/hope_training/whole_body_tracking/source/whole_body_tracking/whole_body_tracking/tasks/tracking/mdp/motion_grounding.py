"""Dependency-light helpers for grounding concatenated motion clips."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def apply_segment_height_offsets(
    body_pos_w: torch.Tensor,
    seg_start: torch.Tensor,
    seg_len: torch.Tensor,
    offsets: Sequence[float],
) -> None:
    """Add one constant world-z offset to every body in each motion segment.

    A constant translation preserves all relative poses and velocities. The
    operation is intentionally in-place so the command's loaded motion tensors
    become the single grounded reference used by imitation and RSI.
    """

    if not offsets:
        return
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError("body_pos_w must have shape (frames, bodies, 3)")
    if len(offsets) != int(seg_len.numel()):
        raise ValueError(
            "ground_height_offset_per_clip must have one value per motion clip: "
            f"got {len(offsets)} offsets for {int(seg_len.numel())} clips"
        )

    for clip_index, raw_offset in enumerate(offsets):
        offset = float(raw_offset)
        if not torch.isfinite(torch.tensor(offset)):
            raise ValueError(f"non-finite ground height offset for clip {clip_index}")
        start = int(seg_start[clip_index].item())
        stop = start + int(seg_len[clip_index].item())
        body_pos_w[start:stop, :, 2].add_(offset)
