# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Shared ActionAdapter: raw policy output -> joint-position targets.

The adapter is a pure, deterministic numeric transform (an optional raw-action
clip/transform, a per-column residual, then a joint clamp). It is NOT a rejection filter and
emits no failure status:

    adapter_action = transform(clip(raw_action))
    q_des = default_q + adapter_action * action_scale
    q_des = clip(q_des, clamp_lower, clamp_upper)

The SAME configuration file drives training and this reference runner, so a user
tunes the mapping in exactly one place. The shipped constants are neutral EXAMPLE
values (see ``config/action_adapter.yaml``) and must be tuned for a real robot.

Vendor hard limits, motor protection, and e-stop remain the responsibility of the
robot backend; this transform neither probes nor bypasses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .joint_order import JOINT_NAMES, NUM_JOINTS


@dataclass
class ActionAdapter:
    default_q: np.ndarray     # (31,) neutral/upright joint positions, rad
    action_scale: np.ndarray  # (31,) per-column residual scale
    clamp_lower: np.ndarray   # (31,) lower joint-position clamp, rad
    clamp_upper: np.ndarray   # (31,) upper joint-position clamp, rad
    raw_action_transform: str = "identity"
    action_clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for field in ("default_q", "action_scale", "clamp_lower", "clamp_upper"):
            v = np.asarray(getattr(self, field), dtype=np.float64).reshape(-1)
            if v.shape[0] != NUM_JOINTS:
                raise ValueError(f"{field} must be length {NUM_JOINTS}, got {v.shape[0]}")
            setattr(self, field, v)
        if np.any(self.clamp_lower > self.clamp_upper):
            raise ValueError("action_adapter clamp_lower must be <= clamp_upper for every joint")
        self.raw_action_transform = _resolve_raw_action_transform(self.raw_action_transform)
        self.action_clip = _resolve_action_clip(self.action_clip)

    def clip_raw_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Clip actor output to the configured applied-action interval."""
        raw = np.asarray(raw_action, dtype=np.float64)
        if raw.ndim == 0 or raw.shape[-1] != NUM_JOINTS:
            raise ValueError(
                f"raw_action trailing dimension must be {NUM_JOINTS}, got {raw.shape}"
            )
        if self.action_clip is None:
            return raw.copy()
        return np.clip(raw, self.action_clip[0], self.action_clip[1])

    def transform_raw_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Map actor output into the normalized residual domain.

        The runner separately calls :meth:`clip_raw_action` when forming
        ``last_action`` so feedback and target decoding use the same accepted value.
        """
        raw = self.clip_raw_action(raw_action)
        if self.raw_action_transform == "identity":
            return raw.copy()
        if self.raw_action_transform == "tanh":
            return np.tanh(raw)
        raise ValueError(f"unsupported raw_action_transform: {self.raw_action_transform!r}")

    def decode(self, raw_action: np.ndarray) -> np.ndarray:
        """Map ``raw_action[31]`` to 31 clamped joint-position targets."""
        raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if raw.shape[0] != NUM_JOINTS:
            raise ValueError(f"raw_action must be length {NUM_JOINTS}, got {raw.shape[0]}")
        adapter_action = self.transform_raw_action(raw)
        q_des = self.default_q + adapter_action * self.action_scale
        return np.clip(q_des, self.clamp_lower, self.clamp_upper)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ActionAdapter":
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)

        default_q = _resolve_per_joint(doc["default_q"], "default_q")
        raw_action_transform = _resolve_raw_action_transform(doc.get("raw_action_transform"))
        action_clip = _resolve_action_clip(doc.get("action_clip"))

        scale_spec = doc["action_scale"]
        if isinstance(scale_spec, (int, float)):
            action_scale = np.full(NUM_JOINTS, float(scale_spec), dtype=np.float64)
        else:
            action_scale = _resolve_per_joint(scale_spec, "action_scale")

        clamp = doc["joint_position_clamp"]
        clamp_lower = _resolve_per_joint(clamp["lower"], "joint_position_clamp.lower")
        clamp_upper = _resolve_per_joint(clamp["upper"], "joint_position_clamp.upper")
        return cls(
            default_q=default_q,
            action_scale=action_scale,
            clamp_lower=clamp_lower,
            clamp_upper=clamp_upper,
            raw_action_transform=raw_action_transform,
            action_clip=action_clip,
        )


def _resolve_per_joint(spec, field_name: str) -> np.ndarray:
    """Accept either an ordered length-31 list or a ``{joint_name: value}`` map."""
    if isinstance(spec, dict):
        missing = [n for n in JOINT_NAMES if n not in spec]
        if missing:
            raise ValueError(f"{field_name} is missing joints: {missing[:3]}...")
        return np.array([float(spec[n]) for n in JOINT_NAMES], dtype=np.float64)
    arr = np.asarray(spec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != NUM_JOINTS:
        raise ValueError(f"{field_name} must be length {NUM_JOINTS}, got {arr.shape[0]}")
    return arr


def _resolve_raw_action_transform(spec) -> str:
    """Normalize the optional transform; omitted means the legacy affine adapter."""
    transform = "identity" if spec is None else str(spec).strip().lower()
    if transform not in {"identity", "tanh"}:
        raise ValueError(
            "raw_action_transform must be one of ['identity', 'tanh'], "
            f"got {spec!r}"
        )
    return transform


def _resolve_action_clip(spec) -> tuple[float, float] | None:
    """Normalize an optional global ``[lower, upper]`` raw-action clip."""
    if spec is None:
        return None
    values = np.asarray(spec, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("action_clip must contain two finite values: [lower, upper]")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError("action_clip lower must be <= upper")
    return lower, upper
