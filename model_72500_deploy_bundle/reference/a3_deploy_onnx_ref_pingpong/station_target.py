# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Derive the lateral base station for a planner strike command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPECTED_BASE_STATION_SEMANTICS = "moving_lateral_target_v1"


def validate_lateral_station_policy_manifest(onnx_path: str | Path) -> None:
    """Reject a shape-compatible actor trained with the old fixed-station input."""

    policy_path = Path(onnx_path).resolve()
    manifest_path = policy_path.with_name("policy_manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(
            "moving lateral station is enabled, but policy_manifest.json is missing next "
            f"to {policy_path}; export the new footwork policy and manifest together"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    declared_file = manifest.get("onnx_file")
    if declared_file and Path(str(declared_file)).name != policy_path.name:
        raise RuntimeError(
            f"{manifest_path} describes {declared_file!r}, not {policy_path.name!r}"
        )
    semantics = manifest.get("base_station_semantics")
    if semantics != EXPECTED_BASE_STATION_SEMANTICS:
        raise RuntimeError(
            "the selected actor was not exported for moving-base station semantics: "
            f"expected {EXPECTED_BASE_STATION_SEMANTICS!r}, got {semantics!r}. "
            "Retrain/export the footwork policy; do not reuse the fixed-station actor."
        )


def derive_lateral_base_target(
    target_pos_w: np.ndarray,
    swing_side: float,
    nominal_station_xy: np.ndarray,
    *,
    base_target_y_range: tuple[float, float],
    forehand_reach_y: float,
    backhand_reach_y: float,
) -> np.ndarray:
    """Return the fixed-x, moving-y station associated with a strike target.

    Training samples a comfortable racket reach relative to the commanded base.
    Deployment performs the inverse operation using the centre reach of the
    selected clip, then clamps the lateral station to the trained range.
    """

    target = np.asarray(target_pos_w, dtype=np.float64)
    nominal = np.asarray(nominal_station_xy, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError("target_pos_w must have shape (3,)")
    if nominal.shape != (2,):
        raise ValueError("nominal_station_xy must have shape (2,)")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(nominal)):
        raise ValueError("station geometry inputs must be finite")

    y_min, y_max = (float(value) for value in base_target_y_range)
    if not np.isfinite([y_min, y_max]).all() or y_min > y_max:
        raise ValueError("base_target_y_range must contain finite min <= max")
    reach_y = float(forehand_reach_y if swing_side >= 0.0 else backhand_reach_y)
    if not np.isfinite(reach_y):
        raise ValueError("clip reach_y must be finite")

    desired_offset_y = target[1] - nominal[1] - reach_y
    station = nominal.copy()
    station[1] += np.clip(desired_offset_y, y_min, y_max)
    return station
