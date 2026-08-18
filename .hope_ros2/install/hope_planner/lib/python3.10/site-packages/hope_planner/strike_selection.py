"""Side-aware strike-region selection for the live planner.

The policy was trained on disjoint forehand/backhand racket-position boxes. A
single global strike plane cannot represent both boxes, so the live planner
predicts one crossing at the midpoint of each side's x range. This module
accepts a crossing only when its full position lies inside the corresponding
training box. It deliberately never clips a physical prediction into range:
an unreachable/OOD ball produces no command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from .ball_trajectory_predictor import StrikeTarget
from .side_selection import BACKHAND, FOREHAND, select_swing_side


def incoming_plane_crossing(
    previous: np.ndarray,
    current: np.ndarray,
    plane_x: float,
) -> Optional[np.ndarray]:
    """Interpolate one measured -X crossing of ``plane_x``.

    This helper is diagnostic only: it reports when the measured ball segment
    actually passes a configured strike plane.  It does not participate in
    trajectory prediction or command admission.
    """

    p0 = np.asarray(previous, dtype=np.float64)
    p1 = np.asarray(current, dtype=np.float64)
    x_hit = float(plane_x)
    if (
        p0.shape != (3,)
        or p1.shape != (3,)
        or not np.all(np.isfinite(p0))
        or not np.all(np.isfinite(p1))
        or not np.isfinite(x_hit)
    ):
        return None
    # Incoming means decreasing canonical X.  Report each exact plane only on
    # the transition from the opponent side to the robot side.
    if not (p0[0] > x_hit and p1[0] <= x_hit):
        return None
    dx = p0[0] - p1[0]
    if dx <= 1.0e-12:
        return None
    alpha = float(np.clip((p0[0] - x_hit) / dx, 0.0, 1.0))
    crossing = p0 + alpha * (p1 - p0)
    crossing[0] = x_hit
    return crossing


@dataclass(frozen=True)
class StrikeRegion:
    """Closed axis-aligned racket-target box in the planner world frame."""

    side: int
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    velocity: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = None

    def __post_init__(self) -> None:
        if self.side not in (FOREHAND, BACKHAND):
            raise ValueError(f"invalid swing side {self.side}; expected +1 or -1")
        for axis, limits in (("x", self.x), ("y", self.y), ("z", self.z)):
            values = np.asarray(limits, dtype=np.float64)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{axis} limits must be two finite values, got {limits}")
            if not values[0] < values[1]:
                raise ValueError(f"{axis} limits must satisfy min < max, got {limits}")
        if self.velocity is not None:
            values = np.asarray(self.velocity, dtype=np.float64)
            if values.shape != (3, 2) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "velocity limits must be a finite (3,2) xyz interval array, "
                    f"got {self.velocity}"
                )
            if not np.all(values[:, 0] < values[:, 1]):
                raise ValueError(
                    f"velocity limits must satisfy min < max per axis, got {self.velocity}"
                )

    @property
    def x_hit(self) -> float:
        """Strike plane at the centre of this side's trained x interval."""

        return 0.5 * (float(self.x[0]) + float(self.x[1]))

    @staticmethod
    def _margin_vector(margin, label: str) -> np.ndarray:
        values = np.asarray(margin, dtype=np.float64)
        if values.shape == ():
            values = np.full(3, float(values), dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} margin must be one finite value or a finite xyz vector")
        if np.any(values < 0.0):
            raise ValueError(f"{label} margin must be non-negative")
        return values

    def contains(
        self,
        position: np.ndarray,
        *,
        margin=(0.0, 0.0, 0.0),
        atol: float = 1.0e-9,
    ) -> bool:
        p = np.asarray(position, dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return False
        margins = self._margin_vector(margin, "position")
        return (
            self.x[0] - margins[0] - atol <= p[0] <= self.x[1] + margins[0] + atol
            and self.y[0] - margins[1] - atol <= p[1] <= self.y[1] + margins[1] + atol
            and self.z[0] - margins[2] - atol <= p[2] <= self.z[1] + margins[2] + atol
        )

    def contains_velocity(
        self,
        velocity: np.ndarray,
        *,
        margin=0.0,
        atol: float = 1.0e-9,
    ) -> bool:
        """True only for finite velocities inside this side's training envelope."""

        if self.velocity is None:
            return False
        v = np.asarray(velocity, dtype=np.float64)
        limits = np.asarray(self.velocity, dtype=np.float64)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            return False
        margins = self._margin_vector(margin, "velocity")
        return bool(
            np.all(v >= limits[:, 0] - margins - atol)
            and np.all(v <= limits[:, 1] + margins + atol)
        )

    def contains_command(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        position_margin=(0.0, 0.0, 0.0),
        velocity_margin=0.0,
        atol: float = 1.0e-9,
    ) -> bool:
        return self.contains(position, margin=position_margin, atol=atol) and self.contains_velocity(
            velocity, margin=velocity_margin, atol=atol
        )

    def rejection_reason(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        *,
        position_margin=(0.0, 0.0, 0.0),
        velocity_margin=0.0,
    ) -> str:
        """Human-readable diagnostics for an out-of-region prediction."""

        p = np.asarray(position, dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return f"non-finite/invalid position {p}"
        position_margins = self._margin_vector(position_margin, "position")
        violations = []
        for index, (axis, configured_limits) in enumerate(
            (("x", self.x), ("y", self.y), ("z", self.z))
        ):
            limits = (
                configured_limits[0] - position_margins[index],
                configured_limits[1] + position_margins[index],
            )
            if not limits[0] <= p[index] <= limits[1]:
                violations.append(
                    f"{axis}={p[index]:.4f} outside [{limits[0]:.4f}, {limits[1]:.4f}]"
                )
        if velocity is not None:
            v = np.asarray(velocity, dtype=np.float64)
            if v.shape != (3,) or not np.all(np.isfinite(v)):
                violations.append(f"non-finite/invalid velocity {v}")
            elif self.velocity is None:
                violations.append("velocity envelope is not configured")
            else:
                velocity_margins = self._margin_vector(velocity_margin, "velocity")
                for index, (axis, configured_limits) in enumerate(zip("xyz", self.velocity)):
                    limits = (
                        configured_limits[0] - velocity_margins[index],
                        configured_limits[1] + velocity_margins[index],
                    )
                    if not limits[0] <= v[index] <= limits[1]:
                        violations.append(
                            f"v{axis}={v[index]:.4f} outside "
                            f"[{limits[0]:.4f}, {limits[1]:.4f}]"
                        )
        if violations:
            return ", ".join(violations)
        return "inside command envelope" if velocity is not None else "inside position region"


def validate_side_regions(
    regions: Mapping[int, StrikeRegion],
    split_y: float,
    hysteresis_y: float = 0.0,
) -> None:
    """Reject a configuration whose split cannot classify both trained boxes."""

    if set(regions) != {FOREHAND, BACKHAND}:
        raise ValueError("strike regions must contain exactly FOREHAND (+1) and BACKHAND (-1)")
    fh = regions[FOREHAND]
    bh = regions[BACKHAND]
    if fh.side != FOREHAND or bh.side != BACKHAND:
        raise ValueError("strike-region mapping keys must match each region's side")
    split_y = float(split_y)
    hysteresis_y = max(0.0, float(hysteresis_y))
    if not np.isfinite(split_y):
        raise ValueError("swing_side_split_y must be finite")
    tolerance = 1.0e-12
    if fh.y[1] > split_y - hysteresis_y + tolerance:
        raise ValueError(
            "forehand y range overlaps split-hysteresis: "
            f"max={fh.y[1]}, split-h={split_y - hysteresis_y}"
        )
    if bh.y[0] < split_y + hysteresis_y - tolerance:
        raise ValueError(
            "backhand y range overlaps split+hysteresis: "
            f"min={bh.y[0]}, split+h={split_y + hysteresis_y}"
        )


def select_strike_candidate(
    candidates: Mapping[int, StrikeTarget],
    regions: Mapping[int, StrikeRegion],
    *,
    split_y: float,
    hysteresis_y: float = 0.0,
    previous_side: int = 0,
    locked_side: int = 0,
    position_margin=(0.0, 0.0, 0.0),
) -> Optional[tuple[int, StrikeTarget]]:
    """Select an in-distribution candidate, or return ``None``.

    ``locked_side`` is strict: once a rally task has chosen a side, revisions
    may update only that side and can never flip it. For a fresh task, bounds
    and the configured lateral split must both agree. If an unusual trajectory
    is valid in both disjoint regions, the previous side wins when available;
    otherwise the earlier strike is selected.
    """

    valid = {}
    for side, strike in candidates.items():
        region = regions.get(side)
        if (
            region is None
            or not strike.valid
            or not region.contains(strike.p_ball, margin=position_margin)
        ):
            continue
        classified = select_swing_side(
            float(strike.p_ball[1]), split_y, hysteresis_y, previous_side
        )
        if classified == side:
            valid[side] = strike

    if locked_side in (FOREHAND, BACKHAND):
        strike = valid.get(locked_side)
        return (locked_side, strike) if strike is not None else None
    if not valid:
        return None
    if previous_side in valid:
        return previous_side, valid[previous_side]
    side, strike = min(valid.items(), key=lambda item: item[1].t_strike)
    return side, strike
