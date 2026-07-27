"""Debounced incoming-ball cycle detection.

The mocap velocity estimate can change sign while a ball is stationary or is
being handled.  A single ``vx >= 0`` followed by ``vx < 0`` therefore cannot
be used as a physical-ball boundary.  This helper uses separated velocity
thresholds (hysteresis), consecutive-sample confirmation, and a minimum x
position before declaring a new incoming cycle.
"""

from dataclasses import dataclass, field
import math


@dataclass
class IncomingCycleDetector:
    """Return ``True`` once for each confirmed incoming cycle after the first.

    The first confirmed incoming motion belongs to the diagnostic ball that
    already exists at node startup (or after a capture-gap reset), so it does
    not request another reset.  A later cycle is accepted only after a stable
    non-incoming interval has armed the detector.
    """

    incoming_vx_threshold: float = -0.50
    nonincoming_vx_threshold: float = -0.10
    incoming_confirm_samples: int = 5
    nonincoming_confirm_samples: int = 10
    min_rearm_x: float = 0.30

    _seen_incoming: bool = field(init=False, default=False)
    _incoming_active: bool = field(init=False, default=False)
    _cycle_passed_rearm_boundary: bool = field(init=False, default=False)
    _ready_for_next: bool = field(init=False, default=False)
    _incoming_count: int = field(init=False, default=0)
    _nonincoming_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        values = (
            self.incoming_vx_threshold,
            self.nonincoming_vx_threshold,
            self.min_rearm_x,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("incoming-cycle thresholds must be finite")
        if self.incoming_vx_threshold >= self.nonincoming_vx_threshold:
            raise ValueError(
                "incoming_vx_threshold must be below nonincoming_vx_threshold"
            )
        if self.incoming_confirm_samples < 1:
            raise ValueError("incoming_confirm_samples must be at least 1")
        if self.nonincoming_confirm_samples < 1:
            raise ValueError("nonincoming_confirm_samples must be at least 1")

    def reset(self) -> None:
        """Forget the current motion cycle after a mocap timestamp gap."""

        self._seen_incoming = False
        self._incoming_active = False
        self._cycle_passed_rearm_boundary = False
        self._ready_for_next = False
        self._incoming_count = 0
        self._nonincoming_count = 0

    def update(self, x: float, vx: float) -> bool:
        """Consume one estimated state and report a confirmed new cycle.

        Incoming evidence requires ``x >= min_rearm_x`` and a sufficiently
        negative velocity.  Non-incoming evidence uses a different velocity
        threshold, creating a dead band that rejects direction chatter.
        """

        if not (math.isfinite(x) and math.isfinite(vx)):
            self._incoming_count = 0
            self._nonincoming_count = 0
            return False

        in_rearm_zone = x >= self.min_rearm_x
        incoming_evidence = (
            in_rearm_zone and vx <= self.incoming_vx_threshold
        )
        nonincoming_evidence = (
            in_rearm_zone and vx >= self.nonincoming_vx_threshold
        )

        if self._incoming_active:
            self._incoming_count = 0
            if x < self.min_rearm_x:
                self._cycle_passed_rearm_boundary = True
            if self._cycle_passed_rearm_boundary and nonincoming_evidence:
                self._nonincoming_count += 1
                if self._nonincoming_count >= self.nonincoming_confirm_samples:
                    self._incoming_active = False
                    self._cycle_passed_rearm_boundary = False
                    self._ready_for_next = True
                    self._nonincoming_count = 0
            else:
                self._nonincoming_count = 0
            return False

        if nonincoming_evidence:
            self._nonincoming_count += 1
            self._incoming_count = 0
            if self._nonincoming_count >= self.nonincoming_confirm_samples:
                self._ready_for_next = True
                self._nonincoming_count = self.nonincoming_confirm_samples
            return False

        self._nonincoming_count = 0
        if not incoming_evidence:
            self._incoming_count = 0
            return False

        self._incoming_count += 1
        if self._incoming_count < self.incoming_confirm_samples:
            return False

        self._incoming_count = 0
        self._incoming_active = True
        self._cycle_passed_rearm_boundary = False
        if not self._seen_incoming:
            self._seen_incoming = True
            self._ready_for_next = False
            return False
        if not self._ready_for_next:
            return False

        self._ready_for_next = False
        return True
