# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Continuous multi-rally swing lifecycle.

Per tick the runner advances one state machine that turns the latest
``RacketCommand`` and the robot state into the strike goal fed to the observation:

    ready -> swing -> follow-through -> recovery -> ready -> (next task_id)

Contract points enforced here:
  * A new ``task_id`` engages a swing (only from ready or recovery). ``swing_side``
    is locked for the whole task.
  * A higher ``task_revision`` under the active ``task_id`` updates the target and
    time-to-strike, but only before contact.
  * There is exactly one swing per ``task_id``.
  * Between balls the robot pose, joint state, and policy history are NOT reset --
    the lifecycle never touches them, it only chooses what goal to observe.
  * Recovery emits a fixed ready reach while the runner commands the nominal
    table-centred base station; a later task may request a new lateral station.

The reference clock: ``time_to_strike`` is seeded from the command and is emitted
for the current pre-physics observation.  :meth:`advance` is called only after the
physics transition, where it counts down by ``dt``.  This matches Isaac's
``obs_t -> action_t -> physics transition -> obs_(t+1)`` convention and avoids
showing the actor a target one control tick earlier than the planner supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .observation import ObsTarget, RobotState
from .racket_command import BACKHAND, FOREHAND, RacketCommand


class Phase(Enum):
    READY = "ready"
    SWING = "swing"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class LifecycleConfig:
    dt: float = 0.02
    # A 91-frame guarded reference clip strikes at frame 50: actor samples run
    # from +1.0 s through 0 to -0.8 s, both endpoints included. Inter-clip hold is
    # command/feed timing, not a separate reference-clock phase.
    follow_through_s: float = 0.8
    recovery_s: float = 0.0
    ready_time_to_strike: float = 1.0
    # Canonical READY actor-conditioning anchor.  When these three fields are set,
    # READY emits exactly
    #   [target_rel_base_w, target_vel_w, ready_time_to_strike, ready_swing_side]
    # in observation[103:111].  The position components are expressed on world axes
    # but remain relative to the live pelvis, so proprioception and station feedback
    # stay live while the policy-conditioning anchor stays fixed.
    ready_target_rel_base_w: tuple[float, float, float] | None = None
    ready_target_vel_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ready_swing_side: int | None = None
    # Backward-compatible fallback for older runtime YAMLs.  If
    # ready_target_rel_base_w is absent, y is mirrored using the last swing side,
    # matching the original reference-runner behavior.
    ready_reach_x: float = 0.40
    ready_reach_y: float = 0.20
    ready_reach_z: float = -0.05    # relative to the pelvis height

    def __post_init__(self) -> None:
        if self.ready_target_rel_base_w is not None:
            rel = np.asarray(self.ready_target_rel_base_w, dtype=np.float64)
            if rel.shape != (3,) or not np.all(np.isfinite(rel)):
                raise ValueError("ready_target_rel_base_w must contain three finite values")
            self.ready_target_rel_base_w = tuple(float(value) for value in rel)

        vel = np.asarray(self.ready_target_vel_w, dtype=np.float64)
        if vel.shape != (3,) or not np.all(np.isfinite(vel)):
            raise ValueError("ready_target_vel_w must contain three finite values")
        self.ready_target_vel_w = tuple(float(value) for value in vel)

        if self.ready_swing_side is not None:
            side = int(self.ready_swing_side)
            if side not in (BACKHAND, 0, FOREHAND):
                raise ValueError(
                    "ready_swing_side must be -1 (backhand), 0 (READY), or +1 (forehand)"
                )
            self.ready_swing_side = side


class SwingLifecycle:
    def __init__(self, cfg: LifecycleConfig | None = None) -> None:
        self.cfg = cfg or LifecycleConfig()
        self.phase = Phase.READY
        self.active_task_id: int | None = None
        self.swing_side: int = FOREHAND         # last locked side (default forehand)
        self._target_pos_w = np.zeros(3)
        self._target_vel_w = np.zeros(3)
        self._tts = self.cfg.ready_time_to_strike
        self._follow_t = 0.0
        self._recover_t = 0.0
        # task_id of the most recently engaged ball; task_ids increase monotonically
        # (one ball, one increasing id), so we only ever engage a strictly newer id --
        # this enforces exactly one swing per task_id.
        self._last_engaged_task_id: int = -1
        # highest task_revision already applied to the active task (pre-contact only).
        self._applied_revision: int = -1

    # -- helpers ------------------------------------------------------------
    def _ready_target_pos_w(self, state: RobotState) -> np.ndarray:
        base = np.asarray(state.base_pos_w, dtype=np.float64)
        if self.cfg.ready_target_rel_base_w is not None:
            return base + np.asarray(
                self.cfg.ready_target_rel_base_w, dtype=np.float64
            )
        side = 1.0 if self.swing_side >= 0 else -1.0
        return base + np.array(
            [self.cfg.ready_reach_x, side * self.cfg.ready_reach_y, self.cfg.ready_reach_z],
            dtype=np.float64,
        )

    def _ready_target_vel_w(self) -> np.ndarray:
        return np.asarray(self.cfg.ready_target_vel_w, dtype=np.float64).copy()

    def _ready_swing_side(self) -> int:
        if self.cfg.ready_swing_side is not None:
            return int(self.cfg.ready_swing_side)
        return int(self.swing_side)

    def _can_engage(self) -> bool:
        return self.phase in (Phase.READY, Phase.RECOVERY)

    @property
    def time_to_strike(self) -> float:
        """Current unadvanced strike clock, primarily for schedulers/diagnostics."""

        return float(self._tts)

    # -- current observation -------------------------------------------------
    def update(self, cmd: RacketCommand | None, state: RobotState) -> ObsTarget:
        """Consume the latest command and emit the goal for the current state.

        This method deliberately does not advance time.  The caller must invoke
        :meth:`advance` once, after the corresponding simulator/robot transition.
        Keeping those two operations explicit prevents a newly received planner
        timestamp from being decremented before the actor has observed it.
        """

        # Engage a new ball, or refine the active one before contact.
        if cmd is not None:
            if cmd.task_id > self._last_engaged_task_id and self._can_engage():
                # A strictly newer ball may engage only while its strike is
                # still in the future. Permanently consume an expired/non-finite
                # task id so a later stale revision cannot trigger a partial
                # swing outside the training clock.
                command_tts = float(cmd.time_to_strike)
                self._last_engaged_task_id = cmd.task_id
                if np.isfinite(command_tts) and command_tts >= 0.0:
                    self.active_task_id = cmd.task_id
                    self._applied_revision = cmd.task_revision
                    self.swing_side = cmd.swing_side      # locked for this task
                    self._target_pos_w = np.asarray(cmd.position, dtype=np.float64).copy()
                    self._target_vel_w = np.asarray(cmd.velocity, dtype=np.float64).copy()
                    self._tts = command_tts
                    self.phase = Phase.SWING
            elif (
                cmd.task_id == self.active_task_id
                and self.phase == Phase.SWING
                and self._tts > 0.0
                and np.isfinite(float(cmd.time_to_strike))
                and float(cmd.time_to_strike) >= 0.0
                and cmd.task_revision > self._applied_revision
            ):
                # Pre-contact revision (must be newer): update where/when, never the
                # locked side.
                self._applied_revision = cmd.task_revision
                self._target_pos_w = np.asarray(cmd.position, dtype=np.float64).copy()
                self._target_vel_w = np.asarray(cmd.velocity, dtype=np.float64).copy()
                self._tts = float(cmd.time_to_strike)

        # Emit the goal to observe at the current state.
        if self.phase in (Phase.SWING, Phase.FOLLOW_THROUGH):
            return ObsTarget(
                pos_w=self._target_pos_w,
                vel_w=self._target_vel_w,
                time_to_strike=self._tts,
                swing_side=float(self.swing_side),
            )
        # READY / RECOVERY -> in-place ready reach, clock pinned.
        return ObsTarget(
            pos_w=self._ready_target_pos_w(state),
            vel_w=self._ready_target_vel_w(),
            time_to_strike=self.cfg.ready_time_to_strike,
            swing_side=float(self._ready_swing_side()),
        )

    # -- post-transition clock ----------------------------------------------
    def advance(self) -> None:
        """Advance exactly one completed control transition.

        Call once after applying the action and stepping the backend.  READY has
        no active strike clock; its display value remains pinned by ``update``.
        """

        c = self.cfg
        if self.phase == Phase.SWING:
            self._tts -= c.dt
            # Whole-tick clocks can leave a tiny positive residue (for example,
            # 1.0 - 50 * 0.02). Snap only numerical zero so the strike sample
            # enters FOLLOW_THROUGH on the same transition as Isaac frame 50.
            if self._tts <= 1.0e-9:
                if abs(self._tts) <= 1.0e-9:
                    self._tts = 0.0
                self.phase = Phase.FOLLOW_THROUGH
                self._follow_t = 0.0
        elif self.phase == Phase.FOLLOW_THROUGH:
            self._tts -= c.dt
            self._follow_t += c.dt
            # ``follow_through_s`` identifies the terminal reference sample
            # (-1.0 s), not a half-open duration. Keep FOLLOW_THROUGH active long
            # enough for the actor to observe and act on that endpoint, then
            # transition on the following completed control step.
            if self._follow_t - c.follow_through_s >= 0.5 * c.dt:
                if c.recovery_s <= 0.0:
                    self.phase = Phase.READY
                    self.active_task_id = None
                else:
                    self.phase = Phase.RECOVERY
                    self._recover_t = 0.0
        elif self.phase == Phase.RECOVERY:
            self._recover_t += c.dt
            if self._recover_t >= c.recovery_s:
                self.phase = Phase.READY
                self.active_task_id = None
