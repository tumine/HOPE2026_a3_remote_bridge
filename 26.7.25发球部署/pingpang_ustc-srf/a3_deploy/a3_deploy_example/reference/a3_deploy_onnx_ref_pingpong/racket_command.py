# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""RacketCommand: the strike goal the planner streams to the runner.

Mirrors the public ``RacketCommand.msg`` payload (the ROS message additionally
carries ``header.stamp``, which defines the time reference):

    uint64 task_id
    uint32 task_revision
    int8   swing_side        # FOREHAND = +1, BACKHAND = -1
    Point  position          # target racket position, world frame, m
    Vector3 velocity         # target racket velocity, world frame, m/s
    float64 time_to_strike   # seconds from header.stamp

Each incoming ball gets a new ``task_id``. Pre-strike trajectory refinements
arrive as an increasing ``task_revision`` under the same ``task_id``. ``swing_side``
is chosen once per ``task_id`` and is locked for that task.

A ``RacketCommandSource`` is the seam between the runner and whatever produces the
command. In a real deployment that is a ROS2 subscriber writing into
``QueueRacketCommandSource``. ``ExampleCommandFeed`` is a self-contained stand-in
so the runner is runnable against the sim WITHOUT a live planner; it is a
demonstration feed only, NOT part of the deploy contract and NOT a scripted swing
(the swing trajectory is always produced by the learned policy).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

FOREHAND: int = 1
BACKHAND: int = -1


@dataclass
class RacketCommand:
    task_id: int
    task_revision: int
    swing_side: int                       # +1 forehand / -1 backhand
    position: np.ndarray                   # (3,) world m
    velocity: np.ndarray                   # (3,) world m/s
    time_to_strike: float                  # s

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        self.swing_side = FOREHAND if self.swing_side >= 0 else BACKHAND


def age_racket_command(cmd: RacketCommand, elapsed_s: float) -> RacketCommand:
    """Return a copy whose strike clock is aged by a non-negative duration.

    ``RacketCommand.time_to_strike`` is defined at the instant represented by
    the command source.  Mailboxes and transports must therefore subtract time
    spent before the 50 Hz runner observes the command.  Invalid, negative, or
    non-finite elapsed values are treated as zero; a clock rollback must never
    make a strike appear farther away.

    The result is intentionally not clamped at zero.  A genuinely stale command
    remains overdue (negative TTS), which is safer and more truthful than
    presenting it to the policy as an on-time strike.
    """

    try:
        elapsed = float(elapsed_s)
    except (TypeError, ValueError, OverflowError):
        elapsed = 0.0
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        elapsed = 0.0
    return RacketCommand(
        task_id=cmd.task_id,
        task_revision=cmd.task_revision,
        swing_side=cmd.swing_side,
        position=cmd.position.copy(),
        velocity=cmd.velocity.copy(),
        time_to_strike=cmd.time_to_strike - elapsed,
    )


class RacketCommandSource(ABC):
    """Latest-command mailbox. ``poll`` returns the newest command or ``None``."""

    @abstractmethod
    def poll(self) -> RacketCommand | None:
        ...


class QueueRacketCommandSource(RacketCommandSource):
    """Thread-safe latest-value mailbox.

    Integration seam for a live planner: a ROS2 (or other) subscriber thread calls
    ``submit(...)`` on each ``RacketCommand.msg``; the 50 Hz runner thread calls
    ``poll()``. Only the newest command is retained.  By default the returned TTS
    is aged from submission to polling with ``time.monotonic``.  Deterministic
    offline simulations that advance their own clock must construct this source
    with ``age_queued_commands=False`` (or inject that simulation clock).
    """

    def __init__(
        self,
        *,
        monotonic_clock=None,
        age_queued_commands: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._latest: RacketCommand | None = None
        self._submitted_at: float | None = None
        self._seen = False
        self._clock = monotonic_clock or time.monotonic
        self._age_queued_commands = bool(age_queued_commands)

    def submit(self, cmd: RacketCommand) -> None:
        try:
            submitted_at = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            submitted_at = None
        if submitted_at is not None and not np.isfinite(submitted_at):
            submitted_at = None
        with self._lock:
            self._latest = cmd
            self._submitted_at = submitted_at
            self._seen = True

    def poll(self) -> RacketCommand | None:
        with self._lock:
            cmd = self._latest
            submitted_at = self._submitted_at
        if cmd is None:
            return None
        try:
            now = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            now = float("nan")
        queue_age = (
            now - submitted_at
            if self._age_queued_commands and submitted_at is not None
            else 0.0
        )
        return age_racket_command(cmd, queue_age)

    def has_any(self) -> bool:
        with self._lock:
            return self._seen


class TranslatedRacketCommandSource(RacketCommandSource):
    """Translate command positions while preserving world-axis vectors.

    The live planner publishes in the canonical table-surface frame, whereas the
    standalone robot MJCF has a floor-based world origin.  Those frames share axis
    directions, so position needs one fixed translation and velocity needs none.
    """

    def __init__(self, source: RacketCommandSource, translation) -> None:
        self.source = source
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)

    def poll(self) -> RacketCommand | None:
        cmd = self.source.poll()
        if cmd is None:
            return None
        return RacketCommand(
            task_id=cmd.task_id,
            task_revision=cmd.task_revision,
            swing_side=cmd.swing_side,
            position=cmd.position + self.translation,
            velocity=cmd.velocity,
            time_to_strike=cmd.time_to_strike,
        )

    def has_any(self) -> bool:
        probe = getattr(self.source, "has_any", None)
        return bool(probe()) if probe is not None else False

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if close is not None:
            close()


@dataclass
class ExampleCommandFeed(RacketCommandSource):
    """Planner-less demonstration feed for the reference sim.

    Emits one new ``task_id`` every ``period_s`` seconds, alternating forehand and
    backhand so all four adjacent side transitions are exercised over a session.
    Within each ball it first pins ``time_to_strike`` at the clip's initial value
    for ``hold_time_s``, then streams increasing ``task_revision`` with a
    decreasing clock across the approach window.  This reproduces the training
    command's pre-swing hold instead of inserting an unrelated READY gap after
    the swing. The targets are fixed example reach points inside the Isaac
    target/velocity boxes; a real deployment replaces this entirely with the
    planner's ``RacketCommand`` stream. Target x/y are specified relative to the
    reset station and converted to MuJoCo world here; z is the floor-world
    absolute height, matching training.
    """

    dt: float = 0.02
    # 50 held actions + 91 clip actions at 50 Hz: mean training cycle = 2.82 s.
    period_s: float = 2.82
    hold_time_s: float = 1.0
    lead_time_s: float = 1.0           # 91-frame clip start -> strike frame 50
    station_xy: tuple = (0.0, 0.0)
    # Midpoints of hope_env_cfg.py's per-clip training boxes.
    forehand_pos: tuple = (0.290, -0.640, 1.105)
    backhand_pos: tuple = (0.600, -0.100, 0.970)
    forehand_vel: tuple = (2.602879, 0.639805, 0.908102)
    backhand_vel: tuple = (1.871409, 0.082040, 1.045623)
    _step: int = field(default=0, init=False)
    _task_id: int = field(default=0, init=False)
    _revision: int = field(default=0, init=False)
    _issued_for_cycle: int = field(default=-1, init=False)
    _side: int = field(default=FOREHAND, init=False)
    _pos: np.ndarray = field(default=None, init=False)
    _vel: np.ndarray = field(default=None, init=False)
    _latest: RacketCommand | None = field(default=None, init=False)

    def poll(self) -> RacketCommand | None:
        period_steps = max(1, int(round(self.period_s / self.dt)))
        cycle = self._step // period_steps
        cycle_step = self._step - cycle * period_steps
        elapsed = cycle_step * self.dt                # time since this ball appeared

        if cycle != self._issued_for_cycle:
            # A fresh ball: new task_id, reset revision, choose side/target.
            self._issued_for_cycle = cycle
            self._task_id += 1
            self._revision = 0
            self._side = FOREHAND if (self._task_id % 2 == 1) else BACKHAND
            if self._side == FOREHAND:
                self._pos = np.array(self.forehand_pos, dtype=np.float64)
                self._vel = np.array(self.forehand_vel, dtype=np.float64)
            else:
                self._pos = np.array(self.backhand_pos, dtype=np.float64)
                self._vel = np.array(self.backhand_vel, dtype=np.float64)
            self._pos[:2] += np.asarray(self.station_xy, dtype=np.float64).reshape(2)
        else:
            self._revision += 1

        # Hold the newly sampled target/clock for the same kind of stand/wrap
        # pre-roll used in training, then count down from clip frame zero.
        swing_elapsed = max(elapsed - self.hold_time_s, 0.0)
        tts = max(self.lead_time_s - swing_elapsed, 0.0)
        self._latest = RacketCommand(
            task_id=self._task_id,
            task_revision=self._revision,
            swing_side=self._side,
            position=self._pos,
            velocity=self._vel,
            time_to_strike=tts,
        )
        self._step += 1
        return self._latest
