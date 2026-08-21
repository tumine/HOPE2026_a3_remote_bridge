"""Admission policy for synchronizing a new planner task with policy timing."""

import math
from enum import Enum


class TaskTimingDecision(Enum):
    """Result of applying the new-task time-to-strike window."""

    ACCEPT = "accept"
    WAIT = "wait"
    DROP = "drop"


def validate_new_task_tts_window(min_s: float, max_s: float) -> tuple[float, float]:
    """Validate and normalize a closed time-to-strike admission window."""

    minimum = float(min_s)
    maximum = float(max_s)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("new-task TTS bounds must be finite")
    if minimum < 0.0:
        raise ValueError("new_task_tts_min_s must be non-negative")
    if maximum < minimum:
        raise ValueError(
            "new_task_tts_max_s must be greater than or equal to "
            "new_task_tts_min_s"
        )
    return minimum, maximum


class NewTaskTTSGate:
    """Stateful new-ball gate that never constrains revisions of an active task.

    A candidate above the window waits for later measurements. A candidate
    below the window expires the whole current ball, so estimator jitter cannot
    bring that same late ball back and accidentally create a task.
    """

    def __init__(self, min_s: float, max_s: float):
        self.min_s, self.max_s = validate_new_task_tts_window(min_s, max_s)
        self._current_ball_dropped = False

    @property
    def current_ball_dropped(self) -> bool:
        return self._current_ball_dropped

    def reset_ball(self) -> None:
        """Release a prior drop latch when a physically new ball begins."""

        self._current_ball_dropped = False

    def decide(
        self,
        time_to_strike_s: float,
        *,
        task_active: bool,
    ) -> TaskTimingDecision:
        """Return admission for one candidate without mutating task counters."""

        tts = float(time_to_strike_s)
        if not math.isfinite(tts):
            raise ValueError("candidate time_to_strike must be finite")

        # Revisions must continue all the way to contact once a task is active.
        if task_active:
            return TaskTimingDecision.ACCEPT

        if self._current_ball_dropped:
            return TaskTimingDecision.DROP
        if tts > self.max_s:
            return TaskTimingDecision.WAIT
        if tts < self.min_s:
            self._current_ball_dropped = True
            return TaskTimingDecision.DROP
        return TaskTimingDecision.ACCEPT
