# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""The 50 Hz reference control loop.

Per tick, in order:
  1. read robot state from the sim bridge;
  2. poll the latest RacketCommand and emit the current swing goal;
  3. assemble the 111-D observation;
  4. run the ONNX actor -> raw_action[31];
  5. clip the raw action to its shared interval, zero the passive head columns
     (idx 3, 4), and feed that APPLIED action back as the next last_action —
     matching training;
  6. pass the applied action through the shared ActionAdapter -> 31 joint targets
     (holding the passive neck at its default);
  7. write the targets, step the sim, then advance the strike clock once.

There are deliberately NO gates, failure checks, rejections, state resets between
tasks, or reference playback here -- a single continuous 111-D control path.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .config import RuntimeConfig
from .joint_order import HEAD_INDICES, NUM_JOINTS
from .lifecycle import Phase, SwingLifecycle
from .observation import build_observation
from .onnx_policy import OnnxPolicy
from .racket_command import RacketCommandSource
from .sim_bridge import SimBridge
from .station_target import (
    derive_lateral_base_target,
    validate_lateral_station_policy_manifest,
)

_HEAD_IDX = list(HEAD_INDICES)


class PingPongReferenceRunner:
    def __init__(
        self,
        cfg: RuntimeConfig,
        bridge: SimBridge,
        command_source: RacketCommandSource,
        policy: OnnxPolicy | None = None,
    ) -> None:
        self.cfg = cfg
        self.bridge = bridge
        self.source = command_source
        if policy is None:
            if cfg.lateral_station.enabled:
                validate_lateral_station_policy_manifest(cfg.onnx_path)
            self.policy = OnnxPolicy(cfg.onnx_path)
        else:
            self.policy = policy
        self.lifecycle = SwingLifecycle(cfg.lifecycle)

        self.default_q = cfg.action_adapter.default_q.copy()
        self.kp = cfg.sim_kp.copy()
        self.kd = cfg.sim_kd.copy()

        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.nominal_station_xy: np.ndarray | None = None
        self.base_target_xy: np.ndarray | None = None

    def run(self, max_ticks: int | None = None, realtime: bool = False,
            status_every: int = 100) -> None:
        dt = self.cfg.control_dt
        self.bridge.reset()

        state = self.bridge.read_state()
        # The table/world transform remains anchored at startup. The policy's base
        # target starts there, then moves laterally for each accepted swing.
        self.nominal_station_xy = np.asarray(state.base_pos_w[:2], dtype=np.float64).copy()
        self.base_target_xy = self.nominal_station_xy.copy()

        tick = 0
        try:
            while max_ticks is None or tick < max_ticks:
                loop_start = time.perf_counter()

                state = self.bridge.read_state()
                cmd = self.source.poll()
                target = self.lifecycle.update(cmd, state)

                # Revisions of an active command may refine the hit point, so update
                # station geometry every active tick.  With no active ball, READY /
                # RECOVERY explicitly command the table-centred nominal station.
                station_cfg = self.cfg.lateral_station
                if self.lifecycle.phase in (Phase.READY, Phase.RECOVERY):
                    self.base_target_xy = self.nominal_station_xy.copy()
                elif station_cfg.enabled and self.lifecycle.phase in (
                    Phase.SWING,
                    Phase.FOLLOW_THROUGH,
                ):
                    self.base_target_xy = derive_lateral_base_target(
                        target.pos_w,
                        target.swing_side,
                        self.nominal_station_xy,
                        base_target_y_range=station_cfg.base_target_y_range,
                        forehand_reach_y=station_cfg.forehand_reach_y,
                        backhand_reach_y=station_cfg.backhand_reach_y,
                    )

                obs = build_observation(
                    state, target, self.last_action, self.default_q, self.base_target_xy
                )
                raw_action = self.policy.infer(obs)
                # The APPLIED action is clipped by the shared adapter before feedback.
                # With a passive neck the head columns are then zeroed; training exposes
                # the same clipped/zeroed columns in its last_action observation.
                applied_action = self.cfg.action_adapter.clip_raw_action(raw_action)
                if self.cfg.passive_neck:
                    applied_action[_HEAD_IDX] = 0.0
                self.last_action = applied_action.copy()

                q_des = self.cfg.action_adapter.decode(applied_action)
                if self.cfg.passive_neck:
                    q_des[_HEAD_IDX] = self.default_q[_HEAD_IDX]

                self.bridge.write_targets(q_des, self.kp, self.kd)
                self.bridge.step()
                # The target above belongs to the pre-transition observation.  Advance
                # its clock only after the commanded physics/control transition.
                self.lifecycle.advance()
                self.bridge.sync_viewer()

                if not self.bridge.is_viewer_running():
                    break

                if status_every and tick % status_every == 0:
                    self._print_status(tick, target)

                tick += 1
                if realtime:
                    self._sleep_to_rate(loop_start, dt)
        finally:
            self.bridge.close()

    def _print_status(self, tick: int, target) -> None:
        side = "forehand" if target.swing_side >= 0 else "backhand"
        print(
            f"[ref] t={tick * self.cfg.control_dt:6.2f}s "
            f"phase={self.lifecycle.phase.value:<14} "
            f"task={self.lifecycle.active_task_id} side={side} "
            f"tts={target.time_to_strike:+.2f}",
            file=sys.stderr,
        )

    @staticmethod
    def _sleep_to_rate(loop_start: float, dt: float) -> None:
        remaining = dt - (time.perf_counter() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
