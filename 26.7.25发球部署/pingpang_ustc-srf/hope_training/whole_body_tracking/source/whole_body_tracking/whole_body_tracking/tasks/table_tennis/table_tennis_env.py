"""Table-tennis environment that adds no-spin ball aerodynamics on top of the manager-based RL env.

Everything except aerodynamics is handled by the standard :class:`~isaaclab.envs.ManagerBasedRLEnv`
machinery configured in :mod:`.table_tennis_env_cfg`. PhysX does not model air drag, so this subclass
updates the ball's no-spin drag force (:func:`.ball.compute_drag_force`) immediately before each control
step. Isaac Lab then applies that buffered force during every decimated physics substep.

The force is intentionally prepared outside ``SimulationContext.step()``. Calling the PhysX tensor API
from an Isaac physics callback can race the asynchronous GPU simulation and cause
``cudaErrorIllegalAddress``.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from . import geometry
from .ball import compute_drag_force
from .table_tennis_env_cfg import TableTennisEnvCfg


class TableTennisEnv(ManagerBasedRLEnv):
    """Manager-based table-tennis env with a per-substep no-spin ball drag force field."""

    cfg: TableTennisEnvCfg

    def __init__(self, cfg: TableTennisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._aero_active = False
        self._setup_ball_aerodynamics()

    def _setup_ball_aerodynamics(self) -> None:
        self._ball = self.scene["ball"]
        self._aero_cfg = self.cfg.ball_aerodynamics
        self._ball_mass = float(geometry.BALL_MASS)
        # Reusable zeroed external-wrench buffers: (num_envs, num_bodies=1, 3). Torque stays zero (no spin).
        self._aero_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._aero_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._aero_active = bool(self._aero_cfg.enabled)

    def _update_ball_aerodynamics(self) -> None:
        """Update the world-frame drag buffer before the manager starts physics stepping."""
        lin_vel_w = self._ball.data.root_lin_vel_w
        force_w = compute_drag_force(lin_vel_w, self._ball_mass, self._aero_cfg)
        self._aero_force[:, 0, :] = force_w
        self._ball.set_external_force_and_torque(
            self._aero_force,
            self._aero_torque,
            is_global=True,
        )

    def step(self, action: torch.Tensor):
        """Apply control-rate drag, then delegate all simulation and RL bookkeeping."""
        if self._aero_active:
            self._update_ball_aerodynamics()
        return super().step(action)
