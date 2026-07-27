"""Phase-aligned manager-based environment for HOPE tracking.

Isaac Lab's generic :class:`~isaaclab.envs.ManagerBasedRLEnv` computes rewards before it
advances command terms. That ordering is harmless for piecewise-constant commands, but
the HOPE command is a sampled motion trajectory: after physics advances from ``t`` to
``t + dt``, the racket state is current while the trajectory phase is still ``t``.
This attributed strike/contact rewards one policy transition late.

This task-local subclass keeps the upstream step implementation intact except for one
intentional ordering change:

``physics -> termination -> command(t + dt) -> reward(t + dt) -> reset -> observation``.

Consequently the reward and returned observation use the same post-physics trajectory
phase. Command terms are still computed exactly once per control step.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class PhaseAlignedManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with post-physics command/reward alignment."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one control step with commands advanced before reward evaluation."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # HOPE's command is a trajectory clock, not a piecewise-constant target.
        # Advance it exactly once after physics so reward and the next observation
        # both describe the state at t + dt.
        self.command_manager.compute(dt=self.step_dt)
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
