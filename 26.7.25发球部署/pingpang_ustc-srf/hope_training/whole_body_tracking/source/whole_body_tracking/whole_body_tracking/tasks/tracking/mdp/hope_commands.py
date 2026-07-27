"""Racket-target command: the ping-pong goal on top of motion imitation.

:class:`RacketTargetCommand` rides on the :class:`~whole_body_tracking.tasks.tracking.mdp.commands.MotionCommand`.
Each swing it samples the quantities the model-based planner supplies at deploy time — a desired racket
position, a desired racket velocity, and a time-to-strike — plus the swing side (forehand/backhand),
which is locked for the duration of that swing. It also:

* holds a FIXED station target (a startup constant = the environment origin): the robot base drifts
  across a rally, and the ``fixed_station_error_xy`` observation feeds that drift back so the policy
  can re-center in place. The station never moves; this is not station planning.
* computes the ACTUAL racket state in simulation by forward kinematics through the fixed racket mount
  (wrist -> paddle center), so the reward can compare actual vs desired.
* derives the strike timing from the reference clip phase, samples an incoming-ball state matching
  the MuJoCo serve distribution, and evaluates the planner's no-spin paddle impulse plus
  drag/gravity flight at the strike for the return rewards.

There is no measured racket feedback at deploy: the racket FK, its face normal, and the ball
evaluation are simulation-only signals used by rewards/critic, never by the actor observation.
Swing side selection is uniform per swing and follows the imitated clip (clip 0 = forehand -> +1,
clip 1 = backhand -> -1), so all four forehand/backhand transitions appear across the batch.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul, sample_uniform

from whole_body_tracking.tasks.tracking.mdp.command_curriculum import (
    contract_box_about_center,
    curriculum_fraction,
    linear_curriculum_progress,
)
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.hope_timing import (
    compute_strike_timing,
    consume_exact_strike,
)
from whole_body_tracking.utils.return_physics_torch import (
    ReturnPhysicsParams,
    integrate_return,
    plan_racket_command,
    predict_paddle_contact,
    relative_normal_speed,
)
from whole_body_tracking.utils.success_metric import load_ball_physics_config

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RacketTargetCommand(CommandTerm):
    """Samples desired racket/station targets and computes the actual racket state by FK."""

    cfg: RacketTargetCommandCfg

    def __init__(self, cfg: RacketTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Racket FK source: prefer a dedicated racket body, else (wrist pose) * (fixed mount offset).
        if cfg.racket_body_name in self.robot.body_names:
            self._racket_mode = "body"
            self._racket_body_index = self.robot.find_bodies(cfg.racket_body_name, preserve_order=True)[0][0]
            self._wrist_body_index = -1
        else:
            assert cfg.wrist_body_name in self.robot.body_names, (
                f"RacketTargetCommand: neither racket body '{cfg.racket_body_name}' nor wrist body "
                f"'{cfg.wrist_body_name}' found on asset '{cfg.asset_name}'."
            )
            self._racket_mode = "wrist_offset"
            self._racket_body_index = -1
            self._wrist_body_index = self.robot.find_bodies(cfg.wrist_body_name, preserve_order=True)[0][0]
        self._mount_offset = torch.tensor(cfg.mount_offset, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._mount_quat = torch.tensor(cfg.mount_quat, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        self._motion_term: MotionCommand | None = None

        # Desired (sampled) targets, world frame.
        self.racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w[:, 2] = 1.0
        self.incoming_ball_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.swing_sign = torch.ones(self.num_envs, device=self.device)

        # Actual racket state (FK), world frame.
        self.racket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.racket_quat_w[:, 0] = 1.0
        self.racket_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w[:, 2] = 1.0

        # Strike timing.
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.pre_strike = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Reward helper signals.
        self.racket_target_distance = torch.zeros(self.num_envs, device=self.device)
        self.feet_contact_frac = torch.zeros(self.num_envs, device=self.device)
        # No-spin return evaluation caches (one-shot at the exact strike frame).
        self.strike_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_net_cross = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Arm once per sampled swing so a frozen/debug clock cannot count the same
        # exact strike frame more than once.
        self._return_armed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._return_attempts = torch.zeros(self.num_envs, device=self.device)
        self._return_contacts = torch.zeros(self.num_envs, device=self.device)
        self._return_net_clears = torch.zeros(self.num_envs, device=self.device)
        self._return_successes = torch.zeros(self.num_envs, device=self.device)

        # Per-clip strike phase / target boxes (resolved lazily once the motion term is available).
        self._strike_phase_per_clip = None
        self._pos_box = _boxes_to_tensor(cfg.racket_pos_range_per_clip, self.device)  # (C,3,2) or None
        self._vel_box = _boxes_to_tensor(cfg.racket_vel_range_per_clip, self.device)
        self._incoming_vel_box = _boxes_to_tensor(cfg.incoming_ball_vel_range_per_clip, self.device)
        self._mount_sign_per_clip = (
            torch.tensor([float(s) for s in cfg.mount_normal_sign_per_clip], device=self.device)
            if cfg.mount_normal_sign_per_clip
            else None
        )
        self._return_physics = ReturnPhysicsParams.from_mapping(load_ball_physics_config())
        self._planner_landing_table = torch.tensor(
            cfg.planner_target_landing_table, dtype=torch.float32, device=self.device
        )
        # A command reset happens before ManagerBasedEnv calls sim.forward(). Mark
        # FK dirty so the first privileged observation refreshes it from the newly
        # forwarded articulation instead of exposing the previous episode's pose.
        self._racket_state_dirty = True

        # Feet resolution for contact fraction (degrades to 0 if it cannot resolve — never crashes).
        try:
            self._contact_sensor = env.scene.sensors["contact_forces"]
        except (KeyError, AttributeError, TypeError):
            self._contact_sensor = None
        self._foot_idx_contact: list[int] = []
        if self._contact_sensor is not None:
            sensor_bodies = list(self._contact_sensor.body_names)
            self._foot_idx_contact = [sensor_bodies.index(n) for n in cfg.feet_body_names if n in sensor_bodies]

        for key in (
            "racket_pos_error",
            "racket_vel_error",
            "racket_normal_error",
            "time_to_strike",
            "return_success",
            "contact_rate",
            "net_clear_rate",
            "return_attempts",
            "target_curriculum_progress",
            "feet_contact_fraction",
            "both_feet_contact",
            "station_error",
            "base_tilt",
            "base_height",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    # --- helpers -------------------------------------------------------------------------------- #
    def _motion(self) -> MotionCommand:
        if self._motion_term is None:
            self._motion_term = self._env.command_manager.get_term(self.cfg.motion_command_name)
        return self._motion_term

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    @property
    def fixed_station_w(self) -> torch.Tensor:
        """Fixed startup station XY = the environment origin plus a nominal offset (constant)."""
        off = torch.tensor(self.cfg.station_nominal_offset_xy, device=self.device)
        return self._env.scene.env_origins[:, :2] + off

    @property
    def command(self) -> torch.Tensor:
        """Raw target vector (world): [pos(3), vel(3), tts(1), station(2), swing(1)]."""
        return torch.cat(
            [
                self.racket_target_pos_w,
                self.racket_target_vel_w,
                self.time_to_strike.unsqueeze(-1),
                self.fixed_station_w,
                self.swing_sign.unsqueeze(-1),
            ],
            dim=-1,
        )

    # --- observation accessors ------------------------------------------------------- #
    def base_forward_xy(self) -> torch.Tensor:
        """Base forward unit vector e_base,x, world XY (2)."""
        fwd = quat_apply(
            self.base_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        )[:, :2]
        return fwd / (torch.norm(fwd, dim=-1, keepdim=True) + 1e-6)

    def fixed_station_error_xy(self) -> torch.Tensor:
        """Fixed startup station XY minus current base XY, world frame (2)."""
        return self.fixed_station_w - self.base_pos_w[:, :2]

    def racket_target_rel_base_w(self) -> torch.Tensor:
        """Target racket position minus base position, world frame (3)."""
        return self.racket_target_pos_w - self.base_pos_w

    # --- sampling ------------------------------------------------------------------------------- #
    def _target_curriculum_progress(self) -> float:
        """Global command difficulty, advanced in policy/control steps."""

        return linear_curriculum_progress(
            getattr(self._env, "common_step_counter", 0),
            self.cfg.target_curriculum_start_step,
            self.cfg.target_curriculum_duration_steps,
        )

    def _curriculum_box(self, box: torch.Tensor, initial_fraction: float) -> torch.Tensor:
        fraction = curriculum_fraction(
            initial_fraction, self._target_curriculum_progress()
        )
        return contract_box_about_center(box, fraction)

    def _sample_targets(self, env_ids: torch.Tensor):
        motion = self._motion()
        n = len(env_ids)
        clip = motion.clip_id[env_ids] if motion._multiseg else torch.zeros(n, dtype=torch.long, device=self.device)
        station = self.fixed_station_w[env_ids]  # (n, 2)

        pos_box = self._resolve_box(self._pos_box, clip, self.cfg.racket_pos_range)  # (n, 3, 2)
        pos_box = self._curriculum_box(
            pos_box, self.cfg.position_curriculum_initial_fraction
        )
        # Position: x/y are STATION-RELATIVE (fixed striking plane in front + side band), z absolute.
        pos = sample_uniform(pos_box[..., 0], pos_box[..., 1], (n, 3), self.device)
        pos[:, 0] = station[:, 0] + pos[:, 0]
        pos[:, 1] = station[:, 1] + pos[:, 1]
        self.racket_target_pos_w[env_ids] = pos

        incoming_box = self._resolve_box(
            self._incoming_vel_box, clip, self.cfg.incoming_ball_vel_range
        )
        incoming_box = self._curriculum_box(
            incoming_box, self.cfg.incoming_velocity_curriculum_initial_fraction
        )
        incoming = sample_uniform(
            incoming_box[..., 0], incoming_box[..., 1], (n, 3), self.device
        )
        self.incoming_ball_vel_w[env_ids] = incoming

        if self.cfg.use_planner_target:
            strike_table = self._world_to_table(pos, env_ids)
            landing = self._planner_landing_table.unsqueeze(0).expand(n, 3)
            vel, normal, _ = plan_racket_command(
                strike_table,
                incoming,
                landing,
                self.cfg.planner_flight_time,
                self._return_physics,
                integration_dt=self.cfg.planner_integration_dt,
            )
        else:
            vel_box = self._resolve_box(self._vel_box, clip, self.cfg.racket_vel_range)
            vel_box = self._curriculum_box(
                vel_box, self.cfg.racket_velocity_curriculum_initial_fraction
            )
            vel = sample_uniform(vel_box[..., 0], vel_box[..., 1], (n, 3), self.device)
            normal = vel / (torch.norm(vel, dim=-1, keepdim=True) + 1e-6)
        self.racket_target_vel_w[env_ids] = vel
        self.racket_target_normal_w[env_ids] = normal

        # Swing side follows the imitated clip (0 = forehand -> +1, 1 = backhand -> -1).
        if motion._multiseg:
            self.swing_sign[env_ids] = torch.where(clip == 0, 1.0, -1.0)
        else:
            self.swing_sign[env_ids] = 1.0

    def _resolve_box(self, per_clip, clip: torch.Tensor, shared_range) -> torch.Tensor:
        """Return an (n, 3, 2) [lo, hi] box per env: per-clip if configured, else the shared box."""
        if per_clip is not None:
            return per_clip[clip]
        shared = torch.tensor(shared_range, dtype=torch.float32, device=self.device)  # (3, 2)
        return shared.unsqueeze(0).expand(len(clip), 3, 2)

    def _world_to_table(self, pos_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """World positions for selected envs -> canonical table frame."""

        local = pos_w - self._env.scene.env_origins[env_ids]
        station_y = self.fixed_station_w[env_ids, 1] - self._env.scene.env_origins[env_ids, 1]
        table = local.clone()
        table[:, 0] -= float(self.cfg.table_near_x)
        table[:, 1] -= station_y + 0.5 * self._return_physics.table_width
        table[:, 2] -= float(self.cfg.table_surface_z)
        return table

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._sample_targets(env_ids)
        self._return_armed[env_ids] = True
        self.strike_fired[env_ids] = False
        self.ball_contact[env_ids] = False
        self.ball_net_cross[env_ids] = False
        self.ball_on_opponent[env_ids] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log true per-episode attempt ratios, then clear their counters."""

        ids = (
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            if env_ids is None
            else env_ids
        )
        attempts = self._return_attempts[ids]
        denominator = attempts.clamp_min(1.0)
        self.metrics["return_success"][ids] = self._return_successes[ids] / denominator
        self.metrics["contact_rate"][ids] = self._return_contacts[ids] / denominator
        self.metrics["net_clear_rate"][ids] = self._return_net_clears[ids] / denominator
        self.metrics["return_attempts"][ids] = attempts
        total_attempts = float(attempts.sum().item())
        aggregate_success = (
            float(self._return_successes[ids].sum().item()) / total_attempts
            if total_attempts > 0.0
            else 0.0
        )
        aggregate_contact = (
            float(self._return_contacts[ids].sum().item()) / total_attempts
            if total_attempts > 0.0
            else 0.0
        )
        aggregate_net_clear = (
            float(self._return_net_clears[ids].sum().item()) / total_attempts
            if total_attempts > 0.0
            else 0.0
        )
        extras = super().reset(ids)
        # CommandManager resets ``motion`` before ``racket_target`` (their
        # declaration order), so the newly sampled clip/time-step is available
        # here. Initialize timing now; otherwise reset's first actor observation
        # contains the constructor default TTS=0.
        self._compute_strike_timing(ids)
        self.strike_fired[ids] = False
        self.ball_contact[ids] = False
        self.ball_net_cross[ids] = False
        self.ball_on_opponent[ids] = False
        self._racket_state_dirty = True
        # CommandTerm's default is an unweighted mean of per-env ratios. Replace
        # it with the actual event-level numerator / denominator for this reset batch.
        extras["return_success"] = aggregate_success
        extras["contact_rate"] = aggregate_contact
        extras["net_clear_rate"] = aggregate_net_clear
        self._return_attempts[ids] = 0.0
        self._return_contacts[ids] = 0.0
        self._return_net_clears[ids] = 0.0
        self._return_successes[ids] = 0.0
        return extras

    # --- per-step updates ----------------------------------------------------------------------- #
    def _compute_strike_timing(self, env_ids: torch.Tensor | None = None):
        motion = self._motion()
        ml = motion.motion
        if self._strike_phase_per_clip is None:
            sp = tuple(self.cfg.strike_phase_per_clip)
            if sp and len(sp) == ml.num_segments:
                self._strike_phase_per_clip = torch.tensor([float(x) for x in sp], device=self.device)
            else:
                self._strike_phase_per_clip = torch.full((ml.num_segments,), float(self.cfg.strike_phase), device=self.device)
        clip = motion.clip_id
        seg_start = ml.seg_start[clip]
        seg_len = ml.seg_len[clip]
        phase = self._strike_phase_per_clip[clip]
        _, time_to_strike, pre_strike, strike_window = compute_strike_timing(
            seg_start=seg_start,
            seg_len=seg_len,
            strike_phase=phase,
            time_steps=motion.time_steps,
            step_dt=self._env.step_dt,
            strike_window_s=self.cfg.strike_window_s,
        )
        if env_ids is None:
            self.time_to_strike.copy_(time_to_strike)
            self.pre_strike.copy_(pre_strike)
            self.strike_window.copy_(strike_window)
        else:
            self.time_to_strike[env_ids] = time_to_strike[env_ids]
            self.pre_strike[env_ids] = pre_strike[env_ids]
            self.strike_window[env_ids] = strike_window[env_ids]

    def _compute_racket_state(self):
        data = self.robot.data
        if self._racket_mode == "body":
            idx = self._racket_body_index
            self.racket_pos_w = data.body_pos_w[:, idx]
            self.racket_quat_w = data.body_quat_w[:, idx]
            # ``body_pos_w`` is the link-frame origin, so its matching velocity is
            # ``body_link_lin_vel_w``. ``body_lin_vel_w`` is the COM velocity and
            # would be inconsistent whenever the link has a non-zero COM offset.
            self.racket_lin_vel_w = data.body_link_lin_vel_w[:, idx]
        else:
            widx = self._wrist_body_index
            wpos = data.body_pos_w[:, widx]
            wquat = data.body_quat_w[:, widx]
            # Start from the wrist link-origin velocity before transporting it to
            # the racket center. Starting from COM velocity would add a spurious
            # ``omega x wrist_com_offset`` term.
            wlin = data.body_link_lin_vel_w[:, widx]
            wang = data.body_ang_vel_w[:, widx]
            offset_w = quat_apply(wquat, self._mount_offset)
            self.racket_pos_w = wpos + offset_w
            self.racket_lin_vel_w = wlin + torch.cross(wang, offset_w, dim=-1)
            self.racket_quat_w = quat_mul(wquat, self._mount_quat)
        # Face normal = a chosen local axis of the racket frame, times the striking-face sign (the
        # forehand and backhand strike with opposite faces).
        axis_w = matrix_from_quat(self.racket_quat_w)[:, :, self.cfg.mount_normal_axis]
        if self._mount_sign_per_clip is not None and self._motion()._multiseg:
            clip = self._motion().clip_id.clamp(max=self._mount_sign_per_clip.shape[0] - 1)
            sign = self._mount_sign_per_clip[clip].unsqueeze(-1)
        else:
            sign = self.cfg.mount_normal_sign
        self.racket_normal_w = axis_w * sign
        self._racket_state_dirty = False

    def refresh_racket_state(self, *, force: bool = False) -> None:
        """Refresh FK before a reward or privileged observation consumes it."""
        if force or self._racket_state_dirty:
            self._compute_racket_state()

    def _update_feet_contact(self):
        if self._contact_sensor is None or not self._foot_idx_contact:
            return
        forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._foot_idx_contact, :], dim=-1)
        in_contact = (forces > self.cfg.contact_force_threshold).float()
        self.feet_contact_frac = in_contact.mean(dim=-1)

    def _evaluate_return(self):
        """One-shot planner-contact + shared drag/gravity return evaluation."""
        exact, next_armed = consume_exact_strike(
            self.time_to_strike,
            step_dt=self._env.step_dt,
            armed=self._return_armed,
        )
        self.strike_fired = exact

        pos_err = torch.norm(self.racket_pos_w - self.racket_target_pos_w, dim=-1)
        self.racket_target_distance = pos_err
        normal_speed = relative_normal_speed(
            self.incoming_ball_vel_w, self.racket_lin_vel_w, self.racket_normal_w
        )
        contact = (
            exact
            & (pos_err <= self.cfg.contact_radius)
            & (normal_speed >= self.cfg.min_relative_normal_speed)
        )

        self.ball_contact.zero_()
        self.ball_net_cross.zero_()
        self.ball_on_opponent.zero_()
        strike_ids = torch.where(exact)[0]
        if len(strike_ids) > 0:
            outgoing = predict_paddle_contact(
                self.incoming_ball_vel_w[strike_ids],
                self.racket_lin_vel_w[strike_ids],
                self.racket_normal_w[strike_ids],
                self._return_physics,
            )
            start_table = self._world_to_table(
                self.racket_target_pos_w[strike_ids], strike_ids
            )
            outcome = integrate_return(
                start_table,
                outgoing,
                self._return_physics,
                dt=self.cfg.return_integration_dt,
                max_time=self.cfg.return_max_time,
            )
            hit = contact[strike_ids]
            net_clear = hit & outcome.net_clear
            on_opponent = net_clear & outcome.on_opponent
            self.ball_contact[strike_ids] = hit
            self.ball_net_cross[strike_ids] = net_clear
            self.ball_on_opponent[strike_ids] = on_opponent

        self._return_attempts += exact.float()
        self._return_contacts += self.ball_contact.float()
        self._return_net_clears += self.ball_net_cross.float()
        self._return_successes += self.ball_on_opponent.float()
        self._return_armed.copy_(next_armed)

    def _update_metrics(self):
        # PhaseAlignedManagerBasedRLEnv calls this after physics and before reward;
        # motion was updated first in CommandManager declaration order.
        self._compute_strike_timing()
        self.refresh_racket_state(force=True)
        self._update_feet_contact()
        self._evaluate_return()
        self.metrics["racket_pos_error"] = torch.where(
            self.strike_window, self.racket_target_distance, self.metrics["racket_pos_error"]
        )
        self.metrics["racket_vel_error"] = torch.where(
            self.strike_window,
            torch.norm(self.racket_lin_vel_w - self.racket_target_vel_w, dim=-1),
            self.metrics["racket_vel_error"],
        )
        normal_cosine = torch.sum(
            self.racket_normal_w * self.racket_target_normal_w, dim=-1
        ).clamp(-1.0, 1.0)
        self.metrics["racket_normal_error"] = torch.where(
            self.strike_window,
            torch.acos(normal_cosine),
            self.metrics["racket_normal_error"],
        )
        self.metrics["time_to_strike"] = self.time_to_strike
        denominator = self._return_attempts.clamp_min(1.0)
        self.metrics["return_success"] = self._return_successes / denominator
        self.metrics["contact_rate"] = self._return_contacts / denominator
        self.metrics["net_clear_rate"] = self._return_net_clears / denominator
        self.metrics["return_attempts"] = self._return_attempts
        self.metrics["target_curriculum_progress"].fill_(
            self._target_curriculum_progress()
        )
        self.metrics["feet_contact_fraction"] = self.feet_contact_frac
        self.metrics["both_feet_contact"] = (self.feet_contact_frac >= 1.0).float()
        self.metrics["station_error"] = torch.linalg.vector_norm(
            self.base_pos_w[:, :2] - self.fixed_station_w, dim=-1
        )
        tilt_sine = torch.linalg.vector_norm(
            self.robot.data.projected_gravity_b[:, :2], dim=-1
        ).clamp(0.0, 1.0)
        self.metrics["base_tilt"] = torch.asin(tilt_sine)
        self.metrics["base_height"] = self.base_pos_w[:, 2]

    def _update_command(self):
        self._compute_strike_timing()
        # Re-sample the target at each new swing (the motion command sets just_resampled this step
        # when it wrapped a swing). Reset-time resampling is handled by the manager's reset -> _resample.
        motion = self._motion()
        wrapped = torch.where(motion.just_resampled)[0]
        if len(wrapped) > 0:
            self._resample_command(wrapped)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


def _boxes_to_tensor(per_clip, device):
    """Convert ((xlo,xhi),(ylo,yhi),(zlo,zhi)) x num_clips into an (C, 3, 2) tensor, or None."""
    if per_clip is None:
        return None
    return torch.tensor(
        [[[float(lo), float(hi)] for (lo, hi) in clip_rng] for clip_rng in per_clip],
        dtype=torch.float32,
        device=device,
    )


@configclass
class RacketTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`RacketTargetCommand`."""

    class_type: type = RacketTargetCommand

    asset_name: str = MISSING
    motion_command_name: str = "motion"
    # Targets are re-sampled per swing (on wrap / reset), not on a timer.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # --- racket mount FK ---
    racket_body_name: str = "pingpang_red_Link"
    wrist_body_name: str = "right_wrist_yaw_Link"
    # Exact fixed-joint origin shared by the vendor URDF and MuJoCo racket site.
    mount_offset: tuple[float, float, float] = (0.21021, 0.032078, 0.032036)
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mount_normal_axis: int = 1  # racket-local +Y is the blade face normal
    mount_normal_sign: float = 1.0
    # Per-clip striking-face sign (forehand and backhand strike with opposite faces), e.g. (1.0, -1.0).
    mount_normal_sign_per_clip: tuple = ()

    # --- fixed station (startup constant) ---
    station_nominal_offset_xy: tuple[float, float] = (0.0, 0.0)

    # --- feet (for the contact fraction used by the follow-through/recovery reward) ---
    feet_body_names: tuple[str, ...] = ("left_ankle_roll_Link", "right_ankle_roll_Link")
    contact_force_threshold: float = 10.0

    # --- strike timing (fraction of the reference clip at which the paddle meets the ball) ---
    strike_phase: float = 0.5
    strike_phase_per_clip: tuple = ()  # e.g. (0.47, 0.33); empty -> scalar strike_phase for every clip
    strike_window_s: float = 0.12  # half-window in which the racket-tracking rewards are active

    # --- racket target boxes ---
    # x/y are STATION-RELATIVE (fixed striking plane in front + swing-side band), z is absolute height.
    racket_pos_range: tuple = ((0.18, 0.75), (-0.76, 0.10), (0.84, 1.21))
    racket_vel_range: tuple = ((1.05, 3.40), (-0.25, 1.10), (0.35, 1.45))
    # Optional per-clip boxes (indexed by clip_id 0=forehand, 1=backhand). None -> shared boxes above.
    racket_pos_range_per_clip: tuple | None = None
    racket_vel_range_per_clip: tuple | None = None

    # --- incoming ball + planner-compatible racket command ---
    # Velocity at the interception point, in world axes. Per-clip defaults can be fitted from the
    # legal one-bounce MuJoCo serve generator plus domain-randomization margin.
    incoming_ball_vel_range: tuple = ((-2.40, -0.80), (-0.35, 0.35), (-2.80, 1.00))
    incoming_ball_vel_range_per_clip: tuple | None = None

    # Expand command boxes linearly from a centered subset to their full configured ranges.
    # A non-positive duration disables the curriculum and preserves legacy full-range sampling.
    target_curriculum_start_step: int = 0
    target_curriculum_duration_steps: int = 0
    position_curriculum_initial_fraction: float = 1.0
    incoming_velocity_curriculum_initial_fraction: float = 1.0
    racket_velocity_curriculum_initial_fraction: float = 1.0

    use_planner_target: bool = True
    # Canonical table-frame landing target and desired post-impact flight time (planner defaults).
    planner_target_landing_table: tuple[float, float, float] = (2.055, -0.7625, 0.020)
    planner_flight_time: float = 0.5
    planner_integration_dt: float = 0.01

    # --- no-spin return evaluation ---
    contact_radius: float = 0.095   # racket radius + ball radius
    min_relative_normal_speed: float = 0.3
    return_integration_dt: float = 0.01
    return_max_time: float = 2.0
    # Table placement is in the tracking environment's floor-based frame. Physical dimensions and
    # paddle/flight parameters come from configs/ball_physics.yaml at runtime.
    table_near_x: float = 0.5       # x of the robot's own table end (robot sits behind it)
    table_surface_z: float = 0.76   # table surface height above the env origin
