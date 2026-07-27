"""The HOPE action term.

:class:`ClampedJointPositionAction` is a 31-D joint-position residual action:

    applied_action = clip(raw_action, action_clip)
    adapter_action = transform(applied_action)
    q_des = default_q + adapter_action * action_scale

followed by a deterministic clamp of the position targets (a numeric transform, not a rejection
gate — it emits no failure status). When ``position_clamp`` is configured (the HOPE task
sets it from the shared deploy ``action_adapter.yaml``) the clamp bounds are those exact per-joint
values, so training and deployment clamp identically; without it the term falls back to the
articulation's soft joint limits. The policy still emits 31 raw values, but the two passive head
joints (idx 3, 4) are held at their articulation default and their applied-action feedback is
zeroed, matching a deploy runner that holds the neck passive. ``applied_raw_actions`` is the
clipped actor output (with passive columns zeroed) and is fed back as the next tick's
``last_action`` observation.

The raw-action clip, transform, residual form, ``action_scale``, ``default_q`` offset, and clamp
are the shared public action adapter: all come from ONE file
(``a3_deploy/a3_deploy_example/config/action_adapter.yaml``) and the reference runner
applies the identical transform, so training and deployment agree
(see ``tests/test_action_adapter_parity.py``).
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


def resolve_action_columns_by_joint_name(action_term, joint_names) -> list[int]:
    """Resolve exact joint names to action columns, raising on any contract mismatch."""
    action_joint_names = getattr(action_term, "_joint_names", None)
    if action_joint_names is None:
        raise RuntimeError("Passive-joint action contract requires the action term's resolved joint names")
    action_joint_names = list(action_joint_names)
    requested = list(joint_names)
    if not requested:
        return []
    missing = [name for name in requested if name not in action_joint_names]
    if missing:
        raise RuntimeError(
            f"Passive joints missing from the configured action term: {missing}; "
            f"resolved action joints are {action_joint_names}"
        )
    return [action_joint_names.index(name) for name in requested]


def _action_joint_ids_in_column_order(action_term) -> list[int]:
    """Return articulation joint ids in action-column order (order-assumption free)."""
    joint_ids = getattr(action_term, "_joint_ids", None)
    if joint_ids is None:
        raise RuntimeError("Passive-joint action contract cannot resolve articulation joint ids")
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(action_term._asset.joint_names)))[joint_ids]
    elif torch.is_tensor(joint_ids):
        joint_ids = joint_ids.detach().cpu().tolist()
    else:
        joint_ids = list(joint_ids)
    return [int(j) for j in joint_ids]


class ClampedJointPositionAction(JointPositionAction):
    """Bounded joint-position residual with a deterministic clamp and optional passive joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_action_transform = str(self.cfg.raw_action_transform).strip().lower()
        if self._raw_action_transform not in {"identity", "tanh"}:
            raise RuntimeError(
                "raw_action_transform must be one of ['identity', 'tanh'], "
                f"got {self.cfg.raw_action_transform!r}"
            )
        self._action_clip: tuple[float, float] | None = None
        if self.cfg.action_clip is not None:
            if len(self.cfg.action_clip) != 2:
                raise RuntimeError("action_clip must contain [lower, upper]")
            lower, upper = (float(value) for value in self.cfg.action_clip)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise RuntimeError("action_clip bounds must be finite")
            if lower > upper:
                raise RuntimeError("action_clip lower must be <= upper")
            self._action_clip = (lower, upper)
        # The clipped actor output accepted this tick (passive columns zeroed) —
        # fed back as ``last_action``.
        self._applied_raw_actions = torch.zeros_like(self._raw_actions)
        # The residual-domain action used for q_des after clipping/transform.
        self._transformed_actions = torch.zeros_like(self._raw_actions)

        # Explicit per-joint clamp from the shared action-adapter config (exact joint names).
        # When absent, process_actions falls back to the articulation soft joint limits.
        self._clamp_lower: torch.Tensor | None = None
        self._clamp_upper: torch.Tensor | None = None
        clamp_cfg = self.cfg.position_clamp
        if clamp_cfg is not None:
            action_joint_names = list(self._joint_names)
            missing = [n for n in action_joint_names if n not in clamp_cfg]
            if missing:
                raise RuntimeError(
                    f"position_clamp is missing joints resolved by the action term: {missing}"
                )
            lo = torch.tensor(
                [float(clamp_cfg[n][0]) for n in action_joint_names],
                dtype=torch.float32, device=self.device,
            )
            hi = torch.tensor(
                [float(clamp_cfg[n][1]) for n in action_joint_names],
                dtype=torch.float32, device=self.device,
            )
            if torch.any(lo > hi):
                raise RuntimeError("position_clamp lower must be <= upper for every joint")
            self._clamp_lower = lo.unsqueeze(0)
            self._clamp_upper = hi.unsqueeze(0)

        passive_names = tuple(self.cfg.passive_joint_names)
        passive_cols = resolve_action_columns_by_joint_name(self, passive_names)
        action_joint_ids = _action_joint_ids_in_column_order(self)
        self._passive_joint_names = passive_names
        self._passive_action_cols = torch.tensor(passive_cols, dtype=torch.long, device=self.device)
        self._passive_joint_ids = torch.tensor(
            [action_joint_ids[c] for c in passive_cols], dtype=torch.long, device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self._action_clip is None:
            self._applied_raw_actions.copy_(self._raw_actions)
        else:
            self._applied_raw_actions.copy_(
                torch.clamp(
                    self._raw_actions,
                    min=self._action_clip[0],
                    max=self._action_clip[1],
                )
            )
        if self._raw_action_transform == "tanh":
            self._transformed_actions.copy_(torch.tanh(self._applied_raw_actions))
        else:
            self._transformed_actions.copy_(self._applied_raw_actions)
        self._processed_actions = self._transformed_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        if self._clamp_lower is not None:
            # Shared-adapter clamp: identical bounds to the deploy ActionAdapter.
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clamp_lower, max=self._clamp_upper
            )
        else:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
            self._processed_actions = torch.clamp(
                self._processed_actions, min=limits[..., 0], max=limits[..., 1]
            )
        if self._passive_action_cols.numel() > 0:
            default_q = self._asset.data.default_joint_pos.index_select(-1, self._passive_joint_ids)
            self._processed_actions.index_copy_(-1, self._passive_action_cols, default_q)
            self._applied_raw_actions.index_fill_(-1, self._passive_action_cols, 0.0)
            self._transformed_actions.index_fill_(-1, self._passive_action_cols, 0.0)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._applied_raw_actions.zero_()
            self._transformed_actions.zero_()
        else:
            self._applied_raw_actions[env_ids] = 0.0
            self._transformed_actions[env_ids] = 0.0

    @property
    def applied_raw_actions(self) -> torch.Tensor:
        """Actor output accepted for this tick, with passive columns set to zero."""
        return self._applied_raw_actions

    @property
    def transformed_actions(self) -> torch.Tensor:
        """Residual-domain action used for q_des, with passive columns set to zero."""
        return self._transformed_actions

    @property
    def passive_joint_names(self) -> tuple[str, ...]:
        return self._passive_joint_names


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction

    raw_action_transform: str = "identity"
    """Transform before residual scaling: ``identity`` (legacy) or bounded ``tanh``."""

    action_clip: tuple[float, float] | None = None
    """Optional global raw-action clip ``(lower, upper)`` applied before transform and feedback."""

    passive_joint_names: tuple[str, ...] = ()
    """Exact joint names held at default q and zeroed in the applied-action feedback (e.g. the head)."""

    position_clamp: dict[str, tuple[float, float]] | None = None
    """Explicit per-joint (lower, upper) clamp in rad, keyed by exact joint name.

    Set from the shared deploy ``action_adapter.yaml`` so training clamps position targets with
    exactly the same bounds as the deploy ActionAdapter. ``None`` falls back to the articulation's
    soft joint limits.
    """
