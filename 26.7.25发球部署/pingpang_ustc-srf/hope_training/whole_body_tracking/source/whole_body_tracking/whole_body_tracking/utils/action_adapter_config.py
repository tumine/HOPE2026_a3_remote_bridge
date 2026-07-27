"""Loader for the SHARED ActionAdapter configuration (training side).

The raw-action -> joint-target mapping is defined in exactly ONE file, read by both
the training package and the reference deploy runner:

    a3_deploy/a3_deploy_example/config/action_adapter.yaml

This module resolves that file (walking up from here to the repo root), parses it
against the canonical 31-joint order, and exposes the resulting ``default_q``,
``raw_action_transform``, raw-action clip, ``action_scale`` and joint-position clamp arrays so
:class:`~whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg.HOPEPingPongEnvCfg`
can wire them into the articulation default pose and the action term. The deploy
runner applies the identical transform through its own ``ActionAdapter``:

    adapter_action = transform(raw_action)
    q_des = default_q + adapter_action * action_scale
    q_des = clip(q_des, clamp_lower, clamp_upper)

Pure NumPy + PyYAML — importable without Isaac / torch (so the parity test in
``tests/test_action_adapter_parity.py`` can compare this loader against the deploy
``ActionAdapter`` directly).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

# Repo-root-relative locations of the two shared contract files.
SHARED_ADAPTER_RELPATH = os.path.join(
    "a3_deploy", "a3_deploy_example", "config", "action_adapter.yaml"
)
JOINT_ORDER_RELPATH = os.path.join("hope_training", "config", "joint_order_agibot_a3.yaml")


def _find_upward(relpath: str, start: str | None = None) -> str:
    """Walk up from ``start`` (default: this file) until ``relpath`` exists."""
    d = os.path.dirname(os.path.abspath(start or __file__))
    for _ in range(16):
        cand = os.path.join(d, relpath)
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        f"shared config not found walking up from {start or __file__}: {relpath}"
    )


def find_shared_action_adapter_config() -> str:
    """Absolute path of the canonical shared ``action_adapter.yaml``."""
    return _find_upward(SHARED_ADAPTER_RELPATH)


def load_joint_order(path: str | None = None) -> tuple[str, ...]:
    """The canonical 31-joint order from ``hope_training/config/joint_order_agibot_a3.yaml``."""
    path = path or _find_upward(JOINT_ORDER_RELPATH)
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    order = tuple(str(n) for n in doc["joint_order"])
    if len(order) != 31 or len(set(order)) != 31:
        raise ValueError(f"joint order must list 31 unique joints, got {len(order)}")
    return order


def resolve_canonical_joint_indices(
    articulation_joint_names: tuple[str, ...] | list[str],
    canonical_joint_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[int, ...]:
    """Map canonical policy columns to an articulation's internal joint indices.

    Isaac/PhysX is free to enumerate an articulation according to its kinematic
    topology.  That internal order is not a deploy contract and, for the bundled
    A3 URDF on Isaac Sim 5.1, differs from HOPE's semantic 31-joint order.  This
    helper validates exact name coverage and returns indices such that::

        articulation_tensor[..., indices]

    is in canonical policy / motion / deployment order.

    The mapping is deliberately name based: no breadth/depth-first traversal
    assumption leaks into the policy interface.
    """
    articulation = tuple(str(name) for name in articulation_joint_names)
    canonical = tuple(str(name) for name in (canonical_joint_names or load_joint_order()))

    duplicate_articulation = sorted({name for name in articulation if articulation.count(name) > 1})
    duplicate_canonical = sorted({name for name in canonical if canonical.count(name) > 1})
    if duplicate_articulation or duplicate_canonical:
        raise ValueError(
            "joint names must be unique; "
            f"articulation duplicates={duplicate_articulation}, canonical duplicates={duplicate_canonical}"
        )

    articulation_set = set(articulation)
    canonical_set = set(canonical)
    missing = [name for name in canonical if name not in articulation_set]
    unexpected = [name for name in articulation if name not in canonical_set]
    if missing or unexpected or len(articulation) != len(canonical):
        raise ValueError(
            "articulation and canonical joint sets differ; "
            f"missing={missing}, unexpected={unexpected}, "
            f"articulation_count={len(articulation)}, canonical_count={len(canonical)}"
        )

    by_name = {name: index for index, name in enumerate(articulation)}
    return tuple(by_name[name] for name in canonical)


def _index_tuple(indices, count: int) -> tuple[int, ...]:
    """Normalize an Isaac index slice/list/tensor without importing Isaac or torch."""
    if isinstance(indices, slice):
        return tuple(range(count))[indices]
    if hasattr(indices, "detach"):
        indices = indices.detach()
    if hasattr(indices, "cpu"):
        indices = indices.cpu()
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    return tuple(int(index) for index in indices)


def validate_live_joint_order_contract(
    env,
    *,
    action_name: str = "joint_pos",
    motion_command_name: str | None = "motion",
    observation_groups: tuple[str, ...] = ("policy", "critic"),
) -> tuple[int, ...]:
    """Validate every live simulator boundary that carries 31 joint columns.

    This intentionally uses duck typing so the contract checker stays importable
    in pure unit tests. It verifies more than tensor dimensions: action columns,
    joint observations, and the motion command must all use the same canonical to
    PhysX name mapping.
    """
    canonical = tuple(load_joint_order())
    robot = env.scene["robot"]
    articulation = tuple(robot.data.joint_names)
    canonical_ids = resolve_canonical_joint_indices(articulation, canonical)

    action_term = env.action_manager.get_term(action_name)
    action_names = tuple(action_term._joint_names)
    action_ids = _index_tuple(action_term._joint_ids, len(articulation))
    if action_names != canonical or action_ids != canonical_ids:
        raise ValueError(
            "action joint columns are not canonical; "
            f"names={list(action_names)}, ids={list(action_ids)}, "
            f"expected_names={list(canonical)}, expected_ids={list(canonical_ids)}"
        )

    observation_manager = env.observation_manager
    for group_name in observation_groups:
        if group_name not in observation_manager.active_terms:
            continue
        term_names = list(observation_manager.active_terms[group_name])
        term_cfgs = observation_manager._group_obs_term_cfgs[group_name]
        for term_name in ("joint_pos", "joint_vel"):
            if term_name not in term_names:
                continue
            term_cfg = term_cfgs[term_names.index(term_name)]
            asset_cfg = term_cfg.params.get("asset_cfg")
            if asset_cfg is None:
                raise ValueError(f"{group_name}.{term_name} has no explicit canonical asset_cfg")
            obs_ids = _index_tuple(asset_cfg.joint_ids, len(articulation))
            if tuple(asset_cfg.joint_names or ()) != canonical or obs_ids != canonical_ids:
                raise ValueError(
                    f"{group_name}.{term_name} is not canonical; "
                    f"names={list(asset_cfg.joint_names or ())}, ids={list(obs_ids)}, "
                    f"expected_ids={list(canonical_ids)}"
                )

    if motion_command_name is not None:
        motion_term = env.command_manager.get_term(motion_command_name)
        motion_ids = _index_tuple(motion_term.joint_indexes, len(articulation))
        if motion_ids != canonical_ids:
            raise ValueError(
                "motion reference columns are not canonical; "
                f"ids={list(motion_ids)}, expected_ids={list(canonical_ids)}"
            )

    return canonical_ids


@dataclass(frozen=True)
class ActionAdapterConfig:
    """The shared adapter constants, resolved into canonical joint order."""

    joint_names: tuple[str, ...]
    default_q: np.ndarray     # (31,) rad
    action_scale: np.ndarray  # (31,) residual scale
    clamp_lower: np.ndarray   # (31,) rad
    clamp_upper: np.ndarray   # (31,) rad
    raw_action_transform: str = "identity"
    action_clip: tuple[float, float] | None = None

    def clip_raw_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Clip actor output to the configured applied-action interval."""
        raw = np.asarray(raw_action, dtype=np.float64)
        if raw.ndim == 0 or raw.shape[-1] != len(self.joint_names):
            raise ValueError(
                f"raw_action trailing dimension must be {len(self.joint_names)}, got {raw.shape}"
            )
        if self.action_clip is None:
            return raw.copy()
        return np.clip(raw, self.action_clip[0], self.action_clip[1])

    def transform_raw_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Apply the configured bounded-domain transform without changing its shape.

        The optional action clip is applied first. ``identity`` is the
        backward-compatible transform for legacy YAML files; ``tanh`` remains
        available for checkpoints trained with that older contract.
        """
        raw = self.clip_raw_action(raw_action)
        if self.raw_action_transform == "identity":
            return raw.copy()
        if self.raw_action_transform == "tanh":
            return np.tanh(raw)
        # The loader validates this field. Keep a defensive branch for direct
        # dataclass construction in downstream tools.
        raise ValueError(f"unsupported raw_action_transform: {self.raw_action_transform!r}")

    def decode(self, raw_action: np.ndarray) -> np.ndarray:
        """Reference transform (identical to the deploy ``ActionAdapter.decode``)."""
        raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if raw.shape[0] != len(self.joint_names):
            raise ValueError(f"raw_action must be length {len(self.joint_names)}")
        adapter_action = self.transform_raw_action(raw)
        q_des = self.default_q + adapter_action * self.action_scale
        return np.clip(q_des, self.clamp_lower, self.clamp_upper)

    # -- dict views used to build the Isaac Lab cfg (exact joint names, no regex) --
    def default_q_by_name(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.joint_names, self.default_q)}

    def action_scale_by_name(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.joint_names, self.action_scale)}

    def position_clamp_by_name(self) -> dict[str, tuple[float, float]]:
        return {
            n: (float(lo), float(hi))
            for n, lo, hi in zip(self.joint_names, self.clamp_lower, self.clamp_upper)
        }


def _resolve_per_joint(spec, joint_names: tuple[str, ...], field_name: str) -> np.ndarray:
    """Accept an ordered length-31 list or a ``{joint_name: value}`` map (same as deploy)."""
    n = len(joint_names)
    if isinstance(spec, dict):
        missing = [name for name in joint_names if name not in spec]
        if missing:
            raise ValueError(f"{field_name} is missing joints: {missing[:3]}...")
        return np.array([float(spec[name]) for name in joint_names], dtype=np.float64)
    arr = np.asarray(spec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{field_name} must be length {n}, got {arr.shape[0]}")
    return arr


def _resolve_raw_action_transform(spec) -> str:
    """Normalize the optional raw-action transform shared by train and deploy."""
    transform = "identity" if spec is None else str(spec).strip().lower()
    if transform not in {"identity", "tanh"}:
        raise ValueError(
            "raw_action_transform must be one of ['identity', 'tanh'], "
            f"got {spec!r}"
        )
    return transform


def _resolve_action_clip(spec) -> tuple[float, float] | None:
    """Normalize an optional global ``[lower, upper]`` raw-action clip."""
    if spec is None:
        return None
    values = np.asarray(spec, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("action_clip must contain two finite values: [lower, upper]")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError("action_clip lower must be <= upper")
    return lower, upper


def load_action_adapter_config(
    path: str | None = None, joint_names: tuple[str, ...] | None = None
) -> ActionAdapterConfig:
    """Parse the shared adapter YAML into canonical-order arrays."""
    path = path or find_shared_action_adapter_config()
    joint_names = joint_names or load_joint_order()
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    default_q = _resolve_per_joint(doc["default_q"], joint_names, "default_q")
    raw_action_transform = _resolve_raw_action_transform(doc.get("raw_action_transform"))
    action_clip = _resolve_action_clip(doc.get("action_clip"))

    scale_spec = doc["action_scale"]
    if isinstance(scale_spec, (int, float)):
        action_scale = np.full(len(joint_names), float(scale_spec), dtype=np.float64)
    else:
        action_scale = _resolve_per_joint(scale_spec, joint_names, "action_scale")

    clamp = doc["joint_position_clamp"]
    clamp_lower = _resolve_per_joint(clamp["lower"], joint_names, "joint_position_clamp.lower")
    clamp_upper = _resolve_per_joint(clamp["upper"], joint_names, "joint_position_clamp.upper")
    if np.any(clamp_lower > clamp_upper):
        raise ValueError("joint_position_clamp lower must be <= upper for every joint")

    return ActionAdapterConfig(
        joint_names=joint_names,
        default_q=default_q,
        action_scale=action_scale,
        clamp_lower=clamp_lower,
        clamp_upper=clamp_upper,
        raw_action_transform=raw_action_transform,
        action_clip=action_clip,
    )
