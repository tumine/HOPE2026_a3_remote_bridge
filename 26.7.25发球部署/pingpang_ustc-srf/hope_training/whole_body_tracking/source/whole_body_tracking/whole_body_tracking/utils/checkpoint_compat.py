"""Compatibility helpers for inference-only use of rsl_rl checkpoints.

Training resume and policy inference have different compatibility requirements.  A resume must
restore actor, critic, optimizer and iteration state exactly.  Evaluation/export only need the
actor and, when it was enabled during training, the actor observation normalizer.  Loading a whole
checkpoint for inference makes harmless critic/optimizer changes break old policies.

The helpers in this module deliberately:

* stage checkpoints on CPU;
* restore only ``model_state_dict["actor.*"]`` into the live actor;
* never touch the critic, exploration standard deviation, optimizer or iteration counter;
* reconstruct actor architecture settings from the checkpoint's adjacent ``params/agent.yaml``;
* require and restore empirical-normalization statistics when training used them.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any


def sha256_file(path: str) -> str:
    """Return the lowercase SHA-256 digest of ``path`` without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config_paths(checkpoint_path: str) -> list[tuple[str, str]]:
    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    candidates = [
        ("agent", os.path.join(run_dir, "params", "agent.yaml")),
        ("environment", os.path.join(run_dir, "params", "env.yaml")),
    ]
    return [(role, path) for role, path in candidates if os.path.isfile(path)]


def _load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    # mmap keeps large critic/optimizer storages off the Python heap unless they are actually read.
    # Fall back for checkpoints written with PyTorch's legacy, non-zip serializer.
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
            mmap=True,
        )
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint root must be a mapping, got {type(payload).__name__}")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint has no mapping-valued 'model_state_dict'")
    return payload


def _actor_linear_shapes(model_state: dict[str, Any]) -> tuple[int, list[int], int] | None:
    """Infer ``input, hidden_dims, output`` from rsl_rl's sequential actor weights."""
    weights: list[tuple[int, Any]] = []
    for name, value in model_state.items():
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "actor" and parts[1].isdigit() and parts[2] == "weight":
            if getattr(value, "ndim", None) == 2:
                weights.append((int(parts[1]), value))
    weights.sort(key=lambda item: item[0])
    if not weights:
        return None
    input_dim = int(weights[0][1].shape[1])
    layer_outputs = [int(value.shape[0]) for _, value in weights]
    return input_dim, layer_outputs[:-1], layer_outputs[-1]


def inspect_checkpoint(
    checkpoint_path: str,
    agent_config_path: str | None = None,
) -> dict[str, Any]:
    """Inspect the inference-relevant checkpoint contract without creating an Isaac environment."""
    payload = _load_checkpoint(checkpoint_path)
    model_state = payload["model_state_dict"]
    actor_keys = sorted(key for key in model_state if key.startswith("actor."))
    if not actor_keys:
        raise ValueError("checkpoint model_state_dict contains no 'actor.*' parameters")

    normalizer_state = payload.get("obs_norm_state_dict")
    if normalizer_state is not None and not isinstance(normalizer_state, dict):
        raise ValueError("checkpoint 'obs_norm_state_dict' must be a mapping")
    has_normalizer = isinstance(normalizer_state, dict) and bool(normalizer_state)
    if isinstance(normalizer_state, dict) and not normalizer_state:
        raise ValueError("checkpoint contains an empty 'obs_norm_state_dict'")

    source_configs = _source_config_paths(checkpoint_path)
    if agent_config_path is not None:
        agent_config_path = os.path.abspath(agent_config_path)
        if not os.path.isfile(agent_config_path):
            raise FileNotFoundError(f"source agent config not found: {agent_config_path}")
        source_configs = [
            ("agent", agent_config_path),
            *((role, path) for role, path in source_configs if role != "agent"),
        ]
    return {
        "path": os.path.abspath(checkpoint_path),
        "iteration": payload.get("iter"),
        "actor_keys": actor_keys,
        "actor_linear_shapes": _actor_linear_shapes(model_state),
        "has_observation_normalizer": has_normalizer,
        "agent_config_path": next((path for role, path in source_configs if role == "agent"), None),
        "source_config_paths": source_configs,
    }


def _load_agent_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import yaml

    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"source agent config must be a mapping: {path}")
    return data


def configure_runner_for_inference(
    agent_cfg: Any,
    checkpoint_path: str,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt a runner config to the checkpoint's actor and normalization contract.

    The saved ``params/agent.yaml`` is authoritative for activation and actor hidden dimensions.
    If it is unavailable, inference is rejected because activation functions cannot be inferred
    from a parameter-only state dictionary.
    """
    inspection = inspection or inspect_checkpoint(checkpoint_path)
    source_cfg = _load_agent_config(inspection.get("agent_config_path"))
    if source_cfg is None:
        raise RuntimeError(
            "no source agent config was found next to the checkpoint. Actor activation cannot be "
            "inferred safely from a state_dict even when the layer sizes match. Keep the original "
            "params/agent.yaml next to the checkpoint or pass its path explicitly."
        )
    saved_policy = source_cfg.get("policy", {})
    if saved_policy and not isinstance(saved_policy, dict):
        raise ValueError("source agent config field 'policy' must be a mapping")

    for required in ("actor_hidden_dims", "activation"):
        if required not in saved_policy:
            raise RuntimeError(
                f"source agent config does not contain policy.{required}; actor architecture "
                "cannot be reconstructed safely"
            )
    shapes = inspection.get("actor_linear_shapes")
    configured_hidden = [int(dim) for dim in saved_policy["actor_hidden_dims"]]
    if shapes is not None and configured_hidden != shapes[1]:
        raise RuntimeError(
            "source agent config does not match the checkpoint actor: "
            f"policy.actor_hidden_dims={configured_hidden}, checkpoint={shapes[1]}"
        )
    for yaml_key, attr_name in (
        ("actor_hidden_dims", "actor_hidden_dims"),
        ("activation", "activation"),
        ("init_noise_std", "init_noise_std"),
    ):
        if yaml_key in saved_policy:
            value = saved_policy[yaml_key]
            if yaml_key == "actor_hidden_dims":
                value = [int(dim) for dim in value]
            setattr(agent_cfg.policy, attr_name, value)

    has_normalizer = bool(inspection["has_observation_normalizer"])
    runner_section = source_cfg.get("runner") or {}
    if not isinstance(runner_section, dict):
        raise ValueError("source agent config field 'runner' must be a mapping")
    source_empirical = source_cfg.get(
        "empirical_normalization",
        runner_section.get("empirical_normalization"),
    )
    if source_empirical is not None and bool(source_empirical) != has_normalizer:
        raise RuntimeError(
            "source agent config and checkpoint disagree about empirical observation "
            f"normalization (config={bool(source_empirical)}, "
            f"checkpoint_has_stats={has_normalizer}). Refusing ambiguous inference."
        )
    agent_cfg.empirical_normalization = has_normalizer
    return inspection


def load_actor_for_inference(
    runner: Any,
    checkpoint_path: str,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load only actor (+ required observation normalizer) into an initialized runner."""
    inspection = inspection or inspect_checkpoint(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path)
    model_state = payload["model_state_dict"]
    actor_state = {
        name[len("actor.") :]: value for name, value in model_state.items() if name.startswith("actor.")
    }
    if not actor_state:
        raise ValueError("checkpoint model_state_dict contains no 'actor.*' parameters")

    try:
        runner.alg.policy.actor.load_state_dict(actor_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint actor is incompatible with the initialized actor. Check the source "
            "params/agent.yaml, actor observation dimension, action dimension and activation."
        ) from exc

    if inspection["has_observation_normalizer"]:
        normalizer = getattr(runner, "obs_normalizer", None)
        if normalizer is None or not getattr(normalizer, "state_dict", lambda: {})():
            raise RuntimeError(
                "checkpoint requires empirical observation normalization, but the runner was not "
                "constructed with an EmpiricalNormalization module"
            )
        try:
            normalizer.load_state_dict(payload["obs_norm_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "checkpoint observation normalizer is incompatible with the actor observation shape"
            ) from exc
        normalizer.eval()

    runner.alg.policy.eval()
    return inspection


def build_checkpoint_provenance(
    checkpoint_path: str,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build portable checkpoint/config provenance for ``policy_manifest.json``."""
    inspection = inspection or inspect_checkpoint(checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    run_dir = os.path.dirname(checkpoint_path)
    configs = []
    for role, config_path in inspection["source_config_paths"]:
        configs.append(
            {
                "role": role,
                "path": os.path.relpath(config_path, run_dir),
                "sha256": sha256_file(config_path),
            }
        )
    return {
        "checkpoint": {
            "file": os.path.basename(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "iteration": inspection.get("iteration"),
        },
        "source_configs": configs,
    }
