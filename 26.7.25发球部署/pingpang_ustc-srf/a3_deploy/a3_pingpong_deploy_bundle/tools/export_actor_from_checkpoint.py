#!/usr/bin/env python3
"""Export the plain HOPE actor from an rsl_rl checkpoint without Isaac or CUDA.

This tool is intentionally strict for the public A3 policy contract:

    111 -> 512 -> 256 -> 128 -> 31, ELU hidden activations

It exports only the deterministic actor. The PPO critic, optimizer, exploration
standard deviation, and training state are not part of deployment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml
from onnx.reference import ReferenceEvaluator


BUNDLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BUNDLE_DIR.parents[1]
POLICY_GENERATION = "full_body_uniform_action_return_physics_v1"
EXPECTED_CHECKPOINT_ITERATION = 21500
SOURCE_RUN = "2026-07-26_01-47-04"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/logs/rsl_rl/hope_pingpong"
    / SOURCE_RUN
    / "model_21500.pt"
)
DEFAULT_SOURCE_DIFF = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/logs/rsl_rl/hope_pingpong"
    / SOURCE_RUN
    / "git/HOPE.diff"
)
EXPORTER_PATH = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking"
    / "whole_body_tracking/utils/exporter.py"
)
JOINT_ORDER_PATH = BUNDLE_DIR / "config/joint_order_agibot_a3.yaml"

EXPECTED_DIMS = (111, 512, 256, 128, 31)
ACTOR_PARAMETER_KEYS = (
    "actor.0.weight",
    "actor.0.bias",
    "actor.2.weight",
    "actor.2.bias",
    "actor.4.weight",
    "actor.4.bias",
    "actor.6.weight",
    "actor.6.bias",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_exporter_module():
    spec = importlib.util.spec_from_file_location("hope_bundle_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import exporter: {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActorHolder(nn.Module):
    """Minimal object exposing the `.actor` attribute expected by the exporter."""

    def __init__(self, actor: nn.Sequential):
        super().__init__()
        self.actor = actor


def _actor_from_checkpoint(checkpoint: Path) -> tuple[ActorHolder, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("checkpoint must contain a model_state_dict mapping")
    state = payload["model_state_dict"]

    missing = [key for key in ACTOR_PARAMETER_KEYS if key not in state]
    if missing:
        raise ValueError(f"checkpoint is missing actor parameters: {missing}")

    actual_dims = (
        int(state["actor.0.weight"].shape[1]),
        int(state["actor.0.weight"].shape[0]),
        int(state["actor.2.weight"].shape[0]),
        int(state["actor.4.weight"].shape[0]),
        int(state["actor.6.weight"].shape[0]),
    )
    if actual_dims != EXPECTED_DIMS:
        raise ValueError(
            f"actor architecture {actual_dims} does not match required {EXPECTED_DIMS}"
        )

    actor = nn.Sequential(
        nn.Linear(111, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 31),
    )
    actor_state = {
        key.removeprefix("actor."): state[key].detach().cpu()
        for key in ACTOR_PARAMETER_KEYS
    }
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    return ActorHolder(actor).eval(), payload


def _make_bundle_metadata(
    onnx_path: Path, joint_order: list[str], checkpoint_iteration: int
) -> None:
    model = onnx.load(onnx_path)
    props = {entry.key: entry.value for entry in model.metadata_props}
    props.update(
        {
            "contract_name": "hope_pingpong",
            "obs_dim": "111",
            "action_dim": "31",
            "control_rate_hz": "50",
            "observation_normalization": "none",
            "action_adapter_config": "config/action_adapter.yaml",
            "joint_order_source": "config/joint_order_agibot_a3.yaml",
            "joint_order": ",".join(joint_order),
            "policy_generation": POLICY_GENERATION,
            "checkpoint_iteration": str(checkpoint_iteration),
        }
    )
    onnx.helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


def _update_manifest(manifest_path: Path, checkpoint_iteration: int) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "action_adapter_config": "config/action_adapter.yaml",
            "joint_order_source": "config/joint_order_agibot_a3.yaml",
            "policy_generation": POLICY_GENERATION,
            "checkpoint_iteration": checkpoint_iteration,
            "provenance_file": "provenance.json",
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _verify_pytorch_onnx_parity(holder: ActorHolder, onnx_path: Path) -> float:
    """Return max absolute actor error on a deterministic dynamic-batch input."""

    rng = np.random.default_rng(20260723)
    observation = rng.normal(0.0, 0.25, size=(8, 111)).astype(np.float32)
    with torch.inference_mode():
        expected = holder.actor(torch.from_numpy(observation)).numpy()
    actual = ReferenceEvaluator(onnx.load(onnx_path)).run(
        None, {"observation": observation}
    )[0]
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(error) or error > 1.0e-5:
        raise ValueError(f"PyTorch/ONNX actor mismatch: max_abs_error={error}")
    return error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE_DIR / "policy")
    parser.add_argument("--source-diff", type=Path, default=DEFAULT_SOURCE_DIFF)
    parser.add_argument(
        "--source-base-commit",
        default="42530cac9717642176a70d43bafdce0a2b0c5e08",
        help="Committed base recorded when the run began; the run also had HOPE.diff.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    source_diff = args.source_diff.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_diff.is_file():
        raise FileNotFoundError(source_diff)

    joint_doc = yaml.safe_load(JOINT_ORDER_PATH.read_text(encoding="utf-8"))
    joint_order = list(joint_doc["joint_order"])
    if len(joint_order) != 31:
        raise ValueError(f"expected 31 joints, got {len(joint_order)}")

    holder, payload = _actor_from_checkpoint(checkpoint)
    checkpoint_iter = int(payload.get("iter", -1))
    if checkpoint_iter != EXPECTED_CHECKPOINT_ITERATION:
        raise ValueError(
            f"expected checkpoint iteration {EXPECTED_CHECKPOINT_ITERATION}, "
            f"got {checkpoint_iter}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = _load_exporter_module()
    onnx_path_str, manifest_path_str = exporter.export_policy(
        holder,
        str(output_dir),
        joint_names=joint_order,
        normalizer=None,
        onnx_filename="hope_pingpong.onnx",
    )
    onnx_path = Path(onnx_path_str)
    manifest_path = Path(manifest_path_str)
    _make_bundle_metadata(onnx_path, joint_order, checkpoint_iter)
    _update_manifest(manifest_path, checkpoint_iter)
    parity_error = _verify_pytorch_onnx_parity(holder, onnx_path)

    provenance = {
        "bundle_contract": "hope_pingpong",
        "policy_status": "current_sim_candidate_not_hardware_certified",
        "policy_generation": POLICY_GENERATION,
        "source_run": SOURCE_RUN,
        "source_checkpoint": _repo_relative(checkpoint),
        "source_checkpoint_iteration": checkpoint_iter,
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_base_commit": args.source_base_commit,
        "deployed_contract_commit": "28d0074a8fb595d4c730cbc13a146fdd53c0a170",
        "source_worktree_diff": _repo_relative(source_diff),
        "source_worktree_diff_sha256": _sha256(source_diff),
        "bundled_training_env": "provenance/training_env.yaml",
        "bundled_training_env_sha256": _sha256(
            BUNDLE_DIR / "provenance/training_env.yaml"
        ),
        "bundled_training_agent": "provenance/training_agent.yaml",
        "bundled_training_agent_sha256": _sha256(
            BUNDLE_DIR / "provenance/training_agent.yaml"
        ),
        "bundled_training_worktree_diff": "provenance/training_worktree.diff",
        "bundled_training_worktree_diff_sha256": _sha256(
            BUNDLE_DIR / "provenance/training_worktree.diff"
        ),
        "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "export_method": "cpu_actor_only_exact_state_dict",
        "actor_architecture": [111, 512, 256, 128, 31],
        "hidden_activation": "elu",
        "observation_normalization": "none",
        "control_rate_hz": 50,
        "onnx_file": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "policy_manifest_sha256": _sha256(manifest_path),
        "pytorch_onnx_max_abs_error": parity_error,
        "training_snapshot_at_iteration_21500": {
            "mean_reward": 14.143362998962402,
            "mean_episode_length": 495.2799987792969,
            "racket_pos_error_m": 0.2804376780986786,
            "racket_vel_error_mps": 0.6911897659301758,
            "racket_normal_error_rad": 0.312211811542511,
            "contact_rate": 0.9962550401687622,
            "net_clear_rate": 0.9605103731155396,
            "return_success": 0.947412371635437,
            "both_feet_contact": 0.9697650671005249,
            "base_tilt_rad": 0.06857311725616455,
            "station_error_m": 0.05009888857603073,
            "target_curriculum_progress": 1.0,
        },
        "mujoco_idle_ready_anchor_10s": {
            "report": "validation/mujoco_idle_ready_anchor_10s.json",
            "min_pelvis_height_m": 1.0653315348318637,
            "max_station_drift_m": 0.021174385941500916,
            "fell": False,
        },
        "mujoco_ready_anchor_smoke": {
            "report": "validation/mujoco_ready_anchor_smoke_diagnostics.json",
            "successes": 8,
            "attempts": 8,
            "falls": 0,
        },
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {onnx_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
