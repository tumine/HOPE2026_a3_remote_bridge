#!/usr/bin/env python3
"""Validate the portable A3 policy bundle without Isaac, CUDA, or onnxruntime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx.reference import ReferenceEvaluator


BUNDLE_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = BUNDLE_DIR / "policy"
ONNX_PATH = POLICY_DIR / "hope_pingpong.onnx"
MANIFEST_PATH = POLICY_DIR / "policy_manifest.json"
PROVENANCE_PATH = POLICY_DIR / "provenance.json"
JOINT_ORDER_PATH = BUNDLE_DIR / "config/joint_order_agibot_a3.yaml"
ACTION_ADAPTER_PATH = BUNDLE_DIR / "config/action_adapter.yaml"
RUNTIME_PATH = BUNDLE_DIR / "config/runtime.yaml"
TRAINING_CONTRACT_PATH = BUNDLE_DIR / "config/policy_training_contract.yaml"
EXPECTED_GENERATION = "full_body_uniform_action_return_physics_v1"
EXPECTED_CHECKPOINT_ITERATION = 21500


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value_info) -> list[int | str | None]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return dims


def main() -> int:
    required = (
        ONNX_PATH,
        MANIFEST_PATH,
        PROVENANCE_PATH,
        JOINT_ORDER_PATH,
        ACTION_ADAPTER_PATH,
        RUNTIME_PATH,
        TRAINING_CONTRACT_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle is missing files: {missing}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    joint_order = list(
        yaml.safe_load(JOINT_ORDER_PATH.read_text(encoding="utf-8"))["joint_order"]
    )
    adapter = yaml.safe_load(ACTION_ADAPTER_PATH.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(RUNTIME_PATH.read_text(encoding="utf-8"))
    training_contract = yaml.safe_load(
        TRAINING_CONTRACT_PATH.read_text(encoding="utf-8")
    )

    if len(joint_order) != 31 or len(set(joint_order)) != 31:
        raise ValueError("joint order must contain 31 unique names")
    if manifest["joint_order"] != joint_order:
        raise ValueError("manifest joint order does not match bundle joint-order YAML")
    if set(adapter["default_q"]) != set(joint_order):
        raise ValueError("ActionAdapter default_q keys do not match joint order")
    if adapter.get("raw_action_transform") != "identity":
        raise ValueError("this policy requires raw_action_transform=identity")
    if [float(value) for value in adapter.get("action_clip", [])] != [-100.0, 100.0]:
        raise ValueError("this policy requires action_clip=[-100, 100]")
    if float(adapter["action_scale"]) != 0.25:
        raise ValueError("this trained policy requires action_scale=0.25")
    for side in ("lower", "upper"):
        if set(adapter["joint_position_clamp"][side]) != set(joint_order):
            raise ValueError(f"ActionAdapter clamp.{side} keys do not match joint order")
    if runtime["policy"]["onnx_path"] != "../policy/hope_pingpong.onnx":
        raise ValueError("runtime does not point to the bundled ONNX")
    if float(runtime["control_hz"]) != 50.0:
        raise ValueError("runtime control_hz must be 50")
    lifecycle = runtime["lifecycle"]
    expected_ready = {
        "ready_target_rel_base_w": [
            0.3201315701007843,
            -0.6952511668205261,
            -0.057795558124780655,
        ],
        "ready_target_vel_w": [
            2.3803303241729736,
            0.6271703243255615,
            1.0374835729599,
        ],
    }
    for key, expected in expected_ready.items():
        if not np.allclose(lifecycle.get(key), expected, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"runtime {key} does not match the validated READY anchor")
    if float(lifecycle["ready_time_to_strike"]) != 1.0:
        raise ValueError("runtime ready_time_to_strike must be 1.0")
    if int(lifecycle["ready_swing_side"]) != 1:
        raise ValueError("runtime ready_swing_side must be +1")
    if training_contract["policy_generation"] != EXPECTED_GENERATION:
        raise ValueError("training contract policy generation is stale")
    if int(training_contract["checkpoint_iteration"]) != EXPECTED_CHECKPOINT_ITERATION:
        raise ValueError("training contract checkpoint iteration is stale")

    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("ONNX must have one input and one output")
    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    if input_info.name != "observation" or _shape(input_info)[-1] != 111:
        raise ValueError(f"bad ONNX input: {input_info.name} {_shape(input_info)}")
    if output_info.name != "raw_action" or _shape(output_info)[-1] != 31:
        raise ValueError(f"bad ONNX output: {output_info.name} {_shape(output_info)}")

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    expected_metadata = {
        "contract_name": "hope_pingpong",
        "obs_dim": "111",
        "action_dim": "31",
        "control_rate_hz": "50",
        "observation_normalization": "none",
        "joint_order": ",".join(joint_order),
        "policy_generation": EXPECTED_GENERATION,
        "checkpoint_iteration": str(EXPECTED_CHECKPOINT_ITERATION),
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"ONNX metadata {key!r}: {metadata.get(key)!r} != {value!r}")

    if provenance["source_checkpoint_iteration"] != EXPECTED_CHECKPOINT_ITERATION:
        raise ValueError(
            f"provenance checkpoint iteration is not {EXPECTED_CHECKPOINT_ITERATION}"
        )
    if provenance["policy_generation"] != EXPECTED_GENERATION:
        raise ValueError("provenance policy generation is stale")
    actual_hash = _sha256(ONNX_PATH)
    if actual_hash != provenance["onnx_sha256"]:
        raise ValueError("ONNX SHA256 does not match provenance")
    if _sha256(MANIFEST_PATH) != provenance["policy_manifest_sha256"]:
        raise ValueError("policy manifest SHA256 does not match provenance")
    for path_key, hash_key in (
        ("bundled_training_env", "bundled_training_env_sha256"),
        ("bundled_training_agent", "bundled_training_agent_sha256"),
        ("bundled_training_worktree_diff", "bundled_training_worktree_diff_sha256"),
    ):
        snapshot = BUNDLE_DIR / provenance[path_key]
        if not snapshot.is_file() or _sha256(snapshot) != provenance[hash_key]:
            raise ValueError(f"bundled provenance snapshot failed validation: {snapshot}")

    # Execute both a single item and a dynamic batch through ONNX's reference backend.
    evaluator = ReferenceEvaluator(model)
    zero_out = evaluator.run(None, {"observation": np.zeros((1, 111), np.float32)})[0]
    rng = np.random.default_rng(20260723)
    batch_out = evaluator.run(
        None, {"observation": rng.normal(0.0, 0.1, size=(4, 111)).astype(np.float32)}
    )[0]
    if zero_out.shape != (1, 31) or batch_out.shape != (4, 31):
        raise ValueError(
            f"unexpected inference shapes: zero={zero_out.shape}, batch={batch_out.shape}"
        )
    if not np.isfinite(zero_out).all() or not np.isfinite(batch_out).all():
        raise ValueError("ONNX inference produced non-finite values")

    report = {
        "status": "ok",
        "policy": str(ONNX_PATH.relative_to(BUNDLE_DIR)),
        "onnx_sha256": actual_hash,
        "input": _shape(input_info),
        "output": _shape(output_info),
        "joint_count": len(joint_order),
        "checkpoint_iteration": provenance["source_checkpoint_iteration"],
        "policy_generation": provenance["policy_generation"],
        "raw_action_transform": adapter["raw_action_transform"],
        "action_clip": adapter["action_clip"],
        "action_scale": adapter["action_scale"],
        "ready_anchor": "validated",
        "reference_inference": "finite",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
