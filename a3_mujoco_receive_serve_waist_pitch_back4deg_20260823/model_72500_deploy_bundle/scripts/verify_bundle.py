#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ITERATION = 72500
EXPECTED_CHECKPOINT_SHA256 = (
    "aa3e99f631f8f56c388329faa1739b2a4cc2dbff2ec35310086d855c84969e0e"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(relative: str, expected: str) -> None:
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"missing required file: {relative}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA256 mismatch for {relative}: {actual} != {expected}")
    print(f"[ok] {relative}: {actual}")


def main() -> int:
    bundle = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    for asset in bundle["hashed_assets"]:
        require_hash(asset["path"], asset["sha256"])

    policy_manifest_path = ROOT / bundle["policy"]["manifest"]
    manifest = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    assert manifest["obs_dim"] == 111
    assert manifest["action_dim"] == 31
    assert manifest["control_rate_hz"] == 50
    assert manifest["observation_normalization"] == "none"
    assert manifest["base_station_semantics"] == "moving_lateral_target_v1"
    assert manifest["onnx_signature"]["input"]["shape"] == ["batch", 111]
    assert manifest["onnx_signature"]["output"]["shape"] == ["batch", 31]
    assert manifest["provenance"]["checkpoint"]["iteration"] == EXPECTED_ITERATION
    assert manifest["provenance"]["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert bundle["policy"]["checkpoint_iteration"] == EXPECTED_ITERATION
    assert bundle["policy"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256

    runtime_path = ROOT / bundle["deployment"]["runtime"]
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert runtime["control_hz"] == 50.0
    assert runtime["observation_normalization"] == "none"
    lifecycle = runtime["lifecycle"]
    assert lifecycle["ready_target_rel_base_w"] == [0.45, -0.25, 0.08]
    assert lifecycle["ready_target_vel_w"] == [0.0, 0.0, 0.0]
    assert lifecycle["ready_time_to_strike"] == 1.0
    assert lifecycle["ready_swing_side"] == 0
    gains = runtime["simulation"]["pd_gains"]["joints"]
    assert gains["waist_yaw_joint"] == {"kp": 85.0, "kd": 3.0}
    assert gains["waist_roll_joint"] == {"kp": 500.0, "kd": 2.0}
    assert gains["waist_pitch_joint"] == {"kp": 500.0, "kd": 2.0}

    sys.path.insert(0, str(ROOT / "reference"))
    from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter

    adapter = ActionAdapter.from_yaml(ROOT / bundle["deployment"]["action_adapter"])
    assert adapter.locked_joint_names == ("waist_roll_joint", "waist_pitch_joint")
    assert adapter.default_q[1] == 0.0 and adapter.default_q[2] == 0.0
    probe = np.linspace(-10.0, 10.0, 31, dtype=np.float64)
    applied = adapter.clip_raw_action(probe)
    desired = adapter.decode(applied)
    assert applied[1] == 0.0 and applied[2] == 0.0
    assert desired[1] == 0.0 and desired[2] == 0.0

    ready_motion = ROOT / bundle["ready_reference"]["motion"]
    with np.load(ready_motion) as motion:
        assert motion["joint_pos"].shape == (1, 31)
        assert motion["joint_vel"].shape == (1, 31)
        assert np.allclose(motion["joint_pos"][:, 1:3], 0.0, atol=1.0e-8)
        assert np.allclose(motion["joint_vel"][:, 1:3], 0.0, atol=1.0e-8)

    ready_report = json.loads(
        (ROOT / bundle["validation"]["ready_report"]).read_text(encoding="utf-8")
    )
    assert ready_report["test"] == "mujoco_deploy_ready_no_command"
    assert ready_report["result"]["fell"] is False
    assert ready_report["result"]["nonfinite_observations"] == 0
    assert ready_report["result"]["nonfinite_actions"] == 0
    for joint in ("waist_roll_joint", "waist_pitch_joint"):
        desired_stats = ready_report["result"]["waist"][joint]["desired_deg"]
        assert desired_stats["min"] == 0.0 and desired_stats["max"] == 0.0

    continuous = json.loads(
        (ROOT / bundle["validation"]["continuous_diagnostics"]).read_text(encoding="utf-8")
    )
    assert continuous["attempts"] == 20
    assert continuous["falls"] == 0
    assert continuous["nonfinite_observations"] == 0
    assert continuous["nonfinite_actions"] == 0
    assert continuous["mujoco_contacts"] == 20
    assert continuous["success_rate"] >= 0.8

    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime unavailable; skipped runtime inference")
    else:
        session = ort.InferenceSession(
            str(ROOT / bundle["policy"]["onnx"]), providers=["CPUExecutionProvider"]
        )
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        assert inp.name == "observation" and inp.shape == ["batch", 111]
        assert out.name == "raw_action" and out.shape == ["batch", 31]
        output = session.run(None, {inp.name: np.zeros((1, 111), dtype=np.float32)})[0]
        assert output.shape == (1, 31) and np.all(np.isfinite(output))
        print("[ok] ONNX signature and zero-input inference")

    print("[ok] model_72500 bundle verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
