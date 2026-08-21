#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    require_hash(bundle["policy"]["onnx"], bundle["policy"]["onnx_sha256"])
    require_hash(bundle["policy"]["manifest"], bundle["policy"]["manifest_sha256"])
    require_hash(bundle["source_configs"]["agent"], bundle["source_configs"]["agent_sha256"])
    require_hash(bundle["source_configs"]["environment"], bundle["source_configs"]["env_sha256"])
    require_hash(bundle["source_configs"]["git_diff"], bundle["source_configs"]["git_diff_sha256"])
    require_hash(bundle["deployment_configs"]["runtime"], bundle["deployment_configs"]["runtime_sha256"])
    require_hash(bundle["deployment_configs"]["strike_box"], bundle["deployment_configs"]["strike_box_sha256"])
    require_hash(bundle["deployment_configs"]["action_adapter"], bundle["deployment_configs"]["action_adapter_sha256"])
    require_hash(bundle["deployment_configs"]["joint_order"], bundle["deployment_configs"]["joint_order_sha256"])
    require_hash(bundle["physics"]["ball_physics"], bundle["physics"]["ball_physics_sha256"])
    require_hash(bundle["simulation_model"]["path"], bundle["simulation_model"]["xml_sha256"])
    require_hash(bundle["ready_reference"]["motion"], bundle["ready_reference"]["motion_sha256"])
    require_hash(bundle["ready_reference"]["metadata"], bundle["ready_reference"]["metadata_sha256"])
    require_hash(bundle["validation"]["ready_report"], bundle["validation"]["ready_report_sha256"])

    manifest = json.loads((ROOT / bundle["policy"]["manifest"]).read_text(encoding="utf-8"))
    assert manifest["obs_dim"] == 111
    assert manifest["action_dim"] == 31
    assert manifest["control_rate_hz"] == 50
    assert manifest["observation_normalization"] == "none"
    assert manifest["base_station_semantics"] == "moving_lateral_target_v1"
    assert manifest["provenance"]["checkpoint"]["iteration"] == 64500
    assert manifest["provenance"]["checkpoint"]["sha256"] == bundle["policy"]["checkpoint_sha256"]

    report = json.loads((ROOT / bundle["validation"]["ready_report"]).read_text(encoding="utf-8"))
    assert report["test"] == "mujoco_deploy_ready_no_command"
    assert report["result"]["fell"] is False
    assert report["result"]["nonfinite_observations"] == 0
    assert report["result"]["nonfinite_actions"] == 0

    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime unavailable; skipped runtime signature inference")
    else:
        session = ort.InferenceSession(
            str(ROOT / bundle["policy"]["onnx"]), providers=["CPUExecutionProvider"]
        )
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        assert inp.name == "observation" and inp.shape[-1] == 111
        assert out.name == "raw_action" and out.shape[-1] == 31
        assert inp.shape[0] in (1, "batch") and out.shape[0] in (1, "batch")
        print(
            "[ok] ONNX signature: observation[batch,111] -> "
            "raw_action[batch,31] (deploy batch=1)"
        )

    print("[ok] bundle verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
