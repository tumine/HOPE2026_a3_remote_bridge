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
    require_hash("provenance/agent.yaml", bundle["source_configs"]["agent_sha256"])
    require_hash("provenance/env.yaml", bundle["source_configs"]["env_sha256"])
    require_hash("config/ball_physics.yaml", bundle["physics"]["ball_physics_sha256"])
    require_hash(bundle["simulation_model"]["path"], bundle["simulation_model"]["xml_sha256"])

    manifest = json.loads((ROOT / bundle["policy"]["manifest"]).read_text(encoding="utf-8"))
    assert manifest["obs_dim"] == 111
    assert manifest["action_dim"] == 31
    assert manifest["control_rate_hz"] == 50
    assert manifest["observation_normalization"] == "none"
    assert manifest["base_station_semantics"] == "moving_lateral_target_v1"

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
