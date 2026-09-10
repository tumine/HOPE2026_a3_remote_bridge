#!/usr/bin/env python3
"""Verify the frozen policy and the tested -4 degree compensation contract."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    package = json.loads((ROOT / "PACKAGE_INFO.json").read_text(encoding="utf-8"))
    policy = ROOT / package["receive_policy"]["onnx"]
    assert sha256(policy) == package["receive_policy"]["onnx_sha256"]

    robot_xml = ROOT / "model_72500_deploy_bundle/robot_model/agibot_a3/xml/agibot_a3.xml"
    assert sha256(robot_xml) == package["physics"]["robot_xml_sha256"]
    ball_config = ROOT / package["physics"]["ball_config"]
    assert sha256(ball_config) == package["physics"]["ball_config_sha256"]

    source_path = ROOT / "pc_tools/a3_serve_receive_mujoco.py"
    source = source_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(source_path))
    required_source_fragments = (
        'default=-4.0',
        '"--auto-receive-first"',
        'target[2] = self.serve_waist_pitch_from + alpha * (',
        'target[2] = self.serve_waist_pitch_rad + alpha * (',
        'if self.auto_serve_track == "1"',
    )
    for fragment in required_source_fragments:
        assert fragment in source, f"missing source contract: {fragment}"

    validation = json.loads(
        (ROOT / package["validated_result"]["file"]).read_text(encoding="utf-8")
    )
    expected = package["validated_result"]
    assert validation["status"] == "ok"
    assert validation["auto_cycle_order"] == "receive_then_serve"
    assert validation["serve_waist_pitch_target_deg"] == -4.0
    assert validation["completed_cycles"] == expected["completed_cycles"]
    assert validation["physical_racket_contacts"] == expected["racket_contacts"]
    assert validation["physical_net_clears"] == expected["net_clears"]
    assert validation["physical_opponent_bounces"] == expected["opponent_bounces"]
    assert len(validation["serve_ready_entries"]) == expected["completed_cycles"]
    print("[成功] model_72500、球物理、状态机和 -4deg 腰 pitch 验证合同一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
