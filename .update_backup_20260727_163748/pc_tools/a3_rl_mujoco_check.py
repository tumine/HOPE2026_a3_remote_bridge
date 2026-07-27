#!/usr/bin/env python3

"""Headless dynamic gate for the deployed no-mocap READY/ball lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from a3_rl_contract import ACTION_DIM, A3RLContract, OnnxActor, base_tilt_radians


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO_ROOT / "26.7.25发球部署/a3_deploy/a3_pingpong_deploy_bundle"
DEFAULT_SOURCE = REPO_ROOT / "26.7.25发球部署/pingpang_ustc-srf"
DEFAULT_RUNTIME = (
    DEFAULT_SOURCE
    / "a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact PC no-mocap WAIT/F/B sequence in headless MuJoCo"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--training-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--max-tilt-rad", type=float, default=0.35)
    parser.add_argument("--min-base-z", type=float, default=0.90)
    parser.add_argument("--max-command-state-error-rad", type=float, default=1.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_dir = args.training_source / "a3_deploy/a3_deploy_example/reference"
    sys.path.insert(0, str(reference_dir))
    try:
        from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
        from a3_deploy_onnx_ref_pingpong.sim_bridge import MujocoDirectBridge
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo/reference runtime is unavailable; run scripts/setup_rl_venv.sh"
        ) from exc

    contract = A3RLContract.load(args.bundle, args.training_source)
    actor = OnnxActor(contract)
    runtime = RuntimeConfig.load(args.runtime)
    bridge = MujocoDirectBridge(
        runtime.model_xml_path,
        control_dt=1.0 / 50.0,
        reset_joint_pos=contract.default_q,
    )
    # Mirrors operator behavior: wait five seconds, inject one F ball, wait,
    # inject one B ball, then prove another ten seconds of no-ball standing.
    segments = (
        ("wait_forehand", 250, "forehand", True),
        ("ball_forehand", contract.clock.total_steps, "forehand", False),
        ("wait_after_forehand", 250, "forehand", True),
        ("ball_backhand", contract.clock.total_steps, "backhand", False),
        ("wait_after_backhand", 500, "backhand", True),
    )
    last_action = np.zeros(ACTION_DIM, dtype=np.float64)
    reports = []
    total_ticks = 0
    try:
        for label, ticks, side, ready in segments:
            max_tilt = 0.0
            min_base_z = float("inf")
            max_raw = 0.0
            max_command_state_error = 0.0
            for local_tick in range(ticks):
                state = bridge.read_state()
                values = (
                    state.base_pos_w,
                    state.base_quat_w,
                    state.base_ang_vel_b,
                    state.q,
                    state.qd,
                )
                if not all(np.isfinite(value).all() for value in values):
                    raise ValueError(f"{label}: MuJoCo state became non-finite")
                tilt = base_tilt_radians(state.base_quat_w)
                base_z = float(state.base_pos_w[2])
                max_tilt = max(max_tilt, tilt)
                min_base_z = min(min_base_z, base_z)
                if tilt > args.max_tilt_rad:
                    raise ValueError(
                        f"{label}: tilt {tilt:.4f} exceeds {args.max_tilt_rad:.4f} rad"
                    )
                if base_z < args.min_base_z:
                    raise ValueError(
                        f"{label}: base z {base_z:.4f} below {args.min_base_z:.4f} m"
                    )
                tts = (
                    1.0
                    if ready
                    else contract.clock.time_to_strike(local_tick)
                )
                observation = contract.build_observation(
                    state.q,
                    state.qd,
                    state.base_quat_w,
                    state.base_ang_vel_b,
                    last_action,
                    side,
                    tts,
                    target_override=(contract.ready_targets[side] if ready else None),
                )
                raw_action = actor.infer(observation)
                max_raw = max(max_raw, float(np.max(np.abs(raw_action))))
                last_action, target_q = contract.decode_action(raw_action, 1.0)
                command_state_error = float(np.max(np.abs(target_q - state.q)))
                max_command_state_error = max(
                    max_command_state_error, command_state_error
                )
                if command_state_error > args.max_command_state_error_rad:
                    raise ValueError(
                        f"{label}: command-to-state error {command_state_error:.4f} "
                        f"exceeds {args.max_command_state_error_rad:.4f} rad"
                    )
                bridge.write_targets(target_q, contract.kp, contract.kd)
                bridge.step()
                total_ticks += 1
            reports.append(
                {
                    "segment": label,
                    "ticks": ticks,
                    "max_tilt_deg": float(np.degrees(max_tilt)),
                    "min_base_z_m": min_base_z,
                    "max_abs_raw_action": max_raw,
                    "max_command_state_error_rad": max_command_state_error,
                }
            )
    finally:
        bridge.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "actuation": "not_created",
                "sequence": "WAIT->F->WAIT->B->WAIT",
                "total_ticks": total_ticks,
                "total_duration_s": total_ticks / 50.0,
                "policy_blend": 1.0,
                "max_command_state_error_rad": args.max_command_state_error_rad,
                "fixed_station_error_xy": [0.0, 0.0],
                "segments": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
