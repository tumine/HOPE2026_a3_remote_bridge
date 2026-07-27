#!/usr/bin/env python3

"""One-time 10 s MuJoCo gate for the current READY observation/action path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from a3_rl_contract import ACTION_DIM, A3RLContract, OnnxActor, base_tilt_radians


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "26.7.25发球部署/pingpang_ustc-srf"
DEFAULT_BUNDLE = DEFAULT_SOURCE / "a3_deploy/a3_pingpong_deploy_bundle"
DEFAULT_RUNTIME = (
    DEFAULT_SOURCE
    / "a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model_21500 READY in MuJoCo")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--training-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--max-tilt-rad", type=float, default=0.35)
    parser.add_argument("--min-base-z", type=float, default=0.90)
    parser.add_argument("--max-station-drift-m", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_dir = args.training_source / "a3_deploy/a3_deploy_example/reference"
    sys.path.insert(0, str(reference_dir))
    try:
        from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
        from a3_deploy_onnx_ref_pingpong.sim_bridge import MujocoDirectBridge
    except ImportError as exc:
        raise RuntimeError("MuJoCo/reference runtime unavailable; run setup_rl_venv.sh") from exc

    contract = A3RLContract.load(args.bundle, args.training_source)
    runtime = RuntimeConfig.load(args.runtime)
    actor = OnnxActor(contract)
    bridge = MujocoDirectBridge(
        runtime.model_xml_path,
        control_dt=1.0 / contract.control_hz,
        reset_joint_pos=contract.default_q,
    )
    ticks = max(1, int(round(args.duration_s * contract.control_hz)))
    last_action = np.zeros(ACTION_DIM, dtype=np.float64)
    max_tilt = 0.0
    min_base_z = float("inf")
    max_drift = 0.0
    max_raw = 0.0
    max_command_state_error = 0.0
    fixed_station = None
    ready_target = None
    try:
        for _ in range(ticks):
            state = bridge.read_state()
            values = (
                state.base_pos_w,
                state.base_quat_w,
                state.base_ang_vel_b,
                state.q,
                state.qd,
            )
            if not all(np.isfinite(value).all() for value in values):
                raise ValueError("MuJoCo state became non-finite")
            if fixed_station is None:
                fixed_station = state.base_pos_w[:2].copy()
                # The simulator's first-frame virtual target stays fixed in
                # world while the pelvis moves underneath it.
                ready_target = contract.ready_target(state.base_pos_w)
            tilt = base_tilt_radians(state.base_quat_w)
            base_z = float(state.base_pos_w[2])
            drift = float(np.linalg.norm(state.base_pos_w[:2] - fixed_station))
            max_tilt = max(max_tilt, tilt)
            min_base_z = min(min_base_z, base_z)
            max_drift = max(max_drift, drift)
            if tilt > args.max_tilt_rad:
                raise ValueError(f"tilt {tilt:.4f} exceeds {args.max_tilt_rad:.4f} rad")
            if base_z < args.min_base_z:
                raise ValueError(f"base z {base_z:.4f} below {args.min_base_z:.4f} m")
            if drift > args.max_station_drift_m:
                raise ValueError(
                    f"station drift {drift:.4f} exceeds {args.max_station_drift_m:.4f} m"
                )
            assert ready_target is not None
            target = ready_target
            observation = contract.build_observation(
                state.q,
                state.qd,
                state.base_pos_w,
                state.base_quat_w,
                state.base_quat_w,
                state.base_ang_vel_b,
                last_action,
                fixed_station,
                target,
            )
            raw = actor.infer(observation)
            last_action, desired = contract.decode_action(raw)
            max_raw = max(max_raw, float(np.max(np.abs(raw))))
            max_command_state_error = max(
                max_command_state_error, float(np.max(np.abs(desired - state.q)))
            )
            bridge.write_targets(desired, contract.kp, contract.kd)
            bridge.step()
    finally:
        bridge.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "actuation": "not_created",
                "policy_generation": contract.policy_generation,
                "checkpoint_iteration": contract.checkpoint_iteration,
                "mode": "fixed_first_frame_ready_world_target",
                "ticks": ticks,
                "duration_s": ticks / contract.control_hz,
                "max_tilt_deg": float(np.degrees(max_tilt)),
                "min_base_z_m": min_base_z,
                "max_station_drift_m": max_drift,
                "max_abs_raw_action": max_raw,
                "max_command_state_error_rad": max_command_state_error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
