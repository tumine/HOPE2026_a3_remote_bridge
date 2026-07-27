#!/usr/bin/env python3

"""Offline gate for the balance_v2 A3 no-mocap deployment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from a3_rl_contract import ACTION_DIM, A3RLContract, OnnxActor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO_ROOT / "26.7.25发球部署/a3_deploy/a3_pingpong_deploy_bundle"
DEFAULT_TRAINING_SOURCE = REPO_ROOT / "26.7.25发球部署/pingpang_ustc-srf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate A3 policy, training/sim2sim parity and offline inference"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--training-source", type=Path, default=DEFAULT_TRAINING_SOURCE
    )
    parser.add_argument("--latency-runs", type=int, default=200)
    args = parser.parse_args()
    if args.latency_runs <= 0:
        parser.error("latency-runs must be positive")
    return args


def standing_observation(contract: A3RLContract, side: str) -> np.ndarray:
    return contract.build_observation(
        q=contract.default_q,
        dq=np.zeros(ACTION_DIM),
        pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        pelvis_gyro_body=(0.0, 0.0, 0.0),
        last_action=np.zeros(ACTION_DIM),
        side=side,
        time_to_strike=1.0,
    )


def assert_example_compatible(
    contract: A3RLContract, side: str, observation: np.ndarray
) -> None:
    path = contract.bundle_dir / f"examples/no_mocap_{side}_observation.json"
    example = np.asarray(
        json.loads(path.read_text(encoding="utf-8"))["observation"],
        dtype=np.float32,
    )
    if example.shape != (111,):
        raise ValueError(f"bad bundled no-mocap example: {path}")
    # The bundle's human-readable example writes base_forward_x as exactly 1.0;
    # the training/reference implementation divides by (norm + 1e-6). All other
    # values must agree, and this one documented epsilon is bounded explicitly.
    difference = np.abs(example - observation)
    allowed = np.zeros(111, dtype=np.float32)
    allowed[99] = 1.1e-6
    if np.any(difference > allowed):
        index = int(np.argmax(difference - allowed))
        raise ValueError(
            f"{side} bundled example differs at observation[{index}]: "
            f"{example[index]} != {observation[index]}"
        )


def main() -> int:
    args = parse_args()
    contract = A3RLContract.load(args.bundle, args.training_source)
    actor = OnnxActor(contract)

    side_reports = {}
    latency_ms: list[float] = []
    maximum_raw = 0.0
    for side in ("forehand", "backhand"):
        initial = standing_observation(contract, side)
        assert_example_compatible(contract, side, initial)
        last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        minimum_target = np.full(ACTION_DIM, np.inf)
        maximum_target = np.full(ACTION_DIM, -np.inf)
        for tick in range(contract.clock.total_steps):
            observation = contract.build_observation(
                q=contract.default_q,
                dq=np.zeros(ACTION_DIM),
                pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                pelvis_gyro_body=(0.0, 0.0, 0.0),
                last_action=last_action,
                side=side,
                time_to_strike=contract.clock.time_to_strike(tick),
            )
            started = time.perf_counter()
            raw_action = actor.infer(observation)
            latency_ms.append((time.perf_counter() - started) * 1000.0)
            last_action, target = contract.decode_action(raw_action, policy_blend=1.0)
            maximum_raw = max(maximum_raw, float(np.max(np.abs(raw_action))))
            minimum_target = np.minimum(minimum_target, target)
            maximum_target = np.maximum(maximum_target, target)
            if not np.all(target >= contract.lower) or not np.all(target <= contract.upper):
                raise ValueError(f"{side} target escaped the mechanical clamp")
            if np.any(last_action[[3, 4]]) or not np.array_equal(
                target[[3, 4]], contract.default_q[[3, 4]]
            ):
                raise ValueError("passive head action contract failed")
        side_reports[side] = {
            "ticks": contract.clock.total_steps,
            "time_to_strike_first_s": contract.clock.time_to_strike(0),
            "time_to_strike_strike_s": contract.clock.time_to_strike(
                contract.clock.hold_steps + contract.clock.strike_lead_steps
            ),
            "time_to_strike_last_s": contract.clock.time_to_strike(
                contract.clock.total_steps - 1
            ),
            "q_target_min_rad": float(np.min(minimum_target)),
            "q_target_max_rad": float(np.max(maximum_target)),
        }

    probe = standing_observation(contract, "forehand")
    for _ in range(10):
        actor.infer(probe)
    timed = []
    for _ in range(args.latency_runs):
        started = time.perf_counter()
        actor.infer(probe)
        timed.append((time.perf_counter() - started) * 1000.0)

    report = {
        "status": "ok",
        "actuation": "not_created",
        "policy_generation": contract.policy_generation,
        "checkpoint_iteration": contract.checkpoint_iteration,
        "onnx": str(contract.onnx_path),
        "control_hz": 50,
        "observation_dim": 111,
        "action_dim": 31,
        "no_mocap_clock": {
            "hold_ticks": contract.clock.hold_steps,
            "guarded_motion_ticks": (
                contract.clock.strike_lead_steps
                + contract.clock.follow_through_steps
                + 1
            ),
            "total_ticks": contract.clock.total_steps,
        },
        "sim2sim_ready_observation": {
            side: {
                "racket_target_rel_base_m": contract.ready_targets[side].rel_base.tolist(),
                "racket_target_vel_world_mps": contract.ready_targets[
                    side
                ].velocity_world.tolist(),
                "time_to_strike_s": 1.0,
                "swing_side": contract.ready_targets[side].swing_side,
                "fixed_station_error_xy": [0.0, 0.0],
            }
            for side in ("forehand", "backhand")
        },
        "gains": {
            "training_equals_checkpoint_snapshot": True,
            "training_equals_sim2sim": True,
            "kp_min": float(np.min(contract.kp)),
            "kp_max": float(np.max(contract.kp)),
            "kd_min": float(np.min(contract.kd)),
            "kd_max": float(np.max(contract.kd)),
        },
        "full_policy_rollout": side_reports,
        "max_abs_raw_action_static_rollout": maximum_raw,
        "inference_ms": {
            "runs": args.latency_runs,
            "mean": float(np.mean(timed)),
            "p95": float(np.percentile(timed, 95)),
            "max": float(np.max(timed)),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
