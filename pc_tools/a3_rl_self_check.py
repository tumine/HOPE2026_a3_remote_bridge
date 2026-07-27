#!/usr/bin/env python3

"""Offline contract/frame/lifecycle gate for the current real deployment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from a3_mocap_frame import CanonicalTableCalibrator, MarkerToPelvis
from a3_rl_contract import (
    ACTION_DIM,
    A3RLContract,
    FOREHAND,
    OnnxActor,
    RacketCommand,
    SwingLifecycle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "26.7.25发球部署/pingpang_ustc-srf"
DEFAULT_BUNDLE = DEFAULT_SOURCE / "a3_deploy/a3_pingpong_deploy_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate current A3 RL deployment")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--training-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--latency-runs", type=int, default=200)
    args = parser.parse_args()
    if args.latency_runs <= 0:
        parser.error("latency-runs must be positive")
    return args


def synthetic_table_gate() -> dict:
    length, width, offset = 2.74, 1.525, 0.01
    table_center = np.asarray((length / 2.0, -width / 2.0, offset))
    base_world = np.asarray((-0.50, -0.7625, 0.30))
    calibrator = CanonicalTableCalibrator(4, length, width, offset, 1e-6, 1e-6)
    frame = None
    for _ in range(4):
        frame = calibrator.add(table_center, (0.0, 0.0, 0.0, 1.0), base_world, True)
    if frame is None:
        raise ValueError("synthetic PPMocap table did not calibrate")
    if not np.allclose(frame.origin_world, 0.0, atol=1e-12):
        raise ValueError(f"canonical near-left origin mismatch: {frame.origin_world}")
    base, quat = frame.pelvis_from_marker(
        base_world,
        (0.0, 0.0, 0.0, 1.0),
        MarkerToPelvis.from_values((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    if not np.allclose(base, base_world, atol=1e-12):
        raise ValueError("synthetic BotA3 position transform mismatch")
    if not np.allclose(quat, (1.0, 0.0, 0.0, 0.0), atol=1e-12):
        raise ValueError("synthetic BotA3 orientation transform mismatch")
    return {"origin_world": frame.origin_world.tolist(), "base_table": base.tolist()}


def main() -> int:
    args = parse_args()
    contract = A3RLContract.load(args.bundle, args.training_source)
    actor = OnnxActor(contract)
    frame_report = synthetic_table_gate()

    # Isaac/MuJoCo first-frame pelvis [0,0,1.0664] translated into the
    # canonical table frame by [-0.5,-0.7625,-0.76].
    base = np.asarray((-0.50, -0.7625, 0.3064))
    target = contract.ready_target(base)
    observation = contract.build_observation(
        contract.default_q,
        np.zeros(ACTION_DIM),
        base,
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        np.zeros(ACTION_DIM),
        base[:2],
        target,
    )
    expected_tail = np.concatenate(
        (
            contract.ready_rel_base_w,
            contract.ready_velocity_w,
            (contract.ready_time_to_strike, float(contract.ready_swing_side)),
        )
    )
    if not np.allclose(observation[103:111], expected_tail, atol=1e-7):
        raise ValueError("READY observation[103:111] differs from runtime anchor")
    if not np.allclose(observation[101:103], 0.0, atol=0.0):
        raise ValueError("startup fixed_station_error_xy must be zero")

    drift = np.asarray((0.012, -0.018, -0.004))
    moved_base = base + drift
    moved_observation = contract.build_observation(
        contract.default_q,
        np.zeros(ACTION_DIM),
        moved_base,
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        np.zeros(ACTION_DIM),
        base[:2],
        target,
    )
    if not np.allclose(
        moved_observation[103:106], contract.ready_rel_base_w - drift, atol=1e-7
    ):
        raise ValueError("READY virtual world target moved with the live pelvis")

    # Hardware deliberately splits the two attitude sources: IMU roll/pitch
    # drives gravity while calibrated PPMocap yaw drives world/table heading.
    yaw_90 = math.sqrt(0.5)
    split_observation = contract.build_observation(
        contract.default_q,
        np.zeros(ACTION_DIM),
        base,
        (1.0, 0.0, 0.0, 0.0),
        (yaw_90, 0.0, 0.0, yaw_90),
        (0.0, 0.0, 0.0),
        np.zeros(ACTION_DIM),
        base[:2],
        target,
    )
    if not np.allclose(split_observation[96:99], (0.0, 0.0, -1.0), atol=1e-7):
        raise ValueError("projected gravity is not isolated to the IMU quaternion")
    if not np.allclose(split_observation[99:101], (0.0, 1.0), atol=2e-6):
        raise ValueError("base forward is not isolated to the PPMocap quaternion")

    # Compare against the updated training repository's clean-room reference
    # implementation, rather than only checking our own expected constants.
    reference_dir = args.training_source / "a3_deploy/a3_deploy_example/reference"
    sys.path.insert(0, str(reference_dir))
    from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter
    from a3_deploy_onnx_ref_pingpong.observation import (
        ObsTarget,
        RobotState,
        build_observation,
    )

    reference_observation = build_observation(
        RobotState(
            base_pos_w=base,
            base_quat_w=np.asarray((1.0, 0.0, 0.0, 0.0)),
            base_ang_vel_b=np.zeros(3),
            q=contract.default_q,
            qd=np.zeros(ACTION_DIM),
        ),
        ObsTarget(
            pos_w=target.position_w,
            vel_w=target.velocity_w,
            time_to_strike=target.time_to_strike,
            swing_side=float(target.swing_side),
        ),
        np.zeros(ACTION_DIM),
        contract.default_q,
        base[:2],
    )
    if not np.array_equal(observation, reference_observation):
        raise ValueError("deployment observation differs from updated reference")

    lifecycle = SwingLifecycle(contract, base)
    command = RacketCommand(
        1,
        0,
        FOREHAND,
        base + np.asarray((0.30, -0.64, 0.35)),
        np.asarray((2.4, 0.6, 0.9)),
        1.0,
    )
    clocks = []
    for tick in range(91):
        strike_target = lifecycle.update(command if tick == 0 else None, base)
        clocks.append(strike_target.time_to_strike)
        lifecycle.advance()
    if not (
        np.isclose(clocks[0], 1.0)
        and np.isclose(clocks[50], 0.0)
        and np.isclose(clocks[90], -0.8)
        and lifecycle.phase == lifecycle.READY
        and lifecycle.active_task_id is None
    ):
        raise ValueError("91-tick +1.0 -> 0 -> -0.8 lifecycle mismatch")
    recovered_ready = lifecycle.update(None, moved_base)
    if not np.allclose(recovered_ready.position_w, target.position_w, atol=1e-12):
        raise ValueError("recovered READY did not restore the first-frame world target")

    raw_probe = np.zeros(ACTION_DIM)
    raw_probe[0], raw_probe[3], raw_probe[4] = 101.0, 50.0, -50.0
    applied, desired = contract.decode_action(raw_probe)
    if applied[0] != 100.0 or np.any(applied[[3, 4]]):
        raise ValueError("clip/passive-head applied-action contract failed")
    if not np.all(desired >= contract.lower) or not np.all(desired <= contract.upper):
        raise ValueError("decoded target escaped mechanical clamp")
    reference_adapter = ActionAdapter.from_yaml(
        contract.bundle_dir / "config/action_adapter.yaml"
    )
    reference_applied = reference_adapter.clip_raw_action(raw_probe)
    reference_applied[[3, 4]] = 0.0
    reference_desired = reference_adapter.decode(reference_applied)
    reference_desired[[3, 4]] = reference_adapter.default_q[[3, 4]]
    if not np.array_equal(applied, reference_applied) or not np.array_equal(
        desired, reference_desired
    ):
        raise ValueError("deployment action decode differs from updated reference")

    raw = actor.infer(observation)
    applied, desired = contract.decode_action(raw)
    if np.any(applied[[3, 4]]) or not np.array_equal(
        desired[[3, 4]], contract.default_q[[3, 4]]
    ):
        raise ValueError("live actor passive-head contract failed")
    for _ in range(10):
        actor.infer(observation)
    timings = []
    for _ in range(args.latency_runs):
        started = time.perf_counter()
        actor.infer(observation)
        timings.append((time.perf_counter() - started) * 1000.0)

    print(
        json.dumps(
            {
                "status": "ok",
                "actuation": "not_created",
                "policy_generation": contract.policy_generation,
                "checkpoint_iteration": contract.checkpoint_iteration,
                "onnx_sha256": __import__("hashlib").sha256(
                    contract.onnx_path.read_bytes()
                ).hexdigest(),
                "action": {
                    "transform": "identity",
                    "clip": list(contract.action_clip),
                    "uniform_scale": float(contract.action_scale[0]),
                    "passive_head": [3, 4],
                },
                "ready_observation_103_111": observation[103:111].tolist(),
                "ready_world_target": {
                    "mode": "fixed_sim_first_frame_target_minus_live_base",
                    "position_w": target.position_w.tolist(),
                    "drift_probe_103_106": moved_observation[103:106].tolist(),
                },
                "lifecycle": {"ticks": 91, "clock": [1.0, 0.0, -0.8]},
                "updated_reference_parity": {
                    "observation_exact": True,
                    "action_exact": True,
                },
                "hardware_attitude_sources": {
                    "projected_gravity": "pelvis_imu_quaternion",
                    "base_forward_xy": "ppmocap_pelvis_quaternion",
                    "tilt_guard": "pelvis_imu_quaternion",
                },
                "synthetic_mocap_frame": frame_report,
                "actor_ready_max_abs_raw": float(np.max(np.abs(raw))),
                "inference_ms": {
                    "runs": args.latency_runs,
                    "mean": float(np.mean(timings)),
                    "p95": float(np.percentile(timings, 95)),
                    "max": float(np.max(timings)),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
