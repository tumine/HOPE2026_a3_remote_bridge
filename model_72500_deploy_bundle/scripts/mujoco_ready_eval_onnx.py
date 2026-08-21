#!/usr/bin/env python3
"""Evaluate the deploy READY/no-command posture of a HOPE ONNX actor in MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_iteration(onnx_path: Path) -> int | None:
    """Recover ``N`` from an exported ``.../model_N/...onnx`` directory."""

    for component in reversed(onnx_path.resolve().parts):
        match = re.fullmatch(r"model_(\d+)", component)
        if match:
            return int(match.group(1))
    return None


def _rotation_pitch(rotation: np.ndarray) -> float:
    return float(math.asin(np.clip(-rotation[2, 0], -1.0, 1.0)))


def _quat_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "final": float(values[-1]),
    }


def _world_geom_aabb(model, data, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    center = model.geom_aabb[geom_id, :3]
    half = model.geom_aabb[geom_id, 3:]
    signs = np.asarray(
        [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    corners = center[None, :] + signs * half[None, :]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world = data.geom_xpos[geom_id][None, :] + corners @ rotation.T
    return np.min(world, axis=0), np.max(world, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--model-xml", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--ready-reference", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--view",
        action="store_true",
        help="Launch the MuJoCo passive viewer for the READY/no-command rollout.",
    )
    parser.add_argument("--table-near-x", type=float, default=0.50)
    parser.add_argument("--table-length", type=float, default=2.74)
    parser.add_argument("--table-width", type=float, default=1.525)
    parser.add_argument("--table-height", type=float, default=0.76)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    for path in (
        args.onnx,
        args.runtime_config,
        args.model_xml,
        args.ready_reference,
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    reference_dir = args.reference_dir.resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    sys.path.insert(0, str(reference_dir))

    import mujoco

    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.joint_order import (
        HEAD_INDICES,
        ISAAC_JOINT_VELOCITY_LIMITS,
        JOINT_NAMES,
    )
    from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
    from a3_deploy_onnx_ref_pingpong.observation import build_observation
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.racket_command import QueueRacketCommandSource
    from a3_deploy_onnx_ref_pingpong.sim_bridge import MujocoDirectBridge

    cfg = RuntimeConfig.load(args.runtime_config.resolve())
    cfg.onnx_path = args.onnx.resolve()
    cfg.model_xml_path = args.model_xml.resolve()
    bridge = MujocoDirectBridge(
        cfg.model_xml_path,
        control_dt=cfg.control_dt,
        launch_viewer=args.view,
        reset_joint_pos=cfg.action_adapter.default_q,
        joint_velocity_limits=np.asarray(ISAAC_JOINT_VELOCITY_LIMITS, dtype=np.float64),
    )
    policy = OnnxPolicy(cfg.onnx_path)
    lifecycle = SwingLifecycle(cfg.lifecycle)
    source = QueueRacketCommandSource()
    default_q = cfg.action_adapter.default_q.copy()
    last_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    head_indices = list(HEAD_INDICES)

    ready_meta = yaml.safe_load(
        args.ready_reference.with_suffix(".yaml").read_text(encoding="utf-8")
    )
    with np.load(args.ready_reference) as ready_data:
        ready_q = ready_data["joint_pos"][0].astype(np.float64)
        ready_body_pos = ready_data["body_pos_w"][0].astype(np.float64)
    ready_body_names = list(ready_meta["tracked_bodies"])
    ready_ground_offset = float(
        ready_meta["neutral_ready_pose"]["ground_height_offset_m"]
    )

    model, data = bridge.model, bridge.data
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (
            "torso_Link",
            "left_ankle_roll_Link",
            "right_ankle_roll_Link",
            "left_wrist_yaw_Link",
            "right_wrist_yaw_Link",
        )
    }
    missing = [name for name, body_id in body_ids.items() if body_id < 0]
    if missing:
        raise ValueError(f"MuJoCo model is missing bodies: {missing}")
    racket_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_racket"
    )
    if racket_site_id < 0:
        raise ValueError("MuJoCo model is missing right_racket site")

    arm_geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        body_name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            )
            or ""
        )
        is_arm = (
            (body_name.startswith("left_") or body_name.startswith("right_"))
            and any(token in body_name for token in ("shoulder", "elbow", "wrist", "hand"))
        ) or "pingpang" in body_name
        if is_arm and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            arm_geom_ids.append(geom_id)

    ticks = int(round(args.duration / cfg.control_dt))
    station_xy = bridge.read_state().base_pos_w[:2].copy()
    table_x = (args.table_near_x, args.table_near_x + args.table_length)
    table_y = (
        float(station_xy[1]) - 0.5 * args.table_width,
        float(station_xy[1]) + 0.5 * args.table_width,
    )
    pelvis_position = []
    station_drift = []
    base_pitch = []
    torso_pitch = []
    q_history = []
    q_des_history = []
    raw_action_history = []
    wrist_height = {"left": [], "right": [], "racket": []}
    foot_tilt = {"left": [], "right": []}
    arm_table_clearance: list[float] = []
    arm_over_table_ticks = 0
    nonfinite_observations = 0
    nonfinite_actions = 0

    bridge.reset()
    try:
        for _ in range(ticks):
            state = bridge.read_state()
            target = lifecycle.update(source.poll(), state)
            obs = build_observation(state, target, last_action, default_q, station_xy)
            if not np.all(np.isfinite(obs)):
                nonfinite_observations += 1
                raise FloatingPointError("non-finite READY observation")
            raw_action = policy.infer(obs)
            if not np.all(np.isfinite(raw_action)):
                nonfinite_actions += 1
                raise FloatingPointError("non-finite READY action")
            applied = cfg.action_adapter.clip_raw_action(raw_action)
            if cfg.passive_neck:
                applied[head_indices] = 0.0
            last_action = applied.copy()
            q_des = cfg.action_adapter.decode(applied)
            if cfg.passive_neck:
                q_des[head_indices] = default_q[head_indices]
            bridge.write_targets(q_des, cfg.sim_kp, cfg.sim_kd)
            bridge.step()
            lifecycle.advance()

            state = bridge.read_state()
            pelvis_position.append(state.base_pos_w.copy())
            station_drift.append(float(np.linalg.norm(state.base_pos_w[:2] - station_xy)))
            base_pitch.append(_rotation_pitch(_quat_rotation(state.base_quat_w)))
            torso_rotation = data.xmat[body_ids["torso_Link"]].reshape(3, 3)
            torso_pitch.append(_rotation_pitch(torso_rotation))
            q_history.append(state.q.copy())
            q_des_history.append(q_des.copy())
            raw_action_history.append(raw_action.copy())
            wrist_height["left"].append(float(data.xpos[body_ids["left_wrist_yaw_Link"], 2]))
            wrist_height["right"].append(float(data.xpos[body_ids["right_wrist_yaw_Link"], 2]))
            wrist_height["racket"].append(float(data.site_xpos[racket_site_id, 2]))
            for side, body_name in (
                ("left", "left_ankle_roll_Link"),
                ("right", "right_ankle_roll_Link"),
            ):
                rotation = data.xmat[body_ids[body_name]].reshape(3, 3)
                foot_tilt[side].append(float(math.acos(np.clip(rotation[2, 2], -1.0, 1.0))))

            tick_clearance = []
            for geom_id in arm_geom_ids:
                lower, upper = _world_geom_aabb(model, data, geom_id)
                overlap = (
                    upper[0] >= table_x[0]
                    and lower[0] <= table_x[1]
                    and upper[1] >= table_y[0]
                    and lower[1] <= table_y[1]
                )
                if overlap:
                    tick_clearance.append(float(lower[2] - args.table_height))
            if tick_clearance:
                arm_over_table_ticks += 1
                arm_table_clearance.append(min(tick_clearance))
    finally:
        bridge.close()

    pelvis = np.asarray(pelvis_position)
    q_actual = np.asarray(q_history)
    q_desired = np.asarray(q_des_history)
    raw_actions = np.asarray(raw_action_history)
    settle_start = max(0, ticks - int(round(2.0 / cfg.control_dt)))
    waist_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    waist = {}
    for name in waist_names:
        index = JOINT_NAMES.index(name)
        waist[name] = {
            "actual_deg": _stats(np.degrees(q_actual[:, index])),
            "desired_deg": _stats(np.degrees(q_desired[:, index])),
            "settled_std_deg": float(np.std(np.degrees(q_actual[settle_start:, index]))),
        }

    reference_error = q_actual - ready_q[None, :]
    reference_body_error = {}
    for body_name in ("left_wrist_yaw_Link", "right_wrist_yaw_Link"):
        ready_index = ready_body_names.index(body_name)
        target = ready_body_pos[ready_index].copy()
        target[2] += ready_ground_offset
        actual = data.xpos[body_ids[body_name]].copy()
        reference_body_error[body_name] = {
            "target_w_m": target.tolist(),
            "final_w_m": actual.tolist(),
            "final_error_m": float(np.linalg.norm(actual - target)),
        }

    report = {
        "test": "mujoco_deploy_ready_no_command",
        "duration_s": float(args.duration),
        "control_ticks": ticks,
        "artifacts": {
            "checkpoint_iteration": _checkpoint_iteration(args.onnx),
            "onnx": {"path": str(args.onnx.resolve()), "sha256": _sha256(args.onnx.resolve())},
            "runtime_config": {
                "path": str(args.runtime_config.resolve()),
                "sha256": _sha256(args.runtime_config.resolve()),
            },
            "model_xml": {
                "path": str(args.model_xml.resolve()),
                "sha256": _sha256(args.model_xml.resolve()),
            },
            "ready_reference": {
                "path": str(args.ready_reference.resolve()),
                "sha256": _sha256(args.ready_reference.resolve()),
            },
        },
        "result": {
            "fell": bool(np.min(pelvis[:, 2]) < 0.75 or np.max(np.abs(base_pitch)) > 0.65),
            "nonfinite_observations": nonfinite_observations,
            "nonfinite_actions": nonfinite_actions,
            "pelvis_height_m": _stats(pelvis[:, 2]),
            "station_drift_m": _stats(np.asarray(station_drift)),
            "base_pitch_deg": _stats(np.degrees(base_pitch)),
            "torso_pitch_deg": _stats(np.degrees(torso_pitch)),
            "wrist_height_m": {
                name: _stats(np.asarray(values)) for name, values in wrist_height.items()
            },
            "foot_tilt_deg": {
                name: _stats(np.degrees(values)) for name, values in foot_tilt.items()
            },
            "waist": waist,
            "ready_joint_tracking": {
                "final_rms_rad": float(np.sqrt(np.mean(reference_error[-1] ** 2))),
                "settled_rms_rad": float(
                    np.sqrt(np.mean(reference_error[settle_start:] ** 2))
                ),
            },
            "ready_wrist_tracking": reference_body_error,
            "max_abs_raw_action": float(np.max(np.abs(raw_actions))),
            "raw_action_clip_fraction": float(np.mean(np.abs(raw_actions) >= 100.0)),
            "arm_horizontal_overlap_with_table_ticks": arm_over_table_ticks,
            "min_arm_aabb_clearance_over_table_m": (
                None if not arm_table_clearance else float(np.min(arm_table_clearance))
            ),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["result"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
