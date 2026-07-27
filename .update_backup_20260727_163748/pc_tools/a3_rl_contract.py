#!/usr/bin/env python3

"""Pure numeric contract for the no-mocap HOPE ping-pong policy.

This module deliberately has no ROS dependency.  It validates the portable
policy bundle, constructs the exact 111-D observation, decodes the 31-D actor
output, and asserts that the real-command gains equal both the Isaac training
snapshot and the MuJoCo/sim2sim runtime configuration.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import yaml


OBS_DIM = 111
ACTION_DIM = 31
HEAD_INDICES = (3, 4)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_isaac_snapshot(path: Path) -> dict:
    """Load an Isaac Lab params/env.yaml without enabling arbitrary constructors."""

    class SnapshotLoader(yaml.SafeLoader):
        pass

    def _sequence(loader, node):
        return loader.construct_sequence(node, deep=True)

    SnapshotLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple", _sequence
    )
    SnapshotLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice", _sequence
    )
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.load(stream, Loader=SnapshotLoader)
    if not isinstance(document, dict):
        raise ValueError(f"expected an Isaac environment mapping: {path}")
    return document


def _call_keyword(path: Path, call_name: str, keyword: str):
    """Read a literal keyword from the current training source without Isaac."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if name != call_name:
            continue
        for item in node.keywords:
            if item.arg == keyword:
                return ast.literal_eval(item.value)
    raise ValueError(f"{path}: missing {call_name}(..., {keyword}=...)")


def _ordered(spec: dict, names: tuple[str, ...], field: str) -> np.ndarray:
    missing = [name for name in names if name not in spec]
    extra = sorted(set(spec) - set(names))
    if missing or extra:
        raise ValueError(f"{field} joint mismatch: missing={missing}, extra={extra}")
    values = np.asarray([float(spec[name]) for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{field} contains non-finite values")
    return values


def normalize_quaternion_wxyz(value: Iterable[float]) -> np.ndarray:
    quat = np.asarray(tuple(value), dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("quaternion must contain four finite WXYZ values")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-9:
        raise ValueError("quaternion norm is zero")
    return quat / norm


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def yaw_radians_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = normalize_quaternion_wxyz(quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def remove_startup_yaw(quat: np.ndarray, startup_yaw: float) -> np.ndarray:
    half = -0.5 * float(startup_yaw)
    yaw_inverse = np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))
    return normalize_quaternion_wxyz(
        quaternion_multiply_wxyz(yaw_inverse, normalize_quaternion_wxyz(quat))
    )


def _rotate(quat: np.ndarray, vector: np.ndarray, inverse: bool = False) -> np.ndarray:
    q = normalize_quaternion_wxyz(quat)
    w = q[0]
    xyz = q[1:4]
    a = vector * (2.0 * w * w - 1.0)
    b = np.cross(xyz, vector) * (2.0 * w)
    c = xyz * (2.0 * float(np.dot(xyz, vector)))
    return a - b + c if inverse else a + b + c


def projected_gravity_body(quat: np.ndarray) -> np.ndarray:
    return _rotate(quat, np.asarray((0.0, 0.0, -1.0)), inverse=True)


def base_forward_xy(quat: np.ndarray) -> np.ndarray:
    forward = _rotate(quat, np.asarray((1.0, 0.0, 0.0)))
    norm = float(np.hypot(forward[0], forward[1])) + 1.0e-6
    return np.asarray((forward[0] / norm, forward[1] / norm))


def base_tilt_radians(quat: np.ndarray) -> float:
    gravity = projected_gravity_body(quat)
    return math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))


@dataclass(frozen=True)
class NoMocapTarget:
    side: str
    rel_base: np.ndarray
    velocity_world: np.ndarray
    swing_side: float


@dataclass(frozen=True)
class NoMocapClock:
    hold_steps: int
    strike_lead_steps: int
    follow_through_steps: int
    dt: float

    @property
    def total_steps(self) -> int:
        # Both the +lead and -follow endpoints receive one actor action.
        return self.hold_steps + self.strike_lead_steps + self.follow_through_steps + 1

    def time_to_strike(self, tick: int) -> float:
        if tick < 0 or tick >= self.total_steps:
            raise IndexError(f"no-mocap tick {tick} outside [0,{self.total_steps})")
        if tick < self.hold_steps:
            return self.strike_lead_steps * self.dt
        return (self.strike_lead_steps - (tick - self.hold_steps)) * self.dt


@dataclass
class A3RLContract:
    bundle_dir: Path
    policy_generation: str
    checkpoint_iteration: int
    joint_names: tuple[str, ...]
    default_q: np.ndarray
    action_scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    targets: dict[str, NoMocapTarget]
    ready_targets: dict[str, NoMocapTarget]
    clock: NoMocapClock
    onnx_path: Path

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        training_source: str | Path | None = None,
    ) -> "A3RLContract":
        bundle = Path(bundle_dir).resolve()
        config = bundle / "config"
        manifest_path = bundle / "policy/policy_manifest.json"
        provenance_path = bundle / "policy/provenance.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        provenance = json.loads(provenance_path.read_text("utf-8"))
        order_doc = _load_yaml(config / "joint_order_agibot_a3.yaml")
        names = tuple(str(name) for name in order_doc["joint_order"])
        if len(names) != ACTION_DIM or len(set(names)) != ACTION_DIM:
            raise ValueError("joint order must contain 31 unique names")
        if tuple(manifest["joint_order"]) != names:
            raise ValueError("policy manifest joint order does not match joint-order YAML")
        if int(manifest["obs_dim"]) != OBS_DIM or int(manifest["action_dim"]) != ACTION_DIM:
            raise ValueError("policy manifest must declare observation[111] -> action[31]")
        if float(manifest["control_rate_hz"]) != 50.0:
            raise ValueError("policy control rate must be exactly 50 Hz")
        if str(manifest["observation_normalization"]).lower() != "none":
            raise ValueError("this policy requires raw, unnormalized observations")
        policy_generation = str(manifest.get("policy_generation", ""))
        checkpoint_iteration = int(manifest.get("checkpoint_iteration", -1))
        if policy_generation != "balance_v2" or checkpoint_iteration != 7000:
            raise ValueError(
                "this no-mocap deploy profile requires balance_v2/model_7000; "
                f"got {policy_generation or 'unknown'}/model_{checkpoint_iteration}"
            )
        cls._assert_bundle_provenance(
            bundle, manifest_path, provenance_path, manifest, provenance
        )

        adapter = _load_yaml(config / "action_adapter.yaml")
        if str(adapter.get("raw_action_transform", "")).lower() != "tanh":
            raise ValueError("this policy requires raw_action_transform=tanh")
        default_q = _ordered(adapter["default_q"], names, "default_q")
        action_scale = _ordered(adapter["action_scale"], names, "action_scale")
        lower = _ordered(adapter["joint_position_clamp"]["lower"], names, "lower")
        upper = _ordered(adapter["joint_position_clamp"]["upper"], names, "upper")
        if np.any(lower > upper):
            raise ValueError("joint lower clamp exceeds upper clamp")

        training = _load_yaml(config / "training_actuator_gains.yaml")
        kp, kd = cls._expand_training_gains(training, names)
        runtime_path = config / "runtime.yaml"
        cls._assert_sim2sim_gain_parity(runtime_path, names, kp, kd)
        ready_targets = cls._ready_targets_from_runtime(runtime_path)
        cls._assert_frozen_training_snapshot(
            bundle, provenance, names, default_q, action_scale, lower, upper, kp, kd
        )

        no_mocap = _load_yaml(config / "no_mocap_observation.yaml")
        if bool(no_mocap.get("enabled_by_default", True)):
            raise ValueError("no-mocap fallback must remain opt-in")
        targets = {}
        for side in ("forehand", "backhand"):
            item = no_mocap["demo_targets"][side]
            targets[side] = NoMocapTarget(
                side=side,
                rel_base=np.asarray(item["racket_target_rel_base_m"], dtype=np.float64),
                velocity_world=np.asarray(
                    item["racket_target_vel_world_mps"], dtype=np.float64
                ),
                swing_side=float(item["swing_side"]),
            )
        timing = no_mocap["timing"]
        clock = NoMocapClock(
            hold_steps=int(timing["pre_swing_hold_steps"]),
            strike_lead_steps=int(timing["strike_lead_steps"]),
            follow_through_steps=int(timing["follow_through_steps"]),
            dt=float(timing["decrement_per_control_step_s"]),
        )
        if not math.isclose(clock.dt, 0.02, abs_tol=1.0e-12):
            raise ValueError("no-mocap clock must use 20 ms steps")

        training_contract = _load_yaml(config / "policy_training_contract.yaml")
        cls._assert_no_mocap_training_parity(
            training_contract, no_mocap, targets, clock
        )
        if training_source is not None:
            cls._assert_current_source_parity(
                Path(training_source).resolve(),
                config,
                training_contract,
                names,
                kp,
                kd,
                ready_targets,
            )

        onnx_path = bundle / "policy" / str(manifest["onnx_file"])
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {onnx_path}")
        return cls(
            bundle_dir=bundle,
            policy_generation=policy_generation,
            checkpoint_iteration=checkpoint_iteration,
            joint_names=names,
            default_q=default_q,
            action_scale=action_scale,
            lower=lower,
            upper=upper,
            kp=kp,
            kd=kd,
            targets=targets,
            ready_targets=ready_targets,
            clock=clock,
            onnx_path=onnx_path,
        )

    @staticmethod
    def _assert_bundle_provenance(
        bundle: Path,
        manifest_path: Path,
        provenance_path: Path,
        manifest: dict,
        provenance: dict,
    ) -> None:
        if provenance.get("bundle_contract") != "hope_pingpong":
            raise ValueError("unexpected policy provenance contract")
        if provenance.get("policy_generation") != manifest.get("policy_generation"):
            raise ValueError("manifest/provenance policy generation mismatch")
        if int(provenance.get("source_checkpoint_iteration", -1)) != int(
            manifest.get("checkpoint_iteration", -2)
        ):
            raise ValueError("manifest/provenance checkpoint mismatch")

        policy = bundle / "policy" / str(manifest["onnx_file"])
        fixed_files = (
            (policy, "onnx_sha256"),
            (manifest_path, "policy_manifest_sha256"),
            (bundle / "config/action_adapter.yaml", "action_adapter_sha256"),
            (bundle / "config/runtime.yaml", "runtime_config_sha256"),
            (
                bundle / "config/policy_training_contract.yaml",
                "policy_training_contract_sha256",
            ),
            (
                bundle / "config/no_mocap_observation.yaml",
                "no_mocap_observation_config_sha256",
            ),
        )
        for path, key in fixed_files:
            expected = provenance.get(key)
            if not path.is_file() or not expected or _sha256(path) != expected:
                raise ValueError(f"bundle provenance check failed: {path.name}")

        for path_key, hash_key in (
            ("bundled_training_env", "bundled_training_env_sha256"),
            ("bundled_training_agent", "bundled_training_agent_sha256"),
            (
                "bundled_training_worktree_diff",
                "bundled_training_worktree_diff_sha256",
            ),
        ):
            path = bundle / str(provenance.get(path_key, ""))
            expected = provenance.get(hash_key)
            if not path.is_file() or not expected or _sha256(path) != expected:
                raise ValueError(f"training provenance check failed: {path_key}")

        # Keep this argument in the signature so callers cannot accidentally skip
        # checking that provenance.json itself exists and was parsed.
        if not provenance_path.is_file():
            raise FileNotFoundError(provenance_path)

    @staticmethod
    def _resolve_pattern_value(spec, joint_name: str, field: str) -> float:
        if isinstance(spec, (int, float)):
            return float(spec)
        if not isinstance(spec, dict):
            raise ValueError(f"training snapshot {field} is not numeric or mapped")
        matches = [
            float(value)
            for pattern, value in spec.items()
            if re.fullmatch(str(pattern), joint_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"training snapshot {field} resolves {len(matches)} times for {joint_name}"
            )
        return matches[0]

    @classmethod
    def _assert_frozen_training_snapshot(
        cls,
        bundle: Path,
        provenance: dict,
        names: tuple[str, ...],
        default_q: np.ndarray,
        action_scale: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        snapshot_path = bundle / str(provenance["bundled_training_env"])
        snapshot = _load_isaac_snapshot(snapshot_path)
        if int(snapshot["decimation"]) != 4 or not math.isclose(
            float(snapshot["sim"]["dt"]), 0.005, abs_tol=1.0e-12
        ):
            raise ValueError("frozen training snapshot is not 200 Hz / decimation 4")
        if snapshot.get("events", {}).get("pd_gains") is not None:
            raise ValueError("frozen checkpoint unexpectedly randomized Kp/Kd")

        robot = snapshot["scene"]["robot"]
        action = snapshot["actions"]["joint_pos"]
        snap_default = _ordered(robot["init_state"]["joint_pos"], names, "snapshot default_q")
        snap_scale = _ordered(action["scale"], names, "snapshot action_scale")
        if not np.array_equal(snap_default, default_q):
            raise ValueError("bundle default_q differs from checkpoint training snapshot")
        if not np.array_equal(snap_scale, action_scale):
            raise ValueError("bundle action_scale differs from checkpoint training snapshot")
        if str(action.get("raw_action_transform", "identity")) != "tanh":
            raise ValueError("checkpoint training snapshot did not use tanh actions")
        snap_clamp = action["position_clamp"]
        snap_lower = np.asarray([float(snap_clamp[name][0]) for name in names])
        snap_upper = np.asarray([float(snap_clamp[name][1]) for name in names])
        if not np.array_equal(snap_lower, lower) or not np.array_equal(
            snap_upper, upper
        ):
            raise ValueError("bundle joint clamp differs from checkpoint snapshot")

        snap_kp: dict[str, float] = {}
        snap_kd: dict[str, float] = {}
        for actuator_name, actuator in robot["actuators"].items():
            patterns = tuple(str(item) for item in actuator["joint_names_expr"])
            for name in names:
                if not any(re.fullmatch(pattern, name) for pattern in patterns):
                    continue
                if name in snap_kp:
                    raise ValueError(
                        f"checkpoint actuator overlap for {name}: {actuator_name}"
                    )
                snap_kp[name] = cls._resolve_pattern_value(
                    actuator["stiffness"], name, "stiffness"
                )
                snap_kd[name] = cls._resolve_pattern_value(
                    actuator["damping"], name, "damping"
                )
        if set(snap_kp) != set(names) or set(snap_kd) != set(names):
            raise ValueError("checkpoint actuator gains do not cover 31 joints")
        snapshot_kp = np.asarray([snap_kp[name] for name in names])
        snapshot_kd = np.asarray([snap_kd[name] for name in names])
        if not np.array_equal(snapshot_kp, kp) or not np.array_equal(snapshot_kd, kd):
            raise ValueError("deployment Kp/Kd differ from frozen checkpoint training Kp/Kd")

        agent_path = bundle / str(provenance["bundled_training_agent"])
        agent = _load_yaml(agent_path)
        if bool(agent.get("empirical_normalization", True)):
            raise ValueError("checkpoint unexpectedly used observation normalization")
        if list(agent["policy"]["actor_hidden_dims"]) != [512, 256, 128]:
            raise ValueError("checkpoint actor architecture mismatch")

    @staticmethod
    def _midpoint_box(document: dict, section: str, side: str) -> np.ndarray:
        box = document[section][side]
        return np.asarray(
            [0.5 * (float(box[axis][0]) + float(box[axis][1])) for axis in "xyz"],
            dtype=np.float64,
        )

    @classmethod
    def _assert_no_mocap_training_parity(
        cls,
        training_contract: dict,
        no_mocap: dict,
        targets: dict[str, NoMocapTarget],
        clock: NoMocapClock,
    ) -> None:
        if training_contract.get("policy_generation") != "balance_v2":
            raise ValueError("no-mocap contract is not the balance_v2 training profile")
        if int(training_contract.get("checkpoint_iteration", -1)) != 7000:
            raise ValueError("no-mocap contract is not for model_7000")
        if no_mocap.get("mode") != "training_midpoint_demo":
            raise ValueError("unsupported no-mocap target mode")
        if not math.isclose(float(no_mocap["control_hz"]), 50.0, abs_tol=1.0e-12):
            raise ValueError("no-mocap source must run at 50 Hz")

        pelvis = np.asarray(
            training_contract["training_reset"]["pelvis_position_world_m"],
            dtype=np.float64,
        )
        for side in ("forehand", "backhand"):
            position_world = cls._midpoint_box(
                training_contract, "racket_position_box", side
            )
            expected_rel = position_world - pelvis
            expected_velocity = cls._midpoint_box(
                training_contract, "racket_velocity_box", side
            )
            target = targets[side]
            if not np.allclose(target.rel_base, expected_rel, atol=1.0e-12):
                raise ValueError(f"{side} no-mocap position is not the training midpoint")
            if not np.allclose(
                target.velocity_world, expected_velocity, atol=1.0e-12
            ):
                raise ValueError(f"{side} no-mocap velocity is not the training midpoint")

        hold_range = training_contract["training_reset"]["hold_steps_range"]
        expected_hold = int(round(0.5 * (int(hold_range[0]) + int(hold_range[1]))))
        if clock.hold_steps != expected_hold:
            raise ValueError("no-mocap hold is not the midpoint of the training range")
        phases = tuple(float(x) for x in training_contract["motion"]["strike_phase_per_clip"])
        if phases != (50.0 / 90.0, 50.0 / 90.0):
            raise ValueError("balance_v2 training strike phase is not frame 50/90")
        if clock.strike_lead_steps != 50 or clock.follow_through_steps != 40:
            raise ValueError("no-mocap clock must replay all 91 guarded actor frames")
        if clock.total_steps != expected_hold + 91:
            raise ValueError("no-mocap clock must contain hold midpoint plus 91 actions")

    @classmethod
    def _assert_current_source_parity(
        cls,
        source: Path,
        bundle_config: Path,
        training_contract: dict,
        names: tuple[str, ...],
        kp: np.ndarray,
        kd: np.ndarray,
        ready_targets: dict[str, NoMocapTarget],
    ) -> None:
        if not source.is_dir():
            raise FileNotFoundError(f"training/sim2sim source not found: {source}")
        deploy_config = source / "a3_deploy/a3_deploy_example/config"
        source_adapter = deploy_config / "action_adapter.yaml"
        if source_adapter.read_bytes() != (bundle_config / "action_adapter.yaml").read_bytes():
            raise ValueError("current training source ActionAdapter differs from policy bundle")
        source_runtime = deploy_config / "hope_pingpong_runtime.yaml"
        cls._assert_sim2sim_gain_parity(source_runtime, names, kp, kd)
        source_ready = cls._ready_targets_from_runtime(source_runtime)
        for side in ("forehand", "backhand"):
            if (
                not np.array_equal(source_ready[side].rel_base, ready_targets[side].rel_base)
                or not np.array_equal(
                    source_ready[side].velocity_world,
                    ready_targets[side].velocity_world,
                )
                or source_ready[side].swing_side != ready_targets[side].swing_side
            ):
                raise ValueError("current sim2sim READY observation differs from bundle runtime")

        env_cfg = (
            source
            / "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
        )
        if tuple(_call_keyword(env_cfg, "MotionCommandCfg", "hold_steps_range")) != (0, 100):
            raise ValueError("current training hold range differs from checkpoint contract")
        phases = tuple(
            _call_keyword(env_cfg, "RacketTargetCommandCfg", "strike_phase_per_clip")
        )
        if phases != (50.0 / 90.0, 50.0 / 90.0):
            raise ValueError("current training strike phase is not frame 50/90")
        source_position_boxes = np.asarray(
            _call_keyword(env_cfg, "RacketTargetCommandCfg", "racket_pos_range_per_clip"),
            dtype=np.float64,
        )
        source_velocity_boxes = np.asarray(
            _call_keyword(env_cfg, "RacketTargetCommandCfg", "racket_vel_range_per_clip"),
            dtype=np.float64,
        )
        contract_positions = np.asarray(
            [
                [training_contract["racket_position_box"][side][axis] for axis in "xyz"]
                for side in ("forehand", "backhand")
            ],
            dtype=np.float64,
        )
        contract_velocities = np.asarray(
            [
                [training_contract["racket_velocity_box"][side][axis] for axis in "xyz"]
                for side in ("forehand", "backhand")
            ],
            dtype=np.float64,
        )
        if not np.array_equal(source_position_boxes, contract_positions):
            raise ValueError("current training position boxes differ from balance_v2 snapshot")
        if not np.array_equal(source_velocity_boxes, contract_velocities):
            raise ValueError("current training velocity boxes differ from balance_v2 snapshot")

        motion_root = source / "hope_training/motions/preprocessed"
        for filename in (
            "ours_forehand_guarded_1p8s_wrist_x.yaml",
            "ours_backhand_guarded_1p8s.yaml",
        ):
            motion = _load_yaml(motion_root / filename)
            if (
                int(motion["frame_count"]) != 91
                or not math.isclose(float(motion["fps"]), 50.0, abs_tol=1.0e-12)
                or int(motion["strike_frame"]) != 50
                or tuple(motion["joint_order"]) != names
            ):
                raise ValueError(f"current motion rhythm/order mismatch: {filename}")

    @staticmethod
    def _expand_training_gains(
        document: dict, names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        if float(document["policy_hz"]) != 50.0 or int(document["decimation"]) != 4:
            raise ValueError("unexpected training control timing")
        mapped: dict[str, tuple[float, float]] = {}
        for group_name, group in document["groups"].items():
            for joint in group["joints"]:
                if joint in mapped:
                    raise ValueError(f"joint appears in multiple gain groups: {joint}")
                mapped[str(joint)] = (float(group["kp"]), float(group["kd"]))
        missing = [name for name in names if name not in mapped]
        extra = sorted(set(mapped) - set(names))
        if missing or extra:
            raise ValueError(
                f"training gain joint mismatch: missing={missing}, extra={extra}"
            )
        kp = np.asarray([mapped[name][0] for name in names], dtype=np.float64)
        kd = np.asarray([mapped[name][1] for name in names], dtype=np.float64)
        return kp, kd

    @staticmethod
    def _assert_sim2sim_gain_parity(
        runtime_path: Path,
        names: tuple[str, ...],
        training_kp: np.ndarray,
        training_kd: np.ndarray,
    ) -> None:
        runtime = _load_yaml(runtime_path)
        joints = runtime["simulation"]["pd_gains"]["joints"]
        runtime_kp = np.asarray([float(joints[name]["kp"]) for name in names])
        runtime_kd = np.asarray([float(joints[name]["kd"]) for name in names])
        if not np.array_equal(runtime_kp, training_kp) or not np.array_equal(
            runtime_kd, training_kd
        ):
            bad = [
                names[i]
                for i in range(ACTION_DIM)
                if runtime_kp[i] != training_kp[i] or runtime_kd[i] != training_kd[i]
            ]
            raise ValueError(
                "training and sim2sim Kp/Kd differ for: " + ", ".join(bad)
            )

    @staticmethod
    def _ready_targets_from_runtime(
        runtime_path: Path,
    ) -> dict[str, NoMocapTarget]:
        runtime = _load_yaml(runtime_path)
        lifecycle = runtime["lifecycle"]
        expected = {
            "follow_through_s": 0.8,
            "recovery_s": 0.0,
            "ready_time_to_strike": 1.0,
        }
        for field, value in expected.items():
            if not math.isclose(float(lifecycle[field]), value, abs_tol=1.0e-12):
                raise ValueError(f"unexpected sim2sim lifecycle {field}")
        x = float(lifecycle["ready_reach_x"])
        y = float(lifecycle["ready_reach_y"])
        z = float(lifecycle["ready_reach_z"])
        values = np.asarray((x, y, z), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("sim2sim READY reach contains non-finite values")
        return {
            "forehand": NoMocapTarget(
                side="forehand",
                rel_base=np.asarray((x, y, z), dtype=np.float64),
                velocity_world=np.zeros(3, dtype=np.float64),
                swing_side=1.0,
            ),
            "backhand": NoMocapTarget(
                side="backhand",
                rel_base=np.asarray((x, -y, z), dtype=np.float64),
                velocity_world=np.zeros(3, dtype=np.float64),
                swing_side=-1.0,
            ),
        }

    def build_observation(
        self,
        q: Iterable[float],
        dq: Iterable[float],
        pelvis_quat_wxyz: Iterable[float],
        pelvis_gyro_body: Iterable[float],
        last_action: Iterable[float],
        side: str,
        time_to_strike: float,
        target_override: NoMocapTarget | None = None,
    ) -> np.ndarray:
        q_array = np.asarray(tuple(q), dtype=np.float64)
        dq_array = np.asarray(tuple(dq), dtype=np.float64)
        gyro = np.asarray(tuple(pelvis_gyro_body), dtype=np.float64)
        previous = np.asarray(tuple(last_action), dtype=np.float64)
        quat = normalize_quaternion_wxyz(pelvis_quat_wxyz)
        if q_array.shape != (ACTION_DIM,) or dq_array.shape != (ACTION_DIM,):
            raise ValueError("q and dq must each contain 31 values")
        if previous.shape != (ACTION_DIM,) or gyro.shape != (3,):
            raise ValueError("last_action must be 31-D and gyro must be 3-D")
        if not all(
            np.isfinite(value).all()
            for value in (q_array, dq_array, gyro, previous)
        ):
            raise ValueError("observation source contains NaN or infinity")
        target = self.targets[side] if target_override is None else target_override
        if target.side != side:
            raise ValueError("observation target side does not match locked side")
        observation = np.empty(OBS_DIM, dtype=np.float64)
        observation[0:3] = gyro
        observation[3:34] = q_array - self.default_q
        observation[34:65] = dq_array
        observation[65:96] = previous
        observation[96:99] = projected_gravity_body(quat)
        observation[99:101] = base_forward_xy(quat)
        observation[101:103] = 0.0
        observation[103:106] = target.rel_base
        observation[106:109] = target.velocity_world
        observation[109] = float(time_to_strike)
        observation[110] = target.swing_side
        if not np.isfinite(observation).all():
            raise ValueError("constructed observation contains NaN or infinity")
        return np.ascontiguousarray(observation, dtype=np.float32)

    def decode_action(
        self, raw_action: Iterable[float], policy_blend: float
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(tuple(raw_action), dtype=np.float64)
        if raw.shape != (ACTION_DIM,) or not np.isfinite(raw).all():
            raise ValueError("raw action must contain 31 finite values")
        if not 0.0 <= policy_blend <= 1.0:
            raise ValueError("policy_blend must be in [0,1]")
        applied = raw.copy()
        applied[list(HEAD_INDICES)] = 0.0
        full_target = self.default_q + np.tanh(applied) * self.action_scale
        full_target = np.clip(full_target, self.lower, self.upper)
        target = self.default_q + policy_blend * (full_target - self.default_q)
        target = np.clip(target, self.lower, self.upper)
        target[list(HEAD_INDICES)] = self.default_q[list(HEAD_INDICES)]
        return applied, target


class OnnxActor:
    def __init__(self, contract: A3RLContract) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is missing; run scripts/setup_rl_venv.sh first"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(contract.onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX actor must have exactly one input and output")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        if self._input_name != "observation" or self._output_name != "raw_action":
            raise ValueError("unexpected ONNX input/output names")
        if inputs[0].shape[-1] != OBS_DIM or outputs[0].shape[-1] != ACTION_DIM:
            raise ValueError("unexpected ONNX input/output dimensions")
        metadata = self._session.get_modelmeta().custom_metadata_map or {}
        embedded_order = tuple(metadata.get("joint_order", "").split(","))
        if embedded_order != contract.joint_names:
            raise ValueError("ONNX joint_order metadata does not match deployment")

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(1, OBS_DIM)
        output = self._session.run(
            [self._output_name], {self._input_name: value}
        )[0]
        action = np.asarray(output, dtype=np.float32).reshape(ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("ONNX actor produced NaN or infinity")
        return action
