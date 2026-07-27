#!/usr/bin/env python3

"""Current model_21500 HOPE policy contract, independent of ROS."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


OBS_DIM = 111
ACTION_DIM = 31
HEAD_INDICES = (3, 4)
FOREHAND = 1
BACKHAND = -1
EXPECTED_GENERATION = "full_body_uniform_action_return_physics_v1"
EXPECTED_CHECKPOINT = 21500


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered(spec, names: tuple[str, ...], field: str) -> np.ndarray:
    if isinstance(spec, (int, float)):
        values = np.full(ACTION_DIM, float(spec), dtype=np.float64)
    elif isinstance(spec, dict):
        missing = [name for name in names if name not in spec]
        extra = sorted(set(spec) - set(names))
        if missing or extra:
            raise ValueError(f"{field} joint mismatch: missing={missing}, extra={extra}")
        values = np.asarray([float(spec[name]) for name in names], dtype=np.float64)
    else:
        values = np.asarray(spec, dtype=np.float64).reshape(-1)
        if values.shape != (ACTION_DIM,):
            raise ValueError(f"{field} must contain 31 values")
    if not np.isfinite(values).all():
        raise ValueError(f"{field} contains NaN or infinity")
    return values


def normalize_quaternion_wxyz(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("quaternion must contain four finite WXYZ values")
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-9:
        raise ValueError("quaternion norm is zero")
    return result / norm


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
class PolicyTarget:
    position_w: np.ndarray
    velocity_w: np.ndarray
    time_to_strike: float
    swing_side: int


@dataclass(frozen=True)
class RacketCommand:
    task_id: int
    task_revision: int
    swing_side: int
    position_w: np.ndarray
    velocity_w: np.ndarray
    time_to_strike: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_w, dtype=np.float64).reshape(3)
        velocity = np.asarray(self.velocity_w, dtype=np.float64).reshape(3)
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("RacketCommand contains non-finite vectors")
        if self.swing_side not in (FOREHAND, BACKHAND):
            raise ValueError("RacketCommand swing_side must be +1 or -1")
        object.__setattr__(self, "position_w", position)
        object.__setattr__(self, "velocity_w", velocity)


class SwingLifecycle:
    READY = "ready"
    SWING = "swing"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"

    def __init__(
        self,
        contract: "A3RLContract",
        ready_reference_base_w: Iterable[float] | None = None,
    ) -> None:
        self.contract = contract
        self.phase = self.READY
        self.active_task_id: int | None = None
        self.swing_side = contract.ready_swing_side
        self._last_engaged_task_id = -1
        self._applied_revision = -1
        self._target_position = np.zeros(3)
        self._target_velocity = np.zeros(3)
        self._tts = contract.ready_time_to_strike
        self._follow_time = 0.0
        self._recovery_time = 0.0
        self._ready_target_position_w: np.ndarray | None = None
        if ready_reference_base_w is not None:
            self.lock_ready_reference(ready_reference_base_w)

    def lock_ready_reference(self, base_position_w: Iterable[float]) -> None:
        """Lock the first-frame virtual target in world coordinates."""

        base = np.asarray(tuple(base_position_w), dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("READY reference base must contain three finite values")
        self._ready_target_position_w = base + self.contract.ready_rel_base_w

    def ready_target(self, base_position_w: Iterable[float]) -> PolicyTarget:
        """Return one fixed target while live base motion changes target-minus-base."""

        if self._ready_target_position_w is None:
            self.lock_ready_reference(base_position_w)
        assert self._ready_target_position_w is not None
        return PolicyTarget(
            self._ready_target_position_w.copy(),
            self.contract.ready_velocity_w.copy(),
            self.contract.ready_time_to_strike,
            self.contract.ready_swing_side,
        )

    def _ready(self, base_position_w: np.ndarray) -> PolicyTarget:
        return self.ready_target(base_position_w)

    def update(
        self, command: RacketCommand | None, base_position_w: np.ndarray
    ) -> PolicyTarget:
        if command is not None:
            if (
                command.task_id > self._last_engaged_task_id
                and self.phase in (self.READY, self.RECOVERY)
            ):
                self._last_engaged_task_id = command.task_id
                tts = float(command.time_to_strike)
                if math.isfinite(tts) and tts >= 0.0:
                    self.active_task_id = command.task_id
                    self._applied_revision = command.task_revision
                    self.swing_side = command.swing_side
                    self._target_position = command.position_w.copy()
                    self._target_velocity = command.velocity_w.copy()
                    self._tts = tts
                    self.phase = self.SWING
            elif (
                command.task_id == self.active_task_id
                and self.phase == self.SWING
                and self._tts > 0.0
                and command.task_revision > self._applied_revision
                and math.isfinite(float(command.time_to_strike))
                and float(command.time_to_strike) >= 0.0
            ):
                self._applied_revision = command.task_revision
                self._target_position = command.position_w.copy()
                self._target_velocity = command.velocity_w.copy()
                self._tts = float(command.time_to_strike)

        if self.phase in (self.SWING, self.FOLLOW_THROUGH):
            return PolicyTarget(
                self._target_position.copy(), self._target_velocity.copy(),
                self._tts, self.swing_side
            )
        return self._ready(base_position_w)

    def advance(self) -> None:
        dt = 1.0 / self.contract.control_hz
        if self.phase == self.SWING:
            self._tts -= dt
            if self._tts <= 1.0e-9:
                if abs(self._tts) <= 1.0e-9:
                    self._tts = 0.0
                self.phase = self.FOLLOW_THROUGH
                self._follow_time = 0.0
        elif self.phase == self.FOLLOW_THROUGH:
            self._tts -= dt
            self._follow_time += dt
            if self._follow_time - self.contract.follow_through_s >= 0.5 * dt:
                if self.contract.recovery_s <= 0.0:
                    self.phase = self.READY
                    self.active_task_id = None
                else:
                    self.phase = self.RECOVERY
                    self._recovery_time = 0.0
        elif self.phase == self.RECOVERY:
            self._recovery_time += dt
            if self._recovery_time >= self.contract.recovery_s:
                self.phase = self.READY
                self.active_task_id = None


@dataclass
class A3RLContract:
    bundle_dir: Path
    policy_generation: str
    checkpoint_iteration: int
    joint_names: tuple[str, ...]
    default_q: np.ndarray
    action_scale: np.ndarray
    action_clip: tuple[float, float]
    lower: np.ndarray
    upper: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    ready_rel_base_w: np.ndarray
    ready_velocity_w: np.ndarray
    ready_time_to_strike: float
    ready_swing_side: int
    follow_through_s: float
    recovery_s: float
    control_hz: float
    onnx_path: Path

    @classmethod
    def load(
        cls, bundle_dir: str | Path, training_source: str | Path | None = None
    ) -> "A3RLContract":
        bundle = Path(bundle_dir).resolve()
        config_dir = bundle / "config"
        manifest_path = bundle / "policy/policy_manifest.json"
        provenance_path = bundle / "policy/provenance.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        generation = str(manifest.get("policy_generation", ""))
        checkpoint = int(manifest.get("checkpoint_iteration", -1))
        if generation != EXPECTED_GENERATION or checkpoint != EXPECTED_CHECKPOINT:
            raise ValueError(
                f"deployment requires {EXPECTED_GENERATION}/model_{EXPECTED_CHECKPOINT}; "
                f"got {generation}/model_{checkpoint}"
            )
        if (
            provenance.get("policy_generation") != generation
            or int(provenance.get("source_checkpoint_iteration", -1)) != checkpoint
        ):
            raise ValueError("manifest/provenance policy identity mismatch")
        if str(manifest.get("observation_normalization", "")).lower() != "none":
            raise ValueError("policy observation normalization must be none")
        if int(manifest.get("obs_dim", -1)) != OBS_DIM or int(
            manifest.get("action_dim", -1)
        ) != ACTION_DIM:
            raise ValueError("policy must have observation[111] -> raw_action[31]")
        control_hz = float(manifest.get("control_rate_hz", 0.0))
        if not math.isclose(control_hz, 50.0, abs_tol=1.0e-12):
            raise ValueError("policy control rate must be 50 Hz")

        order = _yaml(config_dir / "joint_order_agibot_a3.yaml")
        names = tuple(str(name) for name in order["joint_order"])
        if len(names) != ACTION_DIM or len(set(names)) != ACTION_DIM:
            raise ValueError("joint order must contain 31 unique names")
        if tuple(manifest["joint_order"]) != names:
            raise ValueError("manifest and bundle joint order differ")

        onnx_path = bundle / "policy" / str(manifest["onnx_file"])
        if _sha256(onnx_path) != provenance.get("onnx_sha256"):
            raise ValueError("ONNX SHA256 does not match provenance")
        if _sha256(manifest_path) != provenance.get("policy_manifest_sha256"):
            raise ValueError("policy manifest SHA256 does not match provenance")
        for path_key, hash_key in (
            ("bundled_training_env", "bundled_training_env_sha256"),
            ("bundled_training_agent", "bundled_training_agent_sha256"),
            ("bundled_training_worktree_diff", "bundled_training_worktree_diff_sha256"),
        ):
            snapshot = bundle / str(provenance[path_key])
            if not snapshot.is_file() or _sha256(snapshot) != provenance[hash_key]:
                raise ValueError(f"training provenance failed: {snapshot}")

        adapter = _yaml(config_dir / "action_adapter.yaml")
        if str(adapter.get("raw_action_transform", "")).lower() != "identity":
            raise ValueError("current policy requires raw_action_transform=identity")
        clip = tuple(float(value) for value in adapter.get("action_clip", ()))
        if clip != (-100.0, 100.0):
            raise ValueError("current policy requires action_clip=[-100,100]")
        default_q = _ordered(adapter["default_q"], names, "default_q")
        action_scale = _ordered(adapter["action_scale"], names, "action_scale")
        if not np.array_equal(action_scale, np.full(ACTION_DIM, 0.25)):
            raise ValueError("current policy requires uniform action_scale=0.25")
        lower = _ordered(adapter["joint_position_clamp"]["lower"], names, "lower")
        upper = _ordered(adapter["joint_position_clamp"]["upper"], names, "upper")
        if np.any(lower > upper):
            raise ValueError("joint lower clamp exceeds upper clamp")

        gains = _yaml(config_dir / "training_actuator_gains.yaml")
        if (
            float(gains["policy_hz"]) != 50.0
            or int(gains["decimation"]) != 4
            or str(gains["pd_gain_domain_randomization"]).lower() != "none"
        ):
            raise ValueError("unexpected frozen training gain timing")
        mapped: dict[str, tuple[float, float]] = {}
        for group in gains["groups"].values():
            for name in group["joints"]:
                if name in mapped:
                    raise ValueError(f"gain joint appears twice: {name}")
                mapped[str(name)] = (float(group["kp"]), float(group["kd"]))
        if set(mapped) != set(names):
            raise ValueError("training gains do not cover the canonical joint set")
        kp = np.asarray([mapped[name][0] for name in names])
        kd = np.asarray([mapped[name][1] for name in names])

        runtime = _yaml(config_dir / "runtime.yaml")
        runtime_joints = runtime["simulation"]["pd_gains"]["joints"]
        runtime_kp = np.asarray([float(runtime_joints[name]["kp"]) for name in names])
        runtime_kd = np.asarray([float(runtime_joints[name]["kd"]) for name in names])
        if not np.array_equal(kp, runtime_kp) or not np.array_equal(kd, runtime_kd):
            raise ValueError("training and sim2sim Kp/Kd differ")
        lifecycle = runtime["lifecycle"]
        ready_rel = np.asarray(lifecycle["ready_target_rel_base_w"], dtype=np.float64)
        ready_velocity = np.asarray(lifecycle["ready_target_vel_w"], dtype=np.float64)
        ready_tts = float(lifecycle["ready_time_to_strike"])
        ready_side = int(lifecycle["ready_swing_side"])
        follow = float(lifecycle["follow_through_s"])
        recovery = float(lifecycle["recovery_s"])
        if (
            ready_rel.shape != (3,)
            or ready_velocity.shape != (3,)
            or not np.isfinite(ready_rel).all()
            or not np.isfinite(ready_velocity).all()
            or ready_side not in (FOREHAND, BACKHAND)
            or not math.isclose(ready_tts, 1.0, abs_tol=1.0e-12)
            or not math.isclose(follow, 0.8, abs_tol=1.0e-12)
            or not math.isclose(recovery, 0.0, abs_tol=1.0e-12)
        ):
            raise ValueError("runtime READY/lifecycle contract is invalid")

        training_contract = _yaml(config_dir / "policy_training_contract.yaml")
        if (
            training_contract.get("policy_generation") != generation
            or int(training_contract.get("checkpoint_iteration", -1)) != checkpoint
        ):
            raise ValueError("policy training contract identity mismatch")
        anchor = training_contract["ready_observation_anchor"]
        if not np.allclose(anchor["racket_target_rel_base_w"], ready_rel, atol=1e-12):
            raise ValueError("training/runtime READY position mismatch")
        if not np.allclose(anchor["racket_target_vel_w"], ready_velocity, atol=1e-12):
            raise ValueError("training/runtime READY velocity mismatch")
        action = training_contract["action"]
        if (
            action["raw_action_transform"] != "identity"
            or tuple(float(v) for v in action["action_clip"]) != clip
            or float(action["action_scale"]) != 0.25
            or tuple(action["passive_joint_indices"]) != HEAD_INDICES
        ):
            raise ValueError("training and bundled action contracts differ")

        if training_source is not None:
            source = Path(training_source).resolve()
            source_config = source / "a3_deploy/a3_deploy_example/config"
            if not source.is_dir():
                raise FileNotFoundError(f"training source not found: {source}")
            if (source_config / "action_adapter.yaml").read_bytes() != (
                config_dir / "action_adapter.yaml"
            ).read_bytes():
                raise ValueError("updated source and bundle ActionAdapter differ")
            source_runtime = _yaml(source_config / "hope_pingpong_runtime.yaml")
            source_lifecycle = source_runtime["lifecycle"]
            for key in (
                "ready_target_rel_base_w", "ready_target_vel_w",
                "ready_time_to_strike", "ready_swing_side",
                "follow_through_s", "recovery_s",
            ):
                if source_lifecycle[key] != lifecycle[key]:
                    raise ValueError(f"updated source and bundle lifecycle differ: {key}")

        return cls(
            bundle, generation, checkpoint, names, default_q, action_scale, clip,
            lower, upper, kp, kd, ready_rel, ready_velocity, ready_tts,
            ready_side, follow, recovery, control_hz, onnx_path
        )

    def ready_target(self, base_position_w: Iterable[float]) -> PolicyTarget:
        """Build the fixed READY world target from its first-frame reference base."""
        base = np.asarray(tuple(base_position_w), dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("base position must contain three finite values")
        return PolicyTarget(
            base + self.ready_rel_base_w,
            self.ready_velocity_w.copy(),
            self.ready_time_to_strike,
            self.ready_swing_side,
        )

    def build_observation(
        self,
        q: Iterable[float],
        dq: Iterable[float],
        base_position_w: Iterable[float],
        gravity_quaternion_wxyz: Iterable[float],
        heading_quaternion_wxyz: Iterable[float],
        pelvis_gyro_body: Iterable[float],
        last_action: Iterable[float],
        fixed_station_xy: Iterable[float],
        target: PolicyTarget,
    ) -> np.ndarray:
        q = np.asarray(tuple(q), dtype=np.float64)
        dq = np.asarray(tuple(dq), dtype=np.float64)
        base = np.asarray(tuple(base_position_w), dtype=np.float64)
        gyro = np.asarray(tuple(pelvis_gyro_body), dtype=np.float64)
        previous = np.asarray(tuple(last_action), dtype=np.float64)
        station = np.asarray(tuple(fixed_station_xy), dtype=np.float64)
        gravity_quat = normalize_quaternion_wxyz(gravity_quaternion_wxyz)
        heading_quat = normalize_quaternion_wxyz(heading_quaternion_wxyz)
        if q.shape != (ACTION_DIM,) or dq.shape != (ACTION_DIM,):
            raise ValueError("q and dq must each contain 31 values")
        if base.shape != (3,) or gyro.shape != (3,) or station.shape != (2,):
            raise ValueError("base/gyro/station observation shapes are invalid")
        if previous.shape != (ACTION_DIM,):
            raise ValueError("last_action must contain 31 values")
        vectors = (q, dq, base, gyro, previous, station, target.position_w, target.velocity_w)
        if not all(np.isfinite(value).all() for value in vectors):
            raise ValueError("observation source contains NaN or infinity")
        observation = np.empty(OBS_DIM, dtype=np.float64)
        observation[0:3] = gyro
        observation[3:34] = q - self.default_q
        observation[34:65] = dq
        observation[65:96] = previous
        # On hardware gravity comes from the pelvis IMU: yaw does not affect this
        # term, and the IMU is lower-latency and less jittery than optical pose.
        observation[96:99] = projected_gravity_body(gravity_quat)
        # World/table heading must share the planner frame, so this comes from the
        # calibrated BotA3->pelvis PPMocap orientation instead of free-running IMU yaw.
        observation[99:101] = base_forward_xy(heading_quat)
        observation[101:103] = station - base[:2]
        # Position subtraction preserves canonical world-axis components. Do not
        # rotate this vector into the pelvis frame.
        observation[103:106] = target.position_w - base
        observation[106:109] = target.velocity_w
        observation[109] = float(target.time_to_strike)
        observation[110] = float(target.swing_side)
        if not np.isfinite(observation).all():
            raise ValueError("constructed observation contains NaN or infinity")
        return np.ascontiguousarray(observation, dtype=np.float32)

    def decode_action(self, raw_action: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(tuple(raw_action), dtype=np.float64)
        if raw.shape != (ACTION_DIM,) or not np.isfinite(raw).all():
            raise ValueError("raw action must contain 31 finite values")
        applied = np.clip(raw, self.action_clip[0], self.action_clip[1])
        applied[list(HEAD_INDICES)] = 0.0
        desired = self.default_q + applied * self.action_scale
        desired = np.clip(desired, self.lower, self.upper)
        desired[list(HEAD_INDICES)] = self.default_q[list(HEAD_INDICES)]
        return applied, desired


class OnnxActor:
    def __init__(self, contract: A3RLContract) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is missing; run scripts/setup_rl_venv.sh") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(contract.onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX actor must have exactly one input and output")
        if inputs[0].name != "observation" or outputs[0].name != "raw_action":
            raise ValueError("unexpected ONNX input/output names")
        if inputs[0].shape[-1] != OBS_DIM or outputs[0].shape[-1] != ACTION_DIM:
            raise ValueError("unexpected ONNX input/output dimensions")
        metadata = self._session.get_modelmeta().custom_metadata_map or {}
        if tuple(metadata.get("joint_order", "").split(",")) != contract.joint_names:
            raise ValueError("ONNX joint_order metadata differs from deployment")
        if metadata.get("policy_generation") != contract.policy_generation:
            raise ValueError("ONNX policy_generation metadata differs from bundle")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(1, OBS_DIM)
        output = self._session.run([self._output_name], {self._input_name: value})[0]
        action = np.asarray(output, dtype=np.float32).reshape(ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("ONNX actor produced NaN or infinity")
        return action
