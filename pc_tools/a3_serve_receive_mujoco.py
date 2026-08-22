#!/usr/bin/env python3
"""MuJoCo visualization for model_72500 waist/legs + serving arms.

The packaged receive policy keeps running at 50 Hz in every mode.  During a
serve, only its waist and leg targets are sent to MuJoCo; the two arms follow
the tuned ``Serve_A3_leg_model`` serve trajectory and the head holds its captured
pose.  Task-related policy inputs are forced to READY while proprioception and
base state stay live.  The scene contains a regulation table, net and dynamic
40 mm ball.  Serving has no simulated ball; pressing N in receive mode launches
a legal one-bounce incoming ball and scores actual ball/racket/table contacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
# Load observation, lifecycle, ActionAdapter, ONNX and robot model directly from
# the model_72500 candidate bundle.  The serve state machine stays in this file.
RECEIVE_BUNDLE = REPO_ROOT / "model_72500_deploy_bundle"
SERVE_CONFIG = REPO_ROOT / "Serve_A3_leg_model/config/a3_lower_body.yaml"
SERVE_TRACKS_DIR = REPO_ROOT / "Serve_A3_leg_model/tracks"
RECEIVE_CONFIG = RECEIVE_BUNDLE / "config/runtime.yaml"
BALL_PHYSICS_CONFIG = RECEIVE_BUNDLE / "config/ball_physics.yaml"
# This local scene copy contains MuJoCo 3.11 API compatibility fixes. Geometry,
# ball/contact constants, robot MJCF and policy semantics all come from 72500.
PHYSICS_SCENE_DIR = REPO_ROOT / "yfr_hope_pingpong_deploy_20500/sim"
SUCCESS_METRIC_PATH = RECEIVE_BUNDLE / "scripts/success_metric.py"
MUJOCO_EVAL_PATH = RECEIVE_BUNDLE / "scripts/mujoco_eval_onnx.py"

RECEIVE_REFERENCE_DIR = RECEIVE_BUNDLE / "reference"
sys.path.insert(0, str(RECEIVE_REFERENCE_DIR))

from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig  # noqa: E402
from a3_deploy_onnx_ref_pingpong.joint_order import (  # noqa: E402
    HEAD_INDICES,
    ISAAC_JOINT_VELOCITY_LIMITS,
    JOINT_NAMES,
    NUM_JOINTS,
)
from a3_deploy_onnx_ref_pingpong.lifecycle import (  # noqa: E402
    Phase as ReceivePhase,
    SwingLifecycle,
)
from a3_deploy_onnx_ref_pingpong.observation import (  # noqa: E402
    RobotState,
    build_observation,
)
from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy  # noqa: E402
from a3_deploy_onnx_ref_pingpong.racket_command import (  # noqa: E402
    BACKHAND,
    FOREHAND,
    RacketCommand,
)
from a3_deploy_onnx_ref_pingpong.station_target import (  # noqa: E402
    derive_lateral_base_target,
    validate_lateral_station_policy_manifest,
)

sys.path.insert(0, str(PHYSICS_SCENE_DIR))
from mujoco_pingpong_scene import PingPongRealPhysicsScene  # noqa: E402


ARM = slice(5, 19)
HEAD = list(HEAD_INDICES)
LOCKED_WAIST = [1, 2]
MODEL_72500_ONNX_SHA256 = "6ab2e061062455a997afb7c630599d1eafd0a66f90dfcfb5c1da8177c8093ed0"


class Mode(Enum):
    RECEIVE = "接球策略 READY/击球"
    SERVE_HOMING = "进入发球初始姿态"
    SERVE_READY = "发球就绪"
    SERVE_WINDUP = "发球引拍"
    SERVE_SWING = "发球挥拍"
    SERVE_SETTLE = "发球随挥保持"
    SERVE_RETURN = "发球手臂回位"
    TO_RECEIVE = "切换到接球"


class GripperState(Enum):
    OPEN = "打开"
    CLOSED = "关闭"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Python 模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_success_metric():
    return _load_module("a3_serve_receive_success_metric", SUCCESS_METRIC_PATH)


class PhysicsBridge:
    """Small adapter around the real ball/table scene used by the controller."""

    def __init__(self, scene: PingPongRealPhysicsScene) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self._mj = scene._mj
        self._ball_gid = scene._ball_gid
        self._ball_contype = int(self.model.geom_contype[self._ball_gid])
        self._ball_conaffinity = int(self.model.geom_conaffinity[self._ball_gid])
        self.ball_active = False
        self.park_ball()

    def read_state(self):
        return self.scene.read_robot_state()

    def write_targets(self, q_des, kp, kd) -> None:
        self.scene.write_targets(q_des, kp, kd)

    def ball_state(self):
        return self.scene.ball_state()

    def racket_site_state(self):
        return self.scene.racket_site_state()

    def park_ball(self) -> None:
        self.ball_active = False
        self.model.geom_contype[self._ball_gid] = 0
        self.model.geom_conaffinity[self._ball_gid] = 0
        self.scene.set_ball(
            [self.scene.near_edge_x + self.scene.length + 2.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        )

    def launch_ball(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.ball_active = True
        self.model.geom_contype[self._ball_gid] = self._ball_contype
        self.model.geom_conaffinity[self._ball_gid] = self._ball_conaffinity
        self.scene.set_ball(position, velocity)

    def step(self):
        if not self.ball_active:
            self.scene.set_ball(
                [self.scene.near_edge_x + self.scene.length + 2.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            )
        return self.scene.step()

    def close(self) -> None:
        self.scene.close()


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return document


def _load_serve_tracks(base_document: dict, tracks_dir: Path) -> dict[str, dict]:
    """Load the four real-robot tuned numbered tracks into the full loop."""
    tracks: dict[str, dict] = {}
    for number in ("1", "2", "3", "4"):
        path = tracks_dir / f"{number}.yaml"
        document = _load_yaml(path)
        serve = document.get("serve")
        if not isinstance(serve, dict):
            raise ValueError(f"编号轨迹缺少 serve 映射: {path}")
        home = np.asarray(serve.get("home"), dtype=np.float64)
        windup = np.asarray(serve.get("windup_right"), dtype=np.float64)
        hit = np.asarray(serve.get("hit_through_right"), dtype=np.float64)
        if home.shape != (14,) or windup.shape != (7,) or hit.shape != (7,):
            raise ValueError(
                f"编号轨迹关节维度错误: {path}; "
                f"home={home.shape} windup={windup.shape} hit={hit.shape}"
            )
        if not all(np.all(np.isfinite(values)) for values in (home, windup, hit)):
            raise ValueError(f"编号轨迹包含 NaN/Inf: {path}")
        timing = serve.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"编号轨迹缺少 timing: {path}")
        durations = {
            name: float(timing[name])
            for name in ("home_s", "windup_s", "swing_s", "settle_s", "return_s")
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in durations.values()):
            raise ValueError(f"编号轨迹持续时间必须为正数: {path}")
        release_s = float(timing["release_s"])
        if (
            not np.isfinite(release_s)
            or release_s < -durations["windup_s"]
            or release_s > durations["swing_s"]
        ):
            raise ValueError(f"编号轨迹 release_s 超出挥拍区间: {path}")
        # Numbered YAML owns the complete serve block. Simulation, gains and
        # the model contract continue to come from the full-loop base config.
        tracks[number] = {**base_document, "serve": serve}
        print(
            f"[轨迹] 已加载 {number}: {path} "
            f"home={durations['home_s']:.2f}s "
            f"swing={durations['swing_s']:.2f}s release={release_s:+.2f}s",
            flush=True,
        )
    return tracks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReceiveController:
    """model_72500 control tick with its packaged lifecycle and observation."""

    def __init__(
        self,
        bridge: PhysicsBridge,
        config: RuntimeConfig,
        ball_physics,
        table_bounce,
        solve_incoming_velocity,
        *,
        no_mocap_serve_test: bool = False,
    ) -> None:
        self.bridge = bridge
        self.config = config
        self.ball_physics = ball_physics
        self.table_bounce = table_bounce
        self.solve_incoming_velocity = solve_incoming_velocity
        validate_lateral_station_policy_manifest(config.onnx_path)
        self.policy = OnnxPolicy(config.onnx_path)
        initial_state = bridge.read_state()
        self.nominal_station_xy = initial_state.base_pos_w[:2].copy()
        self.synthetic_base_pos_w = initial_state.base_pos_w.copy()
        self.last_physical_base_pos_w = initial_state.base_pos_w.copy()
        self.max_unobserved_base_drift_m = 0.0
        self.no_mocap_serve_test = no_mocap_serve_test
        self.base_target_xy = self.nominal_station_xy.copy()
        self.next_task_id = 1
        self.next_side = FOREHAND
        self.pending_command: RacketCommand | None = None
        self.had_active_ball = False
        self.completed_balls = 0
        self.last_observation: np.ndarray | None = None
        self.last_target = None
        self.ready_contract_checks = 0
        self.max_abs_raw_locked_waist = 0.0
        self.max_abs_applied_locked_waist = 0.0
        self.max_abs_q_des_locked_waist = 0.0
        self.reset()

    def reset(self) -> None:
        self.lifecycle = SwingLifecycle(self.config.lifecycle)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.base_target_xy = self.nominal_station_xy.copy()
        self.pending_command = None
        self.had_active_ball = False

    def queue_example_ball(self) -> bool:
        if self.no_mocap_serve_test:
            return False
        if self.lifecycle.phase is not ReceivePhase.READY:
            return False
        station = self.nominal_station_xy
        if self.next_side == FOREHAND:
            relative_pos = np.array([0.290, -0.750, 1.025])
            velocity = np.array([2.602879, 0.639805, 0.908102])
        else:
            relative_pos = np.array([0.600, 0.200, 1.025])
            velocity = np.array([1.871409, 0.082040, 1.045623])
        position = relative_pos.copy()
        position[:2] += station
        self.pending_command = RacketCommand(
            task_id=self.next_task_id,
            task_revision=0,
            swing_side=self.next_side,
            position=position,
            velocity=velocity,
            time_to_strike=1.0,
        )
        target_table = self.bridge.scene.to_table(position)
        origin_y = float(
            np.clip(
                target_table[1] + (0.08 if self.next_side == FOREHAND else -0.08),
                -self.table_bounce.width + 0.05,
                -0.05,
            )
        )
        origin_table = np.array([2.15, origin_y, 0.25], dtype=np.float64)
        launch_velocity, flight = self.solve_incoming_velocity(
            origin_table,
            target_table,
            1.0,
            self.ball_physics,
            self.table_bounce,
        )
        if len(flight.bounces) != 1:
            raise RuntimeError("接球测试来球未生成合法的单次落台轨迹")
        self.bridge.launch_ball(
            origin_table + self.bridge.scene.offset,
            launch_velocity,
        )
        self.bridge.scene.set_paddle_face_sign(
            1.0 if self.next_side == FOREHAND else -1.0
        )
        self.next_task_id += 1
        self.next_side = BACKHAND if self.next_side == FOREHAND else FOREHAND
        return True

    def infer_target(self) -> tuple[np.ndarray, ReceivePhase]:
        physical_state = self.bridge.read_state()
        self.last_physical_base_pos_w = physical_state.base_pos_w.copy()
        self.max_unobserved_base_drift_m = max(
            self.max_unobserved_base_drift_m,
            float(
                np.linalg.norm(
                    physical_state.base_pos_w[:2]
                    - self.synthetic_base_pos_w[:2]
                )
            ),
        )
        state = physical_state
        command = self.pending_command
        if self.no_mocap_serve_test:
            if self.lifecycle.phase is not ReceivePhase.READY:
                raise RuntimeError("无动捕发球测试模式的接球生命周期必须保持 READY")
            command = None
            # Match the proposed hardware fallback: joint/IMU proprioception
            # remains live, while the unavailable world pelvis position is
            # replaced by the fixed birth-station reference.
            state = RobotState(
                base_pos_w=self.synthetic_base_pos_w.copy(),
                base_quat_w=physical_state.base_quat_w,
                base_ang_vel_b=physical_state.base_ang_vel_b,
                q=physical_state.q,
                qd=physical_state.qd,
            )
        previous_phase = self.lifecycle.phase
        target = self.lifecycle.update(command, state)
        self.pending_command = None
        phase = self.lifecycle.phase
        if phase in (ReceivePhase.SWING, ReceivePhase.FOLLOW_THROUGH):
            self.had_active_ball = True

        station_cfg = self.config.lateral_station
        if self.no_mocap_serve_test:
            self.base_target_xy = self.synthetic_base_pos_w[:2].copy()
        elif phase in (ReceivePhase.READY, ReceivePhase.RECOVERY):
            self.base_target_xy = self.nominal_station_xy.copy()
        elif station_cfg.enabled:
            self.base_target_xy = derive_lateral_base_target(
                target.pos_w,
                target.swing_side,
                self.nominal_station_xy,
                base_target_y_range=station_cfg.base_target_y_range,
                forehand_reach_y=station_cfg.forehand_reach_y,
                backhand_reach_y=station_cfg.backhand_reach_y,
            )

        observation = build_observation(
            state,
            target,
            self.last_action,
            self.config.action_adapter.default_q,
            self.base_target_xy,
        )
        self.last_observation = observation.copy()
        self.last_target = target
        if self.no_mocap_serve_test:
            if phase is not ReceivePhase.READY:
                raise RuntimeError("无动捕发球测试模式意外离开 READY")
            if not np.array_equal(observation[101:103], np.zeros(2, dtype=np.float32)):
                raise RuntimeError(
                    "无动捕发球测试模式的 station_error 必须严格为零: "
                    f"actual={observation[101:103].tolist()}"
                )
        if phase is ReceivePhase.READY:
            lifecycle_cfg = self.config.lifecycle
            expected_rel = np.asarray(
                lifecycle_cfg.ready_target_rel_base_w, dtype=np.float32
            )
            expected_vel = np.asarray(
                lifecycle_cfg.ready_target_vel_w, dtype=np.float32
            )
            expected = np.concatenate(
                (
                    expected_rel,
                    expected_vel,
                    np.array(
                        [
                            lifecycle_cfg.ready_time_to_strike,
                            lifecycle_cfg.ready_swing_side,
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            if not np.allclose(observation[103:111], expected, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "READY observation differs from the model_72500 runtime contract: "
                    f"actual={observation[103:111].tolist()} "
                    f"expected={expected.tolist()}"
                )
            self.ready_contract_checks += 1
        raw_action = self.policy.infer(observation)
        self.max_abs_raw_locked_waist = max(
            self.max_abs_raw_locked_waist,
            float(np.max(np.abs(raw_action[LOCKED_WAIST]))),
        )
        applied = self.config.action_adapter.clip_raw_action(raw_action)
        if self.config.passive_neck:
            applied[HEAD] = 0.0
        self.max_abs_applied_locked_waist = max(
            self.max_abs_applied_locked_waist,
            float(np.max(np.abs(applied[LOCKED_WAIST]))),
        )
        if not np.array_equal(applied[LOCKED_WAIST], np.zeros(2)):
            raise RuntimeError(
                "model_72500 applied_action must lock waist roll/pitch to zero"
            )
        self.last_action = applied.copy()
        q_des = self.config.action_adapter.decode(applied)
        self.max_abs_q_des_locked_waist = max(
            self.max_abs_q_des_locked_waist,
            float(np.max(np.abs(q_des[LOCKED_WAIST]))),
        )
        if not np.array_equal(q_des[LOCKED_WAIST], np.zeros(2)):
            raise RuntimeError(
                "model_72500 q_des must lock waist roll/pitch to 0 rad"
            )
        # Sending-side hard override mirrors the deployment contract even though
        # the packaged ActionAdapter has already produced exact zeros.
        q_des[LOCKED_WAIST] = 0.0
        if self.config.passive_neck:
            q_des[HEAD] = self.config.action_adapter.default_q[HEAD]
        self.lifecycle.advance()
        if (
            previous_phase is ReceivePhase.FOLLOW_THROUGH
            and self.lifecycle.phase is ReceivePhase.READY
        ):
            self.completed_balls += 1
            self.bridge.park_ball()
        return q_des, phase


@dataclass
class Metrics:
    min_base_z: float = math.inf
    max_tilt_rad: float = 0.0
    max_target_step: float = 0.0
    max_arm_target_step: float = 0.0
    max_waist_leg_target_step: float = 0.0
    max_tracking_error: float = 0.0
    racket_contacts: int = 0
    net_clears: int = 0
    opponent_bounces: int = 0
    min_ball_racket_distance: float = math.inf
    closest_ball_pos: tuple[float, float, float] | None = None
    closest_racket_pos: tuple[float, float, float] | None = None
    closest_time_s: float | None = None
    ticks: int = 0


class ClosedLoopRunner:
    def __init__(
        self,
        bridge: PhysicsBridge,
        receive: ReceiveController,
        serve_doc: dict,
        serve_tracks: dict[str, dict],
        *,
        initial_serve_track: str,
        receive_transition_s: float,
        post_serve_settle_s: float,
        serve_time_scale: float,
        auto_cycle: bool,
        cycles: int,
        auto_delay_s: float,
        ready_dwell_s: float,
        hold_after_s: float,
    ) -> None:
        self.bridge = bridge
        self.receive = receive
        self.dt = receive.config.control_dt
        self.receive_transition_s = receive_transition_s
        self.post_serve_settle_s = post_serve_settle_s
        self.auto_cycle = auto_cycle
        self.requested_cycles = cycles
        self.auto_delay_s = auto_delay_s
        self.ready_dwell_s = ready_dwell_s
        self.hold_after_s = hold_after_s
        self.serve_time_scale = serve_time_scale
        self.serve_tracks = serve_tracks
        self.auto_serve_track = initial_serve_track

        all_kp = np.asarray(serve_doc["kps"], dtype=np.float64)
        all_kd = np.asarray(serve_doc["kds"], dtype=np.float64)
        if all_kp.shape != (NUM_JOINTS,) or all_kd.shape != (NUM_JOINTS,):
            raise ValueError("serve kps/kds must contain 31 joints")
        self.serve_arm_kp = all_kp[ARM].copy()
        self.serve_arm_kd = all_kd[ARM].copy()
        if initial_serve_track not in serve_tracks:
            raise ValueError(
                f"--serve-track 必须是 1/2/3/4，实际为 {initial_serve_track}"
            )
        self.active_serve_track = initial_serve_track
        initial_document = serve_tracks[initial_serve_track]
        self._apply_serve_document(initial_document)

        self.mode = Mode.RECEIVE
        self.phase_elapsed = 0.0
        self.mode_elapsed = 0.0
        self.last_target = bridge.read_state().q.copy()
        self.home_from = self.last_target[ARM].copy()
        self.head_hold = self.last_target[HEAD].copy()
        self.transition_from_upper = self.last_target[3:19].copy()
        self.gripper = GripperState.OPEN
        self.release_sent = False
        self.pending_serve_mode = False
        self.completed_cycles = 0
        self.completed_serves = 0
        self.done_since: float | None = None
        self.auto_stage = "enter_serve" if auto_cycle else "manual"
        self.auto_stage_elapsed = 0.0
        self.auto_ball_baseline = 0
        self.metrics = Metrics()
        self.visited: list[str] = [self.mode.name]
        self.events: list[str] = []
        self.ready_contract_logged = False
        self.rally_contacted = False
        self.rally_net_cleared = False
        self.rally_scored = False
        self._log_mode(
            (
                "无动捕测试：固定出生站位、忽略来球，model_72500 始终使用 READY 观测；"
                "按 V 固定使用 1 号、按 1–4 显式选轨迹；发球时仅覆盖双臂"
                if self.receive.no_mocap_serve_test
                else (
                    "model_72500 控制全身；按 V 固定使用 1 号、"
                    "按 1–4 显式选轨迹；发球时仅覆盖双臂，"
                    "腰部和双腿持续使用 READY 推理"
                )
            )
        )
        self._log(
            "发球后快速恢复："
            f"settle={self.post_serve_settle_s * self.serve_time_scale:.2f}s，"
            "跳过发球Home，"
            f"direct_to_receive={self.receive_transition_s:.2f}s"
        )

    def _apply_serve_document(self, document: dict) -> None:
        serve = document["serve"]
        timing = serve["timing"]
        self.serve_home = np.asarray(serve["home"], dtype=np.float64)
        self.windup = self.serve_home.copy()
        self.windup[7:] = np.asarray(serve["windup_right"], dtype=np.float64)
        self.hit = self.serve_home.copy()
        self.hit[7:] = np.asarray(serve["hit_through_right"], dtype=np.float64)
        self.home_s = float(timing["home_s"]) * self.serve_time_scale
        self.windup_s = float(timing["windup_s"]) * self.serve_time_scale
        self.swing_s = float(timing["swing_s"]) * self.serve_time_scale
        self.release_s = float(timing["release_s"]) * self.serve_time_scale
        # The four robot-tuned YAML files own Home/windup/swing/release. The
        # combined serve-receive loop deliberately shortens only the motion
        # after impact so model_72500 can regain the upper body promptly.
        self.settle_s = self.post_serve_settle_s * self.serve_time_scale
        # Retain the YAML value for compatibility with legacy callers. The
        # combined loop now transitions directly from hit-through to receive
        # and therefore does not enter SERVE_RETURN.
        self.return_s = float(timing["return_s"]) * self.serve_time_scale

    def _log(self, text: str) -> None:
        message = f"[闭环 t={self.bridge.data.time:6.2f}s] {text}"
        print(message, flush=True)
        self.events.append(message)

    def _log_mode(self, detail: str = "") -> None:
        suffix = f"；{detail}" if detail else ""
        self._log(f"状态 -> {self.mode.value}{suffix}")

    def _set_mode(self, mode: Mode, detail: str = "") -> None:
        self.mode = mode
        self.phase_elapsed = 0.0
        self.mode_elapsed = 0.0
        self.visited.append(mode.name)
        self._log_mode(detail)

    def _log_receive_observation(self, label: str) -> None:
        observation = self.receive.last_observation
        if observation is None:
            self._log(f"{label}：尚未完成首个策略观测")
            return
        self._log(
            f"{label} [101:111]："
            f"base_error=[{observation[101]:+.3f},{observation[102]:+.3f}] "
            f"racket_rel_base=[{observation[103]:+.3f},"
            f"{observation[104]:+.3f},{observation[105]:+.3f}] "
            f"racket_vel=[{observation[106]:+.3f},"
            f"{observation[107]:+.3f},{observation[108]:+.3f}] "
            f"tts={observation[109]:+.3f} side={observation[110]:+.0f}"
        )

    def handle_key(self, keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key in self.serve_tracks:
            self.request_serve_track(key)
        elif key == "v":
            self.request_serve_mode()
        elif key == "c":
            self.request_gripper_close()
        elif key == "g":
            self.request_gripper_open()
        elif key == "f":
            self.request_serve()
        elif key == "m":
            self.request_receive_mode()
        elif key == "n":
            self.request_ball()
        elif key == "i":
            self._log(
                f"状态={self.mode.value}，夹爪={self.gripper.value}，"
                f"当前发球轨迹={self.active_serve_track}，"
                f"接球phase={self.receive.lifecycle.phase.value}，"
                f"等待发球切换={'是' if self.pending_serve_mode else '否'}"
            )
            if self.receive.no_mocap_serve_test:
                physical = self.receive.last_physical_base_pos_w
                synthetic = self.receive.synthetic_base_pos_w
                drift = float(np.linalg.norm(physical[:2] - synthetic[:2]))
                self._log(
                    "无动捕base："
                    f"观测固定=[{synthetic[0]:+.3f},{synthetic[1]:+.3f}] "
                    f"物理真实=[{physical[0]:+.3f},{physical[1]:+.3f}] "
                    f"未观测漂移={drift:.3f}m"
                )
            self._log_receive_observation("model_72500观测")

    def request_serve_track(self, number: str) -> bool:
        if number not in self.serve_tracks:
            self._log(f"编号轨迹 {number} 不存在；保持轨迹 {self.active_serve_track}")
            return False
        if self.mode is Mode.RECEIVE:
            if self.receive.lifecycle.phase is not ReceivePhase.READY:
                self._log(
                    f"按键 {number} 被拒绝：接球生命周期尚未回到 READY"
                )
                return False
        elif self.mode is not Mode.SERVE_READY:
            self._log(f"按键 {number} 被拒绝：当前状态是{self.mode.value}")
            return False
        self._apply_serve_document(self.serve_tracks[number])
        self.active_serve_track = number
        self._log(
            f"已选择实机编号轨迹 {number}：自动进入该 YAML 的双臂 Home；"
            "到达后按 C/F"
        )
        self._begin_serve_homing(from_target=self.last_target)
        return True

    def request_serve_mode(self) -> bool:
        if self.mode is not Mode.RECEIVE:
            self._log(f"按键 V 被拒绝：当前状态是{self.mode.value}")
            return False
        # V is the fixed default entry and must never inherit the previously
        # selected numbered track.  Keys 1-4 remain the explicit way to enter
        # another tuned trajectory.
        self._apply_serve_document(self.serve_tracks["1"])
        self.active_serve_track = "1"
        if self.receive.lifecycle.phase is not ReceivePhase.READY:
            self.pending_serve_mode = True
            self._log(
                "已接受 V：已固定选择 1 号轨迹；当前接球尚未结束，"
                "将在生命周期回到 READY 后进入发球"
            )
            return True
        self._begin_serve_homing()
        return True

    def _begin_serve_homing(self, from_target: np.ndarray | None = None) -> None:
        self.pending_serve_mode = False
        state = self.bridge.read_state()
        source = self.last_target if from_target is None else from_target
        self.home_from = source[ARM].copy()
        self.head_hold = state.q[HEAD].copy()
        self.release_sent = False
        self._set_mode(
            Mode.SERVE_HOMING,
            f"轨迹={self.active_serve_track}；锁定任务观测为 READY；"
            "腰部/双腿=model_72500，双臂=发球轨迹",
        )

    def request_gripper_close(self) -> bool:
        if self.mode is not Mode.SERVE_READY:
            self._log(f"按键 C 被拒绝：当前状态是{self.mode.value}")
            return False
        self.gripper = GripperState.CLOSED
        self._log("夹爪 CLOSE 成功（仅模拟状态，不生成发球球）")
        return True

    def request_gripper_open(self) -> bool:
        self.gripper = GripperState.OPEN
        self._log("夹爪 OPEN 成功（仅模拟状态）")
        return True

    def request_serve(self) -> bool:
        if self.mode is not Mode.SERVE_READY:
            self._log(f"按键 F 被拒绝：当前状态是{self.mode.value}")
            return False
        if self.mode_elapsed < self.ready_dwell_s:
            remaining = self.ready_dwell_s - self.mode_elapsed
            self._log(f"按键 F 被拒绝：发球姿态尚需稳定 {remaining:.2f}s")
            return False
        if self.gripper is not GripperState.CLOSED:
            self._log("按键 F 被拒绝：请先按 C 关闭夹爪")
            return False
        self.gripper = GripperState.OPEN
        self.rally_contacted = False
        self.rally_net_cleared = False
        self.rally_scored = False
        self.release_sent = False
        self._set_mode(
            Mode.SERVE_WINDUP,
            f"已接受 F，播放编号轨迹 {self.active_serve_track}"
            "（发球阶段不生成球）",
        )
        return True

    def request_receive_mode(self) -> bool:
        if self.mode in (Mode.SERVE_HOMING, Mode.SERVE_READY):
            self._begin_receive_transition("已接受 M，取消发球并返回接球")
            return True
        if self.mode is Mode.RECEIVE:
            if self.pending_serve_mode:
                self.pending_serve_mode = False
                self._log("已取消等待中的发球切换，继续接球")
                return True
            self._log("已经处于接球模式")
            return True
        self._log(f"按键 M 被拒绝：{self.mode.value}过程中不允许中断")
        return False

    def request_ball(self) -> bool:
        if self.receive.no_mocap_serve_test:
            self._log("按键 N 被拒绝：无动捕发球测试模式不接受来球任务")
            return False
        if self.mode is not Mode.RECEIVE:
            self._log(f"按键 N 被拒绝：当前状态是{self.mode.value}")
            return False
        if not self.receive.queue_example_ball():
            self._log("按键 N 被拒绝：上一来球仍在 SWING/FOLLOW_THROUGH")
            return False
        self.rally_contacted = False
        self.rally_net_cleared = False
        self.rally_scored = False
        side = "正手" if self.receive.next_side == BACKHAND else "反手"
        ball_pos, ball_vel = self.bridge.ball_state()
        self._log(
            f"已注入真实物理来球：{side}，TTS=1.00s，"
            f"p=[{ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f}]m "
            f"v=[{ball_vel[0]:.3f},{ball_vel[1]:.3f},{ball_vel[2]:.3f}]m/s"
        )
        return True

    def _begin_receive_transition(self, detail: str) -> None:
        self.transition_from_upper = self.last_target[3:19].copy()
        self._set_mode(Mode.TO_RECEIVE, detail)

    def _set_auto_stage(self, stage: str) -> None:
        self.auto_stage = stage
        self.auto_stage_elapsed = 0.0

    def _auto_actions(self) -> None:
        if not self.auto_cycle or self.done_since is not None:
            return
        if self.auto_stage == "enter_serve":
            if (
                self.auto_stage_elapsed >= self.auto_delay_s
                and self.request_serve_track(self.auto_serve_track)
            ):
                self._set_auto_stage("close_and_fire")
        elif self.auto_stage == "close_and_fire":
            if self.mode is Mode.SERVE_READY and self.mode_elapsed >= self.ready_dwell_s:
                self.request_gripper_close()
                if self.request_serve():
                    self._set_auto_stage("wait_receive")
        elif self.auto_stage == "wait_receive":
            if self.mode is Mode.RECEIVE:
                if self.receive.no_mocap_serve_test:
                    self.completed_cycles += 1
                    if self.completed_cycles >= self.requested_cycles:
                        self.done_since = float(self.bridge.data.time)
                        self._set_auto_stage("done")
                    else:
                        self._set_auto_stage("enter_serve")
                else:
                    self._set_auto_stage("inject_ball")
        elif self.auto_stage == "inject_ball":
            if self.auto_stage_elapsed >= self.auto_delay_s and self.request_ball():
                self.auto_ball_baseline = self.receive.completed_balls
                self._set_auto_stage("wait_ball")
        elif (
            self.auto_stage == "wait_ball"
            and self.receive.completed_balls > self.auto_ball_baseline
        ):
            self.completed_cycles += 1
            if self.completed_cycles >= self.requested_cycles:
                self.done_since = float(self.bridge.data.time)
                self._set_auto_stage("done")
            else:
                self._set_auto_stage("enter_serve")

    def _target_for_mode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Exact model_72500 receive tick in every mode. Entering serve is permitted
        # only from lifecycle READY, so update(None) naturally remains READY;
        # no lifecycle reset or hand-authored target is introduced here.
        policy_target, _ = self.receive.infer_target()
        target = policy_target.copy()
        kp = self.receive.config.sim_kp.copy()
        kd = self.receive.config.sim_kd.copy()

        if self.mode is Mode.RECEIVE:
            if (
                self.pending_serve_mode
                and self.receive.lifecycle.phase is ReceivePhase.READY
            ):
                self._begin_serve_homing(from_target=target)
            return target, kp, kd

        # In every serve-related state, live policy output still owns waist and
        # legs.  Only head/arms are replaced below.
        target[HEAD] = self.head_hold

        if self.mode is Mode.SERVE_HOMING:
            alpha = _smoothstep(self.phase_elapsed / self.home_s)
            target[ARM] = self.home_from + alpha * (self.serve_home - self.home_from)
            kp[ARM] = (
                (1.0 - alpha) * self.receive.config.sim_kp[ARM]
                + alpha * self.serve_arm_kp
            )
            kd[ARM] = (
                (1.0 - alpha) * self.receive.config.sim_kd[ARM]
                + alpha * self.serve_arm_kd
            )
            if self.phase_elapsed >= self.home_s:
                target[ARM] = self.serve_home
                self.gripper = GripperState.OPEN
                self._set_mode(
                    Mode.SERVE_READY,
                    f"编号轨迹 {self.active_serve_track} Home 完成，夹爪 OPEN；"
                    f"按 C 夹球，稳定 {self.ready_dwell_s:.2f}s 后按 F",
                )
            return target, kp, kd

        if self.mode is Mode.SERVE_READY:
            target[ARM] = self.serve_home
            kp[ARM] = self.serve_arm_kp
            kd[ARM] = self.serve_arm_kd
            return target, kp, kd

        if self.mode is Mode.SERVE_WINDUP:
            alpha = _smoothstep(self.phase_elapsed / self.windup_s)
            target[ARM] = self.serve_home + alpha * (self.windup - self.serve_home)
            kp[ARM] = self.serve_arm_kp
            kd[ARM] = self.serve_arm_kd
            if self.phase_elapsed >= self.windup_s:
                target[ARM] = self.windup
                self._set_mode(Mode.SERVE_SWING, "到达引拍姿态")
            return target, kp, kd

        if self.mode is Mode.SERVE_SWING:
            alpha = float(np.clip(self.phase_elapsed / self.swing_s, 0.0, 1.0))
            target[ARM] = self.windup + alpha * (self.hit - self.windup)
            kp[ARM] = self.serve_arm_kp
            kd[ARM] = self.serve_arm_kd
            if not self.release_sent and self.phase_elapsed >= self.release_s:
                self.release_sent = True
                self._log("挥拍释放点：夹爪状态自动切换为 OPEN")
            if self.phase_elapsed >= self.swing_s:
                target[ARM] = self.hit
                self._set_mode(Mode.SERVE_SETTLE, "挥拍完成，进入随挥保持")
            return target, kp, kd

        if self.mode is Mode.SERVE_SETTLE:
            target[ARM] = self.hit
            kp[ARM] = self.serve_arm_kp
            kd[ARM] = self.serve_arm_kd
            if self.phase_elapsed >= self.settle_s:
                self.completed_serves += 1
                self._begin_receive_transition(
                    "短随挥完成，跳过发球 Home，直接恢复接球上肢控制"
                )
            return target, kp, kd

        if self.mode is Mode.SERVE_RETURN:
            alpha = _smoothstep(self.phase_elapsed / self.return_s)
            target[ARM] = self.hit + alpha * (self.serve_home - self.hit)
            kp[ARM] = self.serve_arm_kp
            kd[ARM] = self.serve_arm_kd
            if self.phase_elapsed >= self.return_s:
                target[ARM] = self.serve_home
                self._begin_receive_transition(
                    "发球策略播放结束，开始平滑恢复接球上肢控制"
                )
            return target, kp, kd

        if self.mode is Mode.TO_RECEIVE:
            alpha = _smoothstep(self.phase_elapsed / self.receive_transition_s)
            policy_upper = policy_target[3:19]
            target[3:19] = self.transition_from_upper + alpha * (
                policy_upper - self.transition_from_upper
            )
            kp[ARM] = (
                (1.0 - alpha) * self.serve_arm_kp
                + alpha * self.receive.config.sim_kp[ARM]
            )
            kd[ARM] = (
                (1.0 - alpha) * self.serve_arm_kd
                + alpha * self.receive.config.sim_kd[ARM]
            )
            if self.phase_elapsed >= self.receive_transition_s:
                target = policy_target
                self._set_mode(
                    Mode.RECEIVE,
                    (
                        "model_72500 恢复全身 READY 控制；来球保持禁用，按 V 再次发球"
                        if self.receive.no_mocap_serve_test
                        else "model_72500 恢复全身控制；按 N 注入来球，"
                             "按 1–4 选择下一发球位置"
                    ),
                )
            return target, kp, kd

        raise RuntimeError(f"unhandled mode: {self.mode}")

    @staticmethod
    def _tilt_rad(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return math.acos(float(np.clip(up_z, -1.0, 1.0)))

    def tick(self) -> None:
        self._auto_actions()
        target, kp, kd = self._target_for_mode()
        if not self.ready_contract_logged:
            self._log_receive_observation("model_72500 READY基线")
            self.ready_contract_logged = True
        state = self.bridge.read_state()
        if not all(
            np.all(np.isfinite(value))
            for value in (target, kp, kd, state.q, state.qd, state.base_pos_w)
        ):
            raise RuntimeError("MuJoCo control/state contains NaN/Inf")
        self.metrics.max_target_step = max(
            self.metrics.max_target_step,
            float(np.max(np.abs(target - self.last_target))),
        )
        arm_step = float(np.max(np.abs(target[ARM] - self.last_target[ARM])))
        waist_leg_indices = np.r_[0:3, 19:31]
        waist_leg_step = float(
            np.max(np.abs(target[waist_leg_indices] - self.last_target[waist_leg_indices]))
        )
        self.metrics.max_arm_target_step = max(
            self.metrics.max_arm_target_step, arm_step
        )
        self.metrics.max_waist_leg_target_step = max(
            self.metrics.max_waist_leg_target_step, waist_leg_step
        )
        self.metrics.max_tracking_error = max(
            self.metrics.max_tracking_error,
            float(np.max(np.abs(target - state.q))),
        )
        self.last_target = target.copy()
        self.bridge.write_targets(target, kp, kd)
        physics_events = self.bridge.step()
        if self.mode is Mode.RECEIVE and self.bridge.ball_active:
            ball_pos, _ = self.bridge.ball_state()
            racket_pos, _ = self.bridge.racket_site_state()
            distance = float(np.linalg.norm(ball_pos - racket_pos))
            if distance < self.metrics.min_ball_racket_distance:
                self.metrics.min_ball_racket_distance = distance
                self.metrics.closest_ball_pos = tuple(float(x) for x in ball_pos)
                self.metrics.closest_racket_pos = tuple(float(x) for x in racket_pos)
                self.metrics.closest_time_s = float(self.bridge.data.time)
        if (
            self.mode is Mode.RECEIVE
            and physics_events.ball_racket_contact
            and not self.rally_contacted
        ):
            self.rally_contacted = True
            self.metrics.racket_contacts += 1
            ball_pos, ball_vel = self.bridge.ball_state()
            self._log(
                "接球阶段检测到真实球拍碰撞："
                f"p=[{ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f}]m "
                f"v_out=[{ball_vel[0]:.3f},{ball_vel[1]:.3f},{ball_vel[2]:.3f}]m/s"
            )
        if self.rally_contacted:
            for net_z, direction in physics_events.net_crossings:
                clear_height = self.bridge.scene.net_height + self.bridge.scene.ball_radius
                if direction > 0.0 and net_z > clear_height:
                    if not self.rally_net_cleared:
                        self.metrics.net_clears += 1
                        self._log(
                            f"出球过网：球心高={net_z:.3f}m，"
                            f"门限={clear_height:.3f}m"
                        )
                    self.rally_net_cleared = True
            for table_x, table_y in physics_events.table_contacts:
                on_opponent = (
                    self.bridge.scene.net_x_table < table_x <= self.bridge.scene.length
                    and -self.bridge.scene.width <= table_y <= 0.0
                )
                if self.rally_net_cleared and on_opponent and not self.rally_scored:
                    self.rally_scored = True
                    self.metrics.opponent_bounces += 1
                    self._log(
                        "回球成功：首次有效落台位于对方半台，"
                        f"table_xy=[{table_x:.3f},{table_y:.3f}]m"
                    )
        next_state = self.bridge.read_state()
        self.metrics.min_base_z = min(
            self.metrics.min_base_z, float(next_state.base_pos_w[2])
        )
        self.metrics.max_tilt_rad = max(
            self.metrics.max_tilt_rad, self._tilt_rad(next_state.base_quat_w)
        )
        self.metrics.ticks += 1
        self.phase_elapsed += self.dt
        self.mode_elapsed += self.dt
        self.auto_stage_elapsed += self.dt

    def should_stop(self, duration_s: float | None) -> bool:
        if duration_s is not None and self.bridge.data.time >= duration_s:
            return True
        if self.done_since is not None:
            return self.bridge.data.time - self.done_since >= self.hold_after_s
        return False

    def result(self) -> dict:
        return {
            "status": (
                "ok"
                if not self.auto_cycle or self.completed_cycles >= self.requested_cycles
                else "incomplete"
            ),
            "completed_cycles": self.completed_cycles,
            "completed_serves": self.completed_serves,
            "active_serve_track": self.active_serve_track,
            "post_serve_recovery_s": (
                self.settle_s + self.receive_transition_s
            ),
            "ticks": self.metrics.ticks,
            "simulation_time_s": float(self.bridge.data.time),
            "visited_states": self.visited,
            "receive_completed_balls": self.receive.completed_balls,
            "min_base_z_m": self.metrics.min_base_z,
            "max_tilt_deg": float(np.degrees(self.metrics.max_tilt_rad)),
            "max_target_step_rad": self.metrics.max_target_step,
            "max_arm_target_step_rad": self.metrics.max_arm_target_step,
            "max_waist_leg_target_step_rad": self.metrics.max_waist_leg_target_step,
            "max_tracking_error_rad": self.metrics.max_tracking_error,
            "physical_racket_contacts": self.metrics.racket_contacts,
            "physical_net_clears": self.metrics.net_clears,
            "physical_opponent_bounces": self.metrics.opponent_bounces,
            "ready_contract_checks": self.receive.ready_contract_checks,
            "waist_lock": {
                "raw_action_max_abs": self.receive.max_abs_raw_locked_waist,
                "applied_action_max_abs": self.receive.max_abs_applied_locked_waist,
                "q_des_max_abs_rad": self.receive.max_abs_q_des_locked_waist,
                "kp_roll_pitch": [
                    float(self.receive.config.sim_kp[1]),
                    float(self.receive.config.sim_kp[2]),
                ],
            },
            "min_ball_racket_distance_m": (
                self.metrics.min_ball_racket_distance
                if np.isfinite(self.metrics.min_ball_racket_distance)
                else None
            ),
            "closest_ball_pos_w": self.metrics.closest_ball_pos,
            "closest_racket_pos_w": self.metrics.closest_racket_pos,
            "closest_time_s": self.metrics.closest_time_s,
            "single_command_owner": True,
            "serve_waist_legs_source": "model_72500_ready",
            "serve_arms_source": "Serve_A3_leg_model_tuned_trajectory",
            "gripper_network_io": False,
            "observation_mode": (
                "synthetic_birth_station_ready"
                if self.receive.no_mocap_serve_test
                else "live_mujoco_base_and_optional_ball"
            ),
            "planner_commands_accepted": not self.receive.no_mocap_serve_test,
            "max_unobserved_base_xy_drift_m": (
                self.receive.max_unobserved_base_drift_m
                if self.receive.no_mocap_serve_test
                else None
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-config", type=Path, default=SERVE_CONFIG)
    parser.add_argument(
        "--serve-tracks-dir", type=Path, default=SERVE_TRACKS_DIR,
        help="实机调试后的编号发球 YAML 目录（必须包含 1–4.yaml）",
    )
    parser.add_argument(
        "--serve-track", choices=("1", "2", "3", "4"), default="1",
        help="启动时预选编号轨迹（默认 1）；按 V 始终选择 1，按 1–4 可显式切换",
    )
    parser.add_argument("--receive-config", type=Path, default=RECEIVE_CONFIG)
    parser.add_argument(
        "--ball-physics-config", type=Path, default=BALL_PHYSICS_CONFIG
    )
    parser.add_argument(
        "--table-near-station-x",
        type=float,
        default=0.50,
        help="球台近端相对机器人出生位置的 +X 距离；0.50 与训练一致",
    )
    parser.add_argument("--view", action="store_true", help="打开 MuJoCo 可视化窗口")
    parser.add_argument("--realtime", action="store_true", help="按真实 50 Hz 速度运行")
    parser.add_argument(
        "--no-mocap-serve-test",
        action="store_true",
        help="固定出生站位并持续提供 READY 观测；忽略所有来球任务，只测试 V/C/F 发球",
    )
    parser.add_argument(
        "--auto-cycle",
        action="store_true",
        help="自动执行闭环；无动捕测试只循环 V/C/F，普通模式还会注入来球",
    )
    parser.add_argument("--cycles", type=int, default=1, help="自动闭环次数")
    parser.add_argument("--duration", type=float, default=None, help="最多运行的仿真秒数")
    parser.add_argument(
        "--receive-transition-s", type=float, default=0.20,
        help="短随挥结束后直接切回接球上肢控制的时间",
    )
    parser.add_argument(
        "--post-serve-settle-s", type=float, default=0.05,
        help="挥拍完成后的短暂随挥保持时间；不改变 YAML 的击球段",
    )
    parser.add_argument(
        "--serve-time-scale",
        type=float,
        default=1.0,
        help="仅缩放发球轨迹时间；1.0 为真实配置，慢放可用 3.0",
    )
    parser.add_argument("--auto-delay-s", type=float, default=1.50)
    parser.add_argument(
        "--ready-dwell-s", type=float, default=1.00, help="发球 READY 后最短稳定时间"
    )
    parser.add_argument("--hold-after-s", type=float, default=1.0)
    parser.add_argument("--max-tilt-deg", type=float, default=40.0)
    parser.add_argument("--min-base-z", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    for name in (
        "receive_transition_s",
        "post_serve_settle_s",
        "serve_time_scale",
        "auto_delay_s",
        "ready_dwell_s",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")
    if not args.view and not args.auto_cycle and args.duration is None:
        raise ValueError("headless mode needs --auto-cycle or --duration")

    receive_cfg = RuntimeConfig.load(args.receive_config)
    if tuple(receive_cfg.action_adapter.locked_joint_names) != (
        "waist_roll_joint",
        "waist_pitch_joint",
    ):
        raise RuntimeError(
            "model_72500 requires locked_joint_names=[waist_roll_joint, waist_pitch_joint]"
        )
    if not np.array_equal(receive_cfg.action_adapter.default_q[LOCKED_WAIST], np.zeros(2)):
        raise RuntimeError("model_72500 waist roll/pitch default_q must be exactly zero")
    if not np.array_equal(receive_cfg.sim_kp[LOCKED_WAIST], np.array([500.0, 500.0])):
        raise RuntimeError("model_72500 MuJoCo waist roll/pitch KP must be [500, 500]")
    if not np.array_equal(receive_cfg.sim_kd[LOCKED_WAIST], np.array([2.0, 2.0])):
        raise RuntimeError("model_72500 MuJoCo waist roll/pitch KD must be [2, 2]")
    ready_cfg = receive_cfg.lifecycle
    ready_contract = np.array(
        [
            *ready_cfg.ready_target_rel_base_w,
            *ready_cfg.ready_target_vel_w,
            ready_cfg.ready_time_to_strike,
            ready_cfg.ready_swing_side,
        ],
        dtype=np.float64,
    )
    expected_ready = np.array(
        [0.45, -0.25, 0.08, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    if not np.array_equal(ready_contract, expected_ready):
        raise RuntimeError(
            "model_72500 READY contract mismatch: "
            f"actual={ready_contract.tolist()} expected={expected_ready.tolist()}"
        )
    policy_sha256 = _sha256(Path(receive_cfg.onnx_path))
    if policy_sha256 != MODEL_72500_ONNX_SHA256:
        raise RuntimeError(
            "receive policy is not the model_72500 actor: "
            f"actual={policy_sha256} expected={MODEL_72500_ONNX_SHA256}"
        )
    observation_module = Path(
        sys.modules[build_observation.__module__].__file__
    ).resolve()
    if RECEIVE_BUNDLE.resolve() not in observation_module.parents:
        raise RuntimeError(
            f"observation builder was not loaded from model_72500: {observation_module}"
        )
    print(
        f"[合同] receive_bundle={RECEIVE_BUNDLE} "
        f"onnx_sha256={policy_sha256} observation={observation_module}",
        flush=True,
    )
    serve_doc = _load_yaml(args.serve_config)
    serve_tracks = _load_serve_tracks(serve_doc, args.serve_tracks_dir)
    serve_xml_path = Path(serve_doc["simulation"]["xml_path"])
    if not serve_xml_path.is_absolute():
        serve_xml_path = (args.serve_config.parent / serve_xml_path).resolve()
    # Serve_A3_leg_model is a deployment export and intentionally omits its
    # large sim/ tree. When present, retain the byte-for-byte guard; otherwise
    # the model_72500 bundle remains the single robot/physics source.
    if serve_xml_path.exists():
        if serve_xml_path.read_bytes() != Path(receive_cfg.model_xml_path).read_bytes():
            raise ValueError("serve and receive MuJoCo XML files are not identical")
    else:
        print(
            f"[合同] 发球包未携带 MuJoCo XML，使用接球包模型: {receive_cfg.model_xml_path}",
            flush=True,
        )

    success_metric = _load_success_metric()
    eval_physics = _load_module(
        "a3_serve_receive_mujoco_eval", MUJOCO_EVAL_PATH
    )
    ball_cfg = _load_yaml(args.ball_physics_config)
    ball_physics = success_metric.BallPhysics.from_config(ball_cfg)
    paddle_physics = success_metric.PaddlePhysics.from_config(ball_cfg)
    table_bounce = eval_physics._TableBounceModel.from_config(ball_cfg)

    # The aligned model's stand keyframe is the MuJoCo birth station.  Table
    # placement is fixed to that station, never to a later live pelvis pose.
    import mujoco

    robot_model = mujoco.MjModel.from_xml_path(str(receive_cfg.model_xml_path))
    robot_data = mujoco.MjData(robot_model)
    if robot_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(robot_model, robot_data, 0)
    base_joint = mujoco.mj_name2id(
        robot_model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint"
    )
    if base_joint < 0:
        raise ValueError("MuJoCo 模型缺少 pelvis_free_joint")
    base_qadr = int(robot_model.jnt_qposadr[base_joint])
    birth_xy = robot_data.qpos[base_qadr:base_qadr + 2].copy()

    scene = PingPongRealPhysicsScene(
        str(receive_cfg.model_xml_path),
        ball_cfg,
        JOINT_NAMES,
        control_dt=receive_cfg.control_dt,
        near_edge_x=float(birth_xy[0] + args.table_near_station_x),
        table_center_y=float(birth_xy[1]),
        launch_viewer=False,
        paddle_contact_model=success_metric.predict_paddle_contact,
        paddle_physics=paddle_physics,
        reset_joint_pos=receive_cfg.action_adapter.default_q,
        joint_velocity_limits=np.asarray(ISAAC_JOINT_VELOCITY_LIMITS),
    )
    scene.reset_stand()
    bridge = PhysicsBridge(scene)
    receive = ReceiveController(
        bridge,
        receive_cfg,
        ball_physics,
        table_bounce,
        eval_physics._solve_bounced_serve_velocity,
        no_mocap_serve_test=args.no_mocap_serve_test,
    )
    runner = ClosedLoopRunner(
        bridge,
        receive,
        serve_doc,
        serve_tracks,
        initial_serve_track=args.serve_track,
        receive_transition_s=args.receive_transition_s,
        post_serve_settle_s=args.post_serve_settle_s,
        serve_time_scale=args.serve_time_scale,
        auto_cycle=args.auto_cycle,
        cycles=args.cycles,
        auto_delay_s=args.auto_delay_s,
        ready_dwell_s=args.ready_dwell_s,
        hold_after_s=args.hold_after_s,
    )

    viewer = None
    try:
        if args.view:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                bridge.model, bridge.data, key_callback=runner.handle_key
            )
            viewer.cam.lookat[:] = [
                scene.near_edge_x + 0.55 * scene.length,
                scene.table_center_y,
                0.85,
            ]
            viewer.cam.distance = 4.2
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -18.0
            print(
                "[按键] 1/2/3/4=选择实机发球位置并自动回Home；"
                "V=固定使用1号轨迹进入发球；C=关闭夹爪；F=播放当前轨迹；"
                "G=打开夹爪；"
                "M=返回READY；"
                + (
                    "N=禁用（无动捕测试）；"
                    if args.no_mocap_serve_test
                    else "N=注入测试来球；"
                )
                + "I=状态；关闭窗口=退出",
                flush=True,
            )
        while not runner.should_stop(args.duration):
            if viewer is not None and not viewer.is_running():
                break
            start = time.perf_counter()
            runner.tick()
            if viewer is not None:
                viewer.sync()
            if args.realtime:
                remaining = receive_cfg.control_dt - (time.perf_counter() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()
        bridge.close()

    result = runner.result()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.auto_cycle:
        if result["completed_cycles"] < args.cycles:
            print("[失败] 未完成要求的闭环次数", file=sys.stderr)
            return 2
        if result["min_base_z_m"] < args.min_base_z:
            print("[失败] pelvis 高度低于安全门限", file=sys.stderr)
            return 3
        if result["max_tilt_deg"] > args.max_tilt_deg:
            print("[失败] pelvis 倾角超过安全门限", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
