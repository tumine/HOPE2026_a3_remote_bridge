#!/usr/bin/env python3
"""MuJoCo V/C/F serve test driven by hope_deploy's lower-body policy.

The 225-D -> 12-D TorchScript policy owns both legs at 50 Hz and needs no
world-frame pelvis position. Waist yaw/roll/pitch are held at zero, head and
arms hold their captured positions until V starts the exact serve trajectory
from ``hope_deploy/config/a3_lower_body.yaml``. The gripper is simulated only.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HOPE_DEPLOY_ROOT = Path("/home/bth/workspace/hope_deploy")
LOWER_SIM_DIR = HOPE_DEPLOY_ROOT / "sim/mujoco_deploy"
DEFAULT_CONFIG = HOPE_DEPLOY_ROOT / "config/a3_lower_body.yaml"
DEFAULT_POLICY = HOPE_DEPLOY_ROOT / "models/a3_lower_body.pt"
BALL_CONFIG = REPO_ROOT / "model_72500_deploy_bundle/config/ball_physics.yaml"
PHYSICS_SCENE_DIR = REPO_ROOT / "yfr_hope_pingpong_deploy_20500/sim"
EXPECTED_POLICY_SHA256 = (
    "582180f210852c922179e858c96f188c932d7a384617a622d11e802838dbd38f"
)

sys.path.insert(0, str(LOWER_SIM_DIR))
import deploy_lower_body_mujoco as lower_ref  # noqa: E402

sys.path.insert(0, str(PHYSICS_SCENE_DIR))
from mujoco_pingpong_scene import PingPongRealPhysicsScene  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "pc_tools"))
from a3_serve_receive_mujoco import PhysicsBridge, _load_yaml, _smoothstep  # noqa: E402


ARM = slice(5, 19)
HEAD = slice(3, 5)
WAIST = slice(0, 3)


class Mode(Enum):
    LOWER_READY = "下肢策略站立"
    HOMING = "进入发球初始姿态"
    SERVE_READY = "发球就绪"
    WINDUP = "发球引拍"
    SWING = "发球挥拍"
    SETTLE = "发球随挥保持"
    RETURN = "发球手臂回位"
    CLOSE_WAIT = "回位后夹爪闭合等待"


class Gripper(Enum):
    OPEN = "打开"
    CLOSED = "关闭"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tilt_rad(quat_wxyz: np.ndarray) -> float:
    _, x, y, _ = np.asarray(quat_wxyz, dtype=np.float64)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(up_z, -1.0, 1.0)))


class LowerBodyController:
    """Exact Python reference contract from hope_deploy."""

    def __init__(
        self,
        bridge: PhysicsBridge,
        config: dict,
        indices: lower_ref.ModelIndices,
        policy: torch.jit.ScriptModule,
    ) -> None:
        self.bridge = bridge
        self.config = config
        self.indices = indices
        self.policy = policy
        self.command = config["command_initial"].copy()
        self.held_q = bridge.read_state().q.copy()
        self.history: deque[np.ndarray] = deque(maxlen=lower_ref.HISTORY_LENGTH)
        self.last_action = np.zeros(lower_ref.ACTION_DIM, dtype=np.float32)
        self.inference_count = 0
        self.raw_action_max_abs = 0.0
        self.target_clip_count = 0

    def infer(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = lower_ref.build_frame(
            self.bridge.data,
            self.indices,
            self.config,
            self.command,
            self.last_action,
        )
        observation = lower_ref.push_frame(self.history, frame)
        with torch.inference_mode():
            action = (
                self.policy(torch.from_numpy(observation).unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        if action.shape != (lower_ref.ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("lower-body policy produced an invalid action")
        self.raw_action_max_abs = max(
            self.raw_action_max_abs, float(np.max(np.abs(action)))
        )
        action = np.clip(
            action, -self.config["clip_actions"], self.config["clip_actions"]
        )
        raw_leg_target = (
            self.config["default_leg_q"]
            + self.config["action_scale"] * action
        )
        leg_target = np.clip(
            raw_leg_target,
            self.config["action_lower"],
            self.config["action_upper"],
        )
        self.target_clip_count += int(np.count_nonzero(raw_leg_target != leg_target))

        target = self.held_q.copy()
        target[WAIST] = 0.0
        target[self.indices.leg_control] = leg_target
        self.last_action = action.copy()
        self.inference_count += 1
        return target, self.config["kps"].copy(), self.config["kds"].copy()


@dataclass
class Metrics:
    min_base_z: float = math.inf
    max_tilt_rad: float = 0.0
    max_target_step: float = 0.0
    max_leg_target_step: float = 0.0
    max_arm_target_step: float = 0.0
    max_tracking_error: float = 0.0
    ticks: int = 0


class Runner:
    def __init__(
        self,
        bridge: PhysicsBridge,
        lower: LowerBodyController,
        serve_doc: dict,
        *,
        auto_cycle: bool,
        cycles: int,
        auto_delay_s: float,
        realtime: bool,
    ) -> None:
        self.bridge = bridge
        self.lower = lower
        self.dt = 1.0 / 50.0
        self.auto_cycle = auto_cycle
        self.requested_cycles = cycles
        self.auto_delay_s = auto_delay_s
        self.realtime = realtime

        serve = serve_doc["serve"]
        timing = serve["timing"]
        self.home = np.asarray(serve["home"], dtype=np.float64)
        self.windup = self.home.copy()
        self.windup[7:] = np.asarray(serve["windup_right"], dtype=np.float64)
        self.hit = self.home.copy()
        self.hit[7:] = np.asarray(serve["hit_through_right"], dtype=np.float64)
        self.home_s = float(timing["home_s"])
        self.windup_s = float(timing["windup_s"])
        self.swing_s = float(timing["swing_s"])
        self.release_s = float(timing["release_s"])
        self.settle_s = float(timing["settle_s"])
        self.return_s = float(timing["return_s"])
        self.close_wait_s = float(timing["close_wait_s"])

        self.mode = Mode.LOWER_READY
        self.gripper = Gripper.OPEN
        state = bridge.read_state()
        self.home_from = state.q[ARM].copy()
        self.head_hold = state.q[HEAD].copy()
        self.last_target = state.q.copy()
        self.phase_elapsed = 0.0
        self.auto_elapsed = 0.0
        self.auto_stage = "home" if auto_cycle else "manual"
        self.completed_cycles = 0
        self.release_sent = False
        self.done_since: float | None = None
        self.metrics = Metrics()
        self.visited = [self.mode.name]
        self.events: list[str] = []
        self._log(
            "状态 -> 下肢策略站立；腿=a3_lower_body.pt，腰部=零位，"
            "上肢=捕获保持；无需base位置或动捕"
        )

    def _log(self, message: str) -> None:
        text = f"[下肢发球 t={self.bridge.data.time:6.2f}s] {message}"
        print(text, flush=True)
        self.events.append(text)

    def _set_mode(self, mode: Mode, detail: str) -> None:
        self.mode = mode
        self.phase_elapsed = 0.0
        self.visited.append(mode.name)
        self._log(f"状态 -> {mode.value}；{detail}")

    def handle_key(self, keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "v":
            self.request_home()
        elif key == "c":
            self.request_close()
        elif key == "g":
            self.gripper = Gripper.OPEN
            self._log("夹爪 OPEN（MuJoCo模拟状态）")
        elif key == "f":
            self.request_fire()
        elif key == "i":
            state = self.bridge.read_state()
            self._log(
                f"状态={self.mode.value} 夹爪={self.gripper.value} "
                f"pelvis_z={state.base_pos_w[2]:.3f}m "
                f"tilt={math.degrees(_tilt_rad(state.base_quat_w)):.2f}deg "
                f"lower_runs={self.lower.inference_count} "
                f"raw_action_max={self.lower.raw_action_max_abs:.3f}"
            )

    def request_home(self) -> bool:
        if self.mode not in (Mode.LOWER_READY, Mode.SERVE_READY):
            self._log(f"按键 V 被拒绝：当前状态是{self.mode.value}")
            return False
        state = self.bridge.read_state()
        self.home_from = state.q[ARM].copy()
        self.head_hold = state.q[HEAD].copy()
        self._set_mode(Mode.HOMING, "开始5秒平滑进入仓库发球Home")
        return True

    def request_close(self) -> bool:
        if self.mode is not Mode.SERVE_READY:
            self._log(f"按键 C 被拒绝：当前状态是{self.mode.value}")
            return False
        self.gripper = Gripper.CLOSED
        self._log("夹爪 CLOSE（MuJoCo模拟状态）")
        return True

    def request_fire(self) -> bool:
        if self.mode is not Mode.SERVE_READY:
            self._log(f"按键 F 被拒绝：当前状态是{self.mode.value}")
            return False
        if self.gripper is not Gripper.CLOSED:
            self._log("按键 F 被拒绝：请先按 C")
            return False
        self.release_sent = False
        self._set_mode(Mode.WINDUP, "已接受F，开始仓库发球轨迹")
        return True

    def _auto_actions(self) -> None:
        if not self.auto_cycle or self.done_since is not None:
            return
        if self.auto_stage == "home" and self.auto_elapsed >= self.auto_delay_s:
            if self.request_home():
                self.auto_stage = "fire"
                self.auto_elapsed = 0.0
        elif self.auto_stage == "fire" and self.mode is Mode.SERVE_READY:
            self.request_close()
            if self.request_fire():
                self.auto_stage = "wait"
        elif self.auto_stage == "wait" and self.mode is Mode.SERVE_READY:
            self.completed_cycles += 1
            if self.completed_cycles >= self.requested_cycles:
                self.done_since = float(self.bridge.data.time)
                self.auto_stage = "done"
            else:
                self.auto_stage = "home"
                self.auto_elapsed = 0.0

    def _serve_override(self, target: np.ndarray) -> None:
        if self.mode is Mode.LOWER_READY:
            return
        target[HEAD] = self.head_hold
        if self.mode is Mode.HOMING:
            alpha = _smoothstep(self.phase_elapsed / self.home_s)
            target[ARM] = self.home_from + alpha * (self.home - self.home_from)
            if self.phase_elapsed >= self.home_s:
                target[ARM] = self.home
                self.gripper = Gripper.OPEN
                self._set_mode(Mode.SERVE_READY, "Home完成；按C后按F")
            return
        if self.mode is Mode.SERVE_READY:
            target[ARM] = self.home
            return
        if self.mode is Mode.WINDUP:
            alpha = _smoothstep(self.phase_elapsed / self.windup_s)
            target[ARM] = self.home + alpha * (self.windup - self.home)
            if self.phase_elapsed >= self.windup_s:
                target[ARM] = self.windup
                self._set_mode(Mode.SWING, "引拍完成")
                if self.release_s == 0.0:
                    self.gripper = Gripper.OPEN
                    self.release_sent = True
                    self._log("释放点：夹爪 OPEN")
            return
        if self.mode is Mode.SWING:
            alpha = float(np.clip(self.phase_elapsed / self.swing_s, 0.0, 1.0))
            target[ARM] = self.windup + alpha * (self.hit - self.windup)
            if not self.release_sent and self.phase_elapsed >= self.release_s:
                self.gripper = Gripper.OPEN
                self.release_sent = True
                self._log("释放点：夹爪 OPEN")
            if self.phase_elapsed >= self.swing_s:
                target[ARM] = self.hit
                self._set_mode(Mode.SETTLE, "挥拍完成")
            return
        if self.mode is Mode.SETTLE:
            target[ARM] = self.hit
            if self.phase_elapsed >= self.settle_s:
                self._set_mode(Mode.RETURN, "随挥结束，返回Home")
            return
        if self.mode is Mode.RETURN:
            alpha = _smoothstep(self.phase_elapsed / self.return_s)
            target[ARM] = self.hit + alpha * (self.home - self.hit)
            if self.phase_elapsed >= self.return_s:
                target[ARM] = self.home
                self.gripper = Gripper.CLOSED
                self._set_mode(Mode.CLOSE_WAIT, "回位后夹爪自动CLOSE")
            return
        if self.mode is Mode.CLOSE_WAIT:
            target[ARM] = self.home
            if self.phase_elapsed >= self.close_wait_s:
                self._set_mode(Mode.SERVE_READY, "发球周期完成，可再次V/C/F")
            return
        raise RuntimeError(f"unhandled mode: {self.mode}")

    def tick(self) -> None:
        self._auto_actions()
        target, kp, kd = self.lower.infer()
        self._serve_override(target)
        state = self.bridge.read_state()
        if not all(np.all(np.isfinite(x)) for x in (target, kp, kd, state.q, state.qd)):
            raise RuntimeError("MuJoCo command/state contains NaN or Inf")
        self.metrics.max_target_step = max(
            self.metrics.max_target_step, float(np.max(np.abs(target - self.last_target)))
        )
        self.metrics.max_leg_target_step = max(
            self.metrics.max_leg_target_step,
            float(np.max(np.abs(target[19:31] - self.last_target[19:31]))),
        )
        self.metrics.max_arm_target_step = max(
            self.metrics.max_arm_target_step,
            float(np.max(np.abs(target[ARM] - self.last_target[ARM]))),
        )
        self.metrics.max_tracking_error = max(
            self.metrics.max_tracking_error, float(np.max(np.abs(target - state.q)))
        )
        self.last_target = target.copy()
        self.bridge.write_targets(target, kp, kd)
        self.bridge.step()
        state = self.bridge.read_state()
        self.metrics.min_base_z = min(
            self.metrics.min_base_z, float(state.base_pos_w[2])
        )
        self.metrics.max_tilt_rad = max(
            self.metrics.max_tilt_rad, _tilt_rad(state.base_quat_w)
        )
        self.metrics.ticks += 1
        self.phase_elapsed += self.dt
        self.auto_elapsed += self.dt

    def should_stop(self, duration_s: float | None) -> bool:
        if duration_s is not None and self.bridge.data.time >= duration_s:
            return True
        return self.done_since is not None and self.bridge.data.time - self.done_since >= 0.5

    def result(self) -> dict:
        return {
            "status": (
                "ok"
                if not self.auto_cycle or self.completed_cycles >= self.requested_cycles
                else "incomplete"
            ),
            "completed_cycles": self.completed_cycles,
            "ticks": self.metrics.ticks,
            "simulation_time_s": float(self.bridge.data.time),
            "visited_states": self.visited,
            "min_base_z_m": self.metrics.min_base_z,
            "max_tilt_deg": math.degrees(self.metrics.max_tilt_rad),
            "max_target_step_rad": self.metrics.max_target_step,
            "max_leg_target_step_rad": self.metrics.max_leg_target_step,
            "max_arm_target_step_rad": self.metrics.max_arm_target_step,
            "max_tracking_error_rad": self.metrics.max_tracking_error,
            "lower_body_inference_runs": self.lower.inference_count,
            "lower_body_raw_action_max_abs": self.lower.raw_action_max_abs,
            "lower_body_target_clip_count": self.lower.target_clip_count,
            "lower_body_observation_dim": lower_ref.OBSERVATION_DIM,
            "lower_body_action_dim": lower_ref.ACTION_DIM,
            "base_xy_observation": False,
            "gripper_network_io": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--ball-config", type=Path, default=BALL_CONFIG)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--auto-cycle", action="store_true")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--auto-delay-s", type=float, default=1.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--min-base-z", type=float, default=0.75)
    parser.add_argument("--max-tilt-deg", type=float, default=40.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cycles < 1 or args.auto_delay_s <= 0.0:
        raise ValueError("cycles and auto-delay-s must be positive")
    if not args.view and not args.auto_cycle and args.duration is None:
        raise ValueError("headless mode requires --auto-cycle or --duration")
    config = lower_ref.load_config(args.config)
    policy_path = args.policy.expanduser().resolve()
    policy_sha = _sha256(policy_path)
    if policy_sha != EXPECTED_POLICY_SHA256:
        raise RuntimeError(
            f"unexpected lower-body policy SHA256: {policy_sha}"
        )
    policy = lower_ref.load_policy(
        policy_path, lower_ref.OBSERVATION_DIM, lower_ref.ACTION_DIM
    )
    ball_config = _load_yaml(args.ball_config)

    robot_model = mujoco.MjModel.from_xml_path(str(config["xml_path"]))
    robot_data = mujoco.MjData(robot_model)
    keyframe = mujoco.mj_name2id(
        robot_model, mujoco.mjtObj.mjOBJ_KEY, config["initial_keyframe"]
    )
    if keyframe < 0:
        raise ValueError(f"missing keyframe: {config['initial_keyframe']}")
    mujoco.mj_resetDataKeyframe(robot_model, robot_data, keyframe)
    base_joint = mujoco.mj_name2id(
        robot_model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint"
    )
    base_qadr = int(robot_model.jnt_qposadr[base_joint])
    birth_xy = robot_data.qpos[base_qadr:base_qadr + 2].copy()

    scene = PingPongRealPhysicsScene(
        str(config["xml_path"]),
        ball_config,
        config["controlled_joints"],
        control_dt=1.0 / 50.0,
        near_edge_x=float(birth_xy[0] + 0.50),
        table_center_y=float(birth_xy[1]),
        launch_viewer=False,
        reset_joint_pos=None,
        # The hope_deploy lower-body reference does not project joint velocity.
        # The shared table scene requires a finite vector, so use a deliberately
        # inactive ceiling to preserve that original dynamics contract.
        joint_velocity_limits=np.full(31, 1.0e6, dtype=np.float64),
        contact_physics_dt=float(config["simulation_dt"]),
    )
    scene.reset_stand()
    bridge = PhysicsBridge(scene)
    indices = lower_ref.build_model_indices(scene.model, config)
    lower = LowerBodyController(bridge, config, indices, policy)
    runner = Runner(
        bridge,
        lower,
        _load_yaml(args.config),
        auto_cycle=args.auto_cycle,
        cycles=args.cycles,
        auto_delay_s=args.auto_delay_s,
        realtime=args.realtime,
    )
    print(
        "[合同] lower_body=hope_deploy/a3_lower_body.pt "
        f"sha256={policy_sha} obs=225 action=12 command=[0,0,0] base_xy=unused",
        flush=True,
    )

    viewer = None
    try:
        if args.view:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(
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
                "[按键] V=进入发球Home；C=关闭夹爪；F=播放发球；"
                "G=打开夹爪；I=状态；关闭窗口=退出",
                flush=True,
            )
        while not runner.should_stop(args.duration):
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            runner.tick()
            if viewer is not None:
                viewer.sync()
            if args.realtime:
                remaining = runner.dt - (time.perf_counter() - started)
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
            return 2
        if result["min_base_z_m"] < args.min_base_z:
            print("[失败] pelvis高度低于安全门限", file=sys.stderr)
            return 3
        if result["max_tilt_deg"] > args.max_tilt_deg:
            print("[失败] pelvis倾角超过安全门限", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
