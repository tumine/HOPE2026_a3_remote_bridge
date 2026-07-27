#!/usr/bin/env python3

"""Interactive 50 Hz no-mocap RL deployment runner for the A3 remote bridge.

The process starts in OBSERVE and publishes nothing. The lifecycle is:

    pose                ramp to and hold the training default posture
    policy              enter continuous no-ball READY policy and stand
    control forehand    inject one forehand ball observation, then return READY
    control backhand    inject one backhand ball observation, then return READY
    stop                hold READY policy briefly, send zero gain, exit
    halt                immediately send zero gain and exit

There is no automatic serve/alternation path: F/B each inject exactly one task.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Optional

import numpy as np
import yaml

import rclpy
from joint_msgs.msg import Command, JointCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import UInt32

from a3_rl_contract import (
    ACTION_DIM,
    A3RLContract,
    OnnxActor,
    base_tilt_radians,
    remove_startup_yaw,
    yaw_radians_wxyz,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "26.7.25发球部署/a3_deploy/a3_pingpong_deploy_bundle"
)
DEFAULT_TRAINING_SOURCE = REPO_ROOT / "26.7.25发球部署/pingpang_ustc-srf"
DEFAULT_CONFIG = REPO_ROOT / "config/a3_rl_deploy.yaml"
MAX_PENDING_FRAMES = 64
MDU_MAX_KP = 250.0
MDU_MAX_KD = 8.0
HOTKEY_COMMANDS = {
    "p": "pose",
    "d": "policy",
    "f": "control forehand",
    "b": "control backhand",
    "s": "stop",
    "x": "halt",
    "i": "status",
    "h": "help",
    "?": "help",
    "q": "quit",
}


def stamp_ns(message: object) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def reorder(names, values, canonical: tuple[str, ...]) -> np.ndarray:
    names_tuple = tuple(names)
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(names_tuple) != ACTION_DIM or array.shape != (ACTION_DIM,):
        raise ValueError("joint state must contain exactly 31 names and values")
    if len(set(names_tuple)) != ACTION_DIM:
        raise ValueError("joint state names are not unique")
    lookup = {name: index for index, name in enumerate(names_tuple)}
    missing = [name for name in canonical if name not in lookup]
    if missing:
        raise ValueError("joint state is missing: " + ", ".join(missing))
    ordered = array[[lookup[name] for name in canonical]]
    if not np.isfinite(ordered).all():
        raise ValueError("joint state contains NaN or infinity")
    return ordered


@dataclass
class LiveFrame:
    q: np.ndarray
    dq: np.ndarray
    quat_wxyz: np.ndarray
    gyro: np.ndarray
    stamp_ns: int
    received_at: float


@dataclass(frozen=True)
class RunnerLimits:
    control_hz: float
    default_pose_ramp_s: float
    default_pose_settle_s: float
    default_position_error_rad: float
    default_velocity_rad_s: float
    default_kp_scale: float
    default_kd_scale: float
    policy_warmup_s: float
    stop_ready_duration_s: float
    zero_gain_ticks: int
    max_state_age_s: float
    max_base_tilt_rad: float
    max_abs_raw_action: float
    max_inference_time_ms: float
    max_pending_acks: int
    policy_blend: float

    @classmethod
    def load(cls, path: Path, policy_blend_override: Optional[float]) -> "RunnerLimits":
        with path.open("r", encoding="utf-8") as stream:
            doc = yaml.safe_load(stream)
        safety = doc["safety"]
        default_pose = doc["default_pose"]
        blend = (
            float(policy_blend_override)
            if policy_blend_override is not None
            else float(doc["policy_blend"])
        )
        result = cls(
            control_hz=float(doc["control_hz"]),
            default_pose_ramp_s=float(default_pose["ramp_s"]),
            default_pose_settle_s=float(default_pose["settle_s"]),
            default_position_error_rad=float(
                default_pose["max_position_error_rad"]
            ),
            default_velocity_rad_s=float(default_pose["max_velocity_rad_s"]),
            default_kp_scale=float(default_pose["kp_scale"]),
            default_kd_scale=float(default_pose["kd_scale"]),
            policy_warmup_s=float(doc["policy_warmup_s"]),
            stop_ready_duration_s=float(doc["stop_ready_duration_s"]),
            zero_gain_ticks=int(doc["zero_gain_ticks"]),
            max_state_age_s=float(safety["max_state_age_s"]),
            max_base_tilt_rad=float(safety["max_base_tilt_rad"]),
            max_abs_raw_action=float(safety["max_abs_raw_action"]),
            max_inference_time_ms=float(safety["max_inference_time_ms"]),
            max_pending_acks=int(safety["max_pending_acks"]),
            policy_blend=blend,
        )
        if not math.isclose(result.control_hz, 50.0, abs_tol=1.0e-12):
            raise ValueError("RL runner must use exactly 50 Hz")
        positive = (
            result.default_pose_ramp_s,
            result.default_pose_settle_s,
            result.default_position_error_rad,
            result.default_velocity_rad_s,
            result.default_kp_scale,
            result.default_kd_scale,
            result.policy_warmup_s,
            result.stop_ready_duration_s,
            result.max_state_age_s,
            result.max_base_tilt_rad,
            result.max_abs_raw_action,
            result.max_inference_time_ms,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("runner safety values must be finite and positive")
        if (
            result.zero_gain_ticks <= 0
            or result.max_pending_acks <= 0
            or not 0.0 <= result.policy_blend <= 1.0
        ):
            raise ValueError("invalid zero_gain_ticks or policy_blend")
        return result


class A3RLDeploy(Node):
    OBSERVE = "OBSERVE"
    DEFAULT_RAMP = "DEFAULT_RAMP"
    DEFAULT_HOLD = "DEFAULT_HOLD"
    POLICY_WARMUP = "POLICY_WARMUP"
    POLICY_WAIT = "POLICY_WAIT"
    BALL = "BALL"
    STOP_READY = "STOP_READY"
    ZERO_GAIN = "ZERO_GAIN"
    FAULT = "FAULT"
    DONE = "DONE"

    def __init__(
        self,
        contract: A3RLContract,
        limits: RunnerLimits,
        topic_prefix: str,
        actor: OnnxActor,
        enable_actuation: bool,
        start_console: bool,
        line_console: bool,
    ) -> None:
        super().__init__("a3_pc_rl_deploy")
        self.contract = contract
        self.limits = limits
        self.actor = actor
        self.enable_actuation = bool(enable_actuation)
        self._default_pose_kp = np.minimum(
            self.contract.kp * self.limits.default_kp_scale, MDU_MAX_KP
        )
        self._default_pose_kd = np.minimum(
            self.contract.kd * self.limits.default_kd_scale, MDU_MAX_KD
        )
        self.prefix = topic_prefix.rstrip("/")
        self.mode = self.OBSERVE
        self.done = False
        self.clean_stop_completed = False

        self._joints: "OrderedDict[int, JointState]" = OrderedDict()
        self._pelvis: "OrderedDict[int, Imu]" = OrderedDict()
        self._latest: LiveFrame | None = None
        self._last_matched_stamp = -1
        self._invalid_frames = 0
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._console_thread: threading.Thread | None = None

        self._session = "a3_remote_control/rl-%d-%d" % (
            __import__("os").getpid(),
            time.time_ns(),
        )
        self._sequence = 0
        self._acks: set[int] = set()
        self._last_command_q: np.ndarray | None = None
        self._pose_start_q: np.ndarray | None = None
        self._pose_ready_ticks = 0
        self._pose_ready_announced = False
        self._phase_tick = 0
        self._control_side = "forehand"
        self._completed_swings = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._startup_yaw: float | None = None
        self._zero_ticks_sent = 0
        self._fault_reason = ""
        self._inference_ms: deque[float] = deque(maxlen=1000)
        self._last_report = time.monotonic()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            JointState, self.prefix + "/joint_states", self._on_joints, qos
        )
        self.create_subscription(Imu, self.prefix + "/pelvis_imu", self._on_imu, qos)
        self._publisher = None
        if self.enable_actuation:
            self._publisher = self.create_publisher(
                JointCommand, self.prefix + "/joint_command_control", qos
            )
            self.create_subscription(
                UInt32,
                self.prefix + "/joint_command_control_ack",
                self._on_ack,
                qos,
            )
        self.create_timer(1.0 / self.limits.control_hz, self._tick)

        if start_console:
            self._console_thread = threading.Thread(
                target=(self._line_console_loop if line_console else self._hotkey_loop),
                name="a3-rl-console",
                daemon=True,
            )
            self._console_thread.start()
        self.get_logger().warning(
            "RL runner started in OBSERVE: actuation=%s blend=%.3f no_mocap=yes "
            "policy_Kp/Kd=training==sim2sim; press P for default pose, then D for policy"
            % (
                "ENABLED" if self.enable_actuation else "disabled",
                limits.policy_blend,
            )
        )
        self._print_help()

    def submit_console_command(self, command: str) -> None:
        self._commands.put(command)

    def _line_console_loop(self) -> None:
        while not self.done:
            try:
                line = input("a3-rl> ")
            except EOFError:
                return
            except KeyboardInterrupt:
                self._commands.put("stop")
                return
            self._commands.put(line.strip())

    def _hotkey_loop(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().warning(
                "stdin is not a TTY; falling back to line commands with Enter"
            )
            self._line_console_loop()
            return

        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.done:
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                key = os.read(fd, 1).decode("utf-8", errors="ignore").lower()
                if key in ("\r", "\n", "\t", " "):
                    continue
                command = HOTKEY_COMMANDS.get(key)
                if command is None:
                    continue
                self._commands.put(command)
                # ROS logs can overwrite an ordinary prompt. Echo the accepted key
                # as a complete line so the operator always sees what was queued.
                print(f"\n[KEY {key.upper()}] {command}", flush=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)

    @staticmethod
    def _trim(mapping: OrderedDict) -> None:
        while len(mapping) > MAX_PENDING_FRAMES:
            mapping.popitem(last=False)

    def _on_joints(self, message: JointState) -> None:
        stamp = stamp_ns(message)
        self._joints[stamp] = message
        self._trim(self._joints)
        self._match(stamp)

    def _on_imu(self, message: Imu) -> None:
        stamp = stamp_ns(message)
        self._pelvis[stamp] = message
        self._trim(self._pelvis)
        self._match(stamp)

    def _match(self, stamp: int) -> None:
        if stamp <= self._last_matched_stamp:
            return
        joints = self._joints.get(stamp)
        imu = self._pelvis.get(stamp)
        if joints is None or imu is None:
            return
        self._joints.pop(stamp, None)
        self._pelvis.pop(stamp, None)
        try:
            q = reorder(joints.name, joints.position, self.contract.joint_names)
            dq = reorder(joints.name, joints.velocity, self.contract.joint_names)
            quat = np.asarray(
                (
                    imu.orientation.w,
                    imu.orientation.x,
                    imu.orientation.y,
                    imu.orientation.z,
                ),
                dtype=np.float64,
            )
            gyro = np.asarray(
                (
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ),
                dtype=np.float64,
            )
            if not np.isfinite(quat).all() or not np.isfinite(gyro).all():
                raise ValueError("pelvis IMU contains NaN or infinity")
            # Contract validation/normalization is performed during each policy tick.
            if float(np.linalg.norm(quat)) < 1.0e-9:
                raise ValueError("pelvis IMU quaternion is zero")
        except ValueError as exc:
            self._invalid_frames += 1
            if self._invalid_frames <= 5:
                self.get_logger().error(str(exc))
            return
        self._latest = LiveFrame(q, dq, quat, gyro, stamp, time.monotonic())
        self._last_matched_stamp = stamp

    def _on_ack(self, message: UInt32) -> None:
        sequence = int(message.data)
        if sequence < self._sequence:
            self._acks.add(sequence)

    def _tick(self) -> None:
        self._drain_operator_commands()
        if self.done:
            return
        if self.mode == self.OBSERVE:
            self._report_status()
            return
        if self.mode in (self.ZERO_GAIN, self.FAULT):
            self._tick_zero_gain()
            self._report_status()
            return
        if not self._state_is_safe():
            self._report_status()
            return
        assert self._latest is not None

        if self.mode == self.DEFAULT_RAMP:
            self._tick_default_ramp()
        elif self.mode == self.DEFAULT_HOLD:
            self._tick_default_hold()
        elif self.mode == self.POLICY_WARMUP:
            self._tick_policy_wait(warmup=True)
        elif self.mode == self.POLICY_WAIT:
            self._tick_policy_wait(warmup=False)
        elif self.mode == self.BALL:
            self._tick_ball()
        elif self.mode == self.STOP_READY:
            self._tick_stop_ready()
        self._report_status()

    def _drain_operator_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            fields = command.strip().lower().split()
            if not fields:
                continue
            if fields[0] in ("help", "?"):
                self._print_help()
            elif fields[0] == "status":
                self._report_status(force=True)
            elif fields[0] in ("pose", "initial", "default") and len(fields) == 1:
                self._begin_pose()
            elif fields[0] in ("policy", "ready") and len(fields) == 1:
                self._begin_policy()
            elif fields[0] == "control" and len(fields) == 2 and fields[1] in self.contract.targets:
                self._begin_control(fields[1])
            elif fields[0] == "stop" and len(fields) == 1:
                self._begin_stop()
            elif fields[0] in ("halt", "estop") and len(fields) == 1:
                self._begin_halt("operator immediate halt")
            elif fields[0] in ("quit", "exit") and self.mode == self.OBSERVE:
                self.done = True
                self.clean_stop_completed = True
            else:
                self.get_logger().error("unknown/invalid command: %s" % command)

    def _begin_pose(self) -> None:
        if self.mode == self.DEFAULT_HOLD:
            self.get_logger().info("already holding the training default pose")
            return
        if self.mode != self.OBSERVE:
            self.get_logger().error("P is only allowed from OBSERVE")
            return
        if self._latest is None or self._state_age() > self.limits.max_state_age_s:
            self.get_logger().error("cannot start default pose: no fresh synchronized state")
            return
        self._pose_start_q = self._latest.q.copy()
        self._last_command_q = self._latest.q.copy()
        self._phase_tick = 0
        self._pose_ready_ticks = 0
        self._pose_ready_announced = False
        self.mode = self.DEFAULT_RAMP
        self.get_logger().warning(
            "DEFAULT POSE START: ramp=%.1fs Kp/Kd scale=%.2f/%.2f "
            "capped at %.0f/%.0f; D restores exact training gains; "
            "this stage does not balance the unsupported robot"
            % (
                self.limits.default_pose_ramp_s,
                self.limits.default_kp_scale,
                self.limits.default_kd_scale,
                MDU_MAX_KP,
                MDU_MAX_KD,
            )
        )

    def _begin_policy(self) -> None:
        if self.mode in (self.POLICY_WARMUP, self.POLICY_WAIT):
            self.get_logger().info("policy is already active")
            return
        if self.mode != self.DEFAULT_HOLD:
            self.get_logger().error(
                "D requires DEFAULT_HOLD pose_ready=yes; press P first"
            )
            return
        if not self._pose_is_ready():
            assert self._latest is not None
            pose_error = float(
                np.max(np.abs(self._latest.q - self.contract.default_q))
            )
            max_velocity = float(np.max(np.abs(self._latest.dq)))
            self.get_logger().error(
                "D rejected: pose_ready=no error=%.4frad velocity=%.4frad/s"
                % (pose_error, max_velocity)
            )
            return
        assert self._latest is not None
        self._startup_yaw = yaw_radians_wxyz(self._latest.quat_wxyz)
        self._last_command_q = self._latest.q.copy()
        self._phase_tick = 0
        self._control_side = "forehand"
        self._completed_swings = 0
        self._last_action.fill(0.0)
        self.mode = self.POLICY_WARMUP
        self.get_logger().warning(
            "POLICY ARMED BY D: warmup=%.1fs startup_yaw=%.2fdeg "
            "target=[0.40,+0.20,-0.05] vel=0 TTS=1.0 "
            "gains=training==sim2sim"
            % (self.limits.policy_warmup_s, math.degrees(self._startup_yaw))
        )

    def _begin_control(self, side: str) -> None:
        if self.mode != self.POLICY_WAIT:
            self.get_logger().error(
                "ball input requires POLICY_WAIT ready=yes; press D and wait"
            )
            return
        if self._latest is None:
            self.get_logger().error("ball input requires a fresh robot state")
            return
        self._control_side = side
        self._phase_tick = 0
        self.mode = self.BALL
        self.get_logger().warning(
            "BALL OBSERVATION START: side=%s ticks=%d duration=%.2fs "
            "blend=%.3f state/last_action_preserved=yes"
            % (
                side,
                self.contract.clock.total_steps,
                self.contract.clock.total_steps / self.limits.control_hz,
                self.limits.policy_blend,
            )
        )

    def _begin_stop(self) -> None:
        if self.mode == self.OBSERVE:
            self.done = True
            self.clean_stop_completed = True
            return
        if self.mode in (self.ZERO_GAIN, self.FAULT, self.DONE):
            return
        if self.mode in (self.DEFAULT_RAMP, self.DEFAULT_HOLD):
            self._zero_ticks_sent = 0
            self.mode = self.ZERO_GAIN
            self.get_logger().warning(
                "STOP before policy: sending current-position zero-gain frames"
            )
            return
        if self._latest is None:
            self._begin_halt("stop requested without fresh state")
            return
        self._phase_tick = 0
        self.mode = self.STOP_READY
        self.get_logger().warning(
            "STOP: holding no-ball READY policy for %.1fs before zero gain"
            % self.limits.stop_ready_duration_s
        )

    def _begin_halt(self, reason: str) -> None:
        self._fault_reason = reason
        self._zero_ticks_sent = 0
        self.mode = self.FAULT
        self.get_logger().error("HALT: %s" % reason)

    def _run_policy_tick(self, side: str, tts: float, ready: bool) -> bool:
        assert self._latest is not None
        try:
            policy_quat = self._policy_quaternion()
            observation = self.contract.build_observation(
                self._latest.q,
                self._latest.dq,
                policy_quat,
                self._latest.gyro,
                self._last_action,
                side,
                tts,
                target_override=self.contract.ready_targets[side] if ready else None,
            )
            start = time.perf_counter()
            raw_action = self.actor.infer(observation)
            inference_ms = (time.perf_counter() - start) * 1000.0
            self._inference_ms.append(inference_ms)
            if inference_ms > self.limits.max_inference_time_ms:
                raise RuntimeError(
                    "ONNX inference deadline missed: %.3fms > %.3fms"
                    % (inference_ms, self.limits.max_inference_time_ms)
                )
            max_raw = float(np.max(np.abs(raw_action)))
            if max_raw > self.limits.max_abs_raw_action:
                raw_index = int(np.argmax(np.abs(raw_action)))
                raise ValueError(
                    "raw action exceeds configured limit: %.3f > %.3f "
                    "joint=%s raw=%+.3f gravity=%s forward=%s station=%s "
                    "target=%s tts=%.3f side=%+.0f"
                    % (
                        max_raw,
                        self.limits.max_abs_raw_action,
                        self.contract.joint_names[raw_index],
                        float(raw_action[raw_index]),
                        np.array2string(observation[96:99], precision=3),
                        np.array2string(observation[99:101], precision=3),
                        np.array2string(observation[101:103], precision=3),
                        np.array2string(observation[103:106], precision=3),
                        float(observation[109]),
                        float(observation[110]),
                    )
                )
            applied, desired = self.contract.decode_action(
                raw_action, self.limits.policy_blend
            )
            self._send_position(desired, self.contract.kp, self.contract.kd)
            self._last_action = applied
            return True
        except Exception as exc:  # numeric/runtime faults must immediately de-arm
            self._begin_halt("policy tick failed: %s" % exc)
            return False

    @staticmethod
    def _smoothstep(value: float) -> float:
        clipped = min(max(float(value), 0.0), 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    def _tick_default_ramp(self) -> None:
        assert self._latest is not None
        assert self._pose_start_q is not None
        total = max(
            1,
            int(round(self.limits.default_pose_ramp_s * self.limits.control_hz)),
        )
        alpha = self._smoothstep((self._phase_tick + 1) / total)
        desired = (
            (1.0 - alpha) * self._pose_start_q
            + alpha * self.contract.default_q
        )
        self._send_position(desired, self._default_pose_kp, self._default_pose_kd)
        self._phase_tick += 1
        if self._phase_tick >= total:
            self.mode = self.DEFAULT_HOLD
            self._phase_tick = 0
            self._pose_ready_ticks = 0
            self.get_logger().warning(
                "DEFAULT_HOLD: waiting for position/velocity settling before D"
            )

    def _pose_in_tolerance(self) -> bool:
        if self._latest is None:
            return False
        pose_error = float(
            np.max(np.abs(self._latest.q - self.contract.default_q))
        )
        max_velocity = float(np.max(np.abs(self._latest.dq)))
        return (
            pose_error <= self.limits.default_position_error_rad
            and max_velocity <= self.limits.default_velocity_rad_s
        )

    def _pose_is_ready(self) -> bool:
        required = max(
            1,
            int(round(self.limits.default_pose_settle_s * self.limits.control_hz)),
        )
        return self.mode == self.DEFAULT_HOLD and self._pose_ready_ticks >= required

    def _tick_default_hold(self) -> None:
        self._send_position(
            self.contract.default_q,
            self._default_pose_kp,
            self._default_pose_kd,
        )
        if self._pose_in_tolerance():
            self._pose_ready_ticks += 1
        else:
            self._pose_ready_ticks = 0
            self._pose_ready_announced = False
        if self._pose_is_ready() and not self._pose_ready_announced:
            self._pose_ready_announced = True
            self.get_logger().warning(
                "DEFAULT POSE READY: pose_ready=yes; press D to enter policy"
            )

    def _tick_policy_wait(self, warmup: bool) -> None:
        if not self._run_policy_tick(self._control_side, 1.0, ready=True):
            return
        if warmup:
            self._phase_tick += 1
            total = max(1, int(round(self.limits.policy_warmup_s * 50.0)))
            if self._phase_tick >= total:
                self.mode = self.POLICY_WAIT
                self._phase_tick = 0
                self.get_logger().warning(
                    "POLICY_WAIT ready=yes: no ball; F/B inject exactly one observation task"
                )

    def _tick_ball(self) -> None:
        if self._phase_tick >= self.contract.clock.total_steps:
            self._completed_swings += 1
            self._phase_tick = 0
            self.mode = self.POLICY_WAIT
            self.get_logger().warning(
                "BALL COMPLETE: side=%s completed=%d; POLICY_WAIT restored; "
                "state/last_action preserved"
                % (self._control_side, self._completed_swings)
            )
            self._tick_policy_wait(warmup=False)
            return
        tts = self.contract.clock.time_to_strike(self._phase_tick)
        if self._run_policy_tick(self._control_side, tts, ready=False):
            self._phase_tick += 1

    def _tick_stop_ready(self) -> None:
        total = max(1, int(round(self.limits.stop_ready_duration_s * 50.0)))
        if self._phase_tick >= total:
            self._zero_ticks_sent = 0
            self.mode = self.ZERO_GAIN
            return
        if self._run_policy_tick(self._control_side, 1.0, ready=True):
            self._phase_tick += 1

    def _tick_zero_gain(self) -> None:
        if self._latest is not None:
            zeros = np.zeros(ACTION_DIM, dtype=np.float64)
            self._publish_command(self._latest.q, zeros, zeros)
        self._zero_ticks_sent += 1
        if self._zero_ticks_sent >= self.limits.zero_gain_ticks:
            self.clean_stop_completed = True
            self.done = True
            self.mode = self.DONE
            self.get_logger().warning(
                "zero-gain stop complete; MDU watchdog will latch until gateway restart"
            )

    def _state_is_safe(self) -> bool:
        if self._latest is None:
            self._begin_halt("no synchronized joint/IMU state")
            return False
        age = self._state_age()
        if age > self.limits.max_state_age_s:
            self._begin_halt(
                "state stale: %.1fms > %.1fms"
                % (age * 1000.0, self.limits.max_state_age_s * 1000.0)
            )
            return False
        try:
            tilt = base_tilt_radians(self._policy_quaternion())
        except ValueError as exc:
            self._begin_halt(str(exc))
            return False
        if tilt > self.limits.max_base_tilt_rad:
            self._begin_halt(
                "base tilt %.2fdeg exceeds %.2fdeg"
                % (math.degrees(tilt), math.degrees(self.limits.max_base_tilt_rad))
            )
            return False
        if self.enable_actuation:
            pending = self._sequence - len(self._acks)
            if pending > self.limits.max_pending_acks:
                self._begin_halt(
                    "command ACK backlog %d exceeds %d"
                    % (pending, self.limits.max_pending_acks)
                )
                return False
        return True

    def _policy_quaternion(self) -> np.ndarray:
        if self._latest is None:
            raise ValueError("no IMU quaternion")
        if self._startup_yaw is None:
            self._startup_yaw = yaw_radians_wxyz(self._latest.quat_wxyz)
        # MuJoCo starts with the robot facing world +X.  Without table/world
        # localization, the hardware IMU yaw origin is arbitrary, so calibrate
        # the D-entry heading to MuJoCo +X. Roll/pitch (and therefore projected
        # gravity/tilt) remain unchanged.
        return remove_startup_yaw(self._latest.quat_wxyz, self._startup_yaw)

    def _state_age(self) -> float:
        return float("inf") if self._latest is None else time.monotonic() - self._latest.received_at

    def _send_position(
        self, desired: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> None:
        assert self._latest is not None
        desired = np.clip(np.asarray(desired, dtype=np.float64), self.contract.lower, self.contract.upper)
        # Preserve the exact training/sim2sim action mapping.  Additional
        # per-tick or command-to-state clipping weakens the PD correction and
        # made the no-ball READY policy fall in the matching MuJoCo model.
        # Mechanical limits remain here; the MDU independently enforces a
        # command-to-state ceiling, gains, freshness, order and watchdog.
        self._last_command_q = desired.copy()
        self._publish_command(desired, kp, kd)

    def _publish_command(
        self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> None:
        if not self.enable_actuation:
            self._sequence += 1
            return
        assert self._publisher is not None
        message = JointCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._session
        for index, name in enumerate(self.contract.joint_names):
            row = Command()
            row.name = name
            row.sequence = self._sequence
            row.position = float(q_des[index])
            row.velocity = 0.0
            row.effort = 0.0
            row.stiffness = float(kp[index])
            row.damping = float(kd[index])
            message.joints.append(row)
        self._publisher.publish(message)
        self._sequence += 1

    def emergency_zero_gain(self, count: int = 5) -> None:
        if not self.enable_actuation or self._latest is None or self._publisher is None:
            return
        zeros = np.zeros(ACTION_DIM, dtype=np.float64)
        for _ in range(max(1, count)):
            self._publish_command(self._latest.q, zeros, zeros)

    def close_console(self) -> None:
        """Let the hotkey thread restore the terminal before process exit."""
        self.done = True
        if (
            self._console_thread is not None
            and self._console_thread is not threading.current_thread()
        ):
            self._console_thread.join(timeout=0.5)

    def _report_status(self, force: bool = False) -> None:
        # Avoid interleaving a 1 Hz status stream with the single-key console.
        # State transitions and faults have their own logs; I requests one
        # explicit snapshot through force=True.
        if not force:
            return
        now = time.monotonic()
        self._last_report = now
        age_ms = self._state_age() * 1000.0
        tilt_deg = float("nan")
        default_error = float("nan")
        if self._latest is not None:
            try:
                tilt_deg = math.degrees(base_tilt_radians(self._policy_quaternion()))
            except ValueError:
                pass
            default_error = float(
                np.max(np.abs(self._latest.q - self.contract.default_q))
            )
        ready = (
            self.mode == self.POLICY_WAIT
            and age_ms <= self.limits.max_state_age_s * 1000.0
        )
        pose_ready = self._pose_is_ready()
        p95 = (
            float(np.percentile(np.asarray(self._inference_ms), 95))
            if self._inference_ms
            else 0.0
        )
        self.get_logger().info(
            "mode=%s state_age=%.1fms tilt=%.2fdeg default_err=%.4frad "
            "pose_ready=%s ready=%s side=%s phase=%d/%d balls=%d "
            "sent=%d acked=%d pending=%d infer_p95=%.3fms invalid=%d"
            % (
                self.mode,
                age_ms,
                tilt_deg,
                default_error,
                "yes" if pose_ready else "no",
                "yes" if ready else "no",
                self._control_side,
                self._phase_tick,
                self.contract.clock.total_steps,
                self._completed_swings,
                self._sequence,
                len(self._acks),
                self._sequence - len(self._acks) if self.enable_actuation else 0,
                p95,
                self._invalid_frames,
            )
        )

    def _print_help(self) -> None:
        self.get_logger().info(
            "HOTKEYS (no Enter): P=ramp/hold training default pose; "
            "D=enter no-ball READY policy (only pose_ready=yes); "
            "F=one forehand ball; B=one backhand ball; I=status; "
            "S=normal stop (pre-policy: direct zero; policy: READY then zero); "
            "X=halt(zero+exit); H=help; "
            "Q=quit(OBSERVE only)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 no-mocap HOPE RL deployment runner")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--training-source",
        type=Path,
        default=DEFAULT_TRAINING_SOURCE,
        help="updated training/sim2sim source used for startup parity checks",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topic-prefix", default="/a3_internal")
    parser.add_argument("--policy-blend", type=float)
    parser.add_argument("--enable-actuation", action="store_true")
    parser.add_argument(
        "--line-console",
        action="store_true",
        help="use full text commands followed by Enter instead of single-key control",
    )
    parser.add_argument("--i-understand-this-can-move-the-robot", action="store_true")
    parser.add_argument(
        "--allow-full-policy",
        action="store_true",
        help="required in addition to --policy-blend 1.0",
    )
    args = parser.parse_args()
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS prefix")
    if args.enable_actuation and not args.i_understand_this_can_move_the_robot:
        parser.error("real control requires --i-understand-this-can-move-the-robot")
    if args.policy_blend is not None and not 0.0 <= args.policy_blend <= 1.0:
        parser.error("policy-blend must be in [0,1]")
    if args.policy_blend == 1.0 and not args.allow_full_policy:
        parser.error("policy-blend 1.0 additionally requires --allow-full-policy")
    return args


def main() -> int:
    args = parse_args()
    contract = A3RLContract.load(args.bundle, args.training_source)
    limits = RunnerLimits.load(args.config, args.policy_blend)
    if limits.policy_blend == 1.0 and not args.allow_full_policy:
        raise SystemExit("configured policy_blend=1.0 requires --allow-full-policy")
    if args.enable_actuation and not math.isclose(
        limits.policy_blend, 1.0, abs_tol=1.0e-12
    ):
        raise SystemExit(
            "real control requires policy_blend=1.0: MuJoCo READY standing "
            "failed at derated blend"
        )
    actor = OnnxActor(contract)
    rclpy.init()
    node = A3RLDeploy(
        contract,
        limits,
        args.topic_prefix,
        actor,
        args.enable_actuation,
        start_console=True,
        line_console=args.line_console,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().error("Ctrl-C: immediate zero-gain stop")
    finally:
        if not node.clean_stop_completed:
            node.emergency_zero_gain(5)
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.02)
        node.close_console()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
