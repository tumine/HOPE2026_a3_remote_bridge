#!/usr/bin/env python3

"""Interactive 50 Hz A3 deployment for the current HOPE policy and PPMocap."""

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

import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from hope_msgs.msg import RacketCommand as RacketCommandMessage
from joint_msgs.msg import Command, JointCommand
from ppmocap_msgs.msg import MocapFrame
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import UInt32

from a3_mocap_frame import (
    CanonicalTableCalibrator,
    MarkerToPelvis,
)
from a3_rl_contract import (
    ACTION_DIM,
    A3RLContract,
    OnnxActor,
    RacketCommand,
    SwingLifecycle,
    base_forward_xy,
    base_tilt_radians,
)

USE_FIXED_MOCAP = True  # 设为 True 绕过动捕，False 恢复正常

# 固定虚拟位姿（仿真第一帧 pelvis 在 canonical table frame 中的值）
FIXED_BASE_POS = [-0.5, -0.7625, 0.3064]
FIXED_BASE_QUAT = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
FIXED_BASE_FORWARD = [1.0, 0.0]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_SOURCE = REPO_ROOT / "26.7.25发球部署/pingpang_ustc-srf"
DEFAULT_BUNDLE = (
    DEFAULT_TRAINING_SOURCE / "a3_deploy/a3_pingpong_deploy_bundle"
)
DEFAULT_CONFIG = REPO_ROOT / "config/a3_rl_deploy.yaml"
MAX_PENDING_FRAMES = 64
MDU_MAX_KP = 250.0
MDU_MAX_KD = 8.0
HOTKEY_COMMANDS = {
    "p": "pose",
    "d": "policy",
    "c": "recalibrate",
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


def _point(value) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _quaternion_xyzw(value) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z, value.w), dtype=np.float64)


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
    imu_quat_wxyz: np.ndarray
    gyro: np.ndarray
    stamp_ns: int
    received_at: float


@dataclass
class LiveMocap:
    base_position_w: np.ndarray
    base_quat_wxyz: np.ndarray
    sequence: int
    stamp_ns: int
    received_at: float


@dataclass
class CommandMailbox:
    command: RacketCommand
    received_at: float


@dataclass(frozen=True)
class MocapSettings:
    input_topic: str
    planner_input_topic: str
    planner_command_topic: str
    canonical_frame: str
    calibration_samples: int
    table_length_m: float
    table_width_m: float
    surface_down_offset_m: float
    calibration_translation_jitter_m: float
    calibration_rotation_jitter_deg: float
    marker_to_pelvis: MarkerToPelvis
    expected_station_xy: np.ndarray
    sim_reference_base_table_w: np.ndarray
    station_tolerance_m: float
    base_height_range_m: tuple[float, float]
    max_heading_error_rad: float
    require_planner: bool

    @classmethod
    def from_document(cls, document: dict) -> "MocapSettings":
        mocap = document["mocap"]
        table = mocap["table"]
        marker = mocap["robot_marker_to_pelvis"]
        validation = mocap["validation"]
        station = np.asarray(validation["expected_station_xy"], dtype=np.float64)
        reference_base = np.asarray(
            validation["sim_reference_base_table_xyz"], dtype=np.float64
        )
        height = tuple(float(value) for value in validation["base_height_range_m"])
        if station.shape != (2,) or not np.isfinite(station).all():
            raise ValueError("mocap expected_station_xy must contain two finite values")
        if reference_base.shape != (3,) or not np.isfinite(reference_base).all():
            raise ValueError(
                "mocap sim_reference_base_table_xyz must contain three finite values"
            )
        if not np.allclose(reference_base[:2], station, atol=1.0e-12):
            raise ValueError("READY reference base XY must equal the expected station XY")
        if len(height) != 2 or height[0] >= height[1]:
            raise ValueError("mocap base_height_range_m is invalid")
        return cls(
            input_topic=str(mocap["input_topic"]),
            planner_input_topic=str(mocap["planner_input_topic"]),
            planner_command_topic=str(mocap["planner_command_topic"]),
            canonical_frame=str(mocap["canonical_frame"]),
            calibration_samples=int(mocap["calibration_samples"]),
            table_length_m=float(table["length_m"]),
            table_width_m=float(table["width_m"]),
            surface_down_offset_m=float(table["surface_down_offset_m"]),
            calibration_translation_jitter_m=float(
                mocap["calibration_translation_jitter_m"]
            ),
            calibration_rotation_jitter_deg=float(
                mocap["calibration_rotation_jitter_deg"]
            ),
            marker_to_pelvis=MarkerToPelvis.from_values(
                marker["translation_m"], marker["rpy_rad"]
            ),
            expected_station_xy=station,
            sim_reference_base_table_w=reference_base,
            station_tolerance_m=float(validation["station_tolerance_m"]),
            base_height_range_m=height,
            max_heading_error_rad=math.radians(
                float(validation["max_heading_error_deg"])
            ),
            require_planner=bool(validation.get("require_planner", True)),
        )


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
    max_mocap_age_s: float
    max_base_tilt_rad: float
    max_abs_raw_action: float
    max_inference_time_ms: float
    max_pending_acks: int

    @classmethod
    def load(cls, path: Path) -> tuple["RunnerLimits", MocapSettings]:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        safety = document["safety"]
        default_pose = document["default_pose"]
        if not math.isclose(float(document["policy_blend"]), 1.0, abs_tol=1e-12):
            raise ValueError("current policy requires policy_blend=1.0")
        result = cls(
            control_hz=float(document["control_hz"]),
            default_pose_ramp_s=float(default_pose["ramp_s"]),
            default_pose_settle_s=float(default_pose["settle_s"]),
            default_position_error_rad=float(default_pose["max_position_error_rad"]),
            default_velocity_rad_s=float(default_pose["max_velocity_rad_s"]),
            default_kp_scale=float(default_pose["kp_scale"]),
            default_kd_scale=float(default_pose["kd_scale"]),
            policy_warmup_s=float(document["policy_warmup_s"]),
            stop_ready_duration_s=float(document["stop_ready_duration_s"]),
            zero_gain_ticks=int(document["zero_gain_ticks"]),
            max_state_age_s=float(safety["max_state_age_s"]),
            max_mocap_age_s=float(safety["max_mocap_age_s"]),
            max_base_tilt_rad=float(safety["max_base_tilt_rad"]),
            max_abs_raw_action=float(safety["max_abs_raw_action"]),
            max_inference_time_ms=float(safety["max_inference_time_ms"]),
            max_pending_acks=int(safety["max_pending_acks"]),
        )
        if not math.isclose(result.control_hz, 50.0, abs_tol=1e-12):
            raise ValueError("RL runner must run at exactly 50 Hz")
        numeric = (
            result.default_pose_ramp_s, result.default_pose_settle_s,
            result.default_position_error_rad, result.default_velocity_rad_s,
            result.default_kp_scale, result.default_kd_scale,
            result.policy_warmup_s, result.stop_ready_duration_s,
            result.max_state_age_s, result.max_mocap_age_s,
            result.max_base_tilt_rad, result.max_abs_raw_action,
            result.max_inference_time_ms,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("runner safety values must be finite and positive")
        if result.zero_gain_ticks <= 0 or result.max_pending_acks <= 0:
            raise ValueError("runner tick/backlog limits must be positive")
        return result, MocapSettings.from_document(document)


class A3RLDeploy(Node):
    OBSERVE = "OBSERVE"
    DEFAULT_RAMP = "DEFAULT_RAMP"
    DEFAULT_HOLD = "DEFAULT_HOLD"
    POLICY_WARMUP = "POLICY_WARMUP"
    POLICY_ACTIVE = "POLICY_ACTIVE"
    STOP_READY = "STOP_READY"
    ZERO_GAIN = "ZERO_GAIN"
    FAULT = "FAULT"
    DONE = "DONE"

    def __init__(
        self,
        contract: A3RLContract,
        limits: RunnerLimits,
        mocap: MocapSettings,
        topic_prefix: str,
        actor: OnnxActor,
        enable_actuation: bool,
        line_console: bool,
    ) -> None:
        super().__init__("a3_pc_rl_deploy")
        self.contract = contract
        self.limits = limits
        self.mocap = mocap
        self.actor = actor
        self.enable_actuation = bool(enable_actuation)
        self.prefix = topic_prefix.rstrip("/")
        self.mode = self.OBSERVE
        self.done = False
        self.clean_stop_completed = False
        self._default_pose_kp = np.minimum(
            contract.kp * limits.default_kp_scale, MDU_MAX_KP
        )
        self._default_pose_kd = np.minimum(
            contract.kd * limits.default_kd_scale, MDU_MAX_KD
        )

        self._joints: "OrderedDict[int, JointState]" = OrderedDict()
        self._pelvis: "OrderedDict[int, Imu]" = OrderedDict()
        self._latest: LiveFrame | None = None
        self._latest_mocap: LiveMocap | None = None
        self._latest_command: CommandMailbox | None = None
        self._last_matched_stamp = -1
        self._invalid_frames = 0
        self._invalid_mocap = 0
        self._invalid_commands = 0
        self._last_ball_visible: bool | None = None
        self._ball_lost_at: float | None = None
        self._ball_track_cycle = 0
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._console_thread: threading.Thread | None = None

        self._session = "a3_remote_control/rl-%d-%d" % (os.getpid(), time.time_ns())
        self._sequence = 0
        self._acks: set[int] = set()
        self._pose_start_q: np.ndarray | None = None
        self._pose_ready_ticks = 0
        self._pose_ready_announced = False
        self._phase_tick = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._fixed_station_xy: np.ndarray | None = None
        self._zero_ticks_sent = 0
        self._inference_ms: deque[float] = deque(maxlen=1000)
        self._accept_commands_after = float("inf")
        self._ignore_task_id_through = -1
        self._completed_swings = 0
        self._last_reported_task: int | None = None
        self._lifecycle = SwingLifecycle(contract)

        self._calibrator = CanonicalTableCalibrator(
            mocap.calibration_samples,
            mocap.table_length_m,
            mocap.table_width_m,
            mocap.surface_down_offset_m,
            mocap.calibration_translation_jitter_m,
            mocap.calibration_rotation_jitter_deg,
        )

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        planner_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            JointState, self.prefix + "/joint_states", self._on_joints, reliable_qos
        )
        self.create_subscription(
            Imu, self.prefix + "/pelvis_imu", self._on_imu, reliable_qos
        )
        self.create_subscription(
            MocapFrame, mocap.input_topic, self._on_mocap, sensor_qos
        )
        self._poses_publisher = self.create_publisher(
            PoseArray, mocap.planner_input_topic, planner_qos
        )
        self.create_subscription(
            RacketCommandMessage,
            mocap.planner_command_topic,
            self._on_racket_command,
            command_qos,
        )
        self._publisher = None
        if self.enable_actuation:
            self._publisher = self.create_publisher(
                JointCommand, self.prefix + "/joint_command_control", reliable_qos
            )
            self.create_subscription(
                UInt32,
                self.prefix + "/joint_command_control_ack",
                self._on_ack,
                reliable_qos,
            )
        self.create_timer(1.0 / limits.control_hz, self._tick)
        self._console_thread = threading.Thread(
            target=self._line_console_loop if line_console else self._hotkey_loop,
            name="a3-rl-console",
            daemon=True,
        )
        self._console_thread = threading.Thread(
            target=self._line_console_loop if line_console else self._hotkey_loop,
            name="a3-rl-console",
            daemon=True,
        )
        self._console_thread.start()
        self.get_logger().warning(
            "model_21500 runner in OBSERVE: actuation=%s; waiting for synchronized "
            "MDU state + PPMocap table/base calibration; P then D are both required"
            % ("ENABLED" if self.enable_actuation else "disabled")
        )
        self._print_help()

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
            self.get_logger().warning("stdin is not a TTY; using line commands")
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
                command = HOTKEY_COMMANDS.get(key)
                if command is not None:
                    self._commands.put(command)
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
        joints, imu = self._joints.get(stamp), self._pelvis.get(stamp)
        if joints is None or imu is None:
            return
        self._joints.pop(stamp, None)
        self._pelvis.pop(stamp, None)
        try:
            q = reorder(joints.name, joints.position, self.contract.joint_names)
            dq = reorder(joints.name, joints.velocity, self.contract.joint_names)
            quat = np.asarray(
                (imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z),
                dtype=np.float64,
            )
            gyro = np.asarray(
                (imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z),
                dtype=np.float64,
            )
            if not np.isfinite(quat).all() or not np.isfinite(gyro).all():
                raise ValueError("pelvis IMU contains NaN or infinity")
            if float(np.linalg.norm(quat)) < 1.0e-9:
                raise ValueError("pelvis IMU quaternion is zero")
        except ValueError as exc:
            self._invalid_frames += 1
            if self._invalid_frames <= 5:
                self.get_logger().error(str(exc))
            return
        self._latest = LiveFrame(q, dq, quat, gyro, stamp, time.monotonic())
        self._last_matched_stamp = stamp

    def _on_mocap(self, message: MocapFrame) -> None:
        visible = bool(message.ball.visible)
        now = time.monotonic()
        if self._last_ball_visible is None:
            self._last_ball_visible = visible
            if visible:
                self._ball_track_cycle += 1
                self.get_logger().warning(
                    "BALL_TRACK_ACQUIRED cycle=%d initial=yes sequence=%d"
                    % (self._ball_track_cycle, int(message.sequence))
                )
            else:
                self._ball_lost_at = now
        elif visible != self._last_ball_visible:
            self._last_ball_visible = visible
            if visible:
                self._ball_track_cycle += 1
                gap = 0.0 if self._ball_lost_at is None else now - self._ball_lost_at
                self.get_logger().warning(
                    "BALL_TRACK_ACQUIRED cycle=%d gap=%.3fs sequence=%d"
                    % (self._ball_track_cycle, gap, int(message.sequence))
                )
                self._ball_lost_at = None
            else:
                self._ball_lost_at = now
                self.get_logger().warning(
                    "BALL_TRACK_LOST cycle=%d sequence=%d"
                    % (self._ball_track_cycle, int(message.sequence))
                )
        try:
            table_position = _point(message.table.pose.position)
            table_quaternion = _quaternion_xyzw(message.table.pose.orientation)
            robot_position = _point(message.robot.pose.position)
            robot_quaternion = _quaternion_xyzw(message.robot.pose.orientation)
            vectors = (table_position, table_quaternion, robot_position, robot_quaternion)
            if not all(np.isfinite(value).all() for value in vectors):
                raise ValueError("PPMocap table/robot pose contains NaN or infinity")
            if self._calibrator.frame is None:
                if not message.table.tracked or not message.robot.tracked:
                    return
                frame = self._calibrator.add(
                    table_position,
                    table_quaternion,
                    robot_position,
                    bool(message.table.long_axis_is_x),
                )
                if frame is not None:
                    self.get_logger().warning(
                        "PPMOCAP CALIBRATED: canonical origin=near-left tabletop; "
                        "+X=robot-to-opponent +Y=robot-left +Z=up"
                    )
            frame = self._calibrator.frame
            if frame is None:
                return
            if message.robot.tracked:
                base_position, base_quat = frame.pelvis_from_marker(
                    robot_position, robot_quaternion, self.mocap.marker_to_pelvis
                )
                self._latest_mocap = LiveMocap(
                    base_position,
                    base_quat,
                    int(message.sequence),
                    stamp_ns(message),
                    time.monotonic(),
                )
            if message.ball.visible:
                ball_position = frame.point_from_world(_point(message.ball.position))
                output = PoseArray()
                output.header = message.header
                output.header.frame_id = self.mocap.canonical_frame
                pose = Pose()
                pose.position.x = float(ball_position[0])
                pose.position.y = float(ball_position[1])
                pose.position.z = float(ball_position[2])
                pose.orientation.w = 1.0
                output.poses.append(pose)
                self._poses_publisher.publish(output)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._invalid_mocap += 1
            if self._invalid_mocap <= 5:
                self.get_logger().error(f"invalid PPMocap frame: {exc}")

    def _publish_fixed_mocap(self) -> None:
        """模拟固定虚拟位姿，替代真实的 PPMocap 数据"""
        if self._calibrator.frame is None:
            # 手动创建一个固定的 canonical frame
            # 使用简单的单位变换：桌面原点在 (0,0,0)，无旋转
            from a3_mocap_frame import CanonicalFrame
            self._calibrator._frame = CanonicalFrame(
                origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                x_axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            self.get_logger().warning("FIXED CANONICAL FRAME CREATED (no real PPMocap)")

        self._latest_mocap = LiveMocap(
            base_position_w=np.array(FIXED_BASE_POS, dtype=np.float64),
            base_quat_wxyz=np.array(FIXED_BASE_QUAT, dtype=np.float64),
            sequence=self._mocap_seq_counter,
            stamp_ns=self.get_clock().now().nanoseconds,
            received_at=time.monotonic(),
        )
        self._mocap_seq_counter += 1

        # 发布空的球位姿（没有球，策略保持 READY）
        output = PoseArray()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.mocap.canonical_frame
        self._poses_publisher.publish(output)

    def _on_racket_command(self, message: RacketCommandMessage) -> None:
        try:
            if message.header.frame_id != self.mocap.canonical_frame:
                raise ValueError(
                    f"planner frame {message.header.frame_id!r} != "
                    f"{self.mocap.canonical_frame!r}"
                )
            stamp = stamp_ns(message)
            now_ros_ns = self.get_clock().now().nanoseconds
            transport_age = (
                max(0.0, (now_ros_ns - stamp) * 1.0e-9)
                if stamp > 0 and now_ros_ns >= stamp
                else 0.0
            )
            command = RacketCommand(
                task_id=int(message.task_id),
                task_revision=int(message.task_revision),
                swing_side=int(message.swing_side),
                position_w=_point(message.position),
                velocity_w=_point(message.velocity),
                time_to_strike=float(message.time_to_strike) - transport_age,
            )
            self._latest_command = CommandMailbox(command, time.monotonic())
        except (TypeError, ValueError, OverflowError) as exc:
            self._invalid_commands += 1
            if self._invalid_commands <= 5:
                self.get_logger().error(f"invalid RacketCommand: {exc}")

    def _on_ack(self, message: UInt32) -> None:
        sequence = int(message.data)
        if sequence < self._sequence:
            self._acks.add(sequence)

    def _poll_command(self) -> RacketCommand | None:
        mailbox = self._latest_command
        if mailbox is None or mailbox.received_at < self._accept_commands_after:
            return None
        if mailbox.command.task_id <= self._ignore_task_id_through:
            return None
        age = max(0.0, time.monotonic() - mailbox.received_at)
        command = mailbox.command
        return RacketCommand(
            command.task_id,
            command.task_revision,
            command.swing_side,
            command.position_w,
            command.velocity_w,
            command.time_to_strike - age,
        )

    def _tick(self) -> None:
        self._drain_operator_commands()
        if self.done:
            return
        if self.mode == self.OBSERVE:
            return
        if self.mode in (self.ZERO_GAIN, self.FAULT):
            self._tick_zero_gain()
            return
        require_mocap = self.mode in (
            self.POLICY_WARMUP, self.POLICY_ACTIVE, self.STOP_READY
        )
        if not self._state_is_safe(require_mocap=require_mocap):
            return
        if self.mode == self.DEFAULT_RAMP:
            self._tick_default_ramp()
        elif self.mode == self.DEFAULT_HOLD:
            self._tick_default_hold()
        elif self.mode == self.POLICY_WARMUP:
            self._tick_policy(warmup=True)
        elif self.mode == self.POLICY_ACTIVE:
            self._tick_policy(warmup=False)
        elif self.mode == self.STOP_READY:
            self._tick_stop_ready()

    def _drain_operator_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait().strip().lower()
            except queue.Empty:
                return
            if command in ("help", "?"):
                self._print_help()
            elif command == "status":
                self._report_status()
            elif command in ("pose", "initial", "default"):
                self._begin_pose()
            elif command in ("policy", "ready"):
                self._begin_policy()
            elif command == "recalibrate":
                self._begin_recalibrate()
            elif command == "stop":
                self._begin_stop()
            elif command in ("halt", "estop"):
                self._begin_halt("operator immediate halt")
            elif command in ("quit", "exit") and self.mode == self.OBSERVE:
                self.done = True
                self.clean_stop_completed = True
            else:
                self.get_logger().error(f"unknown/invalid command: {command}")

    def _begin_recalibrate(self) -> None:
        if self.mode != self.OBSERVE:
            self.get_logger().error("C is allowed only in OBSERVE")
            return
        self._calibrator.reset()
        self._latest_mocap = None
        self._latest_command = None
        self.get_logger().warning("PPMocap table calibration cleared")

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
        self._phase_tick = 0
        self._pose_ready_ticks = 0
        self._pose_ready_announced = False
        self.mode = self.DEFAULT_RAMP
        self.get_logger().warning(
            "DEFAULT POSE START: %.1fs ramp; D remains locked until pose and "
            "PPMocap/planner gates are ready" % self.limits.default_pose_ramp_s
        )

    def _mocap_gate_error(self) -> str | None:
        if USE_FIXED_MOCAP:
            return None  # 固定虚拟位姿模式下直接通过
        live = self._latest_mocap
        if self._calibrator.frame is None:
            return f"table calibration {self._calibrator.samples}/{self.mocap.calibration_samples}"
        if live is None:
            return "no tracked BotA3 pose"
        age = time.monotonic() - live.received_at
        if age > self.limits.max_mocap_age_s:
            return f"BotA3 pose stale ({age * 1000.0:.1f}ms)"
        station_error = float(np.linalg.norm(live.base_position_w[:2] - self.mocap.expected_station_xy))
        if station_error > self.mocap.station_tolerance_m:
            return (
                f"station mismatch {station_error:.3f}m; base_xy="
                f"{np.array2string(live.base_position_w[:2], precision=3)} expected="
                f"{np.array2string(self.mocap.expected_station_xy, precision=3)}"
            )
        low, high = self.mocap.base_height_range_m
        if not low <= float(live.base_position_w[2]) <= high:
            return f"base height {live.base_position_w[2]:.3f}m outside [{low:.3f},{high:.3f}]"
        forward = base_forward_xy(live.base_quat_wxyz)
        heading_error = math.atan2(abs(float(forward[1])), float(forward[0]))
        if heading_error > self.mocap.max_heading_error_rad:
            return (
                f"BotA3 marker heading error {math.degrees(heading_error):.1f}deg; "
                "calibrate robot_marker_to_pelvis.rpy_rad"
            )
        if self.mocap.require_planner:
            if self._poses_publisher.get_subscription_count() < 1:
                return "HOPE planner is not subscribed to the canonical ball topic"
            if self.count_publishers(self.mocap.planner_command_topic) < 1:
                return "HOPE planner command publisher is absent"
        return None

    def _begin_policy(self) -> None:
        if self.mode in (self.POLICY_WARMUP, self.POLICY_ACTIVE):
            self.get_logger().info("policy is already active")
            return
        if self.mode != self.DEFAULT_HOLD or not self._pose_is_ready():
            self.get_logger().error("D requires DEFAULT_HOLD pose_ready=yes; press P first")
            return
        gate_error = self._mocap_gate_error()
        if gate_error is not None:
            self.get_logger().error(f"D rejected: PPMocap/planner gate: {gate_error}")
            return
        assert self._latest_mocap is not None
        self._fixed_station_xy = self._latest_mocap.base_position_w[:2].copy()
        self._last_action.fill(0.0)
        self._lifecycle = SwingLifecycle(
            self.contract, self.mocap.sim_reference_base_table_w
        )
        self._phase_tick = 0
        self._completed_swings = 0
        self._last_reported_task = None
        self._accept_commands_after = float("inf")
        self._ignore_task_id_through = (
            -1 if self._latest_command is None else self._latest_command.command.task_id
        )
        self.mode = self.POLICY_WARMUP
        self.get_logger().warning(
            "POLICY ARMED BY D: %.1fs READY warmup; live planner commands remain "
            "blocked until warmup completes; base pose/yaw now come from PPMocap"
            % self.limits.policy_warmup_s
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
            self.get_logger().warning("STOP before policy: sending zero-gain frames")
            return
        self._phase_tick = 0
        self._accept_commands_after = float("inf")
        self.mode = self.STOP_READY
        self.get_logger().warning(
            "STOP: planner commands blocked; holding validated READY for %.1fs"
            % self.limits.stop_ready_duration_s
        )

    def _begin_halt(self, reason: str) -> None:
        if self.mode in (self.FAULT, self.ZERO_GAIN, self.DONE):
            return
        self._zero_ticks_sent = 0
        self.mode = self.FAULT
        self.get_logger().error(f"HALT: {reason}")

    def _run_policy_tick(self, allow_command: bool, force_ready: bool = False) -> bool:
        assert self._latest is not None and self._latest_mocap is not None
        assert self._fixed_station_xy is not None
        live = self._latest_mocap
        previous_task = self._lifecycle.active_task_id
        try:
            if force_ready:
                target = self._lifecycle.ready_target(live.base_position_w)
            else:
                target = self._lifecycle.update(
                    self._poll_command() if allow_command else None,
                    live.base_position_w,
                )
            observation = self.contract.build_observation(
                self._latest.q,
                self._latest.dq,
                live.base_position_w,
                self._latest.imu_quat_wxyz,
                live.base_quat_wxyz,
                self._latest.gyro,
                self._last_action,
                self._fixed_station_xy,
                target,
            )
            started = time.perf_counter()
            raw_action = self.actor.infer(observation)
            inference_ms = (time.perf_counter() - started) * 1000.0
            self._inference_ms.append(inference_ms)
            if inference_ms > self.limits.max_inference_time_ms:
                raise RuntimeError(
                    f"ONNX inference {inference_ms:.3f}ms exceeds "
                    f"{self.limits.max_inference_time_ms:.3f}ms"
                )
            max_raw = float(np.max(np.abs(raw_action)))
            if max_raw > self.limits.max_abs_raw_action:
                index = int(np.argmax(np.abs(raw_action)))
                raise ValueError(
                    f"raw action {max_raw:.3f} exceeds numeric fault limit "
                    f"{self.limits.max_abs_raw_action:.3f} at "
                    f"{self.contract.joint_names[index]}"
                )
            applied, desired = self.contract.decode_action(raw_action)
            self._send_position(desired, self.contract.kp, self.contract.kd)
            self._last_action = applied
            if not force_ready:
                self._lifecycle.advance()
            active_task = self._lifecycle.active_task_id
            if active_task is not None and active_task != self._last_reported_task:
                self._last_reported_task = active_task
                side = "forehand" if self._lifecycle.swing_side > 0 else "backhand"
                self.get_logger().warning(
                    f"LIVE BALL ENGAGED: task={active_task} side={side} "
                    f"tts={target.time_to_strike:.3f}s"
                )
            if previous_task is not None and active_task is None:
                self._completed_swings += 1
                self.get_logger().warning(
                    f"BALL COMPLETE: task={previous_task} total={self._completed_swings}; READY restored"
                )
            return True
        except Exception as exc:
            self._begin_halt(f"policy tick failed: {exc}")
            return False

    @staticmethod
    def _smoothstep(value: float) -> float:
        clipped = min(max(float(value), 0.0), 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    def _tick_default_ramp(self) -> None:
        assert self._latest is not None and self._pose_start_q is not None
        total = max(1, int(round(self.limits.default_pose_ramp_s * 50.0)))
        alpha = self._smoothstep((self._phase_tick + 1) / total)
        desired = (1.0 - alpha) * self._pose_start_q + alpha * self.contract.default_q
        self._send_position(desired, self._default_pose_kp, self._default_pose_kd)
        self._phase_tick += 1
        if self._phase_tick >= total:
            self.mode = self.DEFAULT_HOLD
            self._phase_tick = 0
            self._pose_ready_ticks = 0
            self.get_logger().warning("DEFAULT_HOLD: waiting for pose settling")

    def _pose_in_tolerance(self) -> bool:
        if self._latest is None:
            return False
        return (
            float(np.max(np.abs(self._latest.q - self.contract.default_q)))
            <= self.limits.default_position_error_rad
            and float(np.max(np.abs(self._latest.dq)))
            <= self.limits.default_velocity_rad_s
        )

    def _pose_is_ready(self) -> bool:
        required = max(1, int(round(self.limits.default_pose_settle_s * 50.0)))
        return self.mode == self.DEFAULT_HOLD and self._pose_ready_ticks >= required

    def _tick_default_hold(self) -> None:
        self._send_position(
            self.contract.default_q, self._default_pose_kp, self._default_pose_kd
        )
        if self._pose_in_tolerance():
            self._pose_ready_ticks += 1
        else:
            self._pose_ready_ticks = 0
            self._pose_ready_announced = False
        if self._pose_is_ready() and not self._pose_ready_announced:
            self._pose_ready_announced = True
            gate = self._mocap_gate_error()
            suffix = "mocap/planner_ready=yes" if gate is None else f"D still locked: {gate}"
            self.get_logger().warning(f"DEFAULT POSE READY: pose_ready=yes; {suffix}")

    def _tick_policy(self, warmup: bool) -> None:
        if not self._run_policy_tick(allow_command=not warmup):
            return
        if warmup:
            self._phase_tick += 1
            total = max(1, int(round(self.limits.policy_warmup_s * 50.0)))
            if self._phase_tick >= total:
                self.mode = self.POLICY_ACTIVE
                self._phase_tick = 0
                # Ignore every command received before this point. The next live
                # NEW task is the first one allowed to engage. A revision of a
                # task that started during warmup must not join mid-swing.
                if self._latest_command is not None:
                    self._ignore_task_id_through = max(
                        self._ignore_task_id_through,
                        self._latest_command.command.task_id,
                    )
                self._accept_commands_after = time.monotonic()
                self.get_logger().warning(
                    "POLICY_ACTIVE: validated READY is running; live /racket/command enabled"
                )

    def _tick_stop_ready(self) -> None:
        total = max(1, int(round(self.limits.stop_ready_duration_s * 50.0)))
        if self._phase_tick >= total:
            self._zero_ticks_sent = 0
            self.mode = self.ZERO_GAIN
            return
        if self._run_policy_tick(allow_command=False, force_ready=True):
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
            self.get_logger().warning("zero-gain stop complete")

    def _state_is_safe(self, require_mocap: bool) -> bool:
        if self._latest is None:
            self._begin_halt("no synchronized joint/IMU state")
            return False
        if self._state_age() > self.limits.max_state_age_s:
            self._begin_halt(f"joint/IMU state stale: {self._state_age() * 1000.0:.1f}ms")
            return False
        if require_mocap:
            gate = self._mocap_gate_error()
            if gate is not None:
                self._begin_halt(f"PPMocap/planner gate lost: {gate}")
                return False
        try:
            # Falling is a fast body-attitude event. Always use the pelvis IMU;
            # PPMocap remains an independent position/heading availability gate.
            tilt = base_tilt_radians(self._latest.imu_quat_wxyz)
        except ValueError as exc:
            self._begin_halt(str(exc))
            return False
        if tilt > self.limits.max_base_tilt_rad:
            self._begin_halt(
                f"base tilt {math.degrees(tilt):.2f}deg exceeds "
                f"{math.degrees(self.limits.max_base_tilt_rad):.2f}deg"
            )
            return False
        if self.enable_actuation:
            pending = self._sequence - len(self._acks)
            if pending > self.limits.max_pending_acks:
                self._begin_halt(
                    f"command ACK backlog {pending} exceeds {self.limits.max_pending_acks}"
                )
                return False
        return True

    def _state_age(self) -> float:
        return float("inf") if self._latest is None else time.monotonic() - self._latest.received_at

    def _send_position(self, desired: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        desired = np.clip(np.asarray(desired, dtype=np.float64), self.contract.lower, self.contract.upper)
        self._publish_command(desired, kp, kd)

    def _publish_command(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
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
        self.done = True
        if self._console_thread is not None and self._console_thread is not threading.current_thread():
            self._console_thread.join(timeout=0.5)

    def _report_status(self) -> None:
        state_age_ms = self._state_age() * 1000.0
        mocap_age_ms = (
            float("inf")
            if self._latest_mocap is None
            else (time.monotonic() - self._latest_mocap.received_at) * 1000.0
        )
        base = np.full(3, np.nan) if self._latest_mocap is None else self._latest_mocap.base_position_w
        forward = (
            np.full(2, np.nan)
            if self._latest_mocap is None
            else base_forward_xy(self._latest_mocap.base_quat_wxyz)
        )
        p95 = float(np.percentile(self._inference_ms, 95)) if self._inference_ms else 0.0
        gate = self._mocap_gate_error()
        self.get_logger().info(
            "mode=%s lifecycle=%s state_age=%.1fms mocap_age=%.1fms "
            "base=%s forward=%s pose_ready=%s gate=%s task=%s swings=%d "
            "sent=%d acked=%d pending=%d infer_p95=%.3fms invalid=%d/%d/%d"
            % (
                self.mode,
                self._lifecycle.phase,
                state_age_ms,
                mocap_age_ms,
                np.array2string(base, precision=3),
                np.array2string(forward, precision=3),
                "yes" if self._pose_is_ready() else "no",
                "ready" if gate is None else gate,
                self._lifecycle.active_task_id,
                self._completed_swings,
                self._sequence,
                len(self._acks),
                self._sequence - len(self._acks) if self.enable_actuation else 0,
                p95,
                self._invalid_frames,
                self._invalid_mocap,
                self._invalid_commands,
            )
        )

    def _print_help(self) -> None:
        self.get_logger().info(
            "HOTKEYS (no Enter): P=default pose; D=enter live policy after all gates; "
            "I=status; C=recalibrate table (OBSERVE only); S=normal stop; "
            "X=immediate halt; H=help; Q=quit (OBSERVE only). "
            "There are no F/B synthetic-ball keys: live planner commands select each swing."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A3 model_21500 PPMocap + HOPE planner deployment runner"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--training-source", type=Path, default=DEFAULT_TRAINING_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topic-prefix", default="/a3_internal")
    parser.add_argument("--enable-actuation", action="store_true")
    parser.add_argument("--line-console", action="store_true")
    parser.add_argument("--i-understand-this-can-move-the-robot", action="store_true")
    parser.add_argument("--allow-full-policy", action="store_true")
    args = parser.parse_args()
    if not args.topic_prefix.startswith("/") or args.topic_prefix == "/":
        parser.error("topic-prefix must be an absolute non-root ROS prefix")
    if args.enable_actuation and not args.i_understand_this_can_move_the_robot:
        parser.error("real control requires --i-understand-this-can-move-the-robot")
    if not args.allow_full_policy:
        parser.error("current deployment requires --allow-full-policy")
    return args


def main() -> int:
    args = parse_args()
    contract = A3RLContract.load(args.bundle, args.training_source)
    limits, mocap = RunnerLimits.load(args.config)
    actor = OnnxActor(contract)
    rclpy.init()
    node = A3RLDeploy(
        contract, limits, mocap, args.topic_prefix, actor,
        args.enable_actuation, args.line_console
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
