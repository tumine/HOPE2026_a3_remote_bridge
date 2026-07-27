"""Convert a retargeted motion CSV into the HOPE Agibot-A3 motion format.

The GMR/BeyondMimic CSV layout used by this project is::

    root_xyz (3), root_quat_xyzw (4), Unitree-G1 joint_pos (29)

The HOPE training task instead consumes 31 A3 joints in its canonical
policy/deployment order and 14 tracked A3 link states.  This converter maps the
29 common joints by *name*, fills the two passive A3 head joints from the shared
default pose, and runs A3 forward kinematics in Isaac Lab to produce the body
arrays.

Example (the input FPS must match the source data)::

    python scripts/csv_to_npz.py \
        --input_file ../../motions/preprocessed/ours_forehand.csv \
        --input_fps 50 \
        --output_file ../../motions/preprocessed/ours_forehand.npz \
        --output_fps 50 \
        --swing_side forehand \
        --headless

By default the first root XY is moved to the origin and the initial heading is
rotated to +X.  Relative root motion, root height, roll, and pitch are retained.
Use ``--no-canonicalize_root`` only when the CSV is already expressed in the
HOPE training world frame.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Convert a GMR G1-29 or canonical A3-31 CSV to a local HOPE A3 NPZ."
)
parser.add_argument("--input_file", "--input-file", required=True, type=Path)
parser.add_argument(
    "--input_fps",
    "--input-fps",
    required=True,
    type=float,
    help="Sampling rate of the CSV. CSV files do not contain this metadata.",
)
parser.add_argument("--output_file", "--output-file", required=True, type=Path)
parser.add_argument(
    "--output_fps",
    "--output-fps",
    default=50.0,
    type=float,
    help="Sampling rate stored in the NPZ (HOPE control rate defaults to 50 Hz).",
)
parser.add_argument(
    "--frame_range",
    "--frame-range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional 1-based inclusive CSV frame range.",
)
parser.add_argument(
    "--source_layout",
    "--source-layout",
    choices=("auto", "g1_29", "a3_31"),
    default="auto",
    help="Joint-column layout. auto selects g1_29 for 36 columns and a3_31 for 38 columns.",
)
parser.add_argument(
    "--swing_side",
    "--swing-side",
    choices=("auto", "forehand", "backhand"),
    default="auto",
    help="YAML swing metadata; auto infers it from the output filename.",
)
parser.add_argument("--motion_name", "--motion-name", default=None)
parser.add_argument(
    "--canonicalize_root",
    "--canonicalize-root",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Move initial root XY to zero and align its initial heading with world +X.",
)
parser.add_argument(
    "--root_height_offset",
    "--root-height-offset",
    default=0.0,
    type=float,
    help="Extra clearance added after automatic A3 ground alignment (meters).",
)
parser.add_argument(
    "--align_a3_ground",
    "--align-a3-ground",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Shift root z so the motion's lowest tracked ankle-link height matches the A3 "
        "default standing pose. This compensates for G1/A3 leg-length differences."
    ),
)
parser.add_argument(
    "--clip_joint_limits",
    "--clip-joint-limits",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Clip mapped positions to the shared A3 clamps. Default is to fail on a violation.",
)
parser.add_argument(
    "--strike_frame",
    "--strike-frame",
    default=None,
    type=int,
    help="Optional zero-based output strike frame. Default: peak smoothed right-wrist speed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.input_fps <= 0.0 or args_cli.output_fps <= 0.0:
    parser.error("--input_fps and --output_fps must be positive")
if args_cli.frame_range is not None:
    start, end = args_cli.frame_range
    if start < 1 or end < start:
        parser.error("--frame_range must be 1-based and satisfy 1 <= START <= END")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac/torch imports must happen after AppLauncher has started Isaac Sim.
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from whole_body_tracking.robots.agibot_a3 import (  # noqa: E402
    A3_ANCHOR_BODY,
    A3_MOUNT_OFFSET,
    A3_RACKET_BODY,
    A3_ROOT_BODY,
    A3_TRACKED_BODIES,
    A3_WRIST_BODY,
    AGIBOT_A3_CFG,
    AGIBOT_A3_JOINT_NAMES,
)
from whole_body_tracking.utils.action_adapter_config import (  # noqa: E402
    load_action_adapter_config,
)


# GMR's Unitree G1 29-DOF CSV order.  All names also exist on A3; A3 adds
# head_yaw_joint and head_pitch_joint.
G1_29_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Minimal A3 scene used only for forward kinematics."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2000.0),
    )
    robot: ArticulationCfg = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if torch.any(norm < 1.0e-8):
        raise ValueError("CSV contains a zero-length root quaternion")
    return quat / norm


def _slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Vectorized shortest-path quaternion interpolation (wxyz)."""

    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)
    blend = blend.unsqueeze(-1)

    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    safe = sin_angle.abs() > 1.0e-6
    s0 = torch.sin((1.0 - blend) * angle) / torch.where(safe, sin_angle, torch.ones_like(sin_angle))
    s1 = torch.sin(blend * angle) / torch.where(safe, sin_angle, torch.ones_like(sin_angle))
    spherical = s0 * q0 + s1 * q1
    linear = (1.0 - blend) * q0 + blend * q1
    return _normalize_quaternion(torch.where(safe, spherical, linear))


def _gradient(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] == 1:
        return torch.zeros_like(values)
    return torch.gradient(values, spacing=dt, dim=0)[0]


def _relative_angular_velocity(q_from: torch.Tensor, q_to: torch.Tensor, duration: float) -> torch.Tensor:
    relative = quat_mul(q_to, quat_conjugate(q_from))
    # q and -q represent the same rotation.  Use the short relative arc.
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)
    return axis_angle_from_quat(_normalize_quaternion(relative)) / duration


def _angular_velocity(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """Finite-difference world angular velocity for [T, ..., 4] wxyz quaternions."""

    quat = _normalize_quaternion(quat)
    result = torch.zeros((*quat.shape[:-1], 3), dtype=quat.dtype, device=quat.device)
    if quat.shape[0] == 1:
        return result
    result[0] = _relative_angular_velocity(quat[0], quat[1], dt)
    result[-1] = _relative_angular_velocity(quat[-2], quat[-1], dt)
    if quat.shape[0] > 2:
        result[1:-1] = _relative_angular_velocity(quat[:-2], quat[2:], 2.0 * dt)
    return result


def _initial_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MotionData:
    """Load, map, align, resample, and differentiate one CSV."""

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.source_layout = ""
        self.initial_yaw_rad = 0.0
        self.applied_root_z_shift_m = 0.0
        self.input_frames = 0
        self._load_and_map()
        self._align_root()
        self._resample()
        self._validate_joint_limits()
        self.root_lin_vel = _gradient(self.root_pos, self.output_dt)
        self.root_ang_vel = _angular_velocity(self.root_quat, self.output_dt)
        self.joint_vel = _gradient(self.joint_pos, self.output_dt)

    def _load_and_map(self) -> None:
        input_path = args_cli.input_file.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input CSV not found: {input_path}")
        skiprows = 0
        max_rows = None
        if args_cli.frame_range is not None:
            skiprows = args_cli.frame_range[0] - 1
            max_rows = args_cli.frame_range[1] - args_cli.frame_range[0] + 1
        array = np.loadtxt(input_path, delimiter=",", skiprows=skiprows, max_rows=max_rows)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[0] < 2:
            raise ValueError("at least two CSV frames are required")
        if not np.isfinite(array).all():
            raise ValueError("CSV contains NaN or infinite values")

        if args_cli.source_layout == "auto":
            if array.shape[1] == 36:
                self.source_layout = "g1_29"
            elif array.shape[1] == 38:
                self.source_layout = "a3_31"
            else:
                raise ValueError(
                    f"cannot infer source layout from {array.shape[1]} columns; expected 36 or 38"
                )
        else:
            self.source_layout = args_cli.source_layout
        expected_columns = 36 if self.source_layout == "g1_29" else 38
        if array.shape[1] != expected_columns:
            raise ValueError(
                f"{self.source_layout} requires {expected_columns} columns, got {array.shape[1]}"
            )

        source = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        self.input_frames = int(source.shape[0])
        self.root_pos_input = source[:, :3]
        # CSV uses xyzw; Isaac and the NPZ contract use wxyz.
        self.root_quat_input = _normalize_quaternion(source[:, [6, 3, 4, 5]])

        source_names = G1_29_JOINT_NAMES if self.source_layout == "g1_29" else AGIBOT_A3_JOINT_NAMES
        source_joint_pos = source[:, 7:]
        adapter = load_action_adapter_config()
        mapped = torch.as_tensor(adapter.default_q, dtype=torch.float32, device=self.device).repeat(
            self.input_frames, 1
        )
        target_index = {name: index for index, name in enumerate(AGIBOT_A3_JOINT_NAMES)}
        for source_index, name in enumerate(source_names):
            mapped[:, target_index[name]] = source_joint_pos[:, source_index]
        self.joint_pos_input = mapped

        missing = [name for name in AGIBOT_A3_JOINT_NAMES if name not in source_names]
        print(
            f"[csv_to_npz] loaded {input_path}: frames={self.input_frames}, "
            f"columns={array.shape[1]}, layout={self.source_layout}, input_fps={args_cli.input_fps:g}",
            flush=True,
        )
        if missing:
            print(
                f"[csv_to_npz] filled missing joints from shared default_q: {missing}",
                flush=True,
            )

    def _align_root(self) -> None:
        self.initial_yaw_rad = float(_initial_yaw(self.root_quat_input[0]).item())
        if not args_cli.canonicalize_root:
            aligned_pos = self.root_pos_input.clone()
            aligned_quat = self.root_quat_input.clone()
        else:
            half_yaw = -0.5 * self.initial_yaw_rad
            align_quat = torch.tensor(
                [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
                dtype=torch.float32,
                device=self.device,
            )
            origin = self.root_pos_input[0].clone()
            origin[2] = 0.0
            aligned_pos = quat_apply(align_quat.repeat(self.input_frames, 1), self.root_pos_input - origin)
            aligned_quat = quat_mul(align_quat.repeat(self.input_frames, 1), self.root_quat_input)
        self.root_pos_input = aligned_pos
        self.root_quat_input = _normalize_quaternion(aligned_quat)
        print(
            f"[csv_to_npz] root alignment: canonicalize={args_cli.canonicalize_root}, "
            f"initial_yaw={math.degrees(self.initial_yaw_rad):.3f} deg, "
            f"pre_ground_initial_xyz={self.root_pos_input[0].tolist()}",
            flush=True,
        )

    def _resample(self) -> None:
        self.output_dt = 1.0 / float(args_cli.output_fps)
        if math.isclose(args_cli.input_fps, args_cli.output_fps, rel_tol=0.0, abs_tol=1.0e-9):
            self.root_pos = self.root_pos_input.clone()
            self.root_quat = self.root_quat_input.clone()
            self.joint_pos = self.joint_pos_input.clone()
        else:
            duration = (self.input_frames - 1) / float(args_cli.input_fps)
            output_frames = int(math.floor(duration * args_cli.output_fps + 1.0e-7)) + 1
            times = torch.arange(output_frames, dtype=torch.float32, device=self.device) * self.output_dt
            source_frame = times * float(args_cli.input_fps)
            index_0 = torch.floor(source_frame).long().clamp(max=self.input_frames - 1)
            index_1 = (index_0 + 1).clamp(max=self.input_frames - 1)
            blend = source_frame - index_0.to(source_frame.dtype)
            self.root_pos = torch.lerp(
                self.root_pos_input[index_0], self.root_pos_input[index_1], blend[:, None]
            )
            self.root_quat = _slerp(
                self.root_quat_input[index_0], self.root_quat_input[index_1], blend
            )
            self.joint_pos = torch.lerp(
                self.joint_pos_input[index_0], self.joint_pos_input[index_1], blend[:, None]
            )
        print(
            f"[csv_to_npz] timeline: output_frames={self.root_pos.shape[0]}, "
            f"output_fps={args_cli.output_fps:g}, "
            f"duration={(self.root_pos.shape[0] - 1) / args_cli.output_fps:.3f}s",
            flush=True,
        )

    def _validate_joint_limits(self) -> None:
        adapter = load_action_adapter_config()
        lower = torch.as_tensor(adapter.clamp_lower, dtype=torch.float32, device=self.device)
        upper = torch.as_tensor(adapter.clamp_upper, dtype=torch.float32, device=self.device)
        below = self.joint_pos < lower
        above = self.joint_pos > upper
        violations = below | above
        if not torch.any(violations):
            print("[csv_to_npz] all mapped joint positions are inside the shared A3 clamps", flush=True)
            return
        bad_joint_ids = torch.where(torch.any(violations, dim=0))[0].detach().cpu().tolist()
        details = ", ".join(
            f"{AGIBOT_A3_JOINT_NAMES[index]}="
            f"[{self.joint_pos[:, index].min().item():.3f},{self.joint_pos[:, index].max().item():.3f}] "
            f"limit=[{lower[index].item():.3f},{upper[index].item():.3f}]"
            for index in bad_joint_ids
        )
        if not args_cli.clip_joint_limits:
            raise ValueError(
                "mapped motion violates the shared A3 joint clamps; inspect the mapping or pass "
                f"--clip_joint_limits explicitly. Violations: {details}"
            )
        self.joint_pos = torch.clamp(self.joint_pos, lower, upper)
        print(f"[csv_to_npz] WARNING: clipped joint limit violations: {details}", flush=True)


def _run_forward_kinematics(
    sim: SimulationContext, scene: InteractiveScene, motion: MotionData
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = scene["robot"]
    joint_ids, resolved_joint_names = robot.find_joints(AGIBOT_A3_JOINT_NAMES, preserve_order=True)
    if list(resolved_joint_names) != list(AGIBOT_A3_JOINT_NAMES):
        raise RuntimeError(
            "could not resolve canonical A3 joint order exactly; "
            f"resolved={list(resolved_joint_names)}"
        )
    body_ids, resolved_body_names = robot.find_bodies(A3_TRACKED_BODIES, preserve_order=True)
    if list(resolved_body_names) != list(A3_TRACKED_BODIES):
        raise RuntimeError(
            "could not resolve tracked A3 body order exactly; "
            f"resolved={list(resolved_body_names)}"
        )
    print(f"[csv_to_npz] canonical joint -> PhysX ids: {list(joint_ids)}", flush=True)
    print(f"[csv_to_npz] tracked body -> PhysX ids: {list(body_ids)}", flush=True)

    ankle_motion_indices = [
        A3_TRACKED_BODIES.index("left_ankle_roll_Link"),
        A3_TRACKED_BODIES.index("right_ankle_roll_Link"),
    ]
    # The configured A3 default pose is the training/reset standing reference.  Match its
    # ankle-link ground clearance instead of reusing the G1 pelvis height from the CSV.
    sim.forward()
    scene.update(0.0)
    default_ankle_z = robot.data.body_pos_w[0, [body_ids[index] for index in ankle_motion_indices], 2]
    default_lowest_ankle_z = float(torch.min(default_ankle_z).item())

    body_pos_frames: list[torch.Tensor] = []
    body_quat_frames: list[torch.Tensor] = []
    total = int(motion.joint_pos.shape[0])
    progress_interval = max(1, total // 10)
    root_state = robot.data.default_root_state.clone()

    for frame in range(total):
        root_state[:, :3] = motion.root_pos[frame]
        root_state[:, 3:7] = motion.root_quat[frame]
        root_state[:, 7:10] = motion.root_lin_vel[frame]
        root_state[:, 10:13] = motion.root_ang_vel[frame]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
            motion.joint_pos[frame : frame + 1],
            motion.joint_vel[frame : frame + 1],
            joint_ids=joint_ids,
        )
        # Recompute articulation kinematics without advancing physics.
        sim.forward()
        scene.update(0.0)
        body_pos_frames.append(robot.data.body_pos_w[0, body_ids].clone())
        body_quat_frames.append(robot.data.body_quat_w[0, body_ids].clone())
        if frame == 0 or frame + 1 == total or (frame + 1) % progress_interval == 0:
            print(f"[csv_to_npz] A3 FK {frame + 1}/{total}", flush=True)

    body_pos = torch.stack(body_pos_frames, dim=0)
    body_quat = _normalize_quaternion(torch.stack(body_quat_frames, dim=0))
    raw_lowest_ankle_z = float(torch.min(body_pos[:, ankle_motion_indices, 2]).item())
    z_shift = float(args_cli.root_height_offset)
    if args_cli.align_a3_ground:
        z_shift += default_lowest_ankle_z - raw_lowest_ankle_z
    motion.applied_root_z_shift_m = z_shift
    motion.root_pos[:, 2] += z_shift
    body_pos[:, :, 2] += z_shift
    print(
        f"[csv_to_npz] A3 ground alignment: enabled={args_cli.align_a3_ground}, "
        f"default_lowest_ankle_z={default_lowest_ankle_z:.4f}m, "
        f"raw_motion_lowest_ankle_z={raw_lowest_ankle_z:.4f}m, "
        f"applied_root_z_shift={z_shift:+.4f}m",
        flush=True,
    )
    root_pos_error = torch.max(torch.abs(body_pos[:, 0] - motion.root_pos)).item()
    if root_pos_error > 1.0e-4:
        raise RuntimeError(f"A3 FK root position mismatch: max error={root_pos_error:.6g} m")
    return body_pos, body_quat


def _infer_swing_side(output_path: Path) -> tuple[str, int]:
    side = args_cli.swing_side
    if side == "auto":
        name = output_path.stem.lower()
        if "forehand" in name:
            side = "forehand"
        elif "backhand" in name:
            side = "backhand"
        else:
            raise ValueError(
                "cannot infer swing side from output filename; pass --swing_side forehand|backhand"
            )
    return side, 1 if side == "forehand" else -1


def _detect_strike_frame(body_lin_vel: torch.Tensor) -> int:
    if args_cli.strike_frame is not None:
        if not 0 <= args_cli.strike_frame < body_lin_vel.shape[0]:
            raise ValueError(
                f"--strike_frame must be in [0, {body_lin_vel.shape[0] - 1}]"
            )
        return int(args_cli.strike_frame)
    wrist_index = A3_TRACKED_BODIES.index(A3_WRIST_BODY)
    speed = torch.linalg.vector_norm(body_lin_vel[:, wrist_index], dim=-1)
    if speed.shape[0] >= 5:
        kernel = torch.ones((1, 1, 5), dtype=speed.dtype, device=speed.device) / 5.0
        padded = torch.nn.functional.pad(speed[None, None], (2, 2), mode="replicate")
        speed = torch.nn.functional.conv1d(padded, kernel)[0, 0]
    return int(torch.argmax(speed).item())


def _write_outputs(
    output_path: Path,
    motion: MotionData,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    body_lin_vel = _gradient(body_pos, motion.output_dt)
    body_ang_vel = _angular_velocity(body_quat, motion.output_dt)
    arrays = {
        "fps": np.asarray(args_cli.output_fps, dtype=np.float32),
        "joint_pos": motion.joint_pos.detach().cpu().numpy().astype(np.float32),
        "joint_vel": motion.joint_vel.detach().cpu().numpy().astype(np.float32),
        "body_pos_w": body_pos.detach().cpu().numpy().astype(np.float32),
        "body_quat_w": body_quat.detach().cpu().numpy().astype(np.float32),
        "body_lin_vel_w": body_lin_vel.detach().cpu().numpy().astype(np.float32),
        "body_ang_vel_w": body_ang_vel.detach().cpu().numpy().astype(np.float32),
    }
    np.savez_compressed(output_path, **arrays)

    side_name, side_sign = _infer_swing_side(output_path)
    strike_frame = _detect_strike_frame(body_lin_vel)
    frame_count = int(motion.joint_pos.shape[0])
    duration = (frame_count - 1) / float(args_cli.output_fps)
    metadata = {
        "name": args_cli.motion_name or output_path.stem,
        "swing_side": side_sign,
        "swing_name": side_name,
        "fps": float(args_cli.output_fps),
        "frame_count": frame_count,
        "frame_time_s": 1.0 / float(args_cli.output_fps),
        "duration_s": duration,
        "strike_frame": strike_frame,
        "strike_phase": strike_frame / max(frame_count - 1, 1),
        "strike_detection": "peak_5_frame_smoothed_right_wrist_linear_speed",
        "ready_interval_frames": [0, min(strike_frame, int(round(0.5 * args_cli.output_fps)))],
        "follow_through_end_frame": min(
            frame_count - 1, strike_frame + int(round(0.6 * args_cli.output_fps))
        ),
        "recover_end_frame": frame_count - 1,
        "joint_order": list(AGIBOT_A3_JOINT_NAMES),
        "tracked_bodies": list(A3_TRACKED_BODIES),
        "anchor_body": A3_ANCHOR_BODY,
        "root_body": A3_ROOT_BODY,
        "racket_link": A3_WRIST_BODY,
        "racket_body": A3_RACKET_BODY,
        "mount_offset_xyz": list(A3_MOUNT_OFFSET),
        "blade_normal_axis": "y",
        "blade_normal_sign": side_sign,
        "source": {
            "file": str(args_cli.input_file.expanduser().resolve()),
            "layout": motion.source_layout,
            "fps": float(args_cli.input_fps),
            "frame_range_1_based_inclusive": args_cli.frame_range,
            "root_quaternion_order": "xyzw",
            "joint_units": "radians",
        },
        "root_alignment": {
            "canonicalized_xy_and_heading": bool(args_cli.canonicalize_root),
            "source_initial_yaw_deg": math.degrees(motion.initial_yaw_rad),
            "a3_ground_alignment": bool(args_cli.align_a3_ground),
            "extra_root_height_offset_m": float(args_cli.root_height_offset),
            "applied_root_z_shift_m": float(motion.applied_root_z_shift_m),
        },
    }
    import yaml

    sidecar = output_path.with_suffix(".yaml")
    with sidecar.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False, allow_unicode=True)

    wrist_index = A3_TRACKED_BODIES.index(A3_WRIST_BODY)
    wrist_peak_speed = torch.linalg.vector_norm(body_lin_vel[:, wrist_index], dim=-1).max().item()
    ankle_indices = [
        A3_TRACKED_BODIES.index("left_ankle_roll_Link"),
        A3_TRACKED_BODIES.index("right_ankle_roll_Link"),
    ]
    ankle_z = body_pos[:, ankle_indices, 2]
    print(f"[csv_to_npz] saved NPZ: {output_path}", flush=True)
    print(f"[csv_to_npz] saved YAML: {sidecar}", flush=True)
    print(
        f"[csv_to_npz] strike proxy: frame={strike_frame}/{frame_count - 1}, "
        f"phase={metadata['strike_phase']:.6f}, wrist_peak_speed={wrist_peak_speed:.3f} m/s",
        flush=True,
    )
    print(
        f"[csv_to_npz] tracked ankle-link z range: "
        f"[{ankle_z.min().item():.4f}, {ankle_z.max().item():.4f}] m",
        flush=True,
    )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / float(args_cli.output_fps),
        gravity=(0.0, 0.0, 0.0),
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    print("[csv_to_npz] A3 scene ready", flush=True)

    motion = MotionData(sim.device)
    body_pos, body_quat = _run_forward_kinematics(sim, scene, motion)
    _write_outputs(args_cli.output_file, motion, body_pos, body_quat)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False)
