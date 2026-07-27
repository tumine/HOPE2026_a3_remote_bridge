#!/usr/bin/env python3
"""Apply shape-preserving temporal smoothing to a converted HOPE motion.

Joint positions and the root trajectory are filtered with a centred
Savitzky--Golay window, then all tracked-body poses are rebuilt from the shipped
A3 MuJoCo model so joint and body references remain kinematically consistent.
All velocity arrays are recomputed afterwards.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from crop_motion_npz import _angular_velocity, _gradient


TIME_SERIES_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "a3_deploy").is_dir():
            return parent
    raise RuntimeError(f"could not locate repository root from {here}")


def _default_model_xml() -> Path:
    return (
        _repo_root()
        / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--window-frames", default=15, type=int)
    parser.add_argument("--polyorder", default=3, type=int)
    parser.add_argument("--model-xml", default=_default_model_xml(), type=Path)
    parser.add_argument("--motion-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_filter(window: int, polyorder: int, frame_count: int) -> None:
    if window < 3 or window % 2 != 1:
        raise ValueError("--window-frames must be an odd integer >= 3")
    if window > frame_count:
        raise ValueError(f"filter window {window} exceeds frame count {frame_count}")
    if not 0 <= polyorder < window:
        raise ValueError("--polyorder must satisfy 0 <= polyorder < window-frames")


def _smooth_root_quaternions(
    quat_wxyz: np.ndarray, window_frames: int, polyorder: int
) -> np.ndarray:
    """Filter a short, continuous root-rotation trajectory in a local SO(3) chart."""

    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]].astype(np.float64)
    # Keep the input in one quaternion hemisphere before constructing rotations.
    for frame in range(1, quat_xyzw.shape[0]):
        if np.dot(quat_xyzw[frame - 1], quat_xyzw[frame]) < 0.0:
            quat_xyzw[frame] *= -1.0
    rotations = Rotation.from_quat(quat_xyzw)
    reference = rotations[0]
    relative_rotvec = (reference.inv() * rotations).as_rotvec()
    filtered_rotvec = savgol_filter(
        relative_rotvec,
        window_length=window_frames,
        polyorder=polyorder,
        axis=0,
        mode="mirror",
    )
    filtered_xyzw = (reference * Rotation.from_rotvec(filtered_rotvec)).as_quat()
    filtered_wxyz = filtered_xyzw[:, [3, 0, 1, 2]]
    signs = np.where(np.sum(filtered_wxyz * quat_wxyz, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return (filtered_wxyz * signs).astype(np.float32)


def _load_motion(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    sidecar = path.with_suffix(".yaml")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"motion and YAML sidecar are required: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    with sidecar.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    missing_arrays = [key for key in ("fps", *TIME_SERIES_KEYS) if key not in arrays]
    if missing_arrays:
        raise ValueError(f"input NPZ is missing required arrays: {missing_arrays}")
    missing_metadata = [key for key in ("joint_order", "tracked_bodies", "root_body") if key not in metadata]
    if missing_metadata:
        raise ValueError(f"input YAML is missing required fields: {missing_metadata}")
    frame_count = arrays["joint_pos"].shape[0]
    for key in TIME_SERIES_KEYS:
        if arrays[key].shape[0] != frame_count:
            raise ValueError(f"{key} has an inconsistent frame count: {arrays[key].shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains NaN or infinite values")
    return arrays, metadata


def _name_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"A3 MuJoCo model has no {object_type.name} named {name!r}")
    return object_id


def _forward_kinematics(
    model_xml: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_order: list[str],
    tracked_bodies: list[str],
    reference_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    free_joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")
    root_qadr = int(model.jnt_qposadr[free_joint_id])
    joint_qadr = np.asarray(
        [
            model.jnt_qposadr[_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in joint_order
        ],
        dtype=int,
    )
    body_ids = np.asarray(
        [_name_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in tracked_bodies],
        dtype=int,
    )

    frame_count = joint_pos.shape[0]
    body_pos = np.empty((frame_count, len(body_ids), 3), dtype=np.float32)
    body_quat = np.empty((frame_count, len(body_ids), 4), dtype=np.float32)
    for frame in range(frame_count):
        data.qpos[root_qadr : root_qadr + 3] = root_pos[frame]
        data.qpos[root_qadr + 3 : root_qadr + 7] = root_quat[frame]
        data.qpos[joint_qadr] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        quats = data.xquat[body_ids].copy()
        # q and -q represent the same rotation. Match the source sign so the
        # resulting quaternion stream keeps the same continuous convention.
        signs = np.where(np.sum(quats * reference_quat[frame], axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
        body_quat[frame] = quats * signs
    return body_pos, body_quat


def main() -> int:
    args = _parse_args()
    input_path = args.input_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()
    model_xml = args.model_xml.expanduser().resolve()
    output_sidecar = output_path.with_suffix(".yaml")
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source motion; choose a new output filename")
    if not model_xml.is_file():
        raise FileNotFoundError(f"A3 MuJoCo model not found: {model_xml}")
    if (output_path.exists() or output_sidecar.exists()) and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")

    arrays, metadata = _load_motion(input_path)
    frame_count = int(arrays["joint_pos"].shape[0])
    _validate_filter(args.window_frames, args.polyorder, frame_count)
    fps = float(np.asarray(arrays["fps"]).item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid fps: {fps}")
    dt = 1.0 / fps

    filtered_joint_pos = savgol_filter(
        arrays["joint_pos"].astype(np.float64),
        window_length=args.window_frames,
        polyorder=args.polyorder,
        axis=0,
        mode="mirror",
    ).astype(np.float32)

    tracked_bodies = [str(name) for name in metadata["tracked_bodies"]]
    root_body = str(metadata["root_body"])
    if root_body not in tracked_bodies:
        raise ValueError(f"root_body {root_body!r} is not in tracked_bodies")
    root_index = tracked_bodies.index(root_body)
    root_pos = savgol_filter(
        arrays["body_pos_w"][:, root_index].astype(np.float64),
        window_length=args.window_frames,
        polyorder=args.polyorder,
        axis=0,
        mode="mirror",
    ).astype(np.float32)
    root_quat = _smooth_root_quaternions(
        arrays["body_quat_w"][:, root_index], args.window_frames, args.polyorder
    )
    body_pos, body_quat = _forward_kinematics(
        model_xml,
        root_pos,
        root_quat,
        filtered_joint_pos,
        [str(name) for name in metadata["joint_order"]],
        tracked_bodies,
        arrays["body_quat_w"],
    )

    filtered = {key: value.copy() for key, value in arrays.items()}
    filtered["joint_pos"] = filtered_joint_pos
    filtered["joint_vel"] = _gradient(filtered_joint_pos, dt)
    filtered["body_pos_w"] = body_pos
    filtered["body_quat_w"] = body_quat
    filtered["body_lin_vel_w"] = _gradient(body_pos, dt)
    filtered["body_ang_vel_w"] = _angular_velocity(body_quat, dt)
    for key in TIME_SERIES_KEYS:
        filtered[key] = filtered[key].astype(np.float32, copy=False)

    strike_frame = int(metadata.get("strike_frame", -1))
    if not 0 <= strike_frame < frame_count:
        raise ValueError(f"invalid source strike_frame: {strike_frame}")
    racket_link = str(metadata.get("racket_link", "right_wrist_yaw_Link"))
    peak_frame = None
    if racket_link in tracked_bodies:
        racket_index = tracked_bodies.index(racket_link)
        speed = np.linalg.norm(filtered["body_lin_vel_w"][:, racket_index], axis=-1)
        peak_frame = int(np.argmax(speed))

    updated = copy.deepcopy(metadata)
    updated["name"] = args.motion_name or output_path.stem
    updated["smoothing"] = {
        "method": "centered_savitzky_golay",
        "window_frames": int(args.window_frames),
        "window_seconds": float(args.window_frames / fps),
        "polyorder": int(args.polyorder),
        "edge_mode": "mirror",
        "zero_phase": True,
        "filtered_fields": ["joint_pos", "root_pos_w", "root_quat_w"],
        "root_trajectory_preserved": False,
        "tracked_body_fk": "recomputed_from_a3_mujoco_model",
        "velocities_recomputed": True,
        "source_motion": str(input_path),
        "model_xml": str(model_xml),
        "preserved_strike_frame": strike_frame,
        "filtered_racket_link_speed_peak_frame": peak_frame,
    }
    updated["strike_detection"] = "preserved_semantic_frame_after_centered_savgol_smoothing"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **filtered)
    with output_sidecar.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(updated, stream, sort_keys=False, allow_unicode=True)

    max_joint_delta = float(np.max(np.abs(filtered_joint_pos - arrays["joint_pos"])))
    strike_joint_delta = float(
        np.max(np.abs(filtered_joint_pos[strike_frame] - arrays["joint_pos"][strike_frame]))
    )
    print(
        f"[smooth_motion_npz] {input_path.name} -> {output_path.name}: "
        f"frames={frame_count}, fps={fps:g}, filter=SG({args.window_frames},{args.polyorder}), "
        f"strike={strike_frame}, filtered_peak={peak_frame}, "
        f"max_joint_delta={np.degrees(max_joint_delta):.4f}deg, "
        f"strike_max_joint_delta={np.degrees(strike_joint_delta):.4f}deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
