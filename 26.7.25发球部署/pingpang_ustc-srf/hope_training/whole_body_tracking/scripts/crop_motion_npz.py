#!/usr/bin/env python3
"""Crop a converted HOPE motion without changing its spatial alignment.

This utility operates on the already-retargeted NPZ instead of passing a frame
range back through ``csv_to_npz.py``.  That distinction matters: converting a
CSV frame range would canonicalize the root from a different first frame and
slightly move/rotate the strike pose.

The crop is inclusive at both ends.  Positions and orientations are sliced
exactly; velocities are recomputed at the original FPS so the new endpoint
derivatives are self-consistent.  The matching YAML sidecar is copied and its
timeline metadata is updated.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml


TIME_SERIES_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int, help="Zero-based inclusive.")
    parser.add_argument("--end-frame", required=True, type=int, help="Zero-based inclusive.")
    parser.add_argument(
        "--swing-start-frame",
        required=True,
        type=int,
        help="Conservative first active-swing frame in the source motion.",
    )
    parser.add_argument(
        "--swing-end-frame",
        required=True,
        type=int,
        help="Conservative last active-swing frame in the source motion.",
    )
    parser.add_argument(
        "--strike-frame",
        type=int,
        default=None,
        help="Source-motion strike frame; defaults to the input YAML value.",
    )
    parser.add_argument("--motion-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] == 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, dt, axis=0, edge_order=1).astype(np.float32)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("motion contains a zero-length body quaternion")
    return quat / norm


def _relative_rotvec(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    relative = _quat_multiply(q_to, _quat_conjugate(q_from))
    relative = _normalize_quaternions(relative)
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    sin_half = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(relative[..., :1], 0.0, 1.0))
    scale = np.where(sin_half > 1.0e-8, angle / np.maximum(sin_half, 1.0e-8), 2.0)
    return vector * scale


def _angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    normalized = _normalize_quaternions(quat.astype(np.float64, copy=False))
    result = np.zeros((*quat.shape[:-1], 3), dtype=np.float64)
    if quat.shape[0] == 1:
        return result.astype(np.float32)
    result[0] = _relative_rotvec(normalized[0], normalized[1]) / dt
    result[-1] = _relative_rotvec(normalized[-2], normalized[-1]) / dt
    if quat.shape[0] > 2:
        result[1:-1] = _relative_rotvec(normalized[:-2], normalized[2:]) / (2.0 * dt)
    return result.astype(np.float32)


def _validate_source(
    arrays: dict[str, np.ndarray], metadata: dict, args: argparse.Namespace
) -> tuple[int, float, int]:
    missing = [key for key in ("fps", *TIME_SERIES_KEYS) if key not in arrays]
    if missing:
        raise ValueError(f"input NPZ is missing required arrays: {missing}")
    frame_count = int(arrays["joint_pos"].shape[0])
    for key in TIME_SERIES_KEYS:
        if arrays[key].shape[0] != frame_count:
            raise ValueError(
                f"{key} has {arrays[key].shape[0]} frames, expected {frame_count}"
            )
    fps = float(np.asarray(arrays["fps"]).item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid fps: {fps}")
    if not 0 <= args.start_frame <= args.end_frame < frame_count:
        raise ValueError(
            f"crop [{args.start_frame}, {args.end_frame}] is outside [0, {frame_count - 1}]"
        )
    if not args.start_frame <= args.swing_start_frame <= args.swing_end_frame <= args.end_frame:
        raise ValueError("the full swing interval must be contained in the crop")
    strike_frame = args.strike_frame
    if strike_frame is None:
        if "strike_frame" not in metadata:
            raise ValueError("--strike-frame is required when the input YAML has no strike_frame")
        strike_frame = int(metadata["strike_frame"])
    if not args.swing_start_frame <= strike_frame <= args.swing_end_frame:
        raise ValueError("strike frame must lie inside the full swing interval")
    return frame_count, fps, strike_frame


def main() -> int:
    args = _parse_args()
    input_path = args.input_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source motion; choose a new output filename")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")
    input_sidecar = input_path.with_suffix(".yaml")
    output_sidecar = output_path.with_suffix(".yaml")
    if not input_path.is_file() or not input_sidecar.is_file():
        raise FileNotFoundError(f"motion and YAML sidecar are required: {input_path}")
    if output_sidecar.exists() and not args.overwrite:
        raise FileExistsError(f"output sidecar exists (pass --overwrite): {output_sidecar}")

    with np.load(input_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    with input_sidecar.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}

    _, fps, source_strike = _validate_source(arrays, metadata, args)
    selection = slice(args.start_frame, args.end_frame + 1)
    cropped = {
        key: (value[selection].copy() if key in TIME_SERIES_KEYS else value.copy())
        for key, value in arrays.items()
    }
    dt = 1.0 / fps
    cropped["joint_vel"] = _gradient(cropped["joint_pos"], dt)
    cropped["body_lin_vel_w"] = _gradient(cropped["body_pos_w"], dt)
    cropped["body_ang_vel_w"] = _angular_velocity(cropped["body_quat_w"], dt)
    for key in TIME_SERIES_KEYS:
        cropped[key] = cropped[key].astype(np.float32, copy=False)

    new_count = args.end_frame - args.start_frame + 1
    new_strike = source_strike - args.start_frame
    new_swing_start = args.swing_start_frame - args.start_frame
    new_swing_end = args.swing_end_frame - args.start_frame
    quiet_before = [0, new_swing_start - 1]
    quiet_after = [new_swing_end + 1, new_count - 1]

    updated = copy.deepcopy(metadata)
    updated["name"] = args.motion_name or output_path.stem
    updated["fps"] = fps
    updated["frame_count"] = new_count
    updated["frame_time_s"] = dt
    updated["duration_s"] = (new_count - 1) / fps
    updated["strike_frame"] = new_strike
    updated["strike_phase"] = new_strike / max(new_count - 1, 1)
    updated["ready_interval_frames"] = quiet_before
    updated["full_swing_interval_frames"] = [new_swing_start, new_swing_end]
    updated["follow_through_end_frame"] = min(new_count - 1, new_strike + round(0.6 * fps))
    updated["recover_end_frame"] = new_swing_end
    updated["post_quiet_interval_frames"] = quiet_after
    updated["crop"] = {
        "method": "direct_npz_slice_no_resampling",
        "source_motion": str(input_path),
        "source_frame_range_zero_based_inclusive": [args.start_frame, args.end_frame],
        "source_frame_range_one_based_inclusive": [args.start_frame + 1, args.end_frame + 1],
        "source_strike_frame": source_strike,
        "source_full_swing_interval_frames": [args.swing_start_frame, args.swing_end_frame],
        "velocities_recomputed_after_crop": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cropped)
    with output_sidecar.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(updated, stream, sort_keys=False, allow_unicode=True)

    print(
        f"[crop_motion_npz] {input_path.name}[{args.start_frame}:{args.end_frame}] "
        f"-> {output_path.name}: frames={new_count}, duration={(new_count - 1) / fps:.3f}s, "
        f"strike={new_strike}, phase={updated['strike_phase']:.9f}, "
        f"quiet_before={quiet_before}, quiet_after={quiet_after}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
