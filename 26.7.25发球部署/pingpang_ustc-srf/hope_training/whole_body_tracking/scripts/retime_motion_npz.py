#!/usr/bin/env python3
"""Crop and safely retime a HOPE motion while preserving its strike guard.

The output keeps the source FPS.  A central guard interval is copied at exactly
one source frame per output frame, so the strike pose and its local finite-
difference velocities are unchanged.  The outer wind-up and recovery phases are
accelerated with a bounded, motion-aware inverse-arc-length map shared by the
selected timing references (or an optional monotone C1 cubic-Hermite map). All
velocity arrays are recomputed after interpolation.

Frame ranges and knot indices are zero-based.  The crop is inclusive at both
ends.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import yaml

from crop_motion_npz import TIME_SERIES_KEYS, _angular_velocity, _gradient


_A3_JOINT_VELOCITY_LIMITS_RAD_S = (
    12.0, 22.7, 9.2,
    12.7, 12.7,
    13.6, 13.6, 15.7, 15.7, 15.7, 12.7, 12.7,
    13.6, 13.6, 15.7, 15.7, 15.7, 12.7, 12.7,
    12.0, 12.0, 12.0, 14.6, 10.8, 19.3,
    12.0, 12.0, 12.0, 14.6, 10.8, 19.3,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--guard-start-frame", required=True, type=int)
    parser.add_argument("--guard-end-frame", required=True, type=int)
    parser.add_argument("--output-frame-count", required=True, type=int)
    parser.add_argument("--output-guard-start-frame", required=True, type=int)
    parser.add_argument("--output-guard-end-frame", required=True, type=int)
    parser.add_argument(
        "--mapping-method",
        choices=("motion_aware_arc_length", "cubic_hermite"),
        default="motion_aware_arc_length",
        help=(
            "Outer-phase time allocation. The default uses normalized joint/racket "
            "motion shared by all --timing-reference-file inputs."
        ),
    )
    parser.add_argument(
        "--timing-reference-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Motion contributing to the shared time-allocation cost; repeat for a "
            "forehand/backhand pair. Defaults to --input-file."
        ),
    )
    parser.add_argument(
        "--strike-frame",
        type=int,
        default=None,
        help="Source strike frame; defaults to strike_frame in the YAML sidecar.",
    )
    parser.add_argument("--motion-name", default=None)
    parser.add_argument(
        "--max-endpoint-joint-speed",
        type=float,
        default=0.50,
        help="Maximum absolute joint velocity at either output endpoint, in rad/s.",
    )
    parser.add_argument(
        "--max-endpoint-racket-speed",
        type=float,
        default=0.15,
        help="Maximum racket-link linear speed at either output endpoint, in m/s.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _cubic_hermite(
    x: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    slope0: float,
    slope1: float,
) -> np.ndarray:
    """Evaluate one cubic Hermite segment with dy/dx endpoint slopes."""

    width = x1 - x0
    if width <= 0.0:
        raise ValueError("Hermite segment must have positive width")
    t = (x - x0) / width
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (
        h00 * y0
        + h10 * width * slope0
        + h01 * y1
        + h11 * width * slope1
    )


def build_source_frame_map(
    *,
    start_frame: int,
    end_frame: int,
    guard_start_frame: int,
    guard_end_frame: int,
    output_frame_count: int,
    output_guard_start_frame: int,
    output_guard_end_frame: int,
) -> np.ndarray:
    """Map each integer output frame to a continuous source-frame coordinate."""

    output_end = output_frame_count - 1
    if output_frame_count < 3:
        raise ValueError("output_frame_count must be at least 3")
    if not start_frame < guard_start_frame < guard_end_frame < end_frame:
        raise ValueError(
            "source knots must satisfy start < guard_start < guard_end < end"
        )
    if not 0 < output_guard_start_frame < output_guard_end_frame < output_end:
        raise ValueError(
            "output knots must satisfy 0 < guard_start < guard_end < frame_count - 1"
        )
    source_guard_length = guard_end_frame - guard_start_frame
    output_guard_length = output_guard_end_frame - output_guard_start_frame
    if source_guard_length != output_guard_length:
        raise ValueError(
            "source and output guard intervals must contain the same number of "
            "frame steps to preserve unit-speed strike kinematics"
        )

    output_frames = np.arange(output_frame_count, dtype=np.float64)
    source_frames = np.empty(output_frame_count, dtype=np.float64)

    pre_secant = (guard_start_frame - start_frame) / output_guard_start_frame
    post_secant = (end_frame - guard_end_frame) / (
        output_end - output_guard_end_frame
    )
    if pre_secant < 1.0 or post_secant < 1.0:
        raise ValueError(
            "guarded retiming only supports compressing (not slowing) the outer intervals"
        )

    pre_mask = output_frames <= output_guard_start_frame
    source_frames[pre_mask] = _cubic_hermite(
        output_frames[pre_mask],
        0.0,
        float(output_guard_start_frame),
        float(start_frame),
        float(guard_start_frame),
        pre_secant,
        1.0,
    )

    guard_mask = (output_frames >= output_guard_start_frame) & (
        output_frames <= output_guard_end_frame
    )
    # Assign this interval explicitly so every guarded output frame is exactly
    # the corresponding integer source frame (including the strike).
    source_frames[guard_mask] = guard_start_frame + (
        output_frames[guard_mask] - output_guard_start_frame
    )

    post_mask = output_frames >= output_guard_end_frame
    source_frames[post_mask] = _cubic_hermite(
        output_frames[post_mask],
        float(output_guard_end_frame),
        float(output_end),
        float(guard_end_frame),
        float(end_frame),
        1.0,
        post_secant,
    )

    if not np.isclose(source_frames[0], start_frame, atol=1.0e-12):
        raise AssertionError("retiming map did not preserve the source start knot")
    if not np.isclose(source_frames[-1], end_frame, atol=1.0e-12):
        raise AssertionError("retiming map did not preserve the source end knot")
    if np.any(np.diff(source_frames) <= 0.0):
        raise ValueError("retiming map is not strictly monotone")
    if source_frames.min() < start_frame - 1.0e-9:
        raise ValueError("retiming map undershot the requested crop")
    if source_frames.max() > end_frame + 1.0e-9:
        raise ValueError("retiming map overshot the requested crop")
    return source_frames


def _gaussian_smooth(values: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Gaussian-filter a 1-D cost with nearest-edge padding, without SciPy."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Gaussian smoothing requires a non-empty 1-D array")
    radius = int(4.0 * sigma + 0.5)
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _bounded_time_density(
    edge_cost: np.ndarray,
    target_output_steps: int,
    *,
    density_floor: float = 0.125,
) -> tuple[np.ndarray, float]:
    """Allocate output intervals to source edges with bounded compression.

    Density is ``output_interval / source_interval``. A floor of 0.125 means
    no output step can skip more than eight source-frame intervals. The first
    and last edge remain at unit time scale, preserving endpoint/guard
    finite-difference kinematics.
    """

    edge_cost = np.asarray(edge_cost, dtype=np.float64)
    if edge_cost.ndim != 1 or edge_cost.size < 2:
        raise ValueError("each retimed outer phase needs at least two source edges")
    if target_output_steps <= 1 or target_output_steps >= edge_cost.size:
        raise ValueError(
            "motion-aware retiming requires 1 < output steps < source steps"
        )
    smoothed = _gaussian_smooth(edge_cost)
    mean_cost = float(smoothed.mean())
    if not np.isfinite(mean_cost) or mean_cost < 0.0:
        raise ValueError("motion-aware edge cost must be finite and non-negative")
    if mean_cost <= np.finfo(np.float64).eps:
        smoothed = np.ones_like(smoothed)
    else:
        smoothed += 0.4 * mean_cost

    fixed = np.zeros(smoothed.size, dtype=bool)
    fixed[[0, -1]] = True

    def _density(scale: float) -> np.ndarray:
        density = np.maximum(density_floor, scale * smoothed)
        density[fixed] = 1.0
        return density

    minimum = float(_density(0.0).sum())
    if minimum > target_output_steps + 1.0e-12:
        raise ValueError(
            f"density floor requires {minimum:.6f} output steps, "
            f"but only {target_output_steps} were requested"
        )

    lower, upper = 0.0, 1.0
    while float(_density(upper).sum()) < target_output_steps:
        upper *= 2.0
        if upper > 1.0e12:
            raise ValueError("could not solve motion-aware time density")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if float(_density(middle).sum()) < target_output_steps:
            lower = middle
        else:
            upper = middle

    scale = 0.5 * (lower + upper)
    density = _density(scale)
    free = np.flatnonzero(~fixed)
    correction_index = int(free[np.argmax(density[free])])
    density[correction_index] += target_output_steps - float(density.sum())
    if np.any(density <= 0.0) or np.any(density > 1.0 + 1.0e-10):
        raise ValueError(
            "motion-aware density would locally slow the compressed outer phase"
        )
    return density, scale


def _racket_center_positions(
    arrays: dict[str, np.ndarray], metadata: dict
) -> np.ndarray:
    tracked_bodies = [str(name) for name in metadata.get("tracked_bodies", [])]
    racket_link = str(metadata.get("racket_link", "right_wrist_yaw_Link"))
    if racket_link not in tracked_bodies:
        raise ValueError(f"racket_link {racket_link!r} is not in tracked_bodies")
    racket_index = tracked_bodies.index(racket_link)
    mount_offset = np.asarray(metadata.get("mount_offset_xyz", ()), dtype=np.float64)
    if mount_offset.shape != (3,) or not np.all(np.isfinite(mount_offset)):
        raise ValueError("motion sidecar must provide a finite mount_offset_xyz[3]")
    offset_w = _quat_apply(
        arrays["body_quat_w"][:, racket_index].astype(np.float64), mount_offset
    )
    return arrays["body_pos_w"][:, racket_index].astype(np.float64) + offset_w


def _motion_edge_cost(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    *,
    dt: float,
) -> np.ndarray:
    joint_pos = arrays["joint_pos"].astype(np.float64)
    limits = np.asarray(_A3_JOINT_VELOCITY_LIMITS_RAD_S, dtype=np.float64)
    joint_order = metadata.get("joint_order", ())
    if len(joint_order) != joint_pos.shape[1] or joint_pos.shape[1] != limits.size:
        raise ValueError(
            "motion-aware retiming requires the canonical 31-joint A3 motion order"
        )
    normalized_delta = np.abs(np.diff(joint_pos, axis=0)) / (limits[None, :] * dt)
    max_joint_fraction = np.max(normalized_delta, axis=1)
    normalized_joint_arc = 0.25 * np.linalg.norm(normalized_delta, axis=1)
    racket_position = _racket_center_positions(arrays, metadata)
    normalized_racket_arc = (
        np.linalg.norm(np.diff(racket_position, axis=0), axis=1) / (5.0 * dt)
    )
    return np.maximum.reduce(
        (max_joint_fraction, normalized_joint_arc, normalized_racket_arc)
    )


def _invert_time_density(
    source_start: int,
    density: np.ndarray,
    output_start: int,
) -> np.ndarray:
    output_coordinate = output_start + np.concatenate(
        ([0.0], np.cumsum(density))
    )
    source_coordinate = np.arange(
        source_start, source_start + density.size + 1, dtype=np.float64
    )
    output_end = output_start + int(round(float(density.sum())))
    return np.interp(
        np.arange(output_start, output_end + 1, dtype=np.float64),
        output_coordinate,
        source_coordinate,
    )


def build_motion_aware_source_frame_map(
    *,
    reference_motions: list[tuple[dict[str, np.ndarray], dict]],
    fps: float,
    start_frame: int,
    end_frame: int,
    guard_start_frame: int,
    guard_end_frame: int,
    output_frame_count: int,
    output_guard_start_frame: int,
    output_guard_end_frame: int,
) -> tuple[np.ndarray, dict]:
    """Build a shared, speed-limited inverse-arc-length time map."""

    if not reference_motions:
        raise ValueError("at least one timing reference motion is required")
    # Reuse the structural validations of the simple mapper. Its returned map is
    # discarded; this keeps the public knot contract identical across methods.
    build_source_frame_map(
        start_frame=start_frame,
        end_frame=end_frame,
        guard_start_frame=guard_start_frame,
        guard_end_frame=guard_end_frame,
        output_frame_count=output_frame_count,
        output_guard_start_frame=output_guard_start_frame,
        output_guard_end_frame=output_guard_end_frame,
    )

    dt = 1.0 / float(fps)
    common_cost = None
    for arrays, metadata in reference_motions:
        if arrays["joint_pos"].shape[0] <= end_frame:
            raise ValueError("timing reference is shorter than the requested crop")
        reference_fps = float(np.asarray(arrays["fps"]).item())
        if not np.isclose(reference_fps, fps, atol=1.0e-12):
            raise ValueError(
                f"timing reference fps {reference_fps} does not match input fps {fps}"
            )
        cost = _motion_edge_cost(arrays, metadata, dt=dt)
        common_cost = cost if common_cost is None else np.maximum(common_cost, cost)

    pre_steps = output_guard_start_frame
    post_steps = output_frame_count - 1 - output_guard_end_frame
    pre_density, pre_scale = _bounded_time_density(
        common_cost[start_frame:guard_start_frame], pre_steps
    )
    post_density, post_scale = _bounded_time_density(
        common_cost[guard_end_frame:end_frame], post_steps
    )

    source_frames = np.empty(output_frame_count, dtype=np.float64)
    source_frames[: output_guard_start_frame + 1] = _invert_time_density(
        start_frame, pre_density, 0
    )
    source_frames[
        output_guard_start_frame : output_guard_end_frame + 1
    ] = np.arange(guard_start_frame, guard_end_frame + 1, dtype=np.float64)
    source_frames[output_guard_end_frame:] = _invert_time_density(
        guard_end_frame, post_density, output_guard_end_frame
    )

    # Unit-density outer edges provide a one-frame derivative buffer around
    # the explicit guard. Pin exact integers to remove floating-point tails.
    buffered_source_start = guard_start_frame - 1
    buffered_source_end = guard_end_frame + 1
    buffered_output_start = output_guard_start_frame - 1
    buffered_output_end = output_guard_end_frame + 1
    source_frames[buffered_output_start : buffered_output_end + 1] = np.arange(
        buffered_source_start, buffered_source_end + 1, dtype=np.float64
    )

    if not np.isclose(source_frames[0], start_frame, atol=1.0e-12):
        raise AssertionError("motion-aware map did not preserve the start frame")
    if not np.isclose(source_frames[-1], end_frame, atol=1.0e-12):
        raise AssertionError("motion-aware map did not preserve the end frame")
    if np.any(np.diff(source_frames) < 1.0 - 1.0e-10):
        raise ValueError("motion-aware map is not strictly compressive and monotone")

    diagnostics = {
        "cost": (
            "max(max_abs_joint_delta_over_velocity_limit, "
            "0.25*l2_normalized_joint_arc, racket_center_arc_over_5mps)"
        ),
        "gaussian_sigma_source_frames": 12.0,
        "segment_mean_cost_addition_factor": 0.4,
        "density_floor_output_per_source": 0.125,
        "pre_density_scale": pre_scale,
        "post_density_scale": post_scale,
        "pre_density_range": [float(pre_density.min()), float(pre_density.max())],
        "post_density_range": [float(post_density.min()), float(post_density.max())],
        "max_source_frames_per_output_step": float(np.diff(source_frames).max()),
        "unit_speed_derivative_buffer_source_frames": [
            buffered_source_start,
            buffered_source_end,
        ],
        "unit_speed_derivative_buffer_output_frames": [
            buffered_output_start,
            buffered_output_end,
        ],
    }
    return source_frames, diagnostics


def _interpolate_linear(values: np.ndarray, source_frames: np.ndarray) -> np.ndarray:
    lower = np.floor(source_frames).astype(np.int64)
    upper = np.ceil(source_frames).astype(np.int64)
    fraction_shape = (source_frames.shape[0],) + (1,) * (values.ndim - 1)
    fraction = (source_frames - lower).reshape(fraction_shape)
    result = (
        values[lower].astype(np.float64) * (1.0 - fraction)
        + values[upper].astype(np.float64) * fraction
    )
    integer_mask = np.isclose(source_frames, np.rint(source_frames), atol=1.0e-12)
    result[integer_mask] = values[np.rint(source_frames[integer_mask]).astype(np.int64)]
    return result.astype(np.float32)


def _interpolate_quaternions(
    quaternions: np.ndarray, source_frames: np.ndarray
) -> np.ndarray:
    """Shortest-arc SLERP for a ``(frames, ..., 4)`` wxyz quaternion stream."""

    lower = np.floor(source_frames).astype(np.int64)
    upper = np.ceil(source_frames).astype(np.int64)
    fraction_shape = (source_frames.shape[0],) + (1,) * (quaternions.ndim - 1)
    fraction = (source_frames - lower).reshape(fraction_shape)

    q0 = quaternions[lower].astype(np.float64)
    q1 = quaternions[upper].astype(np.float64)
    q0_norm = np.linalg.norm(q0, axis=-1, keepdims=True)
    q1_norm = np.linalg.norm(q1, axis=-1, keepdims=True)
    if np.any(q0_norm < 1.0e-8) or np.any(q1_norm < 1.0e-8):
        raise ValueError("motion contains a zero-length body quaternion")
    q0 /= q0_norm
    q1 /= q1_norm

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    close = sin_theta < 1.0e-8
    safe_sin_theta = np.where(close, 1.0, sin_theta)
    weight0 = np.sin((1.0 - fraction) * theta) / safe_sin_theta
    weight1 = np.sin(fraction * theta) / safe_sin_theta
    spherical = weight0 * q0 + weight1 * q1
    linear = (1.0 - fraction) * q0 + fraction * q1
    result = np.where(close, linear, spherical)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)

    integer_mask = np.isclose(source_frames, np.rint(source_frames), atol=1.0e-12)
    result[integer_mask] = quaternions[
        np.rint(source_frames[integer_mask]).astype(np.int64)
    ]
    return result.astype(np.float32)


def _load_motion(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    sidecar = path.with_suffix(".yaml")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"motion and YAML sidecar are required: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    with sidecar.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}

    missing = [key for key in ("fps", *TIME_SERIES_KEYS) if key not in arrays]
    if missing:
        raise ValueError(f"input NPZ is missing required arrays: {missing}")
    frame_count = int(arrays["joint_pos"].shape[0])
    for key in TIME_SERIES_KEYS:
        if arrays[key].shape[0] != frame_count:
            raise ValueError(
                f"{key} has {arrays[key].shape[0]} frames, expected {frame_count}"
            )
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains NaN or infinite values")
    return arrays, metadata


def _mapped_interval(
    source_frames: np.ndarray, interval: object
) -> list[int]:
    if (
        not isinstance(interval, (tuple, list))
        or len(interval) != 2
        or int(interval[0]) > int(interval[1])
    ):
        return []
    source_start, source_end = (int(interval[0]), int(interval[1]))
    selected = np.flatnonzero(
        (source_frames >= source_start - 1.0e-9)
        & (source_frames <= source_end + 1.0e-9)
    )
    if selected.size == 0:
        return []
    return [int(selected[0]), int(selected[-1])]


def _nearest_output_frame(source_frames: np.ndarray, source_frame: int) -> int:
    return int(np.argmin(np.abs(source_frames - source_frame)))


def _quat_apply(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate ``vector`` by a wxyz quaternion without constructing a matrix."""

    scalar = quaternion_wxyz[..., :1]
    xyz = quaternion_wxyz[..., 1:]
    first_cross = np.cross(xyz, vector)
    return vector + 2.0 * (
        scalar * first_cross + np.cross(xyz, first_cross)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    input_path = args.input_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()
    input_sidecar = input_path.with_suffix(".yaml")
    output_sidecar = output_path.with_suffix(".yaml")
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source motion")
    if (output_path.exists() or output_sidecar.exists()) and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")

    arrays, metadata = _load_motion(input_path)
    frame_count = int(arrays["joint_pos"].shape[0])
    if not 0 <= args.start_frame < args.end_frame < frame_count:
        raise ValueError(
            f"crop [{args.start_frame}, {args.end_frame}] is outside "
            f"[0, {frame_count - 1}]"
        )

    fps = float(np.asarray(arrays["fps"]).item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid fps: {fps}")
    source_strike = (
        int(metadata.get("strike_frame", -1))
        if args.strike_frame is None
        else args.strike_frame
    )
    if not args.guard_start_frame < source_strike < args.guard_end_frame:
        raise ValueError("strike frame must lie strictly inside the preserved guard")

    timing_reference_paths = [
        path.expanduser().resolve() for path in args.timing_reference_file
    ]
    if not timing_reference_paths:
        timing_reference_paths = [input_path]
    # Preserve caller order for provenance while avoiding duplicate cost loads.
    timing_reference_paths = list(dict.fromkeys(timing_reference_paths))
    timing_references = []
    for reference_path in timing_reference_paths:
        timing_references.append(_load_motion(reference_path))

    mapping_diagnostics = None
    if args.mapping_method == "motion_aware_arc_length":
        source_frames, mapping_diagnostics = build_motion_aware_source_frame_map(
            reference_motions=timing_references,
            fps=fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            guard_start_frame=args.guard_start_frame,
            guard_end_frame=args.guard_end_frame,
            output_frame_count=args.output_frame_count,
            output_guard_start_frame=args.output_guard_start_frame,
            output_guard_end_frame=args.output_guard_end_frame,
        )
    else:
        source_frames = build_source_frame_map(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            guard_start_frame=args.guard_start_frame,
            guard_end_frame=args.guard_end_frame,
            output_frame_count=args.output_frame_count,
            output_guard_start_frame=args.output_guard_start_frame,
            output_guard_end_frame=args.output_guard_end_frame,
        )
    output_strike = _nearest_output_frame(source_frames, source_strike)
    if not np.isclose(source_frames[output_strike], source_strike, atol=1.0e-12):
        raise ValueError(
            f"strike frame {source_strike} is not represented exactly by the output map"
        )

    retimed = {key: value.copy() for key, value in arrays.items()}
    retimed["joint_pos"] = _interpolate_linear(arrays["joint_pos"], source_frames)
    retimed["body_pos_w"] = _interpolate_linear(arrays["body_pos_w"], source_frames)
    retimed["body_quat_w"] = _interpolate_quaternions(
        arrays["body_quat_w"], source_frames
    )
    dt = 1.0 / fps
    retimed["joint_vel"] = _gradient(retimed["joint_pos"], dt)
    retimed["body_lin_vel_w"] = _gradient(retimed["body_pos_w"], dt)
    retimed["body_ang_vel_w"] = _angular_velocity(retimed["body_quat_w"], dt)
    for key in TIME_SERIES_KEYS:
        retimed[key] = retimed[key].astype(np.float32, copy=False)

    tracked_bodies = [str(name) for name in metadata.get("tracked_bodies", [])]
    racket_link = str(metadata.get("racket_link", "right_wrist_yaw_Link"))
    if racket_link not in tracked_bodies:
        raise ValueError(f"racket_link {racket_link!r} is not in tracked_bodies")
    racket_index = tracked_bodies.index(racket_link)
    mount_offset = np.asarray(metadata.get("mount_offset_xyz", ()), dtype=np.float64)
    if mount_offset.shape != (3,) or not np.all(np.isfinite(mount_offset)):
        raise ValueError("motion sidecar must provide a finite mount_offset_xyz[3]")
    endpoint_joint_speed = float(
        np.max(np.abs(retimed["joint_vel"][[0, -1]]))
    )
    racket_offset_w = _quat_apply(
        retimed["body_quat_w"][:, racket_index].astype(np.float64),
        mount_offset,
    )
    racket_center_velocity = (
        retimed["body_lin_vel_w"][:, racket_index].astype(np.float64)
        + np.cross(
            retimed["body_ang_vel_w"][:, racket_index].astype(np.float64),
            racket_offset_w,
        )
    )
    endpoint_racket_center_velocity = racket_center_velocity[[0, -1]]
    endpoint_racket_center_speed = float(
        np.max(
            np.linalg.norm(endpoint_racket_center_velocity, axis=-1)
        )
    )
    joint_acceleration = _gradient(retimed["joint_vel"], dt)
    racket_center_speed = np.linalg.norm(racket_center_velocity, axis=-1)
    peak_racket_frame = int(np.argmax(racket_center_speed))
    max_joint_speed = float(np.max(np.abs(retimed["joint_vel"])))
    max_joint_acceleration = float(np.max(np.abs(joint_acceleration)))
    max_racket_center_speed = float(racket_center_speed[peak_racket_frame])
    if endpoint_joint_speed > args.max_endpoint_joint_speed:
        raise ValueError(
            f"endpoint joint speed {endpoint_joint_speed:.6f} rad/s exceeds "
            f"{args.max_endpoint_joint_speed:.6f} rad/s"
        )
    if endpoint_racket_center_speed > args.max_endpoint_racket_speed:
        raise ValueError(
            f"endpoint racket-center speed {endpoint_racket_center_speed:.6f} m/s exceeds "
            f"{args.max_endpoint_racket_speed:.6f} m/s"
        )

    # The guard has unit time scale.  This is a hard validation that the strike
    # itself (including recomputed derivatives) survived the transform.
    strike_checks = (
        ("joint_pos", 1.0e-7),
        ("joint_vel", 2.0e-5),
        ("body_pos_w", 1.0e-7),
        ("body_quat_w", 1.0e-7),
        ("body_lin_vel_w", 2.0e-5),
        ("body_ang_vel_w", 2.0e-5),
    )
    strike_max_error: dict[str, float] = {}
    for key, tolerance in strike_checks:
        error = float(
            np.max(
                np.abs(
                    retimed[key][output_strike].astype(np.float64)
                    - arrays[key][source_strike].astype(np.float64)
                )
            )
        )
        strike_max_error[key] = error
        if error > tolerance:
            raise ValueError(
                f"retiming changed strike {key}: max error {error:.9g} > {tolerance:g}"
            )

    updated = copy.deepcopy(metadata)
    updated["name"] = args.motion_name or output_path.stem
    updated["fps"] = fps
    updated["frame_count"] = args.output_frame_count
    updated["frame_time_s"] = dt
    updated["duration_s"] = (args.output_frame_count - 1) / fps
    updated["strike_frame"] = output_strike
    updated["strike_phase"] = output_strike / (args.output_frame_count - 1)
    updated["ready_interval_frames"] = _mapped_interval(
        source_frames, metadata.get("ready_interval_frames")
    )
    updated["full_swing_interval_frames"] = _mapped_interval(
        source_frames, metadata.get("full_swing_interval_frames")
    )
    updated["post_quiet_interval_frames"] = _mapped_interval(
        source_frames, metadata.get("post_quiet_interval_frames")
    )
    for key in ("follow_through_end_frame", "recover_end_frame"):
        if key in metadata:
            updated[key] = _nearest_output_frame(
                source_frames, int(metadata[key])
            )

    pre_secant = (
        args.guard_start_frame - args.start_frame
    ) / args.output_guard_start_frame
    post_secant = (
        args.end_frame - args.guard_end_frame
    ) / (
        args.output_frame_count - 1 - args.output_guard_end_frame
    )
    timing_reference_artifacts = []
    for reference_path in timing_reference_paths:
        reference_sidecar = reference_path.with_suffix(".yaml")
        timing_reference_artifacts.append(
            {
                "motion": reference_path.name,
                "npz_sha256": _file_sha256(reference_path),
                "sidecar": reference_sidecar.name,
                "sidecar_sha256": _file_sha256(reference_sidecar),
            }
        )
    updated["retiming"] = {
        "method": (
            "shared_bounded_motion_aware_inverse_arc_length_with_unit_speed_strike_guard"
            if args.mapping_method == "motion_aware_arc_length"
            else "monotone_c1_cubic_hermite_with_unit_speed_strike_guard"
        ),
        "source_motion": input_path.name,
        "source_sidecar": input_sidecar.name,
        "source_npz_sha256": _file_sha256(input_path),
        "source_sidecar_sha256": _file_sha256(input_sidecar),
        "source_frame_range_zero_based_inclusive": [
            args.start_frame,
            args.end_frame,
        ],
        "source_knots_frames": [
            args.start_frame,
            args.guard_start_frame,
            args.guard_end_frame,
            args.end_frame,
        ],
        "output_knots_frames": [
            0,
            args.output_guard_start_frame,
            args.output_guard_end_frame,
            args.output_frame_count - 1,
        ],
        "outer_average_source_frames_per_output_frame": [
            pre_secant,
            post_secant,
        ],
        "timing_reference_artifacts": timing_reference_artifacts,
        "source_frame_map": [float(value) for value in source_frames],
        "guard_time_scale": 1.0,
        "source_strike_frame": source_strike,
        "output_strike_frame": output_strike,
        "position_interpolation": "linear",
        "quaternion_interpolation": "shortest_arc_slerp_wxyz",
        "velocities_recomputed": True,
        "endpoint_validation": {
            "max_abs_joint_speed_rad_s": endpoint_joint_speed,
            "limit_rad_s": float(args.max_endpoint_joint_speed),
            "max_racket_center_linear_speed_m_s": endpoint_racket_center_speed,
            "limit_m_s": float(args.max_endpoint_racket_speed),
        },
        "kinematic_diagnostics": {
            "max_abs_joint_speed_rad_s": max_joint_speed,
            "max_abs_joint_acceleration_rad_s2": max_joint_acceleration,
            "max_racket_center_linear_speed_m_s": max_racket_center_speed,
            "max_racket_center_linear_speed_frame": peak_racket_frame,
        },
        "strike_max_abs_error": strike_max_error,
    }
    if mapping_diagnostics is not None:
        updated["retiming"]["mapping_diagnostics"] = mapping_diagnostics
    else:
        updated["retiming"][
            "outer_endpoint_slopes_source_frames_per_output_frame"
        ] = [pre_secant, 1.0, 1.0, post_secant]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **retimed)
    with output_sidecar.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(updated, stream, sort_keys=False, allow_unicode=True)

    print(
        f"[retime_motion_npz] {input_path.name}[{args.start_frame}:{args.end_frame}] "
        f"-> {output_path.name}: frames={args.output_frame_count}, "
        f"duration={updated['duration_s']:.3f}s, strike={output_strike}, "
        f"guard=[{args.output_guard_start_frame}, {args.output_guard_end_frame}], "
        f"endpoint_joint={endpoint_joint_speed:.4f}rad/s, "
        f"endpoint_racket_center={endpoint_racket_center_speed:.4f}m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
