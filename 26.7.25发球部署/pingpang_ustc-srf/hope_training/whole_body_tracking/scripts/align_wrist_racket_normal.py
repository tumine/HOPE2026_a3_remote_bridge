#!/usr/bin/env python3
"""Align a motion's racket-face normal using only the three wrist joints.

The strike-frame correction is solved against the shipped A3 MuJoCo model, then
blended into a symmetric temporal window with a raised cosine.  The source
motion is never overwritten.  Joint/body velocities and tracked-body FK are
rebuilt so the output NPZ remains internally consistent.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

from crop_motion_npz import _angular_velocity, _gradient
from smooth_motion_npz import TIME_SERIES_KEYS, _forward_kinematics, _load_motion, _name_id


WRIST_JOINTS = (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
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
    parser.add_argument("--model-xml", default=_default_model_xml(), type=Path)
    parser.add_argument("--half-window-frames", default=30, type=int)
    parser.add_argument("--target-normal", default=(1.0, 0.0, 0.0), nargs=3, type=float)
    parser.add_argument("--motion-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class RacketKinematics:
    """Small MuJoCo FK adapter for the A3 racket site."""

    def __init__(
        self,
        model_xml: Path,
        joint_order: list[str],
        root_pos: np.ndarray,
        root_quat: np.ndarray,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        free_joint_id = _name_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint"
        )
        self.root_qadr = int(self.model.jnt_qposadr[free_joint_id])
        self.joint_qadr = np.asarray(
            [
                self.model.jnt_qposadr[
                    _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in joint_order
            ],
            dtype=int,
        )
        self.racket_site_id = _name_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket"
        )
        self.root_pos = root_pos
        self.root_quat = root_quat

    def evaluate(
        self, frame: int, joint_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.root_qadr : self.root_qadr + 3] = self.root_pos[frame]
        self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = self.root_quat[frame]
        self.data.qpos[self.joint_qadr] = joint_pos
        mujoco.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.racket_site_id].copy()
        # The imported A3 paddle is thinnest along local Y, so local +Y is its red-face normal.
        normal = (
            self.data.site_xmat[self.racket_site_id]
            .reshape(3, 3)[:, 1]
            .copy()
        )
        return position, normal

    def joint_bounds(self, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        limits = []
        for name in names:
            joint_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            limits.append(self.model.jnt_range[joint_id].copy())
        limits = np.asarray(limits)
        return limits[:, 0], limits[:, 1]


def _raised_cosine(frame_count: int, strike_frame: int, half_window: int) -> np.ndarray:
    if half_window <= 0:
        raise ValueError("--half-window-frames must be positive")
    frames = np.arange(frame_count)
    distance = np.abs(frames - strike_frame)
    weights = np.zeros(frame_count, dtype=np.float64)
    active = distance <= half_window
    weights[active] = 0.5 * (
        1.0 + np.cos(np.pi * distance[active] / float(half_window))
    )
    return weights


def _solve_wrist_delta(
    kinematics: RacketKinematics,
    strike_frame: int,
    source_joint_pos: np.ndarray,
    joint_order: list[str],
    target_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist_indices = np.asarray([joint_order.index(name) for name in WRIST_JOINTS])
    source_wrist = source_joint_pos[wrist_indices].astype(np.float64)
    lower, upper = kinematics.joint_bounds(WRIST_JOINTS)

    def residual(wrist: np.ndarray) -> np.ndarray:
        candidate = source_joint_pos.astype(np.float64, copy=True)
        candidate[wrist_indices] = wrist
        _, normal = kinematics.evaluate(strike_frame, candidate)
        # Normal alignment dominates; the small regularizer chooses the nearest wrist pose.
        return np.concatenate((10.0 * (normal - target_normal), 0.03 * (wrist - source_wrist)))

    result = least_squares(
        residual,
        source_wrist,
        bounds=(lower, upper),
        max_nfev=3000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    if not result.success:
        raise RuntimeError(f"wrist IK failed: {result.message}")

    corrected = source_joint_pos.astype(np.float64, copy=True)
    corrected[wrist_indices] = result.x
    strike_position, strike_normal = kinematics.evaluate(strike_frame, corrected)
    angle_deg = np.degrees(
        np.arccos(np.clip(float(np.dot(strike_normal, target_normal)), -1.0, 1.0))
    )
    if angle_deg > 0.1:
        raise RuntimeError(
            f"wrist IK left a {angle_deg:.4f} deg face-normal error; refusing output"
        )
    return result.x - source_wrist, strike_position, strike_normal


def _racket_trajectory(
    kinematics: RacketKinematics, joint_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((joint_pos.shape[0], 3), dtype=np.float64)
    normals = np.empty_like(positions)
    for frame in range(joint_pos.shape[0]):
        positions[frame], normals[frame] = kinematics.evaluate(frame, joint_pos[frame])
    return positions, normals


def main() -> int:
    args = _parse_args()
    input_path = args.input_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()
    output_sidecar = output_path.with_suffix(".yaml")
    model_xml = args.model_xml.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source motion")
    if not model_xml.is_file():
        raise FileNotFoundError(f"A3 MuJoCo model not found: {model_xml}")
    if (output_path.exists() or output_sidecar.exists()) and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")

    arrays, metadata = _load_motion(input_path)
    frame_count = int(arrays["joint_pos"].shape[0])
    fps = float(np.asarray(arrays["fps"]).item())
    dt = 1.0 / fps
    strike_frame = int(metadata.get("strike_frame", -1))
    if not 0 <= strike_frame < frame_count:
        raise ValueError(f"invalid strike_frame in sidecar: {strike_frame}")
    half_window = int(args.half_window_frames)
    if strike_frame - half_window < 0 or strike_frame + half_window >= frame_count:
        raise ValueError("the correction window extends outside the motion")

    joint_order = [str(name) for name in metadata["joint_order"]]
    missing_wrist = [name for name in WRIST_JOINTS if name not in joint_order]
    if missing_wrist:
        raise ValueError(f"motion joint_order is missing wrist joints: {missing_wrist}")
    tracked_bodies = [str(name) for name in metadata["tracked_bodies"]]
    root_body = str(metadata["root_body"])
    root_index = tracked_bodies.index(root_body)
    root_pos = arrays["body_pos_w"][:, root_index].astype(np.float64)
    root_quat = arrays["body_quat_w"][:, root_index].astype(np.float64)

    target_normal = np.asarray(args.target_normal, dtype=np.float64)
    target_norm = np.linalg.norm(target_normal)
    if target_norm < 1.0e-8:
        raise ValueError("--target-normal cannot be zero")
    target_normal /= target_norm

    kinematics = RacketKinematics(
        model_xml, joint_order, root_pos, root_quat
    )
    source_strike_position, source_strike_normal = kinematics.evaluate(
        strike_frame, arrays["joint_pos"][strike_frame]
    )
    wrist_delta, _, _ = _solve_wrist_delta(
        kinematics,
        strike_frame,
        arrays["joint_pos"][strike_frame],
        joint_order,
        target_normal,
    )

    weights = _raised_cosine(frame_count, strike_frame, half_window)
    corrected_joint_pos = arrays["joint_pos"].astype(np.float64, copy=True)
    wrist_indices = np.asarray([joint_order.index(name) for name in WRIST_JOINTS])
    corrected_joint_pos[:, wrist_indices] += weights[:, None] * wrist_delta[None, :]

    lower, upper = kinematics.joint_bounds(WRIST_JOINTS)
    corrected_wrist = corrected_joint_pos[:, wrist_indices]
    if np.any(corrected_wrist < lower[None, :] - 1.0e-8) or np.any(
        corrected_wrist > upper[None, :] + 1.0e-8
    ):
        raise ValueError("the blended wrist correction violates an A3 joint limit")

    body_pos, body_quat = _forward_kinematics(
        model_xml,
        root_pos,
        root_quat,
        corrected_joint_pos,
        joint_order,
        tracked_bodies,
        arrays["body_quat_w"],
    )
    racket_pos, racket_normal = _racket_trajectory(
        kinematics, corrected_joint_pos
    )
    racket_vel = _gradient(racket_pos, dt)
    corrected_strike_position = racket_pos[strike_frame]
    corrected_strike_velocity = racket_vel[strike_frame]
    corrected_strike_normal = racket_normal[strike_frame]
    corrected_angle_deg = np.degrees(
        np.arccos(
            np.clip(
                float(np.dot(corrected_strike_normal, target_normal)), -1.0, 1.0
            )
        )
    )

    corrected = {key: value.copy() for key, value in arrays.items()}
    corrected["joint_pos"] = corrected_joint_pos.astype(np.float32)
    corrected["joint_vel"] = _gradient(corrected_joint_pos, dt)
    corrected["body_pos_w"] = body_pos.astype(np.float32)
    corrected["body_quat_w"] = body_quat.astype(np.float32)
    corrected["body_lin_vel_w"] = _gradient(body_pos, dt)
    corrected["body_ang_vel_w"] = _angular_velocity(body_quat, dt)
    for key in TIME_SERIES_KEYS:
        if not np.isfinite(corrected[key]).all():
            raise ValueError(f"corrected {key} contains NaN or infinity")

    updated = copy.deepcopy(metadata)
    updated["name"] = args.motion_name or output_path.stem
    updated["strike_detection"] = "preserved_semantic_frame_after_wrist_face_normal_alignment"
    updated["racket_normal_correction"] = {
        "method": "wrist_only_least_squares_with_raised_cosine_blend",
        "source_motion": str(input_path),
        "model_xml": str(model_xml),
        "strike_frame": strike_frame,
        "blend_frame_range_zero_based_inclusive": [
            strike_frame - half_window,
            strike_frame + half_window,
        ],
        "blend_half_window_frames": half_window,
        "blend_half_window_seconds": half_window / fps,
        "modified_joints": list(WRIST_JOINTS),
        "wrist_delta_degrees_at_strike": np.degrees(wrist_delta).tolist(),
        "target_world_normal": target_normal.tolist(),
        "source_world_normal_at_strike": source_strike_normal.tolist(),
        "corrected_world_normal_at_strike": corrected_strike_normal.tolist(),
        "corrected_normal_error_degrees": float(corrected_angle_deg),
        "source_racket_position_at_strike_m": source_strike_position.tolist(),
        "corrected_racket_position_at_strike_m": corrected_strike_position.tolist(),
        "corrected_racket_velocity_at_strike_mps": corrected_strike_velocity.tolist(),
        "tracked_body_fk_recomputed": True,
        "velocities_recomputed": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **corrected)
    with output_sidecar.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(updated, stream, sort_keys=False, allow_unicode=True)

    print(
        f"[align_wrist_racket_normal] {input_path.name} -> {output_path.name}: "
        f"strike={strike_frame}, blend=[{strike_frame - half_window}, "
        f"{strike_frame + half_window}], delta_deg={np.degrees(wrist_delta)}, "
        f"position={corrected_strike_position}, velocity={corrected_strike_velocity}, "
        f"normal={corrected_strike_normal}, error={corrected_angle_deg:.6f}deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
