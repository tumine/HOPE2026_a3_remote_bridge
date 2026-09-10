# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration loader for the reference runner.

Reads ``config/hope_pingpong_runtime.yaml`` (the clean 111-D runtime config) and
resolves the ActionAdapter, Isaac-matched per-joint simulation PD gains, the
canonical-table to MuJoCo-world transform, and lifecycle timing into ready-to-use
values. All relative paths in the YAML are resolved against the YAML file's own
directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .action_adapter import ActionAdapter
from .joint_order import JOINT_NAMES, NUM_JOINTS
from .lifecycle import LifecycleConfig

# Legacy group ranges retained only so an older user-authored runtime YAML still
# loads.  The shipped config uses explicit per-joint gains matching Isaac.
_GROUP_RANGES = {
    "waist": range(0, 3),
    "neck": range(3, 5),
    "arm": range(5, 19),
    "leg": range(19, 31),
}


@dataclass(frozen=True)
class MujocoTableFrameConfig:
    """Pure translation from the canonical table frame to MuJoCo floor world."""

    near_edge_from_station_x: float = 0.5
    table_width: float = 1.525
    table_surface_z: float = 0.76

    def translation(self, station_xy) -> np.ndarray:
        """Return ``p_mujoco = p_table + translation`` for the nominal startup station."""

        station = np.asarray(station_xy, dtype=np.float64).reshape(2)
        return np.array(
            [
                station[0] + self.near_edge_from_station_x,
                station[1] + 0.5 * self.table_width,
                self.table_surface_z,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LateralStationConfig:
    """Training-matched inverse geometry for the moving base target."""

    enabled: bool = False
    base_target_y_range: tuple[float, float] = (-0.35, 0.6625)
    forehand_reach_y: float = -0.64
    backhand_reach_y: float = -0.10


@dataclass
class RuntimeConfig:
    control_hz: float
    onnx_path: Path
    model_xml_path: Path
    action_adapter: ActionAdapter
    sim_kp: np.ndarray
    sim_kd: np.ndarray
    mujoco_table_frame: MujocoTableFrameConfig
    lateral_station: LateralStationConfig
    lifecycle: LifecycleConfig
    passive_neck: bool = True
    config_dir: Path = field(default_factory=Path)

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        path = Path(path).resolve()
        cfg_dir = path.parent
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)

        norm = str(doc.get("observation_normalization", "none")).lower()
        if norm != "none":
            raise ValueError(
                f"observation_normalization must be 'none' (raw obs), got '{norm}'"
            )

        control_hz = float(doc.get("control_hz", 50.0))
        dt = 1.0 / control_hz

        onnx_path = _resolve(cfg_dir, doc["policy"]["onnx_path"])
        model_xml_path = _resolve(cfg_dir, doc["simulation"]["model_xml_path"])
        adapter_path = _resolve(cfg_dir, doc["action_adapter"]["config_path"])
        adapter = ActionAdapter.from_yaml(adapter_path)

        sim_doc = doc["simulation"]
        sim_kp, sim_kd = _expand_pd_gains(sim_doc["pd_gains"])
        frame_doc = sim_doc.get("table_frame", {})
        mujoco_table_frame = MujocoTableFrameConfig(
            near_edge_from_station_x=float(
                frame_doc.get("near_edge_from_station_x", 0.5)
            ),
            table_width=float(frame_doc.get("table_width", 1.525)),
            table_surface_z=float(frame_doc.get("table_surface_z", 0.76)),
        )
        station_doc = doc.get("lateral_station", {})
        station_range = station_doc.get("base_target_y_range", (-0.35, 0.6625))
        if len(station_range) != 2:
            raise ValueError("lateral_station.base_target_y_range must contain two values")
        lateral_station = LateralStationConfig(
            enabled=bool(station_doc.get("enabled", False)),
            base_target_y_range=(float(station_range[0]), float(station_range[1])),
            forehand_reach_y=float(station_doc.get("forehand_reach_y", -0.64)),
            backhand_reach_y=float(station_doc.get("backhand_reach_y", -0.10)),
        )
        if lateral_station.base_target_y_range[0] > lateral_station.base_target_y_range[1]:
            raise ValueError(
                "lateral_station.base_target_y_range must satisfy min <= max"
            )

        life_doc = doc.get("lifecycle", {})
        lifecycle = LifecycleConfig(
            dt=dt,
            follow_through_s=float(life_doc.get("follow_through_s", 0.8)),
            recovery_s=float(life_doc.get("recovery_s", 0.0)),
            ready_time_to_strike=float(life_doc.get("ready_time_to_strike", 1.0)),
            ready_target_rel_base_w=_optional_float3(
                life_doc.get("ready_target_rel_base_w"),
                "lifecycle.ready_target_rel_base_w",
            ),
            ready_target_vel_w=_float3(
                life_doc.get("ready_target_vel_w", (0.0, 0.0, 0.0)),
                "lifecycle.ready_target_vel_w",
            ),
            ready_swing_side=(
                int(life_doc["ready_swing_side"])
                if life_doc.get("ready_swing_side") is not None
                else None
            ),
            ready_reach_x=float(life_doc.get("ready_reach_x", 0.40)),
            ready_reach_y=float(life_doc.get("ready_reach_y", 0.20)),
            ready_reach_z=float(life_doc.get("ready_reach_z", -0.05)),
        )

        return cls(
            control_hz=control_hz,
            onnx_path=onnx_path,
            model_xml_path=model_xml_path,
            action_adapter=adapter,
            sim_kp=sim_kp,
            sim_kd=sim_kd,
            mujoco_table_frame=mujoco_table_frame,
            lateral_station=lateral_station,
            lifecycle=lifecycle,
            passive_neck=bool(doc.get("passive_neck", True)),
            config_dir=cfg_dir,
        )


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _float3(value, field_name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain exactly three finite values")
    return tuple(float(item) for item in array)


def _optional_float3(value, field_name: str) -> tuple[float, float, float] | None:
    return None if value is None else _float3(value, field_name)


def _expand_pd_gains(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expand explicit per-joint gains into canonical length-31 arrays.

    ``groups`` remains a compatibility fallback for older private configs.  The
    repository config uses ``joints`` so every value is pinned to the same named
    joint as Isaac rather than inheriting a coarse waist/arm/leg approximation.
    """

    kp = np.zeros(NUM_JOINTS, dtype=np.float64)
    kd = np.zeros(NUM_JOINTS, dtype=np.float64)
    joints = spec.get("joints")
    if joints is not None:
        missing = [name for name in JOINT_NAMES if name not in joints]
        extra = sorted(set(joints) - set(JOINT_NAMES))
        if missing or extra:
            raise ValueError(
                "simulation.pd_gains.joints must match the canonical joint order; "
                f"missing={missing}, extra={extra}"
            )
        for i, name in enumerate(JOINT_NAMES):
            gain = joints[name]
            kp[i] = float(gain["kp"])
            kd[i] = float(gain["kd"])
        return kp, kd

    groups = spec.get("groups", {})
    for name, rng in _GROUP_RANGES.items():
        if name not in groups:
            raise ValueError(f"simulation.pd_gains.groups is missing '{name}'")
        g = groups[name]
        for i in rng:
            kp[i] = float(g["kp"])
            kd[i] = float(g["kd"])
    return kp, kd
