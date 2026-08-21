# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo sim-to-sim evaluation of the exported HOPE ONNX policy.

This drives the exported ``hope_pingpong.onnx`` (single layout: observation[1, 111]
-> raw_action[1, 31], no observation normalization) through the SAME 111-D
observation builder, ActionAdapter, swing lifecycle and RacketCommand that the
clean-room reference deploy runner uses, so this is a faithful test of the deploy
contract rather than a second, divergent implementation.

``success_rate`` is measured from an ACTUAL simulated ball. A real ball (free joint,
sphere) is returned from the opponent side toward the robot; it clears the net,
physically bounces exactly once on the robot half, and then reaches the sampled
strike point under gravity + no-spin aerodynamic drag. It also really bounces off
the racket, table and net inside MuJoCo. A return succeeds when ALL of the following
are observed on the real trajectory:

  * **contact**   -- an actual MuJoCo contact between the ball geom and the racket
                     collision geom (or the two pass within the contact radius with
                     positive relative closing speed);
  * **net clear** -- after contact the ball's x crosses the net plane with its centre
                     above the net top; and
  * **opponent-half first bounce** -- the ball's first downward crossing of the table
                     surface (z reaches ball_radius descending) lands on the opponent
                     half, inside the table bounds.

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

There is NO analytic predicted-landing substitute: every event above is read off the
real simulated ball. Forehand, backhand and all incoming balls merge into one number.
Emits only ``{"success_rate": <float>}`` to stdout (and optionally --json-out); a
one-line human summary goes to stderr.

Two evaluation modes (``--eval-mode``):

  * ``continuous`` (default) — deploy-faithful continuous rally: the robot, policy
    ``last_action``, lifecycle and fixed station are initialized ONCE; between serves
    the policy keeps running through its follow-through/recovery with no state reset
    and no teleport, exactly like the deploy runner. The serve side follows the
    pattern FH, FH, BH, BH, ... so all four adjacent transitions (FH->FH, FH->BH,
    BH->BH, BH->FH) are exercised; the transitions actually seen are reported on
    stderr. A fall keeps affecting later serves — that is the honest continuous
    measurement.
  * ``independent`` — independent-strike evaluation: every serve resets the robot to
    its stand with a fresh lifecycle/last_action/station. This measures isolated
    swings only; it does NOT validate between-swing transitions.

Requires ``mujoco``, ``onnxruntime``, ``pyyaml`` and ``numpy`` plus the shipped
reference deploy package (``a3_deploy/a3_deploy_example``) and the
``a3_pingpong`` MJCF. Runs OUTSIDE Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "a3_deploy").is_dir() and (parent / "hope_training").is_dir():
            return parent
    return here.parents[3]


# --- deploy-package override ------------------------------------------------ #
# When this script is shipped inside a standalone deploy package (sim/deps/),
# resolve the packaged dependencies instead of a full training checkout.
def _deploy_pkg_dir() -> pathlib.Path | None:
    here = pathlib.Path(__file__).resolve()
    sim_dir = here.parent
    if (sim_dir / "deps" / "success_metric.py").is_file() and (
        sim_dir / "deps" / "hope_planner"
    ).is_dir():
        return sim_dir.parent
    return None


def _deploy_aware_root() -> tuple[pathlib.Path, str]:
    pkg = _deploy_pkg_dir()
    if pkg is not None:
        return pkg, "deploy"
    return _repo_root(), "repo"


def _reference_pkg_dir(repo_root: pathlib.Path) -> pathlib.Path:
    pkg = _deploy_pkg_dir()
    if pkg is not None:
        return pkg / "reference"
    return repo_root / "a3_deploy" / "a3_deploy_example" / "reference"


def _default_runtime_config(repo_root: pathlib.Path) -> pathlib.Path:
    pkg = _deploy_pkg_dir()
    if pkg is not None:
        return pkg / "config" / "hope_pingpong_runtime.yaml"
    return repo_root / "a3_deploy" / "a3_deploy_example" / "config" / "hope_pingpong_runtime.yaml"


def _load_success_metric(repo_root: pathlib.Path):
    """Import the shared success_metric module by file path (pure NumPy).

    Deploy package: loads sim/deps/success_metric.py directly.
    Full training checkout: loads the training-package source file standalone
    (importing it as a package would execute the training __init__ and drag in
    Isaac Lab gym registration).
    """
    import importlib.util

    pkg = _deploy_pkg_dir()
    if pkg is not None:
        path = pkg / "sim" / "deps" / "success_metric.py"
    else:
        path = (
            repo_root / "hope_training" / "whole_body_tracking" / "source"
            / "whole_body_tracking" / "whole_body_tracking" / "utils"
            / "success_metric.py"
        )
    spec = importlib.util.spec_from_file_location("hope_success_metric", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (sys.modules[__module__]) works.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: str | pathlib.Path) -> str:
    """Return a streaming SHA-256 digest for evaluation provenance."""

    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stand_base_xy(robot_xml_path):
    """Read the robot MJCF stand-keyframe base XY before placing station-relative scenery."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")
    if jid < 0:
        raise ValueError("MuJoCo model has no pelvis_free_joint for station alignment")
    qadr = int(model.jnt_qposadr[jid])
    return data.qpos[qadr:qadr + 2].copy()


@dataclass(frozen=True)
class _TableBounceModel:
    """No-spin table contact and geometry used by the legal incoming-ball solve."""

    horizontal_retention: float
    vertical_restitution: float
    tangential_cap: float
    length: float
    width: float
    net_x: float
    net_clear_z: float
    ball_radius: float

    @classmethod
    def from_config(cls, cfg: dict) -> "_TableBounceModel":
        table = cfg.get("table", {})
        net = cfg.get("net", {})
        ball = cfg.get("ball", {})
        contact = cfg.get("contact", {}).get("table", {})
        radius = float(ball.get("radius", 0.020))
        restitution = float(contact.get("restitution", 0.9215))
        damping = float(contact.get("tangential_damping", 0.369))
        model = cls(
            horizontal_retention=1.0 - damping,
            vertical_restitution=restitution,
            tangential_cap=float(contact.get("tangential_cap", 2.0)),
            length=float(table.get("length", 2.74)),
            width=float(table.get("width", 1.525)),
            net_x=float(
                net.get(
                    "x_position",
                    0.5 * float(table.get("length", 2.74)),
                )
            ),
            net_clear_z=float(net.get("height", 0.1525)) + radius,
            ball_radius=radius,
        )
        if not (
            0.0 < model.horizontal_retention <= 1.0
            and 0.0 < model.vertical_restitution <= 1.0
            and model.tangential_cap >= 0.0
            and model.length > 0.0
            and model.width > 0.0
            and 0.0 < model.net_x < model.length
            and model.ball_radius > 0.0
        ):
            raise ValueError(f"invalid table-bounce model: {model!r}")
        return model

    def apply(self, incoming: np.ndarray) -> np.ndarray:
        incoming = np.asarray(incoming, dtype=np.float64)
        horizontal = incoming[:2]
        speed = float(np.linalg.norm(horizontal))
        removed = min(
            (1.0 - self.horizontal_retention) * speed,
            self.tangential_cap
            * (1.0 + self.vertical_restitution)
            * abs(float(incoming[2])),
        )
        scale = max(0.0, 1.0 - removed / speed) if speed > 1.0e-12 else 0.0
        return np.array(
            [
                horizontal[0] * scale,
                horizontal[1] * scale,
                -self.vertical_restitution * incoming[2],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class _BounceEvent:
    time_s: float
    position_table: np.ndarray
    incoming_velocity: np.ndarray
    outgoing_velocity: np.ndarray


@dataclass(frozen=True)
class _BouncedFlight:
    position_table: np.ndarray
    velocity: np.ndarray
    bounces: tuple[_BounceEvent, ...]
    incoming_net_crossings_z: tuple[float, ...]


def _integrate_bounced_flight(
    origin_table,
    initial_velocity,
    flight_t,
    physics,
    table_bounce: _TableBounceModel,
    integration_dt=0.002,
) -> _BouncedFlight:
    """Integrate flight plus fitted table impulses in the canonical table frame."""

    p = np.asarray(origin_table, dtype=np.float64).copy()
    v = np.asarray(initial_velocity, dtype=np.float64).copy()
    total_time = float(flight_t)
    elapsed = 0.0
    bounces: list[_BounceEvent] = []
    incoming_net_crossings_z: list[float] = []
    while elapsed < total_time - 1.0e-12:
        dt = min(float(integration_dt), total_time - elapsed)
        a = physics.acceleration(v)
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt

        if p[0] > table_bounce.net_x >= p_new[0]:
            dx = p[0] - p_new[0]
            frac = (
                (p[0] - table_bounce.net_x) / dx
                if abs(dx) > 1.0e-12
                else 0.5
            )
            incoming_net_crossings_z.append(
                float(p[2] + frac * (p_new[2] - p[2]))
            )

        if (
            p[2] > table_bounce.ball_radius >= p_new[2]
            and v_new[2] < 0.0
        ):
            dz = p[2] - p_new[2]
            frac = (
                (p[2] - table_bounce.ball_radius) / dz
                if dz > 1.0e-12
                else 0.5
            )
            position = p + frac * (p_new - p)
            on_table = (
                0.0 <= position[0] <= table_bounce.length
                and -table_bounce.width <= position[1] <= 0.0
            )
            if on_table:
                position[2] = table_bounce.ball_radius
                incoming = v + a * (frac * dt)
                outgoing = table_bounce.apply(incoming)
                bounces.append(
                    _BounceEvent(
                        time_s=elapsed + frac * dt,
                        position_table=position.copy(),
                        incoming_velocity=incoming.copy(),
                        outgoing_velocity=outgoing.copy(),
                    )
                )
                remaining_dt = (1.0 - frac) * dt
                a_post = physics.acceleration(outgoing)
                p_new = (
                    position
                    + outgoing * remaining_dt
                    + 0.5 * a_post * remaining_dt * remaining_dt
                )
                v_new = outgoing + a_post * remaining_dt

        p, v = p_new, v_new
        elapsed += dt

    return _BouncedFlight(
        position_table=p,
        velocity=v,
        bounces=tuple(bounces),
        incoming_net_crossings_z=tuple(incoming_net_crossings_z),
    )


def _legal_single_bounce(
    flight: _BouncedFlight,
    target_table: np.ndarray,
    table_bounce: _TableBounceModel,
    *,
    target_tolerance=2.0e-4,
) -> bool:
    return bool(
        np.linalg.norm(flight.position_table - target_table) <= target_tolerance
        and len(flight.bounces) == 1
        and 0.0 < flight.bounces[0].position_table[0] < table_bounce.net_x
        and len(flight.incoming_net_crossings_z) == 1
        and flight.incoming_net_crossings_z[0] > table_bounce.net_clear_z
    )


def _solve_bounced_serve_velocity(
    origin_table,
    strike_table,
    flight_t,
    physics,
    table_bounce: _TableBounceModel,
):
    """Shoot a legal opponent return: clear net, bounce once near-side, hit target."""

    origin = np.asarray(origin_table, dtype=np.float64)
    target = np.asarray(strike_table, dtype=np.float64)
    flight_t = float(flight_t)
    best_error = float("inf")
    best_flight = None
    # Different vertical guesses select different roots of the hybrid bounce
    # trajectory. Trying a bounded set is deterministic and avoids converging to
    # the physically wrong two-bounce solution.
    primary_fractions = (0.60, 0.65, 0.70, 0.55, 0.50, 0.75)
    dense_fractions = tuple(float(value) for value in np.arange(0.25, 0.751, 0.025))
    bounce_fractions = primary_fractions + tuple(
        value
        for value in dense_fractions
        if not any(np.isclose(value, primary) for primary in primary_fractions)
    )
    for bounce_fraction in bounce_fractions:
        bounce_time = bounce_fraction * flight_t
        velocity = (target - origin) / flight_t
        velocity[2] = (
            table_bounce.ball_radius
            - origin[2]
            + 0.5 * physics.gravity * bounce_time * bounce_time
        ) / bounce_time

        for _ in range(18):
            flight = _integrate_bounced_flight(
                origin, velocity, flight_t, physics, table_bounce
            )
            error = target - flight.position_table
            error_norm = float(np.linalg.norm(error))
            if error_norm < best_error:
                best_error = error_norm
                best_flight = flight
            if error_norm < 1.0e-5:
                if _legal_single_bounce(flight, target, table_bounce):
                    return velocity, flight
                break

            jacobian = np.empty((3, 3), dtype=np.float64)
            epsilon = 1.0e-3
            for axis in range(3):
                perturbed = velocity.copy()
                perturbed[axis] += epsilon
                reached = _integrate_bounced_flight(
                    origin, perturbed, flight_t, physics, table_bounce
                ).position_table
                jacobian[:, axis] = (
                    reached - flight.position_table
                ) / epsilon
            try:
                correction = np.linalg.solve(jacobian, error)
            except np.linalg.LinAlgError:
                break
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > 5.0:
                correction *= 5.0 / correction_norm
            velocity += correction

    bounce_summary = (
        "none"
        if best_flight is None
        else [
            event.position_table.tolist()
            for event in best_flight.bounces
        ]
    )
    raise RuntimeError(
        "could not solve a legal one-bounce incoming trajectory: "
        f"best_target_error={best_error:.6g} m, bounces={bounce_summary}"
    )


def _sample_serve(
    rng,
    side,
    scene,
    physics,
    table_bounce,
    args,
    fixed_station_xy,
):
    """Sample a target and a legal physical opponent return through it.

    The strike point is sampled from the exact station-relative end-effector target
    boxes used in Isaac Lab training. The ball starts above the opponent half,
    clears the net, physically bounces exactly once on the robot half, and then
    reaches the sampled point under the shared drag + fitted table-contact model.
    """
    station_xy = np.asarray(fixed_station_xy, dtype=np.float64)
    margin = float(args.strike_range_margin)
    if margin < 0.0:
        raise ValueError("strike_range_margin must be non-negative")

    def _inset_uniform(lower, upper, label):
        lower = float(lower) + margin
        upper = float(upper) - margin
        if lower > upper:
            raise ValueError(f"{label} range is too narrow for strike_range_margin={margin}")
        return rng.uniform(lower, upper)

    if side >= 0:
        x_limits = (args.forehand_strike_forward_min, args.forehand_strike_forward_max)
        y_limits = (args.forehand_strike_lateral_min, args.forehand_strike_lateral_max)
        z_limits = (args.forehand_strike_height_min, args.forehand_strike_height_max)
        label = "forehand"
    else:
        x_limits = (args.backhand_strike_forward_min, args.backhand_strike_forward_max)
        y_limits = (args.backhand_strike_lateral_min, args.backhand_strike_lateral_max)
        z_limits = (args.backhand_strike_height_min, args.backhand_strike_height_max)
        label = "backhand"

    # Backward-compatible shared/symmetric overrides. The new per-side defaults are needed because
    # the retargeted forehand and backhand clips do not have symmetric natural strike locations.
    legacy_forward = (args.strike_forward_min, args.strike_forward_max)
    if any(value is not None for value in legacy_forward):
        if not all(value is not None for value in legacy_forward):
            raise ValueError("--strike-forward-min and --strike-forward-max must be supplied together")
        x_limits = legacy_forward
    legacy_lateral = (args.strike_lateral_offset, args.strike_lateral_jitter)
    if any(value is not None for value in legacy_lateral):
        if not all(value is not None for value in legacy_lateral):
            raise ValueError("--strike-lateral-offset and --strike-lateral-jitter must be supplied together")
        lateral_center = -float(args.strike_lateral_offset) if side >= 0 else float(args.strike_lateral_offset)
        lateral_jitter = float(args.strike_lateral_jitter)
        y_limits = (lateral_center - lateral_jitter, lateral_center + lateral_jitter)

    strike_x = station_xy[0] + _inset_uniform(
        x_limits[0], x_limits[1], f"{label}_strike_forward"
    )
    y = station_xy[1] + _inset_uniform(
        y_limits[0], y_limits[1], f"{label}_strike_lateral"
    )
    z = _inset_uniform(
        z_limits[0], z_limits[1], f"{label}_strike_height"
    )
    strike_pt = np.array([strike_x, y, z], dtype=np.float64)

    origin_y = float(
        np.clip(
            y + rng.uniform(-0.15, 0.15),
            scene.offset[1] - table_bounce.width + 2.0 * table_bounce.ball_radius,
            scene.offset[1] - 2.0 * table_bounce.ball_radius,
        )
    )
    origin = np.array(
        [
            scene.near_edge_x + rng.uniform(1.9, 2.4),
            origin_y,
            scene.table_height + rng.uniform(0.15, 0.35),
        ],
        dtype=np.float64,
    )
    if args.training_aligned_preroll:
        # The policy clock advances in exact 20 ms samples. Quantizing the physical
        # flight time lets launch occur at the same actor sample without a hidden
        # fractional-tick TTS jump.
        min_tick = int(np.ceil(args.serve_flight_time_min / args.control_dt_hint))
        max_tick = int(np.floor(args.serve_flight_time_max / args.control_dt_hint))
        if min_tick > max_tick:
            raise ValueError("serve flight-time range contains no whole control tick")
        # Random permutation makes the selected duration uniform over the
        # physically feasible ticks. An unreachable tick never causes target
        # resampling, which would silently bias the strike-position distribution.
        candidate_flight_times = [
            float(tick * args.control_dt_hint)
            for tick in rng.permutation(np.arange(min_tick, max_tick + 1))
        ]
    else:
        candidate_flight_times = [
            float(rng.uniform(args.serve_flight_time_min, args.serve_flight_time_max))
        ]
    origin_table = scene.to_table(origin)
    strike_table = scene.to_table(strike_pt)
    solve_errors = []
    for flight_t in candidate_flight_times:
        try:
            serve_vel, solved = _solve_bounced_serve_velocity(
                origin_table,
                strike_table,
                flight_t,
                physics,
                table_bounce,
            )
            break
        except RuntimeError as exc:
            solve_errors.append(f"{flight_t:.3f}s: {exc}")
    else:
        raise RuntimeError(
            "no legal one-bounce trajectory exists within the configured flight "
            f"time range for origin_table={origin_table.tolist()}, "
            f"strike_table={strike_table.tolist()}; attempts: "
            + " | ".join(solve_errors)
        )
    bounce = solved.bounces[0]
    return (
        strike_pt,
        origin,
        serve_vel,
        flight_t,
        solved.velocity,
        {
            "time_s": float(bounce.time_s),
            "position_table": bounce.position_table.tolist(),
            "incoming_velocity": bounce.incoming_velocity.tolist(),
            "outgoing_velocity": bounce.outgoing_velocity.tolist(),
            "incoming_net_crossing_z": float(
                solved.incoming_net_crossings_z[0]
            ),
        },
    )


def _whole_control_ticks(seconds: float, dt: float, flag: str) -> int:
    """Convert seconds to an exact non-negative control-tick count."""

    ticks = int(round(float(seconds) / float(dt)))
    if ticks < 0 or not np.isclose(seconds / dt, ticks, atol=1.0e-9):
        raise ValueError(f"{flag} must be a non-negative integer number of control ticks")
    return ticks


def _sample_training_hold_ticks(
    rng,
    *,
    hold_min_seconds: float,
    hold_max_seconds: float,
    stand_min_hold_seconds: float,
    dt: float,
    stand_start: bool,
) -> int:
    """Match Isaac's inclusive ``randint`` followed by stand-start clamping."""

    minimum = _whole_control_ticks(hold_min_seconds, dt, "--hold-min-seconds")
    maximum = _whole_control_ticks(hold_max_seconds, dt, "--hold-max-seconds")
    stand_minimum = _whole_control_ticks(
        stand_min_hold_seconds, dt, "--stand-min-hold-seconds"
    )
    if maximum < minimum:
        raise ValueError("hold range must satisfy min <= max")
    sampled = int(rng.integers(minimum, maximum + 1))
    return max(sampled, stand_minimum) if stand_start else sampled


def _load_legacy_training_contract(
    path,
    *,
    repo_root: pathlib.Path,
    control_dt: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Load the frozen old-policy boxes and derive its reference-clock timing.

    Returns ``(position_boxes, velocity_boxes, metadata)`` and mutates no
    evaluator defaults. Position boxes are applied by
    :func:`_apply_position_boxes`.
    """

    import yaml

    path = pathlib.Path(path).resolve()
    with open(path, "r", encoding="utf-8") as stream:
        doc = yaml.safe_load(stream) or {}

    def _box(section: str, side_name: str) -> np.ndarray:
        source = doc.get(section, {}).get(side_name, {})
        try:
            out = np.asarray([source["x"], source["y"], source["z"]], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: {section}.{side_name} must define x/y/z [min,max]"
            ) from exc
        if out.shape != (3, 2) or not np.all(np.isfinite(out)) or np.any(out[:, 0] > out[:, 1]):
            raise ValueError(f"{path}: invalid {section}.{side_name} box {out!r}")
        return out

    position_boxes = {
        1: _box("racket_position_box", "forehand"),
        -1: _box("racket_position_box", "backhand"),
    }
    velocity_boxes = {
        1: _box("racket_velocity_box", "forehand"),
        -1: _box("racket_velocity_box", "backhand"),
    }

    motion_doc = doc.get("motion", {})
    phases = motion_doc.get("strike_phase_per_clip", ())
    if not isinstance(phases, (list, tuple)) or len(phases) != 2:
        raise ValueError(
            f"{path}: motion.strike_phase_per_clip must contain forehand/backhand phases"
        )

    timing = []
    resolved_motions = {}
    motion_artifacts = {}
    for index, side_name in enumerate(("forehand", "backhand")):
        raw_motion_path = motion_doc.get(side_name)
        if not raw_motion_path:
            raise ValueError(f"{path}: motion.{side_name} is required")
        raw_motion_path = pathlib.Path(str(raw_motion_path))
        candidates = (
            [raw_motion_path]
            if raw_motion_path.is_absolute()
            else [repo_root / raw_motion_path, path.parent / raw_motion_path]
        )
        motion_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if motion_path is None:
            raise FileNotFoundError(
                f"{path}: cannot resolve legacy motion {raw_motion_path}; tried "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        with np.load(motion_path) as motion:
            frame_count = int(motion["joint_pos"].shape[0])
            fps = float(np.asarray(motion["fps"]).reshape(()))
        if frame_count < 2 or fps <= 0.0:
            raise ValueError(f"{motion_path}: invalid frame_count={frame_count}, fps={fps}")
        if not np.isclose(1.0 / fps, control_dt, atol=1.0e-12):
            raise ValueError(
                f"{motion_path}: legacy motion dt={1.0 / fps} does not match "
                f"runtime control dt={control_dt}"
            )
        phase = float(phases[index])
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"{path}: invalid {side_name} strike phase {phase}")
        strike_frame = int(round(phase * (frame_count - 1)))
        timing.append(
            (
                strike_frame / fps,
                (frame_count - 1 - strike_frame) / fps,
                frame_count,
                strike_frame,
            )
        )
        resolved_motions[side_name] = str(motion_path)
        motion_artifact = {
            "path": str(motion_path),
            "sha256": _file_sha256(motion_path),
        }
        sidecar_path = motion_path.with_suffix(".yaml")
        if sidecar_path.is_file():
            motion_artifact["sidecar_path"] = str(sidecar_path)
            motion_artifact["sidecar_sha256"] = _file_sha256(sidecar_path)
        motion_artifacts[side_name] = motion_artifact

    first_timing = np.asarray(timing[0][:2], dtype=np.float64)
    second_timing = np.asarray(timing[1][:2], dtype=np.float64)
    if not np.allclose(first_timing, second_timing, atol=1.0e-12):
        raise ValueError(
            f"{path}: forehand/backhand legacy clocks differ: {timing[0]} vs {timing[1]}"
        )

    metadata = {
        "path": str(path),
        "sha256": _file_sha256(path),
        "policy_generation": str(doc.get("policy_generation", "unknown")),
        "source_run": str(doc.get("source_run", "unknown")),
        "position_boxes": {str(k): v.tolist() for k, v in position_boxes.items()},
        "velocity_boxes": {str(k): v.tolist() for k, v in velocity_boxes.items()},
        "motion_paths": resolved_motions,
        "motion_artifacts": motion_artifacts,
        "frame_count": int(timing[0][2]),
        "strike_frame": int(timing[0][3]),
        "policy_lead_time_s": float(timing[0][0]),
        "follow_through_s": float(timing[0][1]),
    }
    return position_boxes, velocity_boxes, metadata


def _apply_position_boxes(args, boxes: dict[int, np.ndarray]) -> None:
    """Apply frozen per-side position boxes to the existing sampler arguments."""

    forehand = boxes[1]
    backhand = boxes[-1]
    (
        args.forehand_strike_forward_min,
        args.forehand_strike_forward_max,
    ) = forehand[0]
    (
        args.forehand_strike_lateral_min,
        args.forehand_strike_lateral_max,
    ) = forehand[1]
    (
        args.forehand_strike_height_min,
        args.forehand_strike_height_max,
    ) = forehand[2]
    (
        args.backhand_strike_forward_min,
        args.backhand_strike_forward_max,
    ) = backhand[0]
    (
        args.backhand_strike_lateral_min,
        args.backhand_strike_lateral_max,
    ) = backhand[1]
    (
        args.backhand_strike_height_min,
        args.backhand_strike_height_max,
    ) = backhand[2]


def _reject_legacy_shared_overrides(args) -> None:
    """Keep the frozen legacy evaluation profile internally consistent."""

    legacy_shared_overrides = {
        "--strike-forward-min": args.strike_forward_min,
        "--strike-forward-max": args.strike_forward_max,
        "--strike-lateral-offset": args.strike_lateral_offset,
        "--strike-lateral-jitter": args.strike_lateral_jitter,
    }
    conflicting = [
        flag for flag, value in legacy_shared_overrides.items() if value is not None
    ]
    if conflicting:
        raise ValueError(
            "--legacy-training-contract is a frozen evaluation profile and "
            "cannot be combined with shared strike-box overrides: "
            + ", ".join(conflicting)
        )


@dataclass
class _EvaluationDiagnostics:
    """Detailed evidence kept separate from the stable success-only stdout."""

    command_profile: str
    policy_lead_time_s: float
    training_aligned_preroll: bool
    attempts: int = 0
    mujoco_contacts: int = 0
    proximity_contacts: int = 0
    net_clears: int = 0
    opponent_bounces: int = 0
    legal_incoming_bounces: int = 0
    falls: int = 0
    nonfinite_observations: int = 0
    nonfinite_actions: int = 0
    saturated_action_ticks: int = 0
    policy_ticks: int = 0
    max_abs_raw_action: float = 0.0
    min_pelvis_height_m: float = float("inf")
    max_station_drift_m: float = 0.0
    trials: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        total_contacts = self.mujoco_contacts + self.proximity_contacts
        return {
            "command_profile": self.command_profile,
            "policy_lead_time_s": self.policy_lead_time_s,
            "training_aligned_preroll": self.training_aligned_preroll,
            "attempts": self.attempts,
            "mujoco_contacts": self.mujoco_contacts,
            "proximity_contacts": self.proximity_contacts,
            "contact_rate": total_contacts / max(self.attempts, 1),
            "net_clears": self.net_clears,
            "net_clear_rate": self.net_clears / max(self.attempts, 1),
            "opponent_bounces": self.opponent_bounces,
            "legal_incoming_bounces": self.legal_incoming_bounces,
            "legal_incoming_bounce_rate": self.legal_incoming_bounces
            / max(self.attempts, 1),
            "success_rate": self.opponent_bounces / max(self.attempts, 1),
            "falls": self.falls,
            "nonfinite_observations": self.nonfinite_observations,
            "nonfinite_actions": self.nonfinite_actions,
            "saturated_action_ticks": self.saturated_action_ticks,
            "policy_ticks": self.policy_ticks,
            "action_saturation_fraction": self.saturated_action_ticks
            / max(self.policy_ticks, 1),
            "max_abs_raw_action": self.max_abs_raw_action,
            "min_pelvis_height_m": (
                self.min_pelvis_height_m
                if np.isfinite(self.min_pelvis_height_m)
                else None
            ),
            "max_station_drift_m": self.max_station_drift_m,
            "trials": self.trials,
        }


def _predict_command(
    ball_pos,
    ball_vel,
    side,
    scene,
    strike_x,
    task_id,
    revision,
    RacketCommand,
    trajectory_predictor,
    racket_planner,
):
    """Run the shipped bounce-aware planner on the true MuJoCo ball state."""

    strike_x_table = float(strike_x - scene.offset[0])
    strike = trajectory_predictor.predict(
        scene.to_table(ball_pos),
        np.asarray(ball_vel, dtype=np.float64),
        0.0,
        x_hit=strike_x_table,
    )
    if not strike.valid:
        return None
    target_pos = np.asarray(strike.p_ball, dtype=np.float64) + scene.offset
    planner_command = racket_planner.plan(
        strike
    )

    return RacketCommand(
        task_id=task_id,
        task_revision=revision,
        swing_side=side,
        position=target_pos,
        velocity=planner_command.v_racket,
        time_to_strike=float(strike.t_strike),
    )


def run_eval(args) -> dict:
    repo_root, root_kind = _deploy_aware_root()

    # Make the reference deploy package importable (shared 111-D obs / ActionAdapter /
    # lifecycle / RacketCommand / ONNX wrapper).
    ref_dir = pathlib.Path(args.reference_dir) if args.reference_dir else _reference_pkg_dir(repo_root)
    sys.path.insert(0, str(ref_dir))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.joint_order import HEAD_INDICES, JOINT_NAMES
    from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
    from a3_deploy_onnx_ref_pingpong.observation import build_observation
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.racket_command import (
        BACKHAND,
        FOREHAND,
        QueueRacketCommandSource,
        RacketCommand,
    )

    # Shared success metric + physics config (pure NumPy; the DEFINITION only -- fed
    # real simulated events here, never an analytic predicted-landing rollout). Loaded
    # directly by file path so importing it does NOT drag in the Isaac Lab training
    # package (its __init__ registers gym tasks that need gymnasium/Isaac).
    metric = _load_success_metric(repo_root)
    BallPhysics = metric.BallPhysics
    PaddlePhysics = metric.PaddlePhysics
    SuccessRate = metric.SuccessRate
    TableGeometry = metric.TableGeometry
    load_ball_physics_config = metric.load_ball_physics_config
    predict_paddle_contact = metric.predict_paddle_contact

    # The MuJoCo scene builder lives next to this script.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    runtime_config_path = pathlib.Path(
        args.runtime_config or _default_runtime_config(repo_root)
    ).resolve()
    runtime_cfg = RuntimeConfig.load(runtime_config_path)
    onnx_path = pathlib.Path(args.onnx or runtime_cfg.onnx_path).resolve()
    robot_xml = pathlib.Path(args.model_xml or runtime_cfg.model_xml_path).resolve()
    dt = runtime_cfg.control_dt
    # Used by the serve sampler before it otherwise has access to runtime_cfg.
    args.control_dt_hint = dt
    if args.num_serves < 0:
        raise ValueError("--num-serves must be non-negative")
    if args.hold_min_seconds < 0.0 or args.hold_max_seconds < args.hold_min_seconds:
        raise ValueError("hold range must satisfy 0 <= min <= max")
    if (
        args.stand_min_hold_seconds < 0.0
        or args.stand_min_hold_seconds > args.hold_max_seconds
    ):
        raise ValueError("--stand-min-hold-seconds must lie within the hold range")
    if (
        args.serve_flight_time_min <= 0.0
        or args.serve_flight_time_max < args.serve_flight_time_min
    ):
        raise ValueError("serve flight-time range must be positive and ordered")
    legacy_velocity_boxes = None
    profile_metadata = {"name": "current_training_planner"}
    if args.legacy_training_contract:
        _reject_legacy_shared_overrides(args)
        position_boxes, legacy_velocity_boxes, contract_metadata = (
            _load_legacy_training_contract(
                args.legacy_training_contract,
                repo_root=repo_root,
                control_dt=dt,
            )
        )
        _apply_position_boxes(args, position_boxes)
        # A legacy policy must see the exact clock on which it was trained even
        # though the repository runtime now defaults to the guarded 1.8 s motions.
        args.policy_lead_time = float(contract_metadata["policy_lead_time_s"])
        runtime_cfg.lifecycle.ready_time_to_strike = args.policy_lead_time
        runtime_cfg.lifecycle.follow_through_s = float(
            contract_metadata["follow_through_s"]
        )
        profile_metadata = {
            "name": "legacy_independent_velocity",
            **contract_metadata,
        }
    profile_metadata["artifacts"] = {
        "onnx": {
            "path": str(onnx_path),
            "sha256": _file_sha256(onnx_path),
        },
        "runtime_config": {
            "path": str(runtime_config_path),
            "sha256": _file_sha256(runtime_config_path),
        },
        "model_xml": {
            "path": str(robot_xml),
            "sha256": _file_sha256(robot_xml),
        },
    }

    if args.policy_lead_time <= 0.0:
        raise ValueError("--policy-lead-time must be positive")
    if args.training_aligned_preroll and not np.isclose(
        args.policy_lead_time / dt, round(args.policy_lead_time / dt), atol=1.0e-9
    ):
        raise ValueError(
            "--policy-lead-time must be an integer number of control ticks when "
            "--training-aligned-preroll is enabled"
        )
    if args.training_aligned_preroll or args.no_serve:
        for flag, seconds in (
            ("--hold-min-seconds", args.hold_min_seconds),
            ("--hold-max-seconds", args.hold_max_seconds),
            ("--stand-min-hold-seconds", args.stand_min_hold_seconds),
        ):
            _whole_control_ticks(seconds, dt, flag)

    ball_cfg = load_ball_physics_config()
    physics = BallPhysics.from_config(ball_cfg)
    paddle_physics = PaddlePhysics.from_config(ball_cfg)
    table = TableGeometry.from_config(ball_cfg)
    table_bounce = _TableBounceModel.from_config(ball_cfg)
    accumulator = SuccessRate()

    # Pure NumPy planner package (no ROS node import): target generation must use
    # the same drag + paddle-contact inversion as training/deployment.
    if root_kind == "deploy":
        planner_pkg = repo_root / "sim" / "deps" / "hope_planner"
        sys.path.insert(0, str(planner_pkg.parent))  # insert parent so `import hope_planner` resolves
    else:
        planner_pkg = repo_root / "hope_ws" / "src" / "hope_planner"
        sys.path.insert(0, str(planner_pkg))
    from hope_planner.ball_trajectory_predictor import (
        BallTrajectoryPredictor,
        StrikeTarget,
    )
    from hope_planner.constants import (
        PlannerConfig,
        load_ball_physics,
        load_paddle_params,
        load_table_params,
    )
    from hope_planner.racket_target_planner import RacketTargetPlanner

    planner_cfg = PlannerConfig(**load_paddle_params())
    planner_ball_physics = load_ball_physics()
    planner_table = load_table_params()
    trajectory_predictor = BallTrajectoryPredictor(
        planner_ball_physics, planner_cfg, planner_table
    )
    racket_planner = RacketTargetPlanner(
        planner_ball_physics, planner_cfg, planner_table
    )

    policy = OnnxPolicy(str(onnx_path))
    default_q = runtime_cfg.action_adapter.default_q.copy()
    stand_base_xy = _stand_base_xy(robot_xml)
    # Isaac training defines table_near_x relative to the fixed startup station.
    # Convert that contract into MuJoCo world coordinates from the actual MJCF keyframe.
    near_edge_x = (
        float(args.near_edge_x)
        if args.near_edge_x is not None
        else float(stand_base_xy[0] + args.table_near_station_x)
    )
    scene = PingPongRealPhysicsScene(
        str(robot_xml),
        ball_cfg,
        JOINT_NAMES,
        control_dt=runtime_cfg.control_dt,
        near_edge_x=near_edge_x,
        table_center_y=float(stand_base_xy[1]),
        launch_viewer=args.view,
        paddle_contact_model=predict_paddle_contact,
        paddle_physics=paddle_physics,
        reset_joint_pos=default_q,
        video_path=args.video_out,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=1.0 / runtime_cfg.control_dt,
    )

    kp, kd = runtime_cfg.sim_kp.copy(), runtime_cfg.sim_kd.copy()
    head_idx = list(HEAD_INDICES)
    net_clear_z = table.net_height + physics.ball_radius
    contact_radius = args.contact_radius
    max_ticks = max(1, int(round(args.max_trial_seconds / dt)))
    rng = np.random.default_rng(args.seed)
    continuous = args.eval_mode == "continuous"
    diagnostics = _EvaluationDiagnostics(
        command_profile=profile_metadata["name"],
        policy_lead_time_s=float(args.policy_lead_time),
        training_aligned_preroll=bool(args.training_aligned_preroll),
    )

    def _policy_tick(
        lifecycle,
        source,
        last_action,
        fixed_station_xy,
        *,
        advance_clock=True,
    ):
        """One 50 Hz policy step (identical to the deploy runner's tick).

        ``advance_clock=False`` represents Isaac's sampled pre-swing hold: physics
        and policy history advance while the motion/TTS clock stays at clip start.
        """
        state = scene.read_robot_state()
        target = lifecycle.update(source.poll(), state)
        obs = build_observation(state, target, last_action, default_q, fixed_station_xy)
        diagnostics.policy_ticks += 1
        diagnostics.min_pelvis_height_m = min(
            diagnostics.min_pelvis_height_m, float(state.base_pos_w[2])
        )
        diagnostics.max_station_drift_m = max(
            diagnostics.max_station_drift_m,
            float(np.linalg.norm(np.asarray(state.base_pos_w[:2]) - fixed_station_xy)),
        )
        if not np.all(np.isfinite(obs)):
            diagnostics.nonfinite_observations += 1
            raise FloatingPointError("policy observation became non-finite")
        raw_action = policy.infer(obs)
        if not np.all(np.isfinite(raw_action)):
            diagnostics.nonfinite_actions += 1
            raise FloatingPointError("ONNX policy action became non-finite")
        diagnostics.max_abs_raw_action = max(
            diagnostics.max_abs_raw_action, float(np.max(np.abs(raw_action)))
        )
        # Applied action = raw with the passive head columns zeroed, matching both
        # the deploy runner and training's zeroed last_action feedback.
        applied_action = np.asarray(raw_action, dtype=np.float64).copy()
        if runtime_cfg.passive_neck:
            applied_action[head_idx] = 0.0
        unclamped = (
            runtime_cfg.action_adapter.default_q
            + applied_action * runtime_cfg.action_adapter.action_scale
        )
        if np.any(
            (unclamped < runtime_cfg.action_adapter.clamp_lower - 1.0e-12)
            | (unclamped > runtime_cfg.action_adapter.clamp_upper + 1.0e-12)
        ):
            diagnostics.saturated_action_ticks += 1
        q_des = runtime_cfg.action_adapter.decode(applied_action)
        if runtime_cfg.passive_neck:
            q_des[head_idx] = default_q[head_idx]
        scene.write_targets(q_des, kp, kd)
        events = scene.step()
        # ``target`` was observed before this transition.  Advance the lifecycle
        # only after physics, matching the runner and Isaac transition ordering.
        if advance_clock:
            lifecycle.advance()
        return events, applied_action, target

    def _park_ball():
        """Drop the ball out of play (past the far edge) between rally serves."""
        scene.set_ball(
            [
                scene.near_edge_x + scene.length + 1.0,
                scene.table_center_y,
                scene.table_height + 0.5,
            ],
            [0.0, 0.0, 0.0],
        )

    # Continuous mode: ONE initialization for the whole session — robot state,
    # last_action, lifecycle and the fixed station all persist across serves
    # (matching the deploy runner). Independent mode re-creates them per serve.
    scene.reset_stand()
    lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
    # Offline evaluation advances an explicit simulation clock.  Wall-clock
    # mailbox ageing would inject host scheduling jitter into the policy TTS.
    source = QueueRacketCommandSource(age_queued_commands=False)
    last_action = np.zeros(31, dtype=np.float64)
    fixed_station_xy = scene.base_pos_w()[:2].copy()
    table_near_rel_station = float(scene.near_edge_x - fixed_station_xy[0])
    table_center_rel_station_y = float(scene.table_center_y - fixed_station_xy[1])
    if args.near_edge_x is None and not np.isclose(
        table_near_rel_station, args.table_near_station_x, atol=1.0e-6
    ):
        raise RuntimeError(
            "MuJoCo table alignment failed: "
            f"near edge is {table_near_rel_station:.6f} m from station, "
            f"expected {args.table_near_station_x:.6f} m"
        )
    if not np.isclose(table_center_rel_station_y, 0.0, atol=1.0e-6):
        raise RuntimeError(
            "MuJoCo table centreline is not aligned with the fixed station: "
            f"y error={table_center_rel_station_y:.6f} m"
        )
    print(
        "[mujoco_eval] Isaac alignment: "
        f"table_near_rel_station={table_near_rel_station:.4f} m, "
        f"table_center_y_error={table_center_rel_station_y:.4f} m",
        file=sys.stderr,
    )
    max_rest_ticks = max(1, int(round(args.max_rest_seconds / dt)))
    transitions_seen: set = set()
    prev_side = None
    task_counter = 0

    def _planned_target_velocity(side, target_pos, incoming_at_strike):
        if legacy_velocity_boxes is not None:
            box = legacy_velocity_boxes[int(side)]
            return rng.uniform(box[:, 0], box[:, 1])
        return racket_planner.plan(
            StrikeTarget(
                p_ball=scene.to_table(target_pos),
                v_ball=incoming_at_strike,
                t_strike=float(args.policy_lead_time),
                num_bounces=0,
                valid=True,
            )
        ).v_racket

    def _sample_no_serve_target(side):
        if side >= 0:
            position_box = np.array(
                [
                    [args.forehand_strike_forward_min, args.forehand_strike_forward_max],
                    [args.forehand_strike_lateral_min, args.forehand_strike_lateral_max],
                    [args.forehand_strike_height_min, args.forehand_strike_height_max],
                ],
                dtype=np.float64,
            )
            incoming_box = np.array(
                [[-2.40, -0.90], [-0.35, 0.35], [-2.20, 1.00]],
                dtype=np.float64,
            )
        else:
            position_box = np.array(
                [
                    [args.backhand_strike_forward_min, args.backhand_strike_forward_max],
                    [args.backhand_strike_lateral_min, args.backhand_strike_lateral_max],
                    [args.backhand_strike_height_min, args.backhand_strike_height_max],
                ],
                dtype=np.float64,
            )
            incoming_box = np.array(
                [[-2.30, -0.80], [-0.35, 0.35], [-2.80, 1.00]],
                dtype=np.float64,
            )
        local = rng.uniform(position_box[:, 0], position_box[:, 1])
        target_pos = np.array(
            [
                fixed_station_xy[0] + local[0],
                fixed_station_xy[1] + local[1],
                local[2],
            ],
            dtype=np.float64,
        )
        incoming = rng.uniform(incoming_box[:, 0], incoming_box[:, 1])
        return target_pos, _planned_target_velocity(side, target_pos, incoming)

    for trial in range(args.num_serves):
        # Side pattern FH, FH, BH, BH, ... exercises all four adjacent side
        # transitions (FH->FH, FH->BH, BH->BH, BH->FH) across the session.
        side = FOREHAND if (trial % 4) < 2 else BACKHAND
        # Training uses opposite physical faces for the two clips:
        # mount_normal_sign_per_clip=(+1, -1).
        scene.set_paddle_face_sign(1.0 if side >= 0 else -1.0)
        if prev_side is not None:
            transitions_seen.add((prev_side, side))
        prev_side = side

        if not continuous:
            # Independent-strike evaluation: fresh robot + policy state per serve.
            scene.reset_stand()
            lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
            source = QueueRacketCommandSource(age_queued_commands=False)
            last_action = np.zeros(31, dtype=np.float64)
            fixed_station_xy = scene.base_pos_w()[:2].copy()
        elif trial > 0:
            # Continuous rally: keep the policy running (no reset, no teleport)
            # through follow-through/recovery until it is ready for the next ball.
            _park_ball()
            for _ in range(max_rest_ticks):
                _events, last_action, _target = _policy_tick(
                    lifecycle, source, last_action, fixed_station_xy
                )
                if lifecycle.phase.value == "ready":
                    break
            if lifecycle.phase.value != "ready":
                raise RuntimeError(
                    "continuous evaluator could not finish the previous Isaac-length "
                    f"follow-through within --max-rest-seconds={args.max_rest_seconds}"
                )

        task_counter += 1
        task_id = task_counter
        revision = 0
        if args.no_serve:
            strike_pt, target_vel = _sample_no_serve_target(side)
            serve_pos = serve_vel = arrival_vel = None
            planned_incoming_bounce = None
            flight_t = 0.0
        else:
            (
                strike_pt,
                serve_pos,
                serve_vel,
                flight_t,
                arrival_vel,
                planned_incoming_bounce,
            ) = _sample_serve(
                rng,
                side,
                scene,
                physics,
                table_bounce,
                args,
                fixed_station_xy,
            )
            target_vel = _planned_target_velocity(side, strike_pt, arrival_vel)

        initial_tts = (
            float(
                args.no_serve_time_to_strike
                if args.no_serve and args.no_serve_time_to_strike is not None
                else args.policy_lead_time
            )
            if args.training_aligned_preroll or args.no_serve
            else float(flight_t)
        )
        source.submit(
            RacketCommand(
                task_id=task_id,
                task_revision=revision,
                swing_side=side,
                position=strike_pt,
                velocity=target_vel,
                time_to_strike=initial_tts,
            )
        )

        # Training freezes the first clip frame and its clip-start TTS for U[0,2] s.
        # A true stand reset has a 0.5 s minimum; intra-rally wraps do not.
        # Isaac samples an INCLUSIVE integer hold count and then clamps true
        # stand starts to ``stand_start_min_hold``. Sampling continuous seconds
        # here would almost eliminate the atom at exactly 0.5 s (26/101 of
        # training stand starts), so reproduce the discrete distribution.
        hold_ticks = (
            _sample_training_hold_ticks(
                rng,
                hold_min_seconds=args.hold_min_seconds,
                hold_max_seconds=args.hold_max_seconds,
                stand_min_hold_seconds=args.stand_min_hold_seconds,
                dt=dt,
                stand_start=not continuous or trial == 0,
            )
            if args.training_aligned_preroll or args.no_serve
            else 0
        )
        _park_ball()
        for _ in range(hold_ticks):
            _events, last_action, _target = _policy_tick(
                lifecycle,
                source,
                last_action,
                fixed_station_xy,
                advance_clock=False,
            )
            _park_ball()

        if args.no_serve:
            # Complete the entire +TTS -> 0 -> terminal actor trajectory. This is a
            # motion diagnostic only and intentionally contributes no denominator.
            remaining = (
                max(initial_tts, 0.0)
                + runtime_cfg.lifecycle.follow_through_s
                + runtime_cfg.lifecycle.recovery_s
            )
            for _ in range(int(np.ceil(remaining / dt)) + 1):
                _events, last_action, _target = _policy_tick(
                    lifecycle, source, last_action, fixed_station_xy
                )
                _park_ball()
            continue

        if args.training_aligned_preroll:
            if initial_tts + 1.0e-9 < flight_t:
                raise ValueError(
                    f"policy lead time {initial_tts} s is shorter than flight time {flight_t} s"
                )
            prelaunch_ticks = int(round((initial_tts - flight_t) / dt))
            for _ in range(prelaunch_ticks):
                _events, last_action, _target = _policy_tick(
                    lifecycle, source, last_action, fixed_station_xy
                )
                _park_ball()
            if not np.isclose(lifecycle.time_to_strike, flight_t, atol=1.0e-8):
                raise RuntimeError(
                    "policy and physical serve clocks failed to align: "
                    f"policy={lifecycle.time_to_strike:.9f}, flight={flight_t:.9f}"
                )

        scene.set_ball(serve_pos, serve_vel)
        contacted = False
        contact_kind = "none"
        net_clear = False
        first_bounce = None
        incoming_net_clear = False
        incoming_bounces: list[tuple[float, float]] = []
        min_distance = float("inf")
        strike_sample = None
        contact_sample = None
        pending_contact_vx = None  # set on contact frame; watch next few ticks for max +vx

        for _tick in range(max_ticks):
            ball_pos_pre, ball_vel_pre = scene.ball_state()
            racket_pos_pre, racket_vel_pre = scene.racket_site_state()
            # After contact, watch a few ticks and keep the max +x speed (the flip settles
            # in 1-2 physics steps; the post-contact vx can transiently read as the incoming
            # velocity if the collision resolves on a later substep).
            if pending_contact_vx is not None and contact_sample is not None:
                ball_pos_post_here, ball_vel_post_here = scene.ball_state()
                vx_here = float(ball_vel_post_here[0])
                if vx_here > contact_sample.get("outgoing_ball_vx_mps", -1e9):
                    contact_sample["outgoing_ball_vx_mps"] = vx_here
                    contact_sample["outgoing_ball_vel_w"] = [
                        float(ball_vel_post_here[0]),
                        float(ball_vel_post_here[1]),
                        float(ball_vel_post_here[2]),
                    ]
                pending_contact_vx -= 1
                if pending_contact_vx <= 0:
                    pending_contact_vx = None
            pre_distance = float(np.linalg.norm(ball_pos_pre - racket_pos_pre))
            min_distance = min(min_distance, pre_distance)

            if args.stream_planner_revisions:
                cmd = _predict_command(
                    ball_pos_pre,
                    ball_vel_pre,
                    side,
                    scene,
                    strike_pt[0],
                    task_id,
                    revision + 1,
                    RacketCommand,
                    trajectory_predictor,
                    racket_planner,
                )
                if cmd is not None and legacy_velocity_boxes is not None:
                    cmd = RacketCommand(
                        task_id=cmd.task_id,
                        task_revision=cmd.task_revision,
                        swing_side=cmd.swing_side,
                        position=cmd.position,
                        velocity=target_vel,
                        time_to_strike=cmd.time_to_strike,
                    )
                if cmd is not None:
                    revision += 1
                    source.submit(cmd)

            events, last_action, observed_target = _policy_tick(
                lifecycle, source, last_action, fixed_station_xy
            )
            post_transition_tts = float(lifecycle.time_to_strike)

            ball_pos_post, ball_vel_post = scene.ball_state()
            racket_pos_post, racket_vel_post = scene.racket_site_state()
            if (
                strike_sample is None
                or abs(post_transition_tts)
                < strike_sample["abs_time_to_strike_s"]
            ):
                strike_sample = {
                    "abs_time_to_strike_s": abs(post_transition_tts),
                    "action_observation_time_to_strike_s": float(
                        observed_target.time_to_strike
                    ),
                    "post_transition_time_to_strike_s": post_transition_tts,
                    "ball_to_target_pre_transition_m": float(
                        np.linalg.norm(ball_pos_pre - strike_pt)
                    ),
                    "ball_to_target_post_transition_m": float(
                        np.linalg.norm(ball_pos_post - strike_pt)
                    ),
                    "racket_to_target_pre_transition_m": float(
                        np.linalg.norm(racket_pos_pre - strike_pt)
                    ),
                    "racket_to_target_post_transition_m": float(
                        np.linalg.norm(racket_pos_post - strike_pt)
                    ),
                    "racket_velocity_error_pre_transition_mps": float(
                        np.linalg.norm(racket_vel_pre - target_vel)
                    ),
                    "racket_velocity_error_post_transition_mps": float(
                        np.linalg.norm(racket_vel_post - target_vel)
                    ),
                }

            delta = ball_pos_post - racket_pos_post
            distance = float(np.linalg.norm(delta))
            min_distance = min(min_distance, distance)
            direction = delta / distance if distance > 1.0e-9 else np.zeros(3)
            closing_speed = -float(np.dot(ball_vel_post - racket_vel_post, direction))

            if events.ball_racket_contact:
                contacted = True
                contact_kind = "mujoco"
                if contact_sample is None:
                    contact_sample = {
                        "action_observation_time_to_strike_s": float(
                            observed_target.time_to_strike
                        ),
                        "post_transition_time_to_strike_s": post_transition_tts,
                        "racket_to_target_pre_transition_m": float(
                            np.linalg.norm(racket_pos_pre - strike_pt)
                        ),
                        "racket_to_target_post_transition_m": float(
                            np.linalg.norm(racket_pos_post - strike_pt)
                        ),
                        "racket_velocity_error_pre_transition_mps": float(
                            np.linalg.norm(racket_vel_pre - target_vel)
                        ),
                        "racket_velocity_error_post_transition_mps": float(
                            np.linalg.norm(racket_vel_post - target_vel)
                        ),
                    }
                    # Watch up to 4 ticks post-contact for the settled outgoing velocity.
                    pending_contact_vx = 4

            elif (
                not contacted
                and distance <= contact_radius
                and closing_speed > 0.0
            ):
                contacted = True
                contact_kind = "proximity"
                contact_sample = {
                    "action_observation_time_to_strike_s": float(
                        observed_target.time_to_strike
                    ),
                    "post_transition_time_to_strike_s": post_transition_tts,
                    "racket_to_target_pre_transition_m": float(
                        np.linalg.norm(racket_pos_pre - strike_pt)
                    ),
                    "racket_to_target_post_transition_m": float(
                        np.linalg.norm(racket_pos_post - strike_pt)
                    ),
                    "racket_velocity_error_pre_transition_mps": float(
                        np.linalg.norm(racket_vel_pre - target_vel)
                    ),
                    "racket_velocity_error_post_transition_mps": float(
                        np.linalg.norm(racket_vel_post - target_vel)
                    ),
                }
                # Watch up to 4 ticks post-contact for the settled outgoing velocity.
                pending_contact_vx = 4
                

            if not contacted and observed_target.time_to_strike >= 0.0:
                if not incoming_net_clear:
                    incoming_net_clear = any(
                        x_sign < 0.0 and z_cross > net_clear_z
                        for z_cross, x_sign in events.net_crossings
                    )
                for x_t, y_t in events.table_contacts:
                    if (
                        0.0 < x_t < table.net_x
                        and -table.width <= y_t <= 0.0
                    ):
                        incoming_bounces.append((float(x_t), float(y_t)))

            if contacted and first_bounce is None:
                if not net_clear:
                    for z_cross, x_sign in events.net_crossings:
                        if x_sign > 0.0 and z_cross > net_clear_z:
                            net_clear = True
                            break
                if events.table_contacts:
                    x_t, y_t = events.table_contacts[0]
                    first_bounce = (float(x_t), float(y_t))

            if first_bounce is not None:
                break
            if not contacted and ball_pos_post[0] < scene.near_edge_x - 0.8:
                break

        legal_incoming = bool(
            incoming_net_clear and len(incoming_bounces) == 1
        )
        if not legal_incoming:
            raise RuntimeError(
                "MuJoCo incoming trajectory violated the legal one-bounce contract: "
                f"trial={trial}, incoming_net_clear={incoming_net_clear}, "
                f"near_side_bounces={incoming_bounces}"
            )
        success = bool(
            contacted
            and net_clear
            and first_bounce is not None
            and table.on_opponent_half(first_bounce[0], first_bounce[1])
        )
        accumulator.add_bool(success)
        diagnostics.attempts += 1
        if contact_kind == "mujoco":
            diagnostics.mujoco_contacts += 1
        elif contact_kind == "proximity":
            diagnostics.proximity_contacts += 1
        diagnostics.net_clears += int(net_clear)
        diagnostics.opponent_bounces += int(success)
        diagnostics.legal_incoming_bounces += int(legal_incoming)
        fell = scene.base_fallen()
        diagnostics.falls += int(fell)
        diagnostics.trials.append(
            {
                "trial": trial,
                "side": "forehand" if side >= 0 else "backhand",
                "hold_seconds": hold_ticks * dt,
                "flight_time_s": flight_t,
                "planned_incoming_bounce": planned_incoming_bounce,
                "actual_incoming_bounces_table_xy": [
                    list(item) for item in incoming_bounces
                ],
                "incoming_net_clear": incoming_net_clear,
                "legal_incoming": legal_incoming,
                "target_position_w": strike_pt.tolist(),
                "target_velocity_w": np.asarray(target_vel).tolist(),
                "contact": contact_kind,
                "contact_sample": contact_sample,
                "net_clear": net_clear,
                "first_bounce_table_xy": (
                    list(first_bounce) if first_bounce is not None else None
                ),
                "success": success,
                "min_ball_racket_distance_m": min_distance,
                "fell": fell,
                "strike_sample": strike_sample,
            }
        )

    scene.close()
    if continuous:
        _names = {FOREHAND: "FH", BACKHAND: "BH"}
        seen = sorted(f"{_names[a]}->{_names[b]}" for a, b in transitions_seen)
        all_four = len(transitions_seen) == 4
        print(
            f"[mujoco_eval] adjacent side transitions exercised: {', '.join(seen) or 'none'}"
            + ("" if all_four else "  (increase --num-serves >= 5 to cover all four)"),
            file=sys.stderr,
        )
    print(
        f"[mujoco_eval] mode={args.eval_mode} serves={accumulator.attempts} "
        f"contacts={diagnostics.mujoco_contacts}+{diagnostics.proximity_contacts} "
        f"net_clears={diagnostics.net_clears} returns={accumulator.successes} "
        f"success_rate={accumulator.value:.4f}",
        file=sys.stderr,
    )
    diagnostics_doc = diagnostics.as_dict()
    diagnostics_doc["profile_metadata"] = profile_metadata
    if args.diagnostics_out:
        diagnostics_path = pathlib.Path(args.diagnostics_out)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diagnostics_path, "w", encoding="utf-8") as stream:
            json.dump(diagnostics_doc, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return accumulator.as_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", default=None, help="Exported hope_pingpong.onnx (default: from runtime config).")
    parser.add_argument("--model-xml", default=None, help="a3_pingpong MJCF (default: from runtime config).")
    parser.add_argument("--runtime-config", default=None, help="hope_pingpong_runtime.yaml (default: shipped).")
    parser.add_argument("--reference-dir", default=None, help="Dir containing the reference deploy package.")
    parser.add_argument("--num-serves", type=int, default=50, help="Number of served balls (denominator).")
    parser.add_argument(
        "--eval-mode", choices=["continuous", "independent"], default="continuous",
        help="continuous (default): one uninterrupted rally session — robot/policy state persist "
             "across serves and all four adjacent side transitions are exercised. "
             "independent: reset the robot per serve (isolated-swing evaluation only).",
    )
    parser.add_argument(
        "--max-rest-seconds", type=float, default=4.0,
        help="continuous mode: max seconds the policy gets between serves to finish "
             "follow-through/recovery before the next ball is served.",
    )
    parser.add_argument(
        "--max-trial-seconds", type=float, default=3.0, help="Max simulated seconds per serve before scoring."
    )
    parser.add_argument(
        "--contact-radius",
        type=float,
        default=0.095,
        help="Racket-site proximity contact fallback (m); matches the training default.",
    )
    parser.add_argument(
        "--table-near-station-x", type=float, default=0.50,
        help="Table near-edge x relative to the fixed startup station (m); 0.50 matches Isaac training.",
    )
    parser.add_argument(
        "--near-edge-x", type=float, default=None,
        help="Optional absolute MuJoCo-world table near-edge x override (disables automatic station alignment).",
    )
    parser.add_argument(
        "--forehand-strike-forward-min", type=float, default=0.18,
        help="Minimum forehand strike x relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--forehand-strike-forward-max", type=float, default=0.40,
        help="Maximum forehand strike x relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--forehand-strike-lateral-min", type=float, default=-0.76,
        help="Minimum forehand strike y relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--forehand-strike-lateral-max", type=float, default=-0.52,
        help="Maximum forehand strike y relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--strike-range-margin", type=float, default=0.0,
        help="Inset from training-box boundaries (m), keeping actual MuJoCo trajectories in range.",
    )
    parser.add_argument(
        "--forehand-strike-height-min", type=float, default=1.00,
        help="Minimum forehand world-frame strike height (m); matches training.",
    )
    parser.add_argument(
        "--forehand-strike-height-max", type=float, default=1.21,
        help="Maximum forehand world-frame strike height (m); matches training.",
    )
    parser.add_argument(
        "--backhand-strike-forward-min", type=float, default=0.45,
        help="Minimum backhand strike x relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--backhand-strike-forward-max", type=float, default=0.75,
        help="Maximum backhand strike x relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--backhand-strike-lateral-min", type=float, default=-0.30,
        help="Minimum backhand strike y relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--backhand-strike-lateral-max", type=float, default=0.10,
        help="Maximum backhand strike y relative to the fixed startup station (m); matches training.",
    )
    parser.add_argument(
        "--backhand-strike-height-min", type=float, default=0.84,
        help="Minimum backhand world-frame strike height (m); matches training.",
    )
    parser.add_argument(
        "--backhand-strike-height-max", type=float, default=1.10,
        help="Maximum backhand world-frame strike height (m); matches training.",
    )
    parser.add_argument(
        "--strike-forward-min", type=float, default=None,
        help="Legacy shared minimum strike x override; supply together with --strike-forward-max.",
    )
    parser.add_argument(
        "--strike-forward-max", type=float, default=None,
        help="Legacy shared maximum strike x override; supply together with --strike-forward-min.",
    )
    parser.add_argument(
        "--strike-lateral-offset", type=float, default=None,
        help="Legacy symmetric lateral-centre override; supply with --strike-lateral-jitter.",
    )
    parser.add_argument(
        "--strike-lateral-jitter", type=float, default=None,
        help="Legacy symmetric lateral half-range; supply with --strike-lateral-offset.",
    )
    parser.add_argument(
        "--legacy-training-contract",
        default=None,
        help="Frozen legacy_policy_training_contract.yaml. When supplied, its "
             "per-side boxes, independent velocities, frame count and strike phase "
             "replace the current profile and restore that policy's original clock.",
    )
    parser.add_argument(
        "--policy-lead-time",
        type=float,
        default=1.0,
        help="TTS at the current 1.8 s policy clip start (s). A legacy contract "
             "automatically overrides this with its frozen motion timing.",
    )
    parser.add_argument(
        "--training-aligned-preroll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-run the policy from its clip-start TTS and delay physical ball launch "
             "until the sampled flight horizon (default: enabled).",
    )
    parser.add_argument(
        "--stream-planner-revisions",
        action="store_true",
        help="Continuously refine position/TTS from the physical ball after launch. "
             "Default keeps one fixed command, matching training.",
    )
    parser.add_argument(
        "--hold-min-seconds",
        type=float,
        default=0.0,
        help="Minimum sampled clip-start hold before countdown (training default: 0).",
    )
    parser.add_argument(
        "--hold-max-seconds",
        type=float,
        default=2.0,
        help="Maximum sampled clip-start hold before countdown (training default: 2).",
    )
    parser.add_argument(
        "--stand-min-hold-seconds",
        type=float,
        default=0.5,
        help="Minimum hold for a reset/first continuous stand start (training: 0.5 s).",
    )
    parser.add_argument(
        "--serve-flight-time-min",
        type=float,
        default=0.8,
        help="Minimum legal one-bounce incoming-ball flight duration (s).",
    )
    parser.add_argument(
        "--serve-flight-time-max",
        type=float,
        default=1.0,
        help="Maximum physical incoming-ball flight duration (s).",
    )
    parser.add_argument("--no-serve", action="store_true", help="Disable ball serving (ball stays parked, no incoming ball).")
    parser.add_argument(
        "--no-serve-time-to-strike",
        "--ready-time-to-strike",
        dest="no_serve_time_to_strike",
        type=float,
        default=None,
        help="Initial countdown for the single synthetic --no-serve task (seconds); "
             "defaults to the selected policy's training clip-start value. "
             "--ready-time-to-strike is retained as a deprecated alias.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the example serve distribution.")
    parser.add_argument("--view", action="store_true", help="Launch the MuJoCo passive viewer (debug only).")
    parser.add_argument(
        "--video-out",
        default=None,
        help="Record the complete evaluation session to an H.264 MP4.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="Recorded video width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
        help="Recorded video height in pixels (default: 720).",
    )
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    parser.add_argument(
        "--diagnostics-out",
        default=None,
        help="Write detailed contacts, near-misses, strike errors, falls and action statistics as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_eval(args)
    print(json.dumps(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
