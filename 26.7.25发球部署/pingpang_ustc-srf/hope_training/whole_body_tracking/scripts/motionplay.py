"""Play the HOPE forehand/backhand reference motions on Agibot A3 in Isaac Lab.

This is a direct reference-motion viewer: it writes the NPZ root and joint state to the
articulation every physics tick.  It does not load a policy, run rewards, or use the motion
command's random clip lifecycle, so the pose shown in the viewer is the pose stored in the NPZ.

Examples:

    # Show forehand and backhand side by side (default).
    python scripts/motionplay.py --motion both

    # Show one clip at half speed.
    python scripts/motionplay.py --motion forehand --speed 0.5

    # Play once and hold the final frame.
    python scripts/motionplay.py --motion backhand --no-loop

    # Hold the strike frame from each motion's YAML sidecar, side by side.
    python scripts/motionplay.py --motion both --view strike
"""

from __future__ import annotations

import argparse
import pathlib
import time

from isaaclab.app import AppLauncher


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir():
            return parent
    return here.parents[2]


_MOTION_DIR = _repo_root() / "hope_training" / "motions" / "preprocessed"

parser = argparse.ArgumentParser(
    description="Directly replay the HOPE reference motions on the Agibot A3 model."
)
parser.add_argument(
    "--motion",
    choices=("both", "forehand", "backhand"),
    default="both",
    help="Motion to show. 'both' places forehand and backhand robots side by side.",
)
parser.add_argument(
    "--forehand-file",
    type=pathlib.Path,
    default=_MOTION_DIR / "ours_forehand_guarded_1p8s_wrist_x.npz",
    help="Forehand HOPE NPZ file.",
)
parser.add_argument(
    "--backhand-file",
    type=pathlib.Path,
    default=_MOTION_DIR / "ours_backhand_guarded_1p8s.npz",
    help="Backhand HOPE NPZ file.",
)
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (> 0).")
parser.add_argument(
    "--view",
    choices=("motion", "strike"),
    default="motion",
    help=(
        "'motion' plays the clips; 'strike' reads strike_frame from each matching YAML "
        "sidecar and holds those poses."
    ),
)
parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Loop each clip. With --no-loop the final pose remains displayed.",
)
parser.add_argument(
    "--spacing", type=float, default=2.0, help="Distance between the two robots in 'both' mode (m)."
)
parser.add_argument(
    "--max-cycles",
    type=int,
    default=0,
    help="Exit after this many longest-clip cycles; 0 keeps the viewer open (useful for headless checks).",
)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=0.0,
    help="In --view strike, exit after this many seconds; 0 holds until the window is closed.",
)
parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="Print reference ranges and live-vs-reference joint values at quarter-cycle frames.",
)
parser.add_argument(
    "--raw-source-height",
    action="store_true",
    help=(
        "Do not apply the training ground-height correction. By default the viewer "
        "uses the same grounded reference as imitation and RSI."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.speed <= 0.0:
    parser.error("--speed must be greater than zero")
if args_cli.max_cycles < 0:
    parser.error("--max-cycles must be non-negative")
if args_cli.hold_seconds < 0.0:
    parser.error("--hold-seconds must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac/torch-backed modules must be imported after AppLauncher starts Isaac Sim.
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from whole_body_tracking.robots.agibot_a3 import (  # noqa: E402
    A3_BACKHAND_MOTION_GROUND_OFFSET,
    A3_FOREHAND_MOTION_GROUND_OFFSET,
    A3_TRACKED_BODIES,
    AGIBOT_A3_CFG,
    AGIBOT_A3_JOINT_NAMES,
)
from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader  # noqa: E402


@configclass
class MotionPlaybackSceneCfg(InteractiveSceneCfg):
    """Minimal scene for exact, state-driven reference playback."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )
    robot = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_path(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"motion file not found: {path}")
    return path


def _selected_motion_paths() -> list[tuple[str, pathlib.Path]]:
    if args_cli.motion == "forehand":
        selected = [("FOREHAND", args_cli.forehand_file)]
    elif args_cli.motion == "backhand":
        selected = [("BACKHAND", args_cli.backhand_file)]
    else:
        selected = [
            ("FOREHAND", args_cli.forehand_file),
            ("BACKHAND", args_cli.backhand_file),
        ]
    return [(name, _resolve_motion_path(path)) for name, path in selected]


def _load_motion(path: pathlib.Path) -> MotionLoader:
    """Use the training pipeline's loader and sidecar contract validation."""
    motion = MotionLoader(
        str(path),
        len(A3_TRACKED_BODIES),
        AGIBOT_A3_JOINT_NAMES,
        A3_TRACKED_BODIES,
        device=args_cli.device,
    )
    if not args_cli.raw_source_height:
        import yaml

        metadata = yaml.safe_load(path.with_suffix(".yaml").read_text(encoding="utf-8")) or {}
        swing_side = int(metadata.get("swing_side", 0))
        offsets = {
            1: A3_FOREHAND_MOTION_GROUND_OFFSET,
            -1: A3_BACKHAND_MOTION_GROUND_OFFSET,
        }
        if swing_side not in offsets:
            raise ValueError(
                f"{path.with_suffix('.yaml')}: swing_side must be +1/-1 to apply "
                "the training ground-height correction (or pass --raw-source-height)"
            )
        motion.body_pos_w[:, :, 2].add_(offsets[swing_side])
    return motion


def _strike_frame_from_sidecar(path: pathlib.Path, num_frames: int) -> int:
    """Read and validate the zero-based strike frame stored beside one NPZ."""
    import yaml

    sidecar = path.with_suffix(".yaml")
    if not sidecar.is_file():
        raise FileNotFoundError(f"strike view requires a YAML sidecar: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    if "strike_frame" not in metadata:
        raise ValueError(f"strike view requires strike_frame in {sidecar}")
    frame = int(metadata["strike_frame"])
    if not 0 <= frame < num_frames:
        raise ValueError(
            f"{sidecar}: strike_frame={frame} is outside [0, {num_frames - 1}]"
        )
    return frame


def _validate_motion(name: str, motion: MotionLoader) -> None:
    num_joints = len(AGIBOT_A3_JOINT_NAMES)
    if motion.joint_pos.ndim != 2 or motion.joint_pos.shape[1] != num_joints:
        raise ValueError(
            f"{name}: expected joint_pos [T, {num_joints}], got {tuple(motion.joint_pos.shape)}"
        )
    if motion.joint_vel.shape != motion.joint_pos.shape:
        raise ValueError(
            f"{name}: joint_vel shape {tuple(motion.joint_vel.shape)} does not match joint_pos"
        )
    num_frames = motion.joint_pos.shape[0]
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        value = getattr(motion, field)
        if value.shape[0] != num_frames or value.shape[1] < 1:
            raise ValueError(f"{name}: invalid {field} shape {tuple(value.shape)}")
    if motion.fps <= 0.0:
        raise ValueError(f"{name}: fps must be positive, got {motion.fps}")


def _frame_index(source_time: float, motion: MotionLoader) -> int:
    num_frames = int(motion.joint_pos.shape[0])
    duration = num_frames / float(motion.fps)
    if args_cli.loop:
        clip_time = source_time % duration
    else:
        clip_time = min(source_time, (num_frames - 1) / float(motion.fps))
    return min(int(clip_time * float(motion.fps)), num_frames - 1)


def _write_motion_frame(
    scene: InteractiveScene,
    motions: list[MotionLoader],
    source_time: float,
    joint_ids: list[int],
    fixed_frames: list[int] | None = None,
) -> list[int]:
    robot = scene["robot"]
    joint_pos = torch.empty(
        (len(motions), len(AGIBOT_A3_JOINT_NAMES)), dtype=torch.float32, device=args_cli.device
    )
    joint_vel = torch.empty_like(joint_pos)
    root_pose = torch.empty((len(motions), 7), dtype=torch.float32, device=args_cli.device)
    root_velocity = torch.empty((len(motions), 6), dtype=torch.float32, device=args_cli.device)
    frame_ids: list[int] = []

    for env_id, motion in enumerate(motions):
        frame = fixed_frames[env_id] if fixed_frames is not None else _frame_index(source_time, motion)
        frame_ids.append(frame)
        joint_pos[env_id] = motion.joint_pos[frame]
        if fixed_frames is None:
            joint_vel[env_id] = motion.joint_vel[frame] * args_cli.speed
        else:
            joint_vel[env_id] = 0.0
        # tracked_bodies[0] is pelvis_link, i.e. the articulation root. NPZ quaternions are wxyz.
        root_pose[env_id, :3] = motion.body_pos_w[frame, 0] + scene.env_origins[env_id]
        root_pose[env_id, 3:] = motion.body_quat_w[frame, 0]
        if fixed_frames is None:
            root_velocity[env_id, :3] = motion.body_lin_vel_w[frame, 0] * args_cli.speed
            root_velocity[env_id, 3:] = motion.body_ang_vel_w[frame, 0] * args_cli.speed
        else:
            root_velocity[env_id] = 0.0

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)
    # Keep the implicit drives aligned with the state that is written directly each tick.
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)
    robot.set_joint_velocity_target(joint_vel, joint_ids=joint_ids)
    scene.write_data_to_sim()
    return frame_ids


def main() -> None:
    selected = _selected_motion_paths()
    motions = [_load_motion(path) for _, path in selected]
    fixed_frames = None
    if args_cli.view == "strike":
        fixed_frames = [
            _strike_frame_from_sidecar(path, int(motion.joint_pos.shape[0]))
            for (_, path), motion in zip(selected, motions)
        ]
    for (name, path), motion in zip(selected, motions):
        _validate_motion(name, motion)
        print(
            f"[motionplay] {name}: {path} | frames={motion.joint_pos.shape[0]} "
            f"fps={motion.fps:g} duration={motion.joint_pos.shape[0] / motion.fps:.3f}s",
            flush=True,
        )
        if args_cli.diagnostics:
            ranges_deg = torch.rad2deg(torch.max(motion.joint_pos, dim=0).values - torch.min(motion.joint_pos, dim=0).values)
            moving = [
                f"{AGIBOT_A3_JOINT_NAMES[index]}={ranges_deg[index].item():.2f}deg"
                for index in torch.where(ranges_deg > 0.01)[0].detach().cpu().tolist()
            ]
            print(f"[motionplay][diag] {name} moving joints: {', '.join(moving) or 'none'}", flush=True)

    if fixed_frames is not None:
        for (name, path), motion, frame in zip(selected, motions, fixed_frames):
            print(
                f"[motionplay] {name} STRIKE POSE: frame={frame}/{motion.joint_pos.shape[0] - 1} "
                f"time={frame / float(motion.fps):.3f}s sidecar={path.with_suffix('.yaml')}",
                flush=True,
            )

    # Match the training physics rate. Gravity is disabled because this is a kinematic reference
    # viewer rather than a physics rollout; exact NPZ states are written every tick.
    physics_dt = 0.005
    render_interval = 4
    sim_cfg = sim_utils.SimulationCfg(
        dt=physics_dt,
        render_interval=render_interval,
        gravity=(0.0, 0.0, 0.0),
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    scene_cfg = MotionPlaybackSceneCfg(num_envs=len(motions), env_spacing=args_cli.spacing)
    scene = InteractiveScene(scene_cfg)

    center = scene.env_origins.mean(dim=0).detach().cpu().tolist()
    sim.set_camera_view(
        eye=[center[0] + 2.8, center[1] + 3.0, center[2] + 1.8],
        target=[center[0], center[1], center[2] + 0.9],
    )
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    joint_ids, resolved_names = robot.find_joints(AGIBOT_A3_JOINT_NAMES, preserve_order=True)
    if list(resolved_names) != list(AGIBOT_A3_JOINT_NAMES):
        raise RuntimeError(
            "Could not map the canonical motion joint order onto the live A3 articulation.\n"
            f"  resolved: {list(resolved_names)}\n"
            f"  expected: {list(AGIBOT_A3_JOINT_NAMES)}"
        )
    joint_ids = [int(joint_id) for joint_id in joint_ids]
    print(f"[motionplay] canonical joints mapped to PhysX ids: {joint_ids}", flush=True)

    print("[motionplay] Direct reference playback ready (no policy/reward).", flush=True)
    for env_id, ((name, _), origin) in enumerate(zip(selected, scene.env_origins)):
        xyz = origin.detach().cpu().tolist()
        print(f"[motionplay] env_{env_id}: {name} at origin={xyz}", flush=True)
    if fixed_frames is not None:
        print(
            f"[motionplay] Holding strike poses {fixed_frames}; close the Isaac window or press Ctrl+C to stop.",
            flush=True,
        )
    else:
        print("[motionplay] Close the Isaac window or press Ctrl+C to stop.", flush=True)

    source_time = 0.0
    longest_duration = max(m.joint_pos.shape[0] / float(m.fps) for m in motions)
    completed_cycles = 0
    next_tick = time.perf_counter()
    static_view_start = next_tick
    physics_step = 0
    held_final_reported = False
    diagnostic_frames = [
        {0, int(motion.joint_pos.shape[0] // 4), int(motion.joint_pos.shape[0] // 2), int(3 * motion.joint_pos.shape[0] // 4)}
        for motion in motions
    ]
    reported_diagnostics = [set() for _ in motions]

    while simulation_app.is_running():
        frame_ids = _write_motion_frame(scene, motions, source_time, joint_ids, fixed_frames)
        sim.step(render=(physics_step % render_interval == 0))
        scene.update(physics_dt)

        if args_cli.diagnostics:
            live_joint_pos = scene["robot"].data.joint_pos[:, joint_ids]
            for env_id, (motion, frame) in enumerate(zip(motions, frame_ids)):
                if frame in diagnostic_frames[env_id] and frame not in reported_diagnostics[env_id]:
                    reported_diagnostics[env_id].add(frame)
                    reference = motion.joint_pos[frame]
                    error_deg = torch.rad2deg(torch.max(torch.abs(live_joint_pos[env_id] - reference))).item()
                    values = ", ".join(
                        f"{AGIBOT_A3_JOINT_NAMES[index]}="
                        f"{torch.rad2deg(live_joint_pos[env_id, index]).item():.2f}/"
                        f"{torch.rad2deg(reference[index]).item():.2f}deg"
                        for index in (0, 13, 14, 15)
                    )
                    print(
                        f"[motionplay][diag] env_{env_id} frame={frame}: live/ref {values}; "
                        f"max_error={error_deg:.3f}deg",
                        flush=True,
                    )

        previous_cycle = int(source_time / longest_duration)
        if fixed_frames is not None:
            if args_cli.hold_seconds and time.perf_counter() - static_view_start >= args_cli.hold_seconds:
                break
        elif args_cli.loop:
            source_time += physics_dt * args_cli.speed
            current_cycle = int(source_time / longest_duration)
            if current_cycle > previous_cycle:
                completed_cycles = current_cycle
                print(
                    f"[motionplay] completed cycle {completed_cycles}; restarting clips",
                    flush=True,
                )
                if args_cli.max_cycles and completed_cycles >= args_cli.max_cycles:
                    break
        else:
            source_time = min(source_time + physics_dt * args_cli.speed, longest_duration)
            if source_time >= longest_duration and not held_final_reported:
                held_final_reported = True
                print(
                    f"[motionplay] final frame reached {frame_ids}; holding pose",
                    flush=True,
                )
                if args_cli.max_cycles:
                    break

        physics_step += 1
        # SimulationContext is not a wall-clock rate limiter. Pace the viewer at the configured
        # physics frequency; --speed changes source-motion time rather than UI responsiveness.
        next_tick += physics_dt
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[motionplay] stopped", flush=True)
    finally:
        # No Replicator jobs are used by this viewer, so there is nothing to wait for at shutdown.
        simulation_app.close(wait_for_replicator=False)
