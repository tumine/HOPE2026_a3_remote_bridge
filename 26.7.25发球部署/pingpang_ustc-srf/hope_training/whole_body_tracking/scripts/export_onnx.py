"""Export a trained HOPE checkpoint to a deployable ONNX policy + manifest.

Loads a local checkpoint, rebuilds the policy, and writes:

* ``hope_pingpong.onnx``     — single-output actor graph, observation[1, 111] -> raw_action[1, 31]
* ``policy_manifest.json``   — the contract (name, dims, control rate, joint order, obs
                               normalization = none, ActionAdapter config path) plus checkpoint
                               and source-config SHA-256 provenance

Usage:
    python scripts/export_onnx.py --checkpoint logs/rsl_rl/hope_pingpong/<run>/model_<iter>.pt

By default the files are written to ``<checkpoint_dir>/exported/``.
"""

import argparse
import os
import pathlib
import sys


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    return str(rooted.resolve()) if rooted.is_file() else str(rooted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint (.pt) to export.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <ckpt_dir>/exported).")
    parser.add_argument("--task", default="HOPE-PingPong-AgibotA3-v0", help="Gym task id.")
    parser.add_argument("--onnx-name", default="hope_pingpong.onnx", help="Exported ONNX filename.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to build (1 is enough to export).")
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--motion-file",
        default="hope_training/motions/preprocessed/ours_forehand_guarded_1p8s_wrist_x.npz",
        help="Forehand clip (only needed so the env instantiates).",
    )
    parser.add_argument(
        "--motion-file-2",
        default="hope_training/motions/preprocessed/ours_backhand_guarded_1p8s.npz",
        help="Backhand clip (only needed so the env instantiates).",
    )
    parser.add_argument("--experiment-name", default="hope_pingpong", help="rsl_rl experiment name.")
    parser.add_argument(
        "--agent-config",
        default=None,
        help="Original params/agent.yaml (or ppo.yaml) if it is not adjacent to the checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir = args.output_dir or os.path.join(os.path.dirname(checkpoint), "exported")

    # Launch Isaac (headless) before importing isaaclab modules; clear argv so Kit ignores our args.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app

    status = 0
    try:
        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import whole_body_tracking.tasks  # noqa: F401
        from whole_body_tracking.utils.checkpoint_compat import (
            build_checkpoint_provenance,
            configure_runner_for_inference,
            inspect_checkpoint,
            load_actor_for_inference,
        )
        from whole_body_tracking.utils.exporter import export_policy
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        clips = [_resolve_motion_path(c) for c in (args.motion_file, args.motion_file_2) if c]
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        articulation_joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
        # HARD GATE: PhysX's private articulation order may differ. The live action and
        # observation terms map by exact name, while the exported ONNX contract remains canonical.
        from whole_body_tracking.utils.action_adapter_config import (
            load_joint_order,
            validate_live_joint_order_contract,
        )

        expected_order = list(load_joint_order())
        try:
            canonical_ids = validate_live_joint_order_contract(env.unwrapped)
        except ValueError as exc:
            raise RuntimeError(
                "Articulation joints cannot be mapped to the canonical deploy joint order.\n"
                f"  articulation: {articulation_joint_names}\n"
                f"  canonical:    {expected_order}\n"
                f"  reason:       {exc}"
            ) from exc
        print(f"[export_onnx] canonical joint columns map to PhysX ids {list(canonical_ids)}", flush=True)
        env = RslRlVecEnvWrapper(env)

        checkpoint_info = inspect_checkpoint(checkpoint, args.agent_config)
        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(load_ppo_params(), args.experiment_name))
        configure_runner_for_inference(agent_cfg, checkpoint, checkpoint_info)
        agent_cfg.device = args.device
        runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        load_actor_for_inference(runner, checkpoint, checkpoint_info)

        normalizer = runner.obs_normalizer if checkpoint_info["has_observation_normalizer"] else None
        onnx_path, manifest_path = export_policy(
            runner.alg.policy,
            output_dir,
            joint_names=expected_order,
            normalizer=normalizer,
            onnx_filename=args.onnx_name,
            provenance=build_checkpoint_provenance(checkpoint, checkpoint_info),
        )
        print(f"[export_onnx] wrote {onnx_path}", flush=True)
        print(f"[export_onnx] wrote {manifest_path}", flush=True)
        env.close()
    except Exception:
        import traceback

        print("\n[export_onnx] ERROR:", flush=True)
        traceback.print_exc()
        status = 1
    finally:
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
