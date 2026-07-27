# HOPE — whole-body training (Agibot A3)

This Isaac Lab extension trains the **HOPE** whole-body policy for the
[Agibot A3](https://www.zhiyuan-robot.com/) humanoid (31 actuated DOF): a single feed-forward actor,
shared by forehand and backhand, that runs at **50 Hz** and drives a table-tennis swing.

- **Observation:** `float32[111]` (raw — no normalization). See [`docs/POLICY_INTERFACE.md`](../../docs/POLICY_INTERFACE.md).
- **Action:** `float32[31]` raw joint-position residual (joint order in
  [`hope_training/config/joint_order_agibot_a3.yaml`](../config/joint_order_agibot_a3.yaml)).
  The shared adapter clips actions to `[-100, 100]`, keeps a linear (`identity`) transform,
  and uses one `0.25` scale for all columns;
  the two passive head columns remain held at their defaults.
- **Task:** one Gym task, `HOPE-PingPong-AgibotA3-v0`, selected with `task=HOPEPingPong`.

The metric reported by the evaluators is a single number, **`success_rate`** (a returned ball must be
contacted, cross the net, and land its first bounce on the opponent half).

## Install

Requires Isaac Sim + Isaac Lab (with `rsl_rl`), Python 3.10, and an NVIDIA CUDA GPU. Follow the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
first, then install this extension into the Isaac Lab Python:

```bash
python -m pip install -e source/whole_body_tracking
# extra deps used by the Hydra entry points (import them in the Isaac Lab python):
python -m pip install hydra-core omegaconf
```

Optionally `source setup_train_env.sh` (in the GPU/Isaac shell) to put the working-tree source first on
`PYTHONPATH` and get an `isaac_py` launcher.

## Robot asset (bundled URDF)

The starter ships the Agibot-provided A3 ping-pong URDF package under
`agibot/URDF/A3T2.5-URDF-std-pingpang/` (vendor material, no OSS license — see
[`A3_ASSETS.md`](../../A3_ASSETS.md)); the asset-prep step uses it by default. To use your own
vendor-supplied copy, place it under `a3_deploy/URDF/` (see
[`a3_deploy/URDF/README.md`](../../a3_deploy/URDF/README.md)) and pass `--source-root`:

```bash
python scripts/prepare_a3_isaac_asset.py --force
python scripts/prepare_a3_isaac_asset.py --source-root a3_deploy/URDF/<your_a3_package> --force
python scripts/prepare_a3_isaac_asset.py --check   # verify the prepared asset
```

## Motion clips — reference examples only

Training imitates two reference clips (clip 0 = forehand, clip 1 = backhand). The current task
defaults are the 91-frame, 50 Hz `ours_forehand_guarded_1p8s_wrist_x.npz` and
`ours_backhand_guarded_1p8s.npz` motions. They crop source frames 72--320 and
compress the complete outer wind-up, follow-through, and recovery phases. A
shared bounded inverse-arc-length map gives more output samples to sections
with larger normalized joint or racket-centre motion and caps any local skip
at eight source intervals. The
unit-speed strike guard is source 173--187 → output 43--57; inside it, the
reward window source 174--186 → output 44--56 and the strike source 180 →
output 50 are preserved. The sample timestamps span 1.8 s. At 50 Hz the actor
executes all 91 inclusive samples in 91 ticks (1.82 s), with
`time_to_strike` running from +1.0 s through 0 to -0.8 s. The task episode
remains 10 s and therefore contains repeated continuous swing cycles rather
than one reset per clip. The immutable 361-frame motions remain beside these
runtime defaults only as legacy/provenance sources.
The older `hope_forehand.npz` and `hope_backhand.npz` files are only
physically-neutral placeholders for import and shape checks.

## Train

The user runs training. Pick the task/algo and override any field on the CLI:

```bash
python scripts/train.py task=HOPEPingPong algo=ppo headless=true

# common overrides
python scripts/train.py task=HOPEPingPong num_envs=1024 max_iterations=20000 seed=1 \
    motion_file=hope_training/motions/preprocessed/ours_forehand_guarded_1p8s_wrist_x.npz \
    motion_file_2=hope_training/motions/preprocessed/ours_backhand_guarded_1p8s.npz
```

Checkpoints are written locally to `logs/rsl_rl/hope_pingpong/<timestamp>/` (a periodic checkpoint
every `save_interval` iterations and a final one). Resume with `checkpoint_path=<...>/model_<N>.pt`.
Tune training by editing `cfg/task/HOPEPingPong.yaml` (env / motion / overrides) and
`cfg/algo/ppo.yaml` (PPO). Launch from the repository root so the relative motion paths resolve.

The default balance curriculum uses 80% quiet stand starts; the remaining RSI starts use stable
dual-support frames with a 10--20 tick settle. Racket position and incoming-velocity boxes expand
from centered subsets to their complete ranges over 200k control steps. TensorBoard logs
`target_curriculum_progress`, `feet_contact_fraction`, and `both_feet_contact`.
It also records `station_error` (m), `base_tilt` (rad), and `base_height` (m) for balance diagnosis.

The canonical ActionAdapter clips the applied action to `[-100, 100]`, then uses `identity` with a
uniform `0.25` scale and a final mechanical joint clamp. This differs from checkpoints trained with
the former `tanh`/per-joint mapping, so start a new training run rather than resuming one of those
checkpoints. The frozen 29k deployment bundle remains an independent provenance snapshot.

## Play a checkpoint

```bash
python scripts/play.py task=HOPEPingPong num_envs=4 \
    checkpoint=logs/rsl_rl/hope_pingpong/<run>/model_<iter>.pt
```

## Export the deployable policy

```bash
python scripts/export_onnx.py --checkpoint logs/rsl_rl/hope_pingpong/<run>/model_<iter>.pt
```

Writes `hope_pingpong.onnx` (observation[1, 111] -> raw_action[1, 31], single output) and
`policy_manifest.json` (contract name, dims, control rate, joint order, `observation_normalization:
none`, ActionAdapter config path, checkpoint SHA-256, and available source-config SHA-256 values)
to `<run>/exported/`. Export/evaluation/playback load only the actor and its optional observation
normalizer, so changes to the critic or optimizer do not invalidate an inference-compatible old
checkpoint, provided the ActionAdapter contract is also unchanged. If training used empirical normalization, the frozen normalizer is embedded in ONNX;
deployment still supplies raw observations and the manifest records
`embedded_observation_normalization: empirical`. Keep the run's `params/agent.yaml` beside the
checkpoint; for a relocated checkpoint, pass `--agent-config /path/to/original/agent.yaml` so the
actor activation is never guessed.

## Evaluate (success_rate only)

```bash
# in Isaac (runs the torch policy):
python scripts/evaluate.py --checkpoint logs/rsl_rl/hope_pingpong/<run>/model_<iter>.pt --num-envs 256

# MuJoCo sim-to-sim (runs the exported ONNX; needs `mujoco` + `onnxruntime`):
python scripts/mujoco_eval_onnx.py --onnx logs/rsl_rl/hope_pingpong/<run>/exported/hope_pingpong.onnx
```

Both print only `{"success_rate": <float>}`. There is no threshold, best-checkpoint selection, or
exit-code change — the number is descriptive.

## Tests

Pure-Python unit tests (no Isaac / torch needed):

```bash
python tests/test_policy_contract.py        # obs 111 / action 31 / manifest schema
python tests/test_success_metric.py         # no-spin return-success logic
python tests/test_racket_command_msg.py     # RacketCommand.msg field ABI
python tests/test_table_tennis_geometry.py  # ITTF table geometry
```

## Code structure

- `scripts/` — `train.py`, `play.py`, `export_onnx.py`, `evaluate.py`, `mujoco_eval_onnx.py`,
  `prepare_a3_isaac_asset.py`.
- `cfg/` — Hydra configs: `train.yaml` / `play.yaml`, `algo/ppo.yaml`, `base/*`, `task/HOPEPingPong.yaml`.
- `source/whole_body_tracking/whole_body_tracking/`
  - `tasks/tracking/` — the HOPE task, actor observation contract, and MDP terms.
  - `tasks/table_tennis/` — the no-spin ball / ITTF table world and its geometry.
  - `robots/` — the Agibot A3 articulation configuration.
  - `utils/` — `exporter.py` (ONNX + manifest), `success_metric.py` (the `success_rate` core),
    `my_on_policy_runner.py` / `ppo_cfg.py` (rsl_rl glue).

## License

Apache-2.0. Copyright Intelligent Racing Inc. (dba Hitch Interactive). See `LICENSE`.
