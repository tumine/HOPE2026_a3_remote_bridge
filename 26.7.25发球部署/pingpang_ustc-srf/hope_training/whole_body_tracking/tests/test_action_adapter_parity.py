"""Training <-> deployment action-adapter parity (finding: unify the adapter contract).

Both sides must read the SAME shared configuration
(``a3_deploy/a3_deploy_example/config/action_adapter.yaml``) and map an identical
raw action to identical joint-position targets:

    applied_action = clip(raw_action, shared action clip)
    adapter_action = transform(applied_action)
    q_des = default_q + adapter_action * action_scale
    q_des = clip(q_des, shared per-joint clamp)

This test loads the deploy ``ActionAdapter`` and the training-side
``action_adapter_config`` loader from the one canonical YAML and asserts byte-identical
constants plus equivalent transform/decode outputs for representative raw actions
(zero, +/-1, random, and very large extremes). It also pins backward compatibility:
an older YAML with no transform field keeps the legacy identity mapping.

Scope note: the decode-parity leg exercises the deploy adapter against the training
LOADER's reference ``decode`` (the same residual+clamp formula), not the live Isaac
action term — that term needs torch/Isaac and applies the identical constants via
``hope_env_cfg`` (position_clamp / scale / default_q wiring), which these tests pin at
the constants level. Pure NumPy + PyYAML — runs without Isaac / torch / onnxruntime.

Run:  python tests/test_action_adapter_parity.py   (or pytest)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_UTILS = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "utils")
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_SHARED_YAML = os.path.join(
    _REPO, "a3_deploy", "a3_deploy_example", "config", "action_adapter.yaml"
)
_BUNDLE_YAML = os.path.join(
    _REPO, "a3_deploy", "a3_pingpong_deploy_bundle", "config", "action_adapter.yaml"
)


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Training-side loader (pure module, loaded by file path so the Isaac Lab package
# __init__ is never imported).
train_side = _load_by_path(
    "hope_action_adapter_config", os.path.join(_UTILS, "action_adapter_config.py")
)

# Deploy-side ActionAdapter (the reference runner package is import-light).
sys.path.insert(0, _REFERENCE_DIR)
from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter  # noqa: E402
from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES, NUM_JOINTS  # noqa: E402


def _both():
    deploy = ActionAdapter.from_yaml(_SHARED_YAML)
    training = train_side.load_action_adapter_config(path=_SHARED_YAML)
    return deploy, training


def test_shared_config_exists():
    assert os.path.isfile(_SHARED_YAML), _SHARED_YAML
    # The training loader auto-discovers the same canonical file.
    assert os.path.samefile(train_side.find_shared_action_adapter_config(), _SHARED_YAML)


def test_joint_order_matches_deploy():
    training = train_side.load_action_adapter_config(path=_SHARED_YAML)
    assert tuple(training.joint_names) == tuple(JOINT_NAMES)


def test_constants_identical():
    deploy, training = _both()
    assert training.raw_action_transform == deploy.raw_action_transform == "identity"
    assert training.action_clip == deploy.action_clip == (-100.0, 100.0)
    np.testing.assert_array_equal(training.default_q, deploy.default_q)
    np.testing.assert_array_equal(training.action_scale, deploy.action_scale)
    np.testing.assert_array_equal(training.action_scale, np.full(NUM_JOINTS, 0.25))
    np.testing.assert_array_equal(training.clamp_lower, deploy.clamp_lower)
    np.testing.assert_array_equal(training.clamp_upper, deploy.clamp_upper)


def test_decode_parity_on_representative_actions():
    deploy, training = _both()
    rng = np.random.default_rng(0)
    cases = [
        np.zeros(NUM_JOINTS),
        np.ones(NUM_JOINTS),
        -np.ones(NUM_JOINTS),
        rng.uniform(-3.0, 3.0, NUM_JOINTS),
        rng.standard_normal(NUM_JOINTS) * 0.5,
        np.full(NUM_JOINTS, 200.0),    # exercises action clip + positive joint clamp
        np.full(NUM_JOINTS, -200.0),   # exercises action clip + negative joint clamp
    ]
    for raw in cases:
        np.testing.assert_allclose(
            training.transform_raw_action(raw),
            deploy.transform_raw_action(raw),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(training.decode(raw), deploy.decode(raw), rtol=0, atol=0)


def test_linear_mapping_clips_actions_before_scale_and_mechanical_clamp():
    """The shared mapping clips to [-100, 100], then applies the linear 0.25 scale."""
    deploy, training = _both()
    raw = np.linspace(-200.0, 200.0, NUM_JOINTS)
    expected_adapter_action = np.clip(raw, -100.0, 100.0)

    np.testing.assert_array_equal(training.clip_raw_action(raw), expected_adapter_action)
    np.testing.assert_array_equal(deploy.clip_raw_action(raw), expected_adapter_action)
    np.testing.assert_array_equal(
        training.transform_raw_action(raw), expected_adapter_action
    )
    np.testing.assert_array_equal(deploy.transform_raw_action(raw), expected_adapter_action)

    expected_q = np.clip(
        training.default_q + expected_adapter_action * 0.25,
        training.clamp_lower,
        training.clamp_upper,
    )
    np.testing.assert_array_equal(training.decode(raw), expected_q)
    np.testing.assert_array_equal(deploy.decode(raw), expected_q)


def test_uniform_scale_can_reconstruct_both_guarded_reference_motions():
    """Identity + 0.25 can represent every reference pose before the safety clamp."""
    _deploy, training = _both()
    np.testing.assert_array_equal(training.action_scale, np.full(NUM_JOINTS, 0.25))

    motion_dir = os.path.join(_REPO, "hope_training", "motions", "preprocessed")
    clips = (
        "ours_forehand_guarded_1p8s_wrist_x.npz",
        "ours_backhand_guarded_1p8s.npz",
    )
    for clip in clips:
        joint_pos = np.load(os.path.join(motion_dir, clip))["joint_pos"]
        assert np.all(joint_pos >= training.clamp_lower - 1.0e-6), clip
        assert np.all(joint_pos <= training.clamp_upper + 1.0e-6), clip
        raw_action = (joint_pos - training.default_q) / 0.25
        decoded = np.stack([training.decode(raw) for raw in raw_action])
        np.testing.assert_allclose(decoded, joint_pos, rtol=0, atol=1.0e-6)


def test_legacy_yaml_without_transform_or_action_clip_remains_supported():
    """Older configs without the two optional fields retain their original behavior."""
    with open(_SHARED_YAML, "r", encoding="utf-8") as fh:
        legacy = yaml.safe_load(fh)
    legacy.pop("raw_action_transform", None)
    legacy.pop("action_clip", None)
    legacy["action_scale"] = 0.25

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as fh:
        yaml.safe_dump(legacy, fh)
        fh.flush()
        deploy = ActionAdapter.from_yaml(fh.name)
        training = train_side.load_action_adapter_config(path=fh.name)

    assert deploy.raw_action_transform == training.raw_action_transform == "identity"
    assert deploy.action_clip is training.action_clip is None
    np.testing.assert_array_equal(training.action_scale, np.full(NUM_JOINTS, 0.25))
    raw = np.linspace(-2.0, 2.0, NUM_JOINTS)
    expected = np.clip(
        training.default_q + raw * 0.25,
        training.clamp_lower,
        training.clamp_upper,
    )
    np.testing.assert_array_equal(deploy.decode(raw), expected)
    np.testing.assert_array_equal(training.decode(raw), expected)


def test_current_bundle_matches_the_shared_action_contract():
    """The portable model_21500 bundle must ship the same adapter it was trained with."""
    assert os.path.isfile(_BUNDLE_YAML)
    bundle = ActionAdapter.from_yaml(_BUNDLE_YAML)
    shared = ActionAdapter.from_yaml(_SHARED_YAML)
    assert bundle.raw_action_transform == shared.raw_action_transform == "identity"
    assert bundle.action_clip == shared.action_clip == (-100.0, 100.0)
    np.testing.assert_array_equal(bundle.default_q, shared.default_q)
    np.testing.assert_array_equal(bundle.action_scale, shared.action_scale)
    np.testing.assert_array_equal(bundle.action_scale, np.full(NUM_JOINTS, 0.25))
    np.testing.assert_array_equal(bundle.clamp_lower, shared.clamp_lower)
    np.testing.assert_array_equal(bundle.clamp_upper, shared.clamp_upper)


def test_zero_action_returns_default_pose_within_clamp():
    deploy, training = _both()
    np.testing.assert_array_equal(deploy.decode(np.zeros(NUM_JOINTS)), deploy.default_q)
    assert np.all(training.default_q >= training.clamp_lower)
    assert np.all(training.default_q <= training.clamp_upper)


def test_cfg_dict_views_round_trip():
    """The dict views hope_env_cfg feeds into Isaac Lab reproduce the same arrays."""
    _deploy, training = _both()
    dq = training.default_q_by_name()
    sc = training.action_scale_by_name()
    cl = training.position_clamp_by_name()
    assert list(dq.keys()) == list(JOINT_NAMES)
    np.testing.assert_array_equal(
        np.array([dq[n] for n in JOINT_NAMES]), training.default_q
    )
    np.testing.assert_array_equal(
        np.array([sc[n] for n in JOINT_NAMES]), training.action_scale
    )
    lo = np.array([cl[n][0] for n in JOINT_NAMES])
    hi = np.array([cl[n][1] for n in JOINT_NAMES])
    np.testing.assert_array_equal(lo, training.clamp_lower)
    np.testing.assert_array_equal(hi, training.clamp_upper)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} action-adapter parity tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
