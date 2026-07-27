"""Pure unit tests for inference-only rsl_rl checkpoint loading and export provenance."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_UTILS = _ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _UTILS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compat = _load("hope_checkpoint_compat_test", "checkpoint_compat.py")
exporter = _load("hope_exporter_checkpoint_test", "exporter.py")


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(3, 4), nn.ELU(), nn.Linear(4, 2))
        self.critic = nn.Linear(9, 1)


class _Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, 3))
        self.register_buffer("_var", torch.ones(1, 3))
        self.register_buffer("_std", torch.ones(1, 3))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, value):
        return (value - self._mean) / (self._std + 1.0e-2)


def _write_checkpoint(
    run_dir: pathlib.Path,
    *,
    actor: nn.Module,
    normalizer: _Normalizer | None = None,
    source_empirical: bool = False,
) -> pathlib.Path:
    run_dir.mkdir(parents=True)
    state = {f"actor.{key}": value.detach().clone() for key, value in actor.state_dict().items()}
    # Deliberately incompatible critic state and unusable optimizer data. Inference must ignore both.
    state["critic.0.weight"] = torch.randn(17, 23)
    payload = {
        "model_state_dict": state,
        "optimizer_state_dict": {"broken": object()},
        "iter": 123,
        "infos": {},
    }
    if normalizer is not None:
        payload["obs_norm_state_dict"] = normalizer.state_dict()
    checkpoint = run_dir / "model_123.pt"
    torch.save(payload, checkpoint)
    params = run_dir / "params"
    params.mkdir()
    (params / "agent.yaml").write_text(
        "empirical_normalization: "
        + ("true" if source_empirical else "false")
        + "\npolicy:\n"
        + "  actor_hidden_dims: [4]\n"
        + "  activation: elu\n"
        + "  init_noise_std: 0.7\n",
        encoding="utf-8",
    )
    (params / "env.yaml").write_text("task: test\n", encoding="utf-8")
    return checkpoint


def test_actor_only_load_ignores_incompatible_critic_and_optimizer(tmp_path):
    source_actor = _Policy().actor
    with torch.no_grad():
        for parameter in source_actor.parameters():
            parameter.fill_(0.25)
    checkpoint = _write_checkpoint(tmp_path / "run", actor=source_actor)

    live_policy = _Policy()
    critic_before = {key: value.clone() for key, value in live_policy.critic.state_dict().items()}
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=live_policy),
        obs_normalizer=nn.Identity(),
    )
    info = compat.inspect_checkpoint(str(checkpoint))
    compat.load_actor_for_inference(runner, str(checkpoint), info)

    for actual, expected in zip(live_policy.actor.parameters(), source_actor.parameters()):
        torch.testing.assert_close(actual, expected)
    for key, value in live_policy.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key])


def test_configure_and_load_empirical_normalizer_from_checkpoint(tmp_path):
    source_policy = _Policy()
    source_norm = _Normalizer()
    source_norm._mean.fill_(2.0)
    source_norm._std.fill_(0.5)
    source_norm.count.fill_(100)
    checkpoint = _write_checkpoint(
        tmp_path / "run",
        actor=source_policy.actor,
        normalizer=source_norm,
        source_empirical=True,
    )
    cfg = SimpleNamespace(
        empirical_normalization=False,
        policy=SimpleNamespace(actor_hidden_dims=[99], activation="relu", init_noise_std=1.0),
    )
    info = compat.configure_runner_for_inference(cfg, str(checkpoint))
    assert cfg.empirical_normalization is True
    assert cfg.policy.actor_hidden_dims == [4]
    assert cfg.policy.activation == "elu"
    assert cfg.policy.init_noise_std == 0.7

    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=_Policy()),
        obs_normalizer=_Normalizer(),
    )
    compat.load_actor_for_inference(runner, str(checkpoint), info)
    torch.testing.assert_close(runner.obs_normalizer._mean, source_norm._mean)
    torch.testing.assert_close(runner.obs_normalizer._std, source_norm._std)
    assert not runner.obs_normalizer.training


def test_missing_required_empirical_stats_is_rejected(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path / "run",
        actor=_Policy().actor,
        source_empirical=True,
    )
    cfg = SimpleNamespace(
        empirical_normalization=False,
        policy=SimpleNamespace(actor_hidden_dims=[4], activation="elu", init_noise_std=1.0),
    )
    with pytest.raises(RuntimeError, match="disagree about empirical"):
        compat.configure_runner_for_inference(cfg, str(checkpoint))


def test_missing_source_agent_config_is_rejected_instead_of_guessing_activation(tmp_path):
    source_policy = _Policy()
    checkpoint = tmp_path / "orphan_model.pt"
    torch.save(
        {
            "model_state_dict": {
                f"actor.{key}": value for key, value in source_policy.actor.state_dict().items()
            },
            "iter": 1,
        },
        checkpoint,
    )
    cfg = SimpleNamespace(
        empirical_normalization=False,
        policy=SimpleNamespace(actor_hidden_dims=[4], activation="elu", init_noise_std=1.0),
    )
    with pytest.raises(RuntimeError, match="activation cannot be inferred safely"):
        compat.configure_runner_for_inference(cfg, str(checkpoint))


def test_explicit_nested_ppo_config_recovers_orphan_checkpoint(tmp_path):
    source_policy = _Policy()
    checkpoint = tmp_path / "orphan_model.pt"
    torch.save(
        {
            "model_state_dict": {
                f"actor.{key}": value for key, value in source_policy.actor.state_dict().items()
            },
            "iter": 2,
        },
        checkpoint,
    )
    ppo_yaml = tmp_path / "original_ppo.yaml"
    ppo_yaml.write_text(
        "runner:\n  empirical_normalization: false\n"
        "policy:\n  actor_hidden_dims: [4]\n  activation: elu\n  init_noise_std: 0.8\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        empirical_normalization=True,
        policy=SimpleNamespace(actor_hidden_dims=[99], activation="relu", init_noise_std=1.0),
    )
    info = compat.inspect_checkpoint(str(checkpoint), str(ppo_yaml))
    compat.configure_runner_for_inference(cfg, str(checkpoint), info)
    assert cfg.empirical_normalization is False
    assert cfg.policy.actor_hidden_dims == [4]
    assert cfg.policy.activation == "elu"
    assert info["agent_config_path"] == str(ppo_yaml)


def test_manifest_provenance_has_checkpoint_and_source_config_hashes(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "run", actor=_Policy().actor)
    provenance = compat.build_checkpoint_provenance(str(checkpoint))
    assert provenance["checkpoint"]["file"] == "model_123.pt"
    assert provenance["checkpoint"]["iteration"] == 123
    assert provenance["checkpoint"]["sha256"] == compat.sha256_file(str(checkpoint))
    assert {entry["role"] for entry in provenance["source_configs"]} == {"agent", "environment"}
    assert all(len(entry["sha256"]) == 64 for entry in provenance["source_configs"])

    manifest = exporter.build_manifest(
        joint_names=[f"j{i}" for i in range(31)],
        embedded_observation_normalization="empirical",
        provenance=provenance,
    )
    assert manifest["observation_normalization"] == "none"
    assert manifest["embedded_observation_normalization"] == "empirical"
    assert manifest["provenance"] == provenance
    # Prove the optional extension remains ordinary portable JSON.
    json.dumps(manifest)


def test_external_empirical_normalization_contract_is_rejected():
    with pytest.raises(ValueError, match="raw observations"):
        exporter.build_manifest(observation_normalization="empirical")


def test_export_folds_empirical_normalizer_but_keeps_raw_external_contract(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    class ContractPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(nn.Linear(111, 31))

    class ContractNormalizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.linspace(-1.0, 1.0, 111))
            self.register_buffer("scale", torch.linspace(0.5, 1.5, 111))

        def forward(self, value):
            return (value - self.mean) / self.scale

    policy = ContractPolicy().eval()
    normalizer = ContractNormalizer().eval()
    onnx_path, manifest_path = exporter.export_policy(
        policy,
        str(tmp_path),
        joint_names=[f"j{i}" for i in range(31)],
        normalizer=normalizer,
    )
    raw_obs = torch.linspace(-2.0, 2.0, 111).reshape(1, -1)
    expected = policy.actor(normalizer(raw_obs)).detach().numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual = session.run(["raw_action"], {"observation": raw_obs.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected), rtol=1e-5, atol=1e-6)

    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["observation_normalization"] == "none"
    assert manifest["embedded_observation_normalization"] == "empirical"
    metadata = {item.key: item.value for item in onnx.load(onnx_path).metadata_props}
    assert metadata["observation_normalization"] == "none"
    assert metadata["embedded_observation_normalization"] == "empirical"
