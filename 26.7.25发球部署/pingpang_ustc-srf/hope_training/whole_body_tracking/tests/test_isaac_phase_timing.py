"""Deterministic tests for Isaac-side reset/transition strike timing.

These tests deliberately avoid launching Isaac Sim. The timing helpers are loaded by
file path, and the task-local environment step is driven with manager fakes so its
observable call order is tested rather than inferred from comments.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACKING = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
)


def _load_file(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


timing = _load_file("hope_timing_test_module", TRACKING / "mdp/hope_timing.py")


def test_reset_tts_uses_new_motion_frame_immediately():
    """Stand starts at +1 s; RSI starts at its sampled frame, never at stale zero."""
    strike, tts, pre, window = timing.compute_strike_timing(
        seg_start=torch.tensor([0, 91, 91]),
        seg_len=torch.tensor([91, 91, 91]),
        strike_phase=torch.tensor([50.0 / 90.0] * 3),
        time_steps=torch.tensor([0, 91 + 23, 91 + 50]),
        step_dt=0.02,
        strike_window_s=0.12,
    )
    assert strike.tolist() == [50, 141, 141]
    torch.testing.assert_close(tts, torch.tensor([1.0, 0.54, 0.0]))
    assert pre.tolist() == [True, True, False]
    assert window.tolist() == [False, False, True]


def test_exact_strike_event_is_consumed_once():
    armed = torch.tensor([True, True, False])
    tts = torch.tensor([0.0, 0.009, 0.0])

    first, armed = timing.consume_exact_strike(tts, step_dt=0.02, armed=armed)
    second, armed = timing.consume_exact_strike(tts, step_dt=0.02, armed=armed)

    assert first.tolist() == [True, True, False]
    assert second.tolist() == [False, False, False]
    assert armed.tolist() == [False, False, False]


def _load_tracking_env_with_isaac_stub():
    """Load only the local step override without importing Omniverse."""
    names = ("isaaclab", "isaaclab.envs", "isaaclab.envs.common")
    saved = {name: sys.modules.get(name) for name in names}
    isaaclab = types.ModuleType("isaaclab")
    envs = types.ModuleType("isaaclab.envs")
    common = types.ModuleType("isaaclab.envs.common")

    class _ManagerBasedRLEnv:
        pass

    envs.ManagerBasedRLEnv = _ManagerBasedRLEnv
    common.VecEnvStepReturn = tuple
    isaaclab.envs = envs
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.envs": envs,
            "isaaclab.envs.common": common,
        }
    )
    try:
        return _load_file("hope_tracking_env_test_module", TRACKING / "tracking_env.py")
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class _ActionManager:
    def __init__(self, log):
        self.log = log

    def process_action(self, _action):
        self.log.append("process_action")

    def apply_action(self):
        self.log.append("apply_action")


class _RecorderManager:
    active_terms = ()

    def __init__(self, log):
        self.log = log

    def record_pre_step(self):
        self.log.append("record_pre_step")


class _Sim:
    def __init__(self, log):
        self.log = log

    def has_gui(self):
        return False

    def has_rtx_sensors(self):
        return False

    def step(self, render=False):
        assert render is False
        self.log.append("physics")


class _Scene:
    def __init__(self, log):
        self.log = log
        self.physics_generation = 0

    def write_data_to_sim(self):
        self.log.append("write")

    def update(self, dt):
        assert dt == 0.005
        self.physics_generation += 1
        self.log.append("scene_update")


class _TerminationManager:
    def __init__(self, log):
        self.log = log
        self.terminated = torch.tensor([False])
        self.time_outs = torch.tensor([False])

    def compute(self):
        self.log.append("termination")
        return torch.tensor([False])


class _CommandManager:
    def __init__(self, fake):
        self.fake = fake
        self.calls = 0

    def compute(self, dt):
        assert dt == 0.02
        assert self.fake.scene.physics_generation == 1
        self.calls += 1
        self.fake.command_phase += 1
        self.fake.log.append("command")


class _RewardManager:
    def __init__(self, fake):
        self.fake = fake
        self.seen_phase = None

    def compute(self, dt):
        assert dt == 0.02
        self.seen_phase = self.fake.command_phase
        self.fake.log.append("reward")
        return torch.tensor([1.0])


class _ObservationManager:
    def __init__(self, fake):
        self.fake = fake
        self.seen_phase = None

    def compute(self, update_history=False):
        assert update_history is True
        self.seen_phase = self.fake.command_phase
        self.fake.log.append("observation")
        return {"policy": torch.zeros(1, 111)}


class _EventManager:
    available_modes = ()


class _FakeEnv:
    def __init__(self):
        self.log = []
        self.device = "cpu"
        self.cfg = types.SimpleNamespace(
            decimation=1,
            sim=types.SimpleNamespace(render_interval=1),
            rerender_on_reset=False,
        )
        self.physics_dt = 0.005
        self.step_dt = 0.02
        self._sim_step_counter = 0
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.common_step_counter = 0
        self.command_phase = 17
        self.action_manager = _ActionManager(self.log)
        self.recorder_manager = _RecorderManager(self.log)
        self.sim = _Sim(self.log)
        self.scene = _Scene(self.log)
        self.termination_manager = _TerminationManager(self.log)
        self.command_manager = _CommandManager(self)
        self.reward_manager = _RewardManager(self)
        self.observation_manager = _ObservationManager(self)
        self.event_manager = _EventManager()
        self.extras = {}


def test_post_physics_command_reward_observation_share_one_phase():
    module = _load_tracking_env_with_isaac_stub()
    fake = _FakeEnv()
    result = module.PhaseAlignedManagerBasedRLEnv.step(fake, torch.zeros(1, 31))

    assert fake.log.index("physics") < fake.log.index("command") < fake.log.index("reward")
    assert fake.log.index("reward") < fake.log.index("observation")
    assert fake.command_manager.calls == 1
    assert fake.reward_manager.seen_phase == 18
    assert fake.observation_manager.seen_phase == 18
    assert result[1].item() == 1.0


def test_racket_reset_initializes_timing_after_base_resample():
    """Structural guard for the order-sensitive CommandTerm.reset override."""
    source = (TRACKING / "mdp/hope_commands.py").read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RacketTargetCommand")
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset")

    calls = []
    for node in ast.walk(reset):
        if isinstance(node, ast.Call):
            calls.append((getattr(node.func, "attr", ""), node.lineno))
    super_reset_line = next(line for name, line in calls if name == "reset")
    timing_line = next(line for name, line in calls if name == "_compute_strike_timing")
    assert super_reset_line < timing_line


def test_public_task_uses_phase_aligned_environment():
    registration = (TRACKING / "config/agibot_a3/__init__.py").read_text()
    assert "tracking_env:PhaseAlignedManagerBasedRLEnv" in registration
    assert 'entry_point="isaaclab.envs:ManagerBasedRLEnv"' not in registration
