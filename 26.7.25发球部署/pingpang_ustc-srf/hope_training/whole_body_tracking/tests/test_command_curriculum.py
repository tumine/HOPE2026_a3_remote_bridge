"""Deterministic tests for reset filtering and target-range curricula."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
HOPE_COMMANDS = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "hope_commands.py"
)
HELPERS = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "command_curriculum.py"
)


def _load_helpers():
    spec = importlib.util.spec_from_file_location("hope_command_curriculum_test_module", HELPERS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


def test_linear_curriculum_clamps_boundaries_and_zero_duration_disables_it():
    assert helpers.linear_curriculum_progress(99, 100, 200) == 0.0
    assert helpers.linear_curriculum_progress(100, 100, 200) == 0.0
    assert helpers.linear_curriculum_progress(200, 100, 200) == 0.5
    assert helpers.linear_curriculum_progress(300, 100, 200) == 1.0
    assert helpers.linear_curriculum_progress(999, 100, 0) == 1.0


def test_curriculum_fraction_and_box_contraction_are_center_preserving():
    box = torch.tensor(
        [
            [[0.0, 4.0], [-4.0, 2.0], [1.0, 2.0]],
            [[10.0, 14.0], [-3.0, 3.0], [-2.0, 6.0]],
        ]
    )
    contracted = helpers.contract_box_about_center(box, 0.25)
    torch.testing.assert_close(contracted.mean(dim=-1), box.mean(dim=-1))
    torch.testing.assert_close(
        contracted[..., 1] - contracted[..., 0],
        0.25 * (box[..., 1] - box[..., 0]),
    )
    assert helpers.curriculum_fraction(0.25, 0.0) == 0.25
    assert helpers.curriculum_fraction(0.25, 0.5) == pytest.approx(0.625)
    assert helpers.curriculum_fraction(0.25, 1.0) == 1.0
    torch.testing.assert_close(helpers.contract_box_about_center(box, 1.0), box)


def test_stable_rsi_filter_requires_dual_level_feet_and_low_motion():
    # Bodies: root, left foot, right foot.  Frames exercise one failed criterion each.
    body_pos = torch.zeros(6, 3, 3)
    body_lin = torch.zeros_like(body_pos)
    body_ang = torch.zeros_like(body_pos)
    body_pos[:, 1, 2] = 0.1
    body_pos[:, 2, 2] = 0.1
    body_pos[1, 2, 2] = 0.2  # unequal foot heights
    body_lin[2, 1, 0] = 0.3  # moving foot
    body_lin[3, 0, 0] = 0.5  # moving root
    body_ang[4, 0, 2] = 0.9  # rotating root

    (frames,) = helpers.stable_rsi_frame_indices(
        body_pos_w=body_pos,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        seg_start=torch.tensor([0]),
        seg_len=torch.tensor([6]),
        foot_body_indices=(1, 2),
        root_body_index=0,
        phase_range=(0.0, 1.0),
        max_foot_height_delta=0.04,
        max_foot_speed=0.25,
        max_root_lin_speed=0.4,
        max_root_ang_speed=0.8,
    )
    assert frames.tolist() == [0, 5]


def test_stable_rsi_filter_obeys_per_segment_phase_window_and_empty_result():
    body_pos = torch.zeros(10, 3, 3)
    body_lin = torch.zeros_like(body_pos)
    body_ang = torch.zeros_like(body_pos)
    # Make the second segment unstable so the command can exercise its configured fallback.
    body_lin[5:, 1, 0] = 1.0

    first, second = helpers.stable_rsi_frame_indices(
        body_pos_w=body_pos,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        seg_start=torch.tensor([0, 5]),
        seg_len=torch.tensor([5, 5]),
        foot_body_indices=(1, 2),
        root_body_index=0,
        phase_range=(0.25, 0.75),
        max_foot_height_delta=0.04,
        max_foot_speed=0.25,
        max_root_lin_speed=0.4,
        max_root_ang_speed=0.8,
    )
    assert first.tolist() == [1, 2, 3]
    assert second.numel() == 0


@pytest.mark.parametrize("phase_range", [(-0.1, 1.0), (0.8, 0.2), (0.0, 1.1)])
def test_stable_rsi_filter_rejects_invalid_phase_ranges(phase_range):
    bodies = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match="rsi_phase_range"):
        helpers.stable_rsi_frame_indices(
            body_pos_w=bodies,
            body_lin_vel_w=bodies,
            body_ang_vel_w=bodies,
            seg_start=torch.tensor([0]),
            seg_len=torch.tensor([1]),
            foot_body_indices=(1, 2),
            root_body_index=0,
            phase_range=phase_range,
            max_foot_height_delta=0.04,
            max_foot_speed=0.25,
            max_root_lin_speed=0.4,
            max_root_ang_speed=0.8,
        )


def test_racket_command_logs_curriculum_and_two_foot_contact_diagnostics():
    source = HOPE_COMMANDS.read_text(encoding="utf-8")
    assert '"target_curriculum_progress"' in source
    assert '"feet_contact_fraction"' in source
    assert '"both_feet_contact"' in source
    assert '"station_error"' in source
    assert '"base_tilt"' in source
    assert '"base_height"' in source
    assert "(self.feet_contact_frac >= 1.0).float()" in source


def test_shipped_clips_have_stable_rsi_candidates_under_training_thresholds():
    motion_dir = ROOT.parent / "motions/preprocessed"
    clips = (
        ("ours_forehand_guarded_1p8s_wrist_x.npz", 51),
        ("ours_backhand_guarded_1p8s.npz", 82),
    )
    for filename, expected_count in clips:
        path = motion_dir / filename
        metadata = yaml.safe_load(path.with_suffix(".yaml").read_text(encoding="utf-8"))
        body_names = metadata["tracked_bodies"]
        with np.load(path) as motion:
            body_pos = torch.from_numpy(motion["body_pos_w"])
            body_lin = torch.from_numpy(motion["body_lin_vel_w"])
            body_ang = torch.from_numpy(motion["body_ang_vel_w"])

        (frames,) = helpers.stable_rsi_frame_indices(
            body_pos_w=body_pos,
            body_lin_vel_w=body_lin,
            body_ang_vel_w=body_ang,
            seg_start=torch.tensor([0]),
            seg_len=torch.tensor([body_pos.shape[0]]),
            foot_body_indices=(
                body_names.index("left_ankle_roll_Link"),
                body_names.index("right_ankle_roll_Link"),
            ),
            root_body_index=body_names.index("pelvis_link"),
            phase_range=(0.0, 0.9),
            max_foot_height_delta=0.04,
            max_foot_speed=0.15,
            max_root_lin_speed=0.25,
            max_root_ang_speed=0.60,
        )
        assert frames.numel() == expected_count
