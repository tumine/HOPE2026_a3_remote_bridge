"""Regressions for the guarded 1.8-second HOPE motion assets."""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
MOTIONS = REPO / "hope_training/motions/preprocessed"
sys.path.insert(0, str(SCRIPTS))

import retime_motion_npz as retime  # noqa: E402


MOTION_CASES = (
    (
        "ours_forehand_cropped_smooth15_wrist_x",
        "ours_forehand_guarded_1p8s_wrist_x",
    ),
    ("ours_backhand_cropped", "ours_backhand_guarded_1p8s"),
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _quat_apply(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    scalar = quaternion_wxyz[..., :1]
    xyz = quaternion_wxyz[..., 1:]
    first_cross = np.cross(xyz, vector)
    return vector + 2.0 * (
        scalar * first_cross + np.cross(xyz, first_cross)
    )


def _racket_center_state(
    motion: dict[str, np.ndarray], metadata: dict, frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    racket_index = metadata["tracked_bodies"].index(metadata["racket_link"])
    quaternion = motion["body_quat_w"][frame, racket_index].astype(np.float64)
    offset_w = _quat_apply(
        quaternion, np.asarray(metadata["mount_offset_xyz"], dtype=np.float64)
    )
    position = (
        motion["body_pos_w"][frame, racket_index].astype(np.float64) + offset_w
    )
    velocity = motion["body_lin_vel_w"][frame, racket_index].astype(
        np.float64
    ) + np.cross(
        motion["body_ang_vel_w"][frame, racket_index].astype(np.float64),
        offset_w,
    )
    return position, quaternion, velocity


def test_cubic_source_map_is_monotone_and_preserves_explicit_guard():
    source_frames = retime.build_source_frame_map(
        start_frame=72,
        end_frame=320,
        guard_start_frame=174,
        guard_end_frame=186,
        output_frame_count=91,
        output_guard_start_frame=44,
        output_guard_end_frame=56,
    )

    assert source_frames.shape == (91,)
    assert source_frames[0] == 72
    assert source_frames[-1] == 320
    assert np.all(np.diff(source_frames) > 0.0)
    np.testing.assert_array_equal(source_frames[44:57], np.arange(174, 187))
    assert source_frames[50] == 180


def test_motion_aware_map_shares_cost_and_adds_derivative_buffer():
    references = []
    for source_stem, _ in MOTION_CASES:
        references.append(retime._load_motion(MOTIONS / f"{source_stem}.npz"))
    source_frames, diagnostics = retime.build_motion_aware_source_frame_map(
        reference_motions=references,
        fps=50.0,
        start_frame=72,
        end_frame=320,
        guard_start_frame=174,
        guard_end_frame=186,
        output_frame_count=91,
        output_guard_start_frame=44,
        output_guard_end_frame=56,
    )

    np.testing.assert_array_equal(source_frames[43:58], np.arange(173, 188))
    assert source_frames[0] == 72
    assert source_frames[50] == 180
    assert source_frames[-1] == 320
    assert np.diff(source_frames).min() >= 1.0 - 1.0e-10
    assert np.diff(source_frames).max() <= 8.0 + 1.0e-10
    assert diagnostics["unit_speed_derivative_buffer_source_frames"] == [173, 187]
    assert diagnostics["unit_speed_derivative_buffer_output_frames"] == [43, 57]


@pytest.mark.parametrize(("source_stem", "output_stem"), MOTION_CASES)
def test_guarded_motion_asset_preserves_strike_and_has_safe_endpoints(
    source_stem: str, output_stem: str
):
    source_path = MOTIONS / f"{source_stem}.npz"
    output_path = MOTIONS / f"{output_stem}.npz"
    source_metadata = yaml.safe_load(
        source_path.with_suffix(".yaml").read_text(encoding="utf-8")
    )
    output_metadata = yaml.safe_load(
        output_path.with_suffix(".yaml").read_text(encoding="utf-8")
    )
    with np.load(source_path, allow_pickle=False) as loaded:
        source = {key: loaded[key].copy() for key in loaded.files}
    with np.load(output_path, allow_pickle=False) as loaded:
        output = {key: loaded[key].copy() for key in loaded.files}

    assert float(output["fps"]) == 50.0
    assert output_metadata["frame_count"] == 91
    assert output_metadata["duration_s"] == 1.8
    assert output_metadata["strike_frame"] == 50
    assert output_metadata["strike_phase"] == pytest.approx(50.0 / 90.0)
    assert output_metadata["retiming"]["source_strike_frame"] == 180
    assert output_metadata["retiming"]["source_knots_frames"] == [
        72,
        174,
        186,
        320,
    ]
    assert output_metadata["retiming"]["output_knots_frames"] == [0, 44, 56, 90]
    assert output_metadata["retiming"]["guard_time_scale"] == 1.0
    assert output_metadata["retiming"]["method"].startswith(
        "shared_bounded_motion_aware"
    )
    assert output_metadata["retiming"]["source_npz_sha256"] == _sha256(source_path)
    assert output_metadata["retiming"]["source_sidecar_sha256"] == _sha256(
        source_path.with_suffix(".yaml")
    )

    for key in retime.TIME_SERIES_KEYS:
        assert output[key].shape[0] == 91
        assert output[key].dtype == np.float32
        assert np.isfinite(output[key]).all()

    # The unit-speed guard has a one-frame buffer around the actual +/-0.12 s
    # strike reward window.  Consequently every pose and finite-difference
    # velocity in output frames 44..56 is the exact source sample 174..186.
    for key in ("joint_pos", "body_pos_w", "body_quat_w"):
        np.testing.assert_array_equal(output[key][43:58], source[key][173:188])
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_allclose(
            output[key][44:57], source[key][174:187], rtol=0.0, atol=2.0e-5
        )
    for key in retime.TIME_SERIES_KEYS:
        np.testing.assert_allclose(
            output[key][50], source[key][180], rtol=0.0, atol=2.0e-5
        )

    source_racket = _racket_center_state(source, source_metadata, 180)
    output_racket = _racket_center_state(output, output_metadata, 50)
    for source_value, output_value in zip(
        source_racket, output_racket, strict=True
    ):
        np.testing.assert_allclose(
            output_value, source_value, rtol=0.0, atol=2.0e-5
        )

    quaternion_norm = np.linalg.norm(output["body_quat_w"], axis=-1)
    np.testing.assert_allclose(quaternion_norm, 1.0, rtol=0.0, atol=2.0e-6)

    endpoint_validation = output_metadata["retiming"]["endpoint_validation"]
    max_joint_speed = float(np.max(np.abs(output["joint_vel"][[0, -1]])))
    endpoint_racket_speed = max(
        np.linalg.norm(_racket_center_state(output, output_metadata, frame)[2])
        for frame in (0, -1)
    )
    assert max_joint_speed == pytest.approx(
        endpoint_validation["max_abs_joint_speed_rad_s"]
    )
    assert endpoint_racket_speed == pytest.approx(
        endpoint_validation["max_racket_center_linear_speed_m_s"]
    )
    assert max_joint_speed <= 0.5
    assert endpoint_racket_speed <= 0.15

    diagnostics = output_metadata["retiming"]["kinematic_diagnostics"]
    max_all_joint_speed = float(np.max(np.abs(output["joint_vel"])))
    max_all_joint_acceleration = float(
        np.max(np.abs(np.gradient(output["joint_vel"], 1.0 / 50.0, axis=0, edge_order=2)))
    )
    all_racket_speeds = np.array(
        [
            np.linalg.norm(_racket_center_state(output, output_metadata, frame)[2])
            for frame in range(output["joint_pos"].shape[0])
        ]
    )
    assert max_all_joint_speed == pytest.approx(
        diagnostics["max_abs_joint_speed_rad_s"]
    )
    assert max_all_joint_acceleration == pytest.approx(
        diagnostics["max_abs_joint_acceleration_rad_s2"]
    )
    assert float(all_racket_speeds.max()) == pytest.approx(
        diagnostics["max_racket_center_linear_speed_m_s"]
    )
    assert int(all_racket_speeds.argmax()) == diagnostics[
        "max_racket_center_linear_speed_frame"
    ]
    velocity_limits = np.asarray(
        retime._A3_JOINT_VELOCITY_LIMITS_RAD_S, dtype=np.float64
    )
    assert float(np.max(np.abs(output["joint_vel"]) / velocity_limits)) <= 0.50
    assert max_all_joint_acceleration <= 85.0
    assert float(all_racket_speeds.max()) <= 3.0


def test_all_guarded_clip_transitions_have_bounded_joint_jump():
    joint_positions = []
    for _, output_stem in MOTION_CASES:
        with np.load(MOTIONS / f"{output_stem}.npz", allow_pickle=False) as motion:
            joint_positions.append(motion["joint_pos"].copy())

    for current in joint_positions:
        for following in joint_positions:
            max_joint_jump = float(np.max(np.abs(following[0] - current[-1])))
            assert max_joint_jump <= 0.25


@pytest.mark.parametrize(("source_stem", "output_stem"), MOTION_CASES)
def test_guarded_motion_inverse_dynamics_keeps_effort_headroom(
    source_stem: str, output_stem: str
):
    """A prescribed free-base trajectory stays below 2/3 nominal effort.

    This is a conservative MuJoCo inverse-dynamics diagnostic, not a hardware
    certification. Lifting the root by 1 m removes floor constraints while
    retaining gravity; the remaining margin covers the training mass/COM
    randomization and closed-loop tracking error.
    """

    del source_stem
    mujoco = pytest.importorskip("mujoco")
    model_path = (
        REPO
        / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    metadata = yaml.safe_load(
        (MOTIONS / f"{output_stem}.yaml").read_text(encoding="utf-8")
    )
    with np.load(MOTIONS / f"{output_stem}.npz", allow_pickle=False) as loaded:
        motion = {key: loaded[key].copy() for key in loaded.files}

    joint_qpos_addresses = []
    joint_dof_addresses = []
    effort_limits = []
    for joint_name in metadata["joint_order"]:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        assert joint_id >= 0
        joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        assert actuator_ids.size == 1
        effort_limits.append(
            float(np.max(np.abs(model.actuator_ctrlrange[actuator_ids[0]])))
        )

    frame_count = motion["joint_pos"].shape[0]
    qpos = np.tile(model.qpos0, (frame_count, 1))
    root_index = metadata["tracked_bodies"].index(metadata["root_body"])
    qpos[:, :3] = motion["body_pos_w"][:, root_index]
    qpos[:, 2] += 1.0
    qpos[:, 3:7] = motion["body_quat_w"][:, root_index]
    qpos[:, joint_qpos_addresses] = motion["joint_pos"]

    dt = 1.0 / float(motion["fps"])
    qvel = np.zeros((frame_count, model.nv), dtype=np.float64)
    for frame in range(frame_count):
        if frame == 0:
            previous, following, interval = 0, 1, dt
        elif frame == frame_count - 1:
            previous, following, interval = frame - 1, frame, dt
        else:
            previous, following, interval = frame - 1, frame + 1, 2.0 * dt
        mujoco.mj_differentiatePos(
            model,
            qvel[frame],
            interval,
            qpos[previous],
            qpos[following],
        )
    qacc = np.gradient(qvel, dt, axis=0, edge_order=2)

    max_effort_ratio = 0.0
    effort_limits = np.asarray(effort_limits, dtype=np.float64)
    for frame in range(frame_count):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        data.qacc[:] = qacc[frame]
        mujoco.mj_inverse(model, data)
        max_effort_ratio = max(
            max_effort_ratio,
            float(
                np.max(
                    np.abs(data.qfrc_inverse[joint_dof_addresses])
                    / effort_limits
                )
            ),
        )
    assert max_effort_ratio <= 2.0 / 3.0


@pytest.mark.parametrize(("source_stem", "output_stem"), MOTION_CASES)
def test_checked_in_asset_is_reproducible_from_retime_script(
    tmp_path: pathlib.Path, source_stem: str, output_stem: str
):
    regenerated = tmp_path / f"{output_stem}.npz"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "retime_motion_npz.py"),
            "--input-file",
            str(MOTIONS / f"{source_stem}.npz"),
            "--output-file",
            str(regenerated),
            "--start-frame",
            "72",
            "--end-frame",
            "320",
            "--guard-start-frame",
            "174",
            "--guard-end-frame",
            "186",
            "--output-frame-count",
            "91",
            "--output-guard-start-frame",
            "44",
            "--output-guard-end-frame",
            "56",
            "--strike-frame",
            "180",
            "--motion-name",
            output_stem,
            "--timing-reference-file",
            str(MOTIONS / f"{MOTION_CASES[0][0]}.npz"),
            "--timing-reference-file",
            str(MOTIONS / f"{MOTION_CASES[1][0]}.npz"),
        ],
        check=True,
        cwd=REPO,
    )

    with np.load(regenerated, allow_pickle=False) as actual, np.load(
        MOTIONS / f"{output_stem}.npz", allow_pickle=False
    ) as expected:
        assert actual.files == expected.files
        for key in actual.files:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert regenerated.with_suffix(".yaml").read_text(
        encoding="utf-8"
    ) == (MOTIONS / f"{output_stem}.yaml").read_text(encoding="utf-8")
