"""MuJoCo integration check for the calibrated racket impulse and outgoing flight."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import yaml


mujoco = pytest.importorskip("mujoco")

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "hope_training/whole_body_tracking/scripts"
sys.path.insert(0, str(SCRIPTS))

from mujoco_pingpong_scene import PingPongRealPhysicsScene  # noqa: E402


def _load_success_metric():
    path = (
        ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/utils/success_metric.py"
    )
    spec = importlib.util.spec_from_file_location("success_metric_mujoco_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_racket_contact_and_flight_match_shared_model():
    metric = _load_success_metric()
    motion_path = (
        ROOT
        / "hope_training/motions/preprocessed/"
        "ours_forehand_guarded_1p8s_wrist_x.npz"
    )
    metadata = yaml.safe_load(motion_path.with_suffix(".yaml").read_text())
    motion = np.load(motion_path)
    model_path = (
        ROOT
        / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    )
    ball_cfg = metric.load_ball_physics_config()
    physics = metric.BallPhysics.from_config(ball_cfg)
    paddle = metric.PaddlePhysics.from_config(ball_cfg)
    table = metric.TableGeometry.from_config(ball_cfg)
    scene = PingPongRealPhysicsScene(
        str(model_path),
        ball_cfg,
        metadata["joint_order"],
        control_dt=0.02,
        near_edge_x=0.5,
        table_center_y=0.0,
        paddle_contact_model=metric.predict_paddle_contact,
        paddle_physics=paddle,
    )
    try:
        scene.reset_stand()
        frame = int(metadata["strike_frame"])
        root_index = metadata["tracked_bodies"].index(metadata["root_body"])
        data = scene.data
        data.qpos[scene._base_qadr:scene._base_qadr + 3] = motion["body_pos_w"][
            frame, root_index
        ]
        data.qpos[scene._base_qadr + 3:scene._base_qadr + 7] = motion["body_quat_w"][
            frame, root_index
        ]
        data.qpos[scene._q_adr] = motion["joint_pos"][frame]
        data.qvel[:] = 0.0
        data.qvel[scene._v_adr] = motion["joint_vel"][frame]
        scene._mj.mj_forward(scene.model, data)

        racket_pos, racket_vel = scene.racket_site_state()
        racket_normal = data.site_xmat[scene._racket_sid].reshape(3, 3)[:, 1].copy()
        incoming = np.array([-2.23, 0.0, -3.95])
        side = (
            racket_normal
            if np.dot(incoming - racket_vel, racket_normal) < 0.0
            else -racket_normal
        )
        scene.set_ball(racket_pos + side * 0.045, incoming)
        events = scene.step()
        assert events.ball_racket_contact
        assert scene.last_paddle_contact is not None
        contact = scene.last_paddle_contact

        expected_outgoing = metric.predict_paddle_contact(
            contact["incoming_ball_velocity"],
            contact["racket_velocity"],
            contact["racket_normal"],
            paddle,
        )
        assert np.allclose(
            contact["outgoing_ball_velocity"], expected_outgoing, atol=1.0e-12
        )

        predicted = metric.integrate_outgoing_ball(
            scene.to_table(contact["resolved_ball_position"]),
            contact["outgoing_ball_velocity"],
            physics,
            table,
            dt=0.0005,
            max_time=2.0,
        )
        actual_landing = None
        actual_net_clear = False
        for _ in range(100):
            step_events = scene.step()
            actual_net_clear |= any(
                direction > 0.0
                and z > table.net_height + physics.ball_radius
                for z, direction in step_events.net_crossings
            )
            for x, y, direction in step_events.surface_crossings:
                if direction < 0.0:
                    actual_landing = (x, y)
                    break
            if actual_landing is not None:
                break

        assert predicted.landing_xy is not None
        assert actual_landing is not None
        assert np.linalg.norm(
            np.asarray(actual_landing) - np.asarray(predicted.landing_xy)
        ) < 0.005
        assert actual_net_clear == predicted.net_clear
    finally:
        scene.close()


def test_contact_model_receives_opposite_forehand_backhand_face_normals():
    """The fitted impulse must see +Y for FH and -Y for BH, as training does."""

    metric = _load_success_metric()
    motion_path = (
        ROOT
        / "hope_training/motions/preprocessed/"
        "ours_forehand_guarded_1p8s_wrist_x.npz"
    )
    metadata = yaml.safe_load(motion_path.with_suffix(".yaml").read_text())
    motion = np.load(motion_path)
    model_path = (
        ROOT
        / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    )
    ball_cfg = metric.load_ball_physics_config()
    paddle = metric.PaddlePhysics.from_config(ball_cfg)
    received_normals = []

    def _capturing_contact(ball_velocity, racket_velocity, racket_normal, physics):
        received_normals.append(np.asarray(racket_normal).copy())
        return metric.predict_paddle_contact(
            ball_velocity, racket_velocity, racket_normal, physics
        )

    scene = PingPongRealPhysicsScene(
        str(model_path),
        ball_cfg,
        metadata["joint_order"],
        control_dt=0.02,
        near_edge_x=0.5,
        table_center_y=0.0,
        paddle_contact_model=_capturing_contact,
        paddle_physics=paddle,
    )
    try:
        frame = int(metadata["strike_frame"])
        root_index = metadata["tracked_bodies"].index(metadata["root_body"])
        incoming = np.array([-2.23, 0.0, -3.95])

        for face_sign in (1.0, -1.0):
            # Select before reset to prove reset_stand cannot lose the trial side.
            scene.set_paddle_face_sign(face_sign)
            scene.reset_stand()
            assert scene._paddle_face_sign == face_sign

            data = scene.data
            data.qpos[scene._base_qadr:scene._base_qadr + 3] = motion["body_pos_w"][
                frame, root_index
            ]
            data.qpos[scene._base_qadr + 3:scene._base_qadr + 7] = motion["body_quat_w"][
                frame, root_index
            ]
            data.qpos[scene._q_adr] = motion["joint_pos"][frame]
            data.qvel[:] = 0.0
            data.qvel[scene._v_adr] = motion["joint_vel"][frame]
            scene._mj.mj_forward(scene.model, data)

            racket_pos, racket_vel = scene.racket_site_state()
            geometric_axis = (
                data.site_xmat[scene._racket_sid].reshape(3, 3)[:, 1].copy()
            )
            contact_side = (
                geometric_axis
                if np.dot(incoming - racket_vel, geometric_axis) < 0.0
                else -geometric_axis
            )
            scene.set_ball(racket_pos + contact_side * 0.045, incoming)
            assert scene.step().ball_racket_contact

        assert len(received_normals) == 2
        np.testing.assert_allclose(
            received_normals[0], -received_normals[1], rtol=0.0, atol=1.0e-10
        )
    finally:
        scene.close()


def test_real_table_contact_uses_fitted_no_spin_bounce():
    """A physical MuJoCo table hit must apply the shared fitted impulse once."""

    metric = _load_success_metric()
    motion_path = (
        ROOT
        / "hope_training/motions/preprocessed/"
        "ours_forehand_guarded_1p8s_wrist_x.npz"
    )
    metadata = yaml.safe_load(motion_path.with_suffix(".yaml").read_text())
    model_path = (
        ROOT
        / "a3_deploy/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    )
    ball_cfg = metric.load_ball_physics_config()
    table_contact = ball_cfg["contact"]["table"]
    scene = PingPongRealPhysicsScene(
        str(model_path),
        ball_cfg,
        metadata["joint_order"],
        control_dt=0.02,
        near_edge_x=0.5,
        table_center_y=0.0,
    )
    try:
        scene.reset_stand()
        scene.set_ball(
            [1.0, -0.5, scene.table_height + 0.06],
            [-1.5, 0.2, -2.0],
        )
        contact_events = []
        for _ in range(10):
            events = scene.step()
            contact_events.extend(events.table_contacts)
            if contact_events:
                break

        assert len(contact_events) == 1
        assert scene.last_table_contact is not None
        contact = scene.last_table_contact
        incoming = contact["incoming_ball_velocity"]
        outgoing = contact["outgoing_ball_velocity"]
        horizontal_retention = 1.0 - float(
            table_contact["tangential_damping"]
        )
        np.testing.assert_allclose(
            outgoing[:2],
            horizontal_retention * incoming[:2],
            rtol=0.0,
            atol=1.0e-12,
        )
        assert outgoing[2] == pytest.approx(
            -float(table_contact["restitution"]) * incoming[2],
            abs=1.0e-12,
        )
        assert incoming[2] < 0.0 < outgoing[2]
    finally:
        scene.close()
