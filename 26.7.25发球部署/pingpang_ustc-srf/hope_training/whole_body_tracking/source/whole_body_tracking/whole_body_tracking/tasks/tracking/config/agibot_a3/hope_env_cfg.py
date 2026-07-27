"""Agibot A3 — the single HOPE whole-body task.

One environment config, :class:`HOPEPingPongEnvCfg`, wiring:

* motion imitation (:class:`MotionCommand`) over a forehand + backhand clip pair (clip 0 / clip 1),
  ``wrap_teleport=False`` so the policy physically transitions between swings (continuous rally);
* the ping-pong goal (:class:`RacketTargetCommand`): sampled interception/incoming-ball state,
  planner-derived racket velocity/normal + time-to-strike + swing side, a fixed startup station,
  and a no-spin paddle-contact/outgoing-ball evaluation for the return rewards;
* the 111-D actor observation (``hope_pingpong`` contract) and a privileged critic that adds
  the 62-D reference joint stream, reference errors, and the actual racket FK state (value function only);
* racket-task rewards plus whole-body reference imitation and always-on stability shaping;
* a [-100, 100]-clipped linear joint-position residual with one shared 0.25 scale (passive head);
* physical-fall / time-out terminations and light domain randomization.

Control runs at 50 Hz. The default motion set is the equal-length 1.80 s
guarded-retime ``ours_forehand`` / ``ours_backhand`` pair under
``hope_training/motions/preprocessed``; the forehand uses a centred 15-frame
smoothing filter. The strike-centred guard and its finite-difference buffer
remain at the source speed.
"""

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.agibot_a3 import (
    A3_ANCHOR_BODY,
    A3_FEET_BODIES,
    A3_ROOT_BODY,
    A3_STAND_PELVIS_HEIGHT,
    A3_TRACKED_BODIES,
    AGIBOT_A3_CFG,
    AGIBOT_A3_JOINT_NAMES,
    AGIBOT_A3_PASSIVE_HEAD_JOINT_NAMES,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg
from whole_body_tracking.utils.action_adapter_config import load_action_adapter_config


def _find_motion_clip(name: str) -> str:
    """Locate a motion clip under ``hope_training/motions/preprocessed`` (walk up from here)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(14):
        cand = os.path.join(d, "hope_training", "motions", "preprocessed", name)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # Fall back to a relative path; the task YAML can override motion_file explicitly.
    return os.path.join("hope_training", "motions", "preprocessed", name)


FOREHAND_CLIP = _find_motion_clip("ours_forehand_guarded_1p8s_wrist_x.npz")
BACKHAND_CLIP = _find_motion_clip("ours_backhand_guarded_1p8s.npz")


def _canonical_joint_entity() -> SceneEntityCfg:
    """Joint selector that presents PhysX state in the deploy contract order."""
    return SceneEntityCfg(
        "robot", joint_names=list(AGIBOT_A3_JOINT_NAMES), preserve_order=True
    )


@configclass
class CommandsCfg:
    """Motion imitation + racket target commands."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        anchor_body_name=A3_ANCHOR_BODY,
        body_names=A3_TRACKED_BODIES,
        joint_names=AGIBOT_A3_JOINT_NAMES,
        motion_file=[FOREHAND_CLIP, BACKHAND_CLIP],  # clip 0 = forehand, clip 1 = backhand
        # Constant lifts measured from the lowest foot collision-mesh vertex
        # over every frame. Each leaves 1 mm minimum clearance and is also used
        # by RSI root initialization through the grounded motion tensor.
        ground_height_offset_per_clip=(0.0774785241, 0.0767081053),
        wrap_teleport=False,
        # Most resets now reproduce the real deploy entry: a quiet default stand.
        # The remaining RSI starts are restricted to kinematically stable dual-support
        # frames and held briefly before their reference clock starts advancing.
        stand_start_prob=0.80,
        stand_start_min_hold=25,
        rsi_stable_frame_only=True,
        rsi_foot_body_names=tuple(A3_FEET_BODIES),
        rsi_root_body_name=A3_ROOT_BODY,
        rsi_phase_range=(0.0, 0.9),
        rsi_max_foot_height_delta=0.04,
        rsi_max_foot_speed=0.15,
        rsi_max_root_lin_speed=0.25,
        rsi_max_root_ang_speed=0.60,
        rsi_fallback_to_segment_start=True,
        rsi_settle_steps_range=(10, 20),
        rsi_settle_zero_velocity=True,
        hold_steps_range=(0, 100),
        pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.005, 0.005),
                    "roll": (-0.04, 0.04), "pitch": (-0.04, 0.04), "yaw": (-0.08, 0.08)},
        velocity_range={"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (-0.05, 0.05),
                        "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15), "yaw": (-0.20, 0.20)},
        joint_position_range=(-0.03, 0.03),
    )

    racket_target = mdp.RacketTargetCommandCfg(
        asset_name="robot",
        motion_command_name="motion",
        debug_vis=False,
        mount_normal_axis=1,          # racket-local +Y blade face
        mount_normal_sign_per_clip=(1.0, -1.0),  # forehand/backhand strike with opposite faces
        # The 1.80 s clips are both 91 frames and preserve the strike at frame 50.
        strike_phase_per_clip=(0.5555555555555556, 0.5555555555555556),
        strike_window_s=0.12,
        # STATION-RELATIVE racket target boxes (x forward reach, y lateral reach, z absolute height).
        # These boxes are deliberately wider than the retargeted clips' natural strike poses while
        # remaining inside the legal one-bounce MuJoCo serve domain at a 0.8--1.0 s flight horizon.
        # Dense tracking rewards let the residual policy exploit this reach volume without requiring
        # a separate physical-ball actor in Isaac.
        racket_pos_range_per_clip=(
            ((0.18, 0.40), (-0.76, -0.52), (1.00, 1.21)),  # forehand, grounded
            ((0.45, 0.75), (-0.30, 0.10), (0.84, 1.10)),   # backhand, grounded
        ),
        # Broad post-bounce arrival domains. They contain the complete extrema measured from the
        # current legal MuJoCo serve generator and add substantial independent axis margins for
        # robustness instead of training only on a fitted 1st--99th percentile tube.
        incoming_ball_vel_range_per_clip=(
            ((-2.40, -0.90), (-0.35, 0.35), (-2.20, 1.00)),  # forehand
            ((-2.30, -0.80), (-0.35, 0.35), (-2.80, 1.00)),  # backhand
        ),
        # These are authoritative conservative envelopes for the planner output and are mirrored
        # by the live admission gate. They also serve as the explicit sampling fallback when
        # use_planner_target=False. In normal training, incoming velocity and desired opponent
        # landing pass through the same planner equations used at deployment.
        racket_vel_range_per_clip=(
            # Conservative outward bounds over every source-domain corner plus deterministic
            # interior samples, with numerical margin around the Torch/NumPy planner union.
            ((1.75, 3.40), (0.25, 1.10), (0.35, 1.45)),    # forehand planner envelope
            ((1.05, 3.00), (-0.25, 0.50), (0.45, 1.45)),   # backhand planner envelope
        ),
        # Start from centered, easier target/serve subsets, then expose the complete
        # legal boxes over 200k control steps (about 8.3k PPO iterations at 24 steps).
        target_curriculum_start_step=0,
        target_curriculum_duration_steps=200_000,
        position_curriculum_initial_fraction=0.35,
        incoming_velocity_curriculum_initial_fraction=0.40,
        racket_velocity_curriculum_initial_fraction=0.50,
        feet_body_names=tuple(A3_FEET_BODIES),
    )


@configclass
class ActionsCfg:
    """31-D clipped linear joint-position residual with one shared scale (passive head)."""

    joint_pos = mdp.ClampedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(AGIBOT_A3_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=True,
        passive_joint_names=AGIBOT_A3_PASSIVE_HEAD_JOINT_NAMES,
    )


@configclass
class ObservationsCfg:
    """111-D actor observation + privileged critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Order is fixed — it is the hope_pingpong observation contract.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _canonical_joint_entity()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _canonical_joint_entity()},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.applied_last_action, params={"action_name": "joint_pos"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_forward_xy = ObsTerm(
            func=mdp.base_forward_xy, params={"command_name": "racket_target"}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        fixed_station_error_xy = ObsTerm(
            func=mdp.fixed_station_error_xy, params={"command_name": "racket_target"}, noise=Unoise(n_min=-0.03, n_max=0.03)
        )
        racket_target_rel_base = ObsTerm(
            func=mdp.racket_target_rel_base, params={"command_name": "racket_target"}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        swing_side = ObsTerm(func=mdp.swing_side, params={"command_name": "racket_target"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Actor terms (noise-free) ...
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _canonical_joint_entity()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _canonical_joint_entity()})
        last_action = ObsTerm(func=mdp.applied_last_action, params={"action_name": "joint_pos"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_forward_xy = ObsTerm(func=mdp.base_forward_xy, params={"command_name": "racket_target"})
        fixed_station_error_xy = ObsTerm(func=mdp.fixed_station_error_xy, params={"command_name": "racket_target"})
        racket_target_rel_base = ObsTerm(func=mdp.racket_target_rel_base, params={"command_name": "racket_target"})
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        swing_side = ObsTerm(func=mdp.swing_side, params={"command_name": "racket_target"})
        # ... plus privileged (sim-only) signals for the value function.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # 62-D ref stream
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        robot_body_pos_b = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        robot_body_ori_b = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_w = ObsTerm(func=mdp.racket_lin_vel_w, params={"command_name": "racket_target"})
        racket_normal_w = ObsTerm(func=mdp.racket_normal_w, params={"command_name": "racket_target"})
        racket_target_normal_w = ObsTerm(func=mdp.racket_target_normal_w, params={"command_name": "racket_target"})
        episode_time_left = ObsTerm(func=mdp.episode_time_left)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Racket-task rewards plus always-on balance and lower-body regularization."""

    # Survival/balance. Timeout is not penalized; physical fall terminations are.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_tracking_exp,
        weight=0.5,
        params={
            "target_height": A3_STAND_PELVIS_HEIGHT,
            "std": 0.08,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_planar_velocity = RewTerm(
        func=mdp.base_planar_velocity_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    base_roll_pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # Both feet must support the robot; one planted foot receives no partial bonus.
    both_feet_contact = RewTerm(
        func=mdp.both_feet_contact,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=list(A3_FEET_BODIES), preserve_order=True
            ),
            "force_threshold": 10.0,
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_l2,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(A3_FEET_BODIES), preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=list(A3_FEET_BODIES), preserve_order=True
            ),
            "force_threshold": 10.0,
        },
    )
    foot_flat = RewTerm(
        func=mdp.foot_flat_orientation,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(A3_FEET_BODIES), preserve_order=True
            ),
            "sole_normal_axis": 2,
            "sole_normal_sign": 1.0,
        },
    )
    feet_stance = RewTerm(
        func=mdp.feet_stance_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(A3_FEET_BODIES), preserve_order=True
            ),
            "target_width": 0.244,
            "midpoint_weight": 0.5,
            "target_midpoint_xy_b": (-0.005, 0.0),
        },
    )

    # Forehand/backhand sample imitation (pelvis, waist, legs and arms; swing-gated).
    imitation = RewTerm(
        func=mdp.sample_imitation,
        weight=1.0,
        params={
            "command_name": "motion",
            "std_pos": 0.3,
            "std_ori": 0.4,
            "body_names": A3_TRACKED_BODIES,
        },
    )
    # 3. racket position (strike window)
    racket_position = RewTerm(
        func=mdp.racket_position, weight=4.0, params={"command_name": "racket_target", "std": 0.12}
    )
    # 4. racket velocity (strike window)
    racket_velocity = RewTerm(
        func=mdp.racket_velocity, weight=2.0, params={"command_name": "racket_target", "std": 0.6}
    )
    # 5. simplified blade direction (strike window)
    blade_direction = RewTerm(
        func=mdp.racket_blade_direction, weight=1.0, params={"command_name": "racket_target", "std": 0.3}
    )
    # 6. actual ball contact (one-shot at strike)
    ball_contact = RewTerm(func=mdp.ball_contact, weight=2.0, params={"command_name": "racket_target"})
    # 7. net crossing (one-shot at strike)
    net_cross = RewTerm(func=mdp.ball_net_cross, weight=2.0, params={"command_name": "racket_target"})
    # 8. opponent-half first bounce (one-shot at strike)
    opponent_bounce = RewTerm(func=mdp.ball_opponent_bounce, weight=4.0, params={"command_name": "racket_target"})
    # 9. in-place follow-through / recovery
    follow_through_recovery = RewTerm(
        func=mdp.follow_through_recovery,
        weight=1.0,
        params={"command_name": "racket_target", "std": 0.5, "station_std": 0.3},
    )
    # Raw-action regularization prevents large constant actor outputs from hiding
    # behind the bounded target mapping; leg acceleration damps lower-body chatter.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.002)
    leg_joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
            )
        },
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )


@configclass
class TerminationsCfg:
    """Time-out and physical-fall resets (ordinary env lifecycle, not a gate)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_tilted thresholds the horizontal projected-gravity norm: 0.65 ~= 40.5 deg.
    base_tilted = DoneTerm(func=mdp.base_tilted, params={"threshold": 0.65})
    base_too_low = DoneTerm(func=mdp.base_too_low, params={"min_height": 0.75})


@configclass
class EventCfg:
    """Light domain randomization for sim-to-real robustness."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=A3_ANCHOR_BODY),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class HOPEPingPongEnvCfg(ManagerBasedRLEnvCfg):
    """The single public HOPE task (gym id ``HOPE-PingPong-AgibotA3-v0``)."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 50 Hz control (decimation 4 over a 200 Hz physics step).
        self.decimation = 4
        # A full 91-frame guarded swing spans 1.80 s, before the U[0, 2] s hold.
        # Keep enough horizon for repeated strikes and all transition types.
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Robot + shared action adapter. Transform/default_q/scale/joint clamp all come
        # from the ONE shared config the deploy runner reads (action_adapter.yaml), so the same raw
        # action produces the same joint targets in training and deployment. The actor observation
        # remains the established 111-D contract (see tests/test_action_adapter_parity.py).
        adapter = load_action_adapter_config()
        self.scene.robot = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = adapter.default_q_by_name()
        self.actions.joint_pos.raw_action_transform = adapter.raw_action_transform
        self.actions.joint_pos.action_clip = adapter.action_clip
        self.actions.joint_pos.scale = adapter.action_scale_by_name()
        self.actions.joint_pos.position_clamp = adapter.position_clamp_by_name()

        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
