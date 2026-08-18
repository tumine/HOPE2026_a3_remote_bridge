#!/usr/bin/env python3

"""Generate the model_50000 observation parity fixture from its reference."""

from pathlib import Path
import sys

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = (
    REPO_ROOT
    / "model_50000_mujoco_sim2sim_bundle/a3_deploy/a3_deploy_example/reference"
)
sys.path.insert(0, str(REFERENCE_ROOT))

from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter  # noqa: E402
from a3_deploy_onnx_ref_pingpong.observation import (  # noqa: E402
    ObsTarget,
    RobotState,
    build_observation,
)
from a3_deploy_onnx_ref_pingpong.quaternion import (  # noqa: E402
    normalize,
    projected_gravity_body,
)


def main() -> int:
    bundle = REPO_ROOT / "model_50000_mujoco_sim2sim_bundle"
    adapter = ActionAdapter.from_yaml(bundle / "config/action_adapter.yaml")
    indices = np.arange(31, dtype=np.float64)
    observation = build_observation(
        state=RobotState(
            base_pos_w=np.asarray((0.2, -0.4, 0.7)),
            base_quat_w=np.asarray((0.7, -0.1, 0.2, 0.65)),
            base_ang_vel_b=np.asarray((0.11, -0.22, 0.33)),
            q=-0.25 + indices * 0.025,
            qd=-0.3 + indices * 0.02,
        ),
        target=ObsTarget(
            pos_w=np.asarray((0.45, -1.1, 0.25)),
            vel_w=np.asarray((2.2, 0.4, 0.9)),
            time_to_strike=0.7345,
            swing_side=-1,
        ),
        last_action=(indices - 15.0) * 0.03,
        default_q=adapter.default_q,
        base_target_xy=np.asarray((-0.5, -0.7625)),
    )
    # Real-hardware safety override: tilt comes from the pelvis IMU while the
    # PPMocap quaternion above remains the world/table heading source.
    observation[96:99] = projected_gravity_body(
        normalize(np.asarray((0.9, 0.1, -0.2, 0.3), dtype=np.float64))
    )
    for value in observation:
        print(format(float(value), ".9g"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
