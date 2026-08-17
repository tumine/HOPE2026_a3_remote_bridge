#!/usr/bin/env python3

"""Generate the model_21500 observation parity fixture from Python."""

from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "pc_tools"))

from a3_rl_contract import A3RLContract, PolicyTarget  # noqa: E402


def main() -> int:
    bundle = (
        REPO_ROOT
        / "26.7.25发球部署/pingpang_ustc-srf"
        / "a3_deploy/a3_pingpong_deploy_bundle/config"
    )
    order = yaml.safe_load(
        (bundle / "joint_order_agibot_a3.yaml").read_text(encoding="utf-8")
    )["joint_order"]
    adapter = yaml.safe_load(
        (bundle / "action_adapter.yaml").read_text(encoding="utf-8")
    )
    default_q = np.asarray(
        [adapter["default_q"][name] for name in order], dtype=np.float64
    )

    contract = A3RLContract.__new__(A3RLContract)
    contract.default_q = default_q
    indices = np.arange(31, dtype=np.float64)
    observation = contract.build_observation(
        q=-0.25 + indices * 0.025,
        dq=-0.3 + indices * 0.02,
        base_position_w=(0.2, -0.4, 0.7),
        gravity_quaternion_wxyz=(0.9, 0.1, -0.2, 0.3),
        heading_quaternion_wxyz=(0.7, -0.1, 0.2, 0.65),
        pelvis_gyro_body=(0.11, -0.22, 0.33),
        last_action=(indices - 15.0) * 0.03,
        fixed_station_xy=(-0.5, -0.7625),
        target=PolicyTarget(
            position_w=np.asarray((0.45, -1.1, 0.25)),
            velocity_w=np.asarray((2.2, 0.4, 0.9)),
            time_to_strike=0.7345,
            swing_side=-1,
        ),
    )
    for value in observation:
        print(format(float(value), ".9g"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
