#!/usr/bin/env python3

"""Generate the model_21500 raw-action fixture with Python ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = (
    REPO_ROOT
    / "26.7.25发球部署/pingpang_ustc-srf/a3_deploy"
    / "a3_pingpong_deploy_bundle/policy/hope_pingpong.onnx"
)
OBSERVATION = Path(__file__).with_name("observation_golden.txt")


def main() -> int:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    observation = np.loadtxt(OBSERVATION, dtype=np.float32).reshape(1, 111)
    raw_action = session.run(
        ["raw_action"], {"observation": observation}
    )[0].reshape(31)
    for value in raw_action:
        print(format(float(value), ".9g"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
