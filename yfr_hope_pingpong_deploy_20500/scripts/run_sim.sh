#!/usr/bin/env bash
# HOPE PingPong 部署包 - MuJoCo 仿真 (reference runner)
# 用法:
#   ./run_sim.sh              # headless 跑 20 秒
#   ./run_sim.sh --view       # 打开可视化窗口
#   ./run_sim.sh --realtime   # 真实时间 50Hz
#   ./run_sim.sh --duration 60 --onnx /path/to/model.onnx
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/mimi/miniconda3/envs/isaaclab/bin/python}"
export PYTHONPATH="${HERE}/../reference:${PYTHONPATH:-}"
exec "$PY" -m a3_deploy_onnx_ref_pingpong \
  --config "${HERE}/../config/hope_pingpong_runtime.yaml" \
  --backend mujoco "$@"
