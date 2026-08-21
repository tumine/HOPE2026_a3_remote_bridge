#!/usr/bin/env bash
# HOPE PingPong 部署包 - MuJoCo sim2sim 评估（真实物理球）
# 用法: ./run_eval.sh [seed] [num_serves]
# 输出: success_rate + 可选视频 (viz/eval_seed<seed>.json + .mp4)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-0}"
SERVES="${2:-50}"
ONNX="${HERE}/../models/hope_pingpong.onnx"
export PYTHONPATH="${HERE}/../reference:${HERE}/../sim/deps:${PYTHONPATH:-}"
FFMPEG_BIN="$(/home/mimi/miniconda3/envs/isaaclab/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo ffmpeg)"
MUJOCO_GL=egl FFMPEG_BIN="$FFMPEG_BIN" \
  /home/mimi/miniconda3/envs/isaaclab/bin/python "${HERE}/../sim/mujoco_eval_onnx.py" \
  --onnx "$ONNX" \
  --num-serves "$SERVES" \
  --video-out "${HERE}/../viz/eval_seed${SEED}.mp4" \
  --json-out "${HERE}/../viz/eval_seed${SEED}.json" \
  --seed "$SEED"
