# model_50000 MuJoCo sim2sim 对齐包

这是 `model_50000.pt` 的自包含 MuJoCo 评估和部署对齐快照。目录内包含 ONNX、checkpoint、GJC 机器人模型及 mesh、球台/球物理、planner 核心、reference runner、站位计算、状态机、111 维观测、动作适配和训练参数。

## 快照结论

- 来源运行：`2026-08-18_20-57-32_resume_48000_waist_kp150_fast`
- checkpoint：`model_50000.pt`
- 输入/输出：111 维观测 / 31 维动作
- 控制频率：50 Hz，周期 20 ms
- 腰部 yaw/pitch KP：150/150
- 腰部课程：20,000 控制步；该 checkpoint 已经历 48,000 控制步，课程已完成并固定在 150
- 移动站位 y 范围：`[-0.35, 0.6625]` m
- 挥拍时钟：击球前 1.0 s，击球后 0.8 s

## 快速运行

推荐使用已有环境：

```bash
cd /home/fsr/HOPE/a3_deploy/model_50000_mujoco_sim2sim_bundle
export PYTHON=/home/fsr/miniconda3/envs/hope/bin/python

./scripts/run_mujoco.sh \
  --view \
  --eval-mode continuous \
  --num-serves 10 \
  --serve-side-pattern alternating \
  --seed 0
```

无图形界面验证时去掉 `--view`。独立环境可安装：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

左右最远交替压力测试：

```bash
export PYTHON=/home/fsr/miniconda3/envs/hope/bin/python
./scripts/run_hard_edges.sh --view
```

短 TTS 动作诊断：

```bash
./scripts/run_tts_probe.sh 1.0 --view
./scripts/run_tts_probe.sh 0.25 --view
```

`run_tts_probe.sh` 是无球动作诊断，只比较收到目标后可用的动作时间，不计算回球成功率。

## 文档

- `docs/DEPLOYMENT_ALIGNMENT.md`：部署端必须逐项一致的坐标、时钟、观测和动作合同
- `docs/STATION_AND_LIFECYCLE.md`：站位公式、状态切换、task/revision 规则和 TTS 行为
- `docs/OBSERVATION_ACTION_CONTRACT.md`：111 维观测、31 维动作和 50 Hz tick 顺序

## 关键源代码

```text
站位计算:
  a3_deploy/a3_deploy_example/reference/
  a3_deploy_onnx_ref_pingpong/station_target.py

状态机:
  a3_deploy/a3_deploy_example/reference/
  a3_deploy_onnx_ref_pingpong/lifecycle.py

部署 tick 与观测:
  a3_deploy/.../runner.py
  a3_deploy/.../observation.py

动作适配和关节顺序:
  a3_deploy/.../action_adapter.py
  a3_deploy/.../joint_order.py

MuJoCo 评估与场景:
  hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py
  hope_training/whole_body_tracking/scripts/mujoco_pingpong_scene.py

planner:
  hope_ws/src/hope_planner/hope_planner/
```

## 目录边界

这个包可以独立运行 MuJoCo 评估，但不会直接驱动真实机器人。真实部署端必须自行实现可靠的关节/IMU/base 位姿读取、硬件安全限制和关节目标下发，并严格复用这里的观测、动作、状态和站位语义。
