# HOPE PingPong 部署包 — model_20500

AgiBot A3 乒乓对拉策略部署包。策略为 **HOPEPingPong** 训练产物，从 `model_19000.pt` 续训至 **model_20500.pt**（run `2026-08-20_12-37-16_resume_19000_upright_feet_push`），导出为 ONNX 推理图。

## 目录结构

```
hope_pingpong_deploy_20500/
├── models/
│   ├── hope_pingpong.onnx       # obs[1,111] -> raw_action[1,31], 单输出, 无观测归一化
│   └── policy_manifest.json     # 部署契约: 维度/控制率/关节顺序/SHA-256 溯源
├── config/
│   ├── hope_pingpong_runtime.yaml   # 111-D 运行时配置 (50Hz, PD 增益, 生命周期)
│   └── action_adapter.yaml          # 共享 ActionAdapter (q_des = default_q + raw*0.25 + clamp)
├── sim/
│   ├── mjcf/a3_pingpong.xml         # A3 乒乓机器人 MJCF (自带球拍碰撞 geom)
│   ├── mjcf/meshes/                 # 机器人 STL 网格 (meshdir=meshes)
│   ├── deps/                        # 纯 NumPy 依赖: success_metric + hope_planner
│   ├── mujoco_pingpong_scene.py     # 仿真场景组装 (球台/网/球/物理)
│   └── mujoco_eval_onnx.py          # sim2sim 评估 (真实物理球 + 录像)
├── reference/                       # clean-room 参考部署 runner (契约实现)
├── scripts/
│   ├── run_sim.sh                   # MuJoCo 仿真运行
│   └── run_eval.sh                  # sim2sim 成功率评估
└── README.md
```

## 部署契约

| 项 | 值 |
|---|---|
| 观测 | 111-D: base_ang_vel(3) + joint_pos(31, q-default_q) + joint_vel(31) + last_action(31) + projected_gravity(3) + base_forward_xy(2) + fixed_station_error_xy(2) + racket_target_rel_base(3) + racket_target_vel_w(3) + time_to_strike(1) + swing_side(1) |
| 动作 | 31-D raw_action; idx 3,4 (head) 置零后作为 applied action 反馈 next-tick last_action |
| 控制率 | 50 Hz |
| 归一化 | none (raw) |
| ActionAdapter | q_des = default_q + raw_action × 0.25, 然后关节限位 clamp |
| 生命周期 | ready → swing → follow-through(0.8s) → recovery(0s) → ready; time_to_strike +1.0 → 0 → -0.8 |
| 击球窗 | ±0.12 s 围绕击球帧 (91 帧 @50Hz clip, 击球对齐 frame 50) |

## 快速开始

### 依赖

```bash
pip install -r reference/requirements.txt   # numpy pyyaml onnxruntime mujoco
```

脚本默认使用 `/home/mimi/miniconda3/envs/isaaclab/bin/python`（含 mujoco/onnxruntime/imageio-ffmpeg）。
其他环境用 `PYTHON_BIN=/path/to/python ./scripts/run_sim.sh` 覆盖。

### 仿真 (in-process MuJoCo)

```bash
./scripts/run_sim.sh                # headless 20 s
./scripts/run_sim.sh --view --realtime   # 可视化窗口, 真实时间
./scripts/run_sim.sh --duration 60  # 60 s
```

### sim2sim 成功率评估 (真实物理球)

```bash
./scripts/run_eval.sh 0 50   # seed=0, 50 次发球 (FH/BH 交替)
./scripts/run_eval.sh 1 20   # seed=1, 20 次 (快速验证)
```

输出: `viz/eval_seed<seed>.json` (success_rate) + `viz/eval_seed<seed>.mp4` (录像)。
若中断报 `MuJoCo incoming trajectory violated the legal one-bounce contract`，换 seed 重跑即可(发球采样边界问题, 非策略问题)。

### 真机部署

用你自己的 AgiBot A3 厂商部署 runner，加载 `models/hope_pingpong.onnx` + `config/*.yaml`。
参考实现见 `reference/`（契约实现, 非厂商 runner）。

## 训练来源

| 项 | 值 |
|---|---|
| 任务 | HOPEPingPong (rsl_rl) |
| checkpoint | model_20500.pt (sha256 a2e9de09...) |
| 训练 run | 2026-08-20_12-37-16_resume_19000_upright_feet_push |
| 奖励要点 | upright(torso) -5.0 / imitation 1.0 / racket_pos 4.0 / racket_vel 2.0 / blade_dir 1.0 / ball_contact 2.0 / net_cross 2.0 / opponent_bounce 4.0 / outgoing_speed 1.0 / action_rate -0.1 / joint_limit -10.0 / feet_anchor -2.0 / hip_orientation 0.75 |
| 训练末段指标 | return_success 0.908 / contact_rate 0.963 / net_clear_rate 0.917 |

## 物理参数

- 球: 自由关节 + 重力 + 气动阻力 (drag_k 0.1261)
- 台面: 长 2.74 m, 宽 1.525 m, 高 0.76 m (table_frame)
- 碰撞: restitution 0.654, tangential_damping 0.52, cap 0.5
- PD 增益: 见 config/hope_pingpong_runtime.yaml (Isaac Lab nominal, 仿真值)
