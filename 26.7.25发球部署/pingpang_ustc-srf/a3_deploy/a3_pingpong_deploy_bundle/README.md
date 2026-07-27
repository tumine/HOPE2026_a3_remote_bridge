# 智元 A3 乒乓球策略部署包

这个目录是当前 HOPE A3 乒乓球策略的精简交接包，包含 ONNX actor、观测与动作合同、
关节顺序、训练配置快照、MuJoCo 参考配置、Planner 接口说明和部署前检查清单。

## 当前策略

| 项目 | 值 |
|---|---|
| 策略文件 | `policy/hope_pingpong.onnx` |
| 来源 checkpoint | `2026-07-26_01-47-04/model_21500.pt` |
| checkpoint iteration | 21500 |
| checkpoint SHA256 | `7939b3a2e4716e764271126668602b0826f6f4fc9615a74ccdb7a0761ee19987` |
| 网络 | `111 → 512 → 256 → 128 → 31`，隐藏层 ELU |
| ONNX | IR 6，opset 11，动态 batch |
| 输入 | `observation: float32[batch,111]` |
| 输出 | `raw_action: float32[batch,31]` |
| 控制频率 | 50 Hz |
| 观测归一化 | 无，直接输入原始量 |
| 策略代际 | `full_body_uniform_action_return_physics_v1` |
| 状态 | **当前 MuJoCo/sim-to-sim 候选，尚未通过实机安全认证** |

这版策略包含：

- pelvis、腰部、双腿和双臂共 14 个 body 的全身参考动作模仿；
- 正反手分离的击球位置/速度窗口和完整回球物理目标；
- `identity + clip[-100,100] + 全关节 scale=0.25 + 机械限位` 的统一动作接口；
- 91 帧、50 Hz 的正反手动作，触球前 1.0 秒、触球后 0.8 秒；
- 无击球命令时使用经过 10 秒 MuJoCo 稳定性验证的 READY 观测锚点。

训练 run 在提交 `42530cac9717642176a70d43bafdce0a2b0c5e08` 上启动，并带有
`provenance/training_worktree.diff`；部署接口和 READY 修正最终收录于提交
`28d0074a8fb595d4c730cbc13a146fdd53c0a170`。

正反手 NPZ 只用于训练模仿，actor 推理时不读取 motion 文件。它们的名称和训练窗口保存在
`config/policy_training_contract.yaml`，完整训练配置保存在 `provenance/`。

## 目录

```text
a3_pingpong_deploy_bundle/
├── README.md
├── policy/
│   ├── hope_pingpong.onnx
│   ├── policy_manifest.json
│   └── provenance.json
├── config/
│   ├── runtime.yaml
│   ├── action_adapter.yaml
│   ├── joint_order_agibot_a3.yaml
│   ├── policy_training_contract.yaml
│   ├── training_actuator_gains.yaml
│   └── hardware_gains_TEMPLATE.yaml
├── docs/
│   ├── OBSERVATION_ACTION.md
│   ├── CONTROL_GAINS.md
│   ├── PLANNER_AND_FRAMES.md
│   ├── VALIDATION.md
│   └── DEPLOYMENT_CHECKLIST.md
├── provenance/
│   ├── training_env.yaml
│   ├── training_agent.yaml
│   └── training_worktree.diff
├── validation/
│   ├── mujoco_idle_ready_anchor_10s.json
│   ├── mujoco_ready_anchor_smoke.json
│   └── mujoco_ready_anchor_smoke_diagnostics.json
└── tools/
    ├── export_actor_from_checkpoint.py
    └── verify_bundle.py
```

## 每周期的唯一接口

```text
机器人/IMU/世界位姿 + Planner RacketCommand
                       ↓
              observation[111]
                       ↓
             hope_pingpong.onnx
                       ↓
                raw_action[31]
                       ↓
       clip(raw_action, -100, 100)
                       ↓
        head 索引 3、4 置零，得到 applied_action
                       ↓
 q_des = default_q + 0.25 * applied_action
                       ↓
             31 关节机械角限位
                       ↓
          交给 A3 厂商位置伺服后端
```

下一周期 `last_action` 必须反馈经过 `clip` 且 head 清零后的 `applied_action`，不能反馈
原始网络输出，也不能额外使用 `tanh`。

## 先验证部署包

```bash
cd /home/fsr/HOPE
/home/fsr/miniconda3/envs/hope/bin/python \
  a3_deploy/a3_pingpong_deploy_bundle/tools/verify_bundle.py
```

验证器会检查 ONNX 图、输入输出、metadata、关节顺序、ActionAdapter、READY 锚点、来源
快照哈希，并执行动态 batch ONNX reference inference。重新导出时还会比较 PyTorch
checkpoint actor 与 ONNX 输出。

## 在 MuJoCo 参考运行器中播放

部署包已经把同一模型放入 `a3_deploy_example/models/hope_pingpong.onnx`，因此默认命令可
直接运行：

```bash
cd /home/fsr/HOPE/a3_deploy/a3_deploy_example
pip install -r reference/requirements.txt

# 正反手示例命令
scripts/run_pingpong_sim.sh --view --realtime

# 不发球，只保持经过验证的 READY 输入
scripts/run_pingpong_sim.sh --idle --view --realtime
```

也可以显式使用本 bundle 的模型和配置：

```bash
cd /home/fsr/HOPE/a3_deploy/a3_deploy_example/reference
PYTHONPATH="$PWD" python -m a3_deploy_onnx_ref_pingpong \
  --config ../../a3_pingpong_deploy_bundle/config/runtime.yaml \
  --view --realtime
```

连接 ROS 2 Planner：

```bash
PYTHONPATH="$PWD" python -m a3_deploy_onnx_ref_pingpong \
  --config ../../a3_pingpong_deploy_bundle/config/runtime.yaml \
  --planner --view --realtime
```

当前 Planner 已使用正反手独立击球区域，并与本策略训练窗口做了表面坐标到机器人站位
坐标的平移对齐；实机仍需完成动捕、球桌、基座世界位姿和消息延迟的现场标定。详见
`docs/PLANNER_AND_FRAMES.md`。

## 实机部署边界

ONNX 和 111/31 维策略合同已经具备；仓库中没有智元厂商实机后端的权威 KP/KD，也没有
完整厂商控制接线。因此不能把本目录直接接电机后宣称完成实机部署。

实机前至少需要：

1. 用厂商 SDK 的 31 关节顺序逐项核对名称、方向、零位和单位。
2. 由厂商/机器人负责人填写 `hardware_gains_TEMPLATE.yaml`，配置电流、力矩、速度、
   通信超时和急停保护。
3. 提供经过标定的基座世界位置和姿态，使其与球桌、动捕和 Planner 使用同一坐标系。
4. 实测并补偿消息年龄、推理和执行延迟。
5. 依次完成离线 ONNX、MuJoCo、悬空低增益、系绳站立、无球挥拍和软球低速测试。

详细步骤见 `docs/DEPLOYMENT_CHECKLIST.md`。

## 参数权威顺序

出现冲突时按以下顺序处理：

1. 真机安全限制、KP/KD：智元厂商后端和经过审批的机器人参数。
2. 策略观测、动作、关节列顺序：本目录的 manifest、观测文档和 joint-order YAML。
3. `raw_action_transform`、`action_clip`、`default_q`、`action_scale` 和关节角 clamp：
   `config/action_adapter.yaml`。
4. Isaac 训练动力学参数：`config/training_actuator_gains.yaml`。
5. MuJoCo 参考运行增益和 READY 输入：`config/runtime.yaml`，仅用于仿真/接口参考。

训练增益和 MuJoCo 增益都不是可直接照抄的真机增益。
