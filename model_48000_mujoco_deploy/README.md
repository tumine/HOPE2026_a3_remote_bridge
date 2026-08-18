# model_48000 MuJoCo 部署包

这个目录是 `model_48000.pt` 的独立部署包，包含可部署 ONNX 策略、动作适配配置、`model_48000` 的冻结击球盒和使用 GJC 精确机器人 XML 的 MuJoCo sim2sim 脚本。

## 策略来源

- 训练运行：`hope_pingpong/2026-08-18_14-17-33_resume_41500_full_table_fast`
- 检查点：`model_48000.pt`
- 策略输入：111 维
- 策略输出：31 维
- 控制频率：50 Hz
- ONNX：`policy/hope_pingpong.onnx`
- MuJoCo 模型：`/home/fsr/HOPE/gjc_robot_model/agibot_a3/xml/agibot_a3.xml`

ONNX 不包含观测归一化。部署端必须保持 111 维观测顺序、坐标系、动作适配和 20 ms 控制周期一致。策略 manifest 和动作适配配置分别位于 `policy/policy_manifest.json` 和 `config/action_adapter.yaml`。

## model_48000 的击球盒

这个 checkpoint 已经是全台移动 base、全高度击球盒版本。`model_48000_training_contract.yaml` 是从该运行保存的 `params/env.yaml` 提取的冻结合同，不能替换成 `model_41500` 的旧合同。

相对于移动 base 站位，策略实际追踪的球拍目标盒为：

| 动作 | x | y | z |
| --- | --- | --- | --- |
| 正手 | `[0.18, 0.40]` | `[-0.76, -0.52]` | `[0.84, 1.21]` |
| 反手 | `[0.45, 0.75]` | `[-0.30, 0.10]` | `[0.84, 1.21]` |

移动 base 站位范围为 `[-0.35, 0.6625]` m，正反手分界线为 `y=-0.41` m，滞回空带为 `0` m。动作时钟为击球前 `1.0 s`、击球后 `0.8 s`，击球窗为 `0.12 s`。

每侧的球拍速度盒也已经固定在合同中：

- 正手：x `[1.75,3.40]`、y `[0.20,1.10]`、z `[0.35,1.60]`
- 反手：x `[1.05,3.10]`、y `[-1.10,0.65]`、z `[0.35,1.50]`

这里的 z 范围已经统一为 `[0.84,1.21]`。相对固定启动站位叠加 base 站位后的几何包络记录在 `config/strike_box.yaml`；物理来球测试仍需经过球台边界和一次落台约束，不能直接把包络外的点当成合法来球。

## MuJoCo 标准测试

在仓库根目录执行：

```bash
cd /home/fsr/HOPE

a3_deploy/model_48000_mujoco_deploy/scripts/run_mujoco.sh \
  --view \
  --eval-mode continuous \
  --num-serves 10 \
  --serve-side-pattern alternating \
  --seed 0
```

这个命令使用：

- GJC 精确机器人模型；
- `model_48000` 实际训练过的正手/反手位置和速度盒；
- 全台移动 base runtime 配置；
- 连续多球模式，击球之间不重置机器人。

无窗口检查命令：

```bash
cd /home/fsr/HOPE

a3_deploy/model_48000_mujoco_deploy/scripts/run_mujoco.sh \
  --eval-mode continuous \
  --num-serves 10 \
  --serve-side-pattern alternating \
  --seed 0
```

## 左右边缘压力测试

如果要把来球固定到球台横向两侧边缘，直接调用评估器并显式指定合法的球台内目标范围。这个压力测试不加载冻结合同，因为命令行需要覆盖默认采样盒；它仍然使用同一个 `model_48000` ONNX 和全台 runtime。

```bash
cd /home/fsr/HOPE/hope_training/whole_body_tracking

/home/fsr/miniconda3/envs/hope/bin/python scripts/mujoco_eval_onnx.py \
  --onnx logs/rsl_rl/hope_pingpong/2026-08-18_14-17-33_resume_41500_full_table_fast/exported/model_48000/hope_pingpong.onnx \
  --runtime-config /home/fsr/HOPE/a3_deploy/model_48000_mujoco_deploy/config/hope_pingpong_runtime.yaml \
  --model-xml /home/fsr/HOPE/gjc_robot_model/agibot_a3/xml/agibot_a3.xml \
  --eval-mode continuous \
  --num-serves 20 \
  --serve-side-pattern alternating \
  --forehand-strike-lateral-min=-0.75 \
  --forehand-strike-lateral-max=-0.70 \
  --backhand-strike-lateral-min=0.70 \
  --backhand-strike-lateral-max=0.75 \
  --forehand-strike-height-min=0.84 \
  --forehand-strike-height-max=1.21 \
  --backhand-strike-height-min=0.84 \
  --backhand-strike-height-max=1.21 \
  --serve-flight-time-min=0.8 \
  --serve-flight-time-max=1.0 \
  --max-rest-seconds=1.0 \
  --max-trial-seconds=2.5 \
  --hold-min-seconds=0.0 \
  --hold-max-seconds=0.0 \
  --stand-min-hold-seconds=0.0 \
  --start-base-x-jitter=0.10 \
  --start-base-y-jitter=0.20 \
  --seed=7 \
  --view
```

正手边缘范围对应球台左侧，反手边缘范围对应球台右侧。`0.8-1.0 s` 是为了满足该策略的 `1.0 s` lead time，同时给物理发球器保留合法一次落台轨迹的候选空间。

## 部署注意事项

1. 真实部署必须由来球预测器提供同一坐标系下的球拍目标位置、速度、击球时间和 swing side。
2. 不要给 `model_48000` 加载 `model_41500_mujoco_deploy/config/model_41500_training_contract.yaml`；那会把 z 范围和 base 站位解释切回旧版本。
3. MuJoCo 首次测试建议保留碰撞检测、动作限幅和急停，并先使用无窗口或低速测试确认 GJC 模型的关节顺序和碰撞几何。
4. `--view` 需要有效的 X11/Wayland OpenGL 上下文。远程或无显示环境请去掉 `--view`，评估仍会正常输出实际接触、过网和回球结果。

## 文件结构

```text
model_48000_mujoco_deploy/
├── README.md
├── provenance.json
├── policy/
│   ├── hope_pingpong.onnx
│   └── policy_manifest.json
├── config/
│   ├── action_adapter.yaml
│   ├── hope_pingpong_runtime.yaml
│   ├── model_48000_training_contract.yaml
│   └── strike_box.yaml
└── scripts/
    └── run_mujoco.sh
```

