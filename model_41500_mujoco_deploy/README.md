# model_41500 MuJoCo 部署包

这个目录是 `model_41500.pt` 对应的独立部署包，包含导出的 ONNX 策略、运行配置、动作适配配置、该检查点实际训练过的击球盒，以及使用 GJC 精确机器人模型进行 MuJoCo sim2sim 的脚本。

## 策略来源

- 训练运行：`hope_pingpong/2026-08-17_22-03-09_resume_25000_gjc_robot_model`
- 检查点：`model_41500.pt`
- 策略输入：111 维
- 策略输出：31 维
- 推理频率：50 Hz
- ONNX：`policy/hope_pingpong.onnx`
- 机器人模型：`/home/fsr/HOPE/gjc_robot_model/agibot_a3/xml/agibot_a3.xml`

策略的 ONNX 导出不包含观测归一化；部署端必须保持训练时的观测顺序、坐标系、动作适配和 50 Hz 控制周期。具体接口记录在 `policy/policy_manifest.json` 和 `config/action_adapter.yaml` 中。

## model_41500 的击球盒

`model_41500.pt` 保存时已经包含移动 base 语义，但它保存的环境配置仍然是该检查点实际训练过的击球盒。运行 MuJoCo 时，脚本会自动加载 `config/model_41500_training_contract.yaml`，不会把后续尚未由该检查点学到的全台、全高度范围误当成当前策略能力。

以下坐标是相对于移动 base 站位的球拍目标盒，单位为米：

| 动作 | x | y | z |
| --- | --- | --- | --- |
| 正手 | `[0.18, 0.40]` | `[-0.76, -0.52]` | `[1.00, 1.21]` |
| 反手 | `[0.45, 0.75]` | `[-0.30, 0.10]` | `[0.84, 1.10]` |

对应的 base 横向目标范围是 `[-0.35, 0.35]` m，动作分界线是 `y = -0.41` m；原训练配置还使用了 `0.04` m 的滞回空带。训练时的球拍速度盒也已原样保存在 `model_41500_training_contract.yaml`。

按当前 MuJoCo 地面坐标和球台范围裁剪后，实际世界坐标击球区域约为：

| 动作 | world y | world z |
| --- | --- | --- |
| 正手 | `[-0.7625, -0.45]` | `[1.00, 1.21]` |
| 反手 | `[-0.37, 0.45]` | `[0.84, 1.10]` |

后续继续训练配置的全台、全高度目标盒记录在 `config/strike_box.yaml` 的 `current_continuation_profile` 中，但它不是 `model_41500` 已经学到的能力。做当前策略部署或回放时使用 `checkpoint_profile`。

## MuJoCo 可视化

在仓库根目录执行：

```bash
cd /home/fsr/HOPE
python3 -m pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt

a3_deploy/model_41500_mujoco_deploy/scripts/run_mujoco.sh \
  --view \
  --eval-mode continuous \
  --num-serves 10 \
  --seed 0
```

脚本优先使用 GJC 精确机器人 XML、`model_41500` 的冻结训练击球盒、当前 runtime 的球台和多球参数，并在连续多球模式下运行。本工作区未提供 GJC XML 时，会自动使用仓库内带有 `isaac_joint_velocity_limits` 的 aligned MuJoCo XML；正式 sim2sim 或真实部署仍应通过 `A3_MODEL_XML` 指定与训练一致的 GJC 模型。

只检查脚本参数和文件路径：

```bash
cd /home/fsr/HOPE
a3_deploy/model_41500_mujoco_deploy/scripts/run_mujoco.sh --help
```

无球稳定状态测试：

```bash
cd /home/fsr/HOPE
a3_deploy/model_41500_mujoco_deploy/scripts/run_mujoco.sh \
  --view \
  --eval-mode independent \
  --no-serve \
  --num-serves 1 \
  --seed 0
```

实际训练盒内的正反手交替，并随机化初始 base 位置：

```bash
cd /home/fsr/HOPE
a3_deploy/model_41500_mujoco_deploy/scripts/run_mujoco.sh \
  --view \
  --eval-mode continuous \
  --num-serves 20 \
  --start-base-x-jitter 0.10 \
  --start-base-y-jitter 0.35 \
  --seed 0
```

`model_41500_training_contract.yaml` 会让每一球在该检查点实际训练过的正手、反手位置和速度盒内采样；上面的命令只增加初始化位置扰动，不会改变策略训练盒。需要测试后续全台、全高度盒时，应使用继续训练版本的策略和对应 planner 配置，不能仅替换当前检查点的测试参数。

## 部署检查项

1. 真实部署前确认控制周期稳定为 20 ms，并确认机器人 XML、关节顺序、关节限位和动作适配配置一致。
2. 目标站位使用 `model_41500` 的 `base_target_y_range: [-0.35, 0.35]`。后续配置中的 `0.6625` 上界属于继续训练版本，不能直接用于解释当前检查点。
3. 运行时保留自身与球台碰撞检查；首次测试使用低速、限幅和急停，确认机器人不会把球台作为可穿越空间。
4. 真实机器人部署不能直接把 MuJoCo 的球体位置当作传感器输入，必须由来球预测器生成同一坐标系下的目标位置、速度和击球时间。

## 文件结构

```text
model_41500_mujoco_deploy/
├── README.md
├── provenance.json
├── policy/
│   ├── hope_pingpong.onnx
│   └── policy_manifest.json
├── config/
│   ├── action_adapter.yaml
│   ├── hope_pingpong_runtime.yaml
│   ├── model_41500_training_contract.yaml
│   └── strike_box.yaml
└── scripts/
    └── run_mujoco.sh
```
