# HOPE Ping-Pong model_72500 部署候选包

本包冻结训练节点 `model_72500.pt`。它包含 ONNX 策略、完整 111 维观测与
31 维动作参考实现、planner 命令/站位/状态机代码、waist roll/pitch 锁定配置、
GJC MuJoCo 模型、独立 READY 参考动作、验证脚本和训练溯源文件。

这是训练仍在进行时冻结的中间部署候选：`model_72500` 的躯干推力课程约完成
45%，课程强度约 50.5%。它不是最终 `model_78000`，也不是实机安全认证版本。

## 固定策略合同

- ONNX：`observation float32[batch,111] -> raw_action float32[batch,31]`
- ONNX 支持动态 batch；实时部署固定使用 batch `1`
- 控制频率：`50 Hz`，周期 `20 ms`
- 观测归一化：无
- canonical 关节顺序：`config/joint_order_agibot_a3.yaml`
- 动作：identity，先裁剪到 `[-100,100]`，再做 residual position decode
- 站位：正/反手击球点反推横向 `base_target_y`；READY 回启动中央站位

## 本节点最重要的腰部变化

| 关节/索引 | 策略动作 | q_des | MuJoCo nominal PD |
|---|---|---|---|
| `waist_yaw_joint` / 0 | 正常执行 | `0 + 0.25*a[0]` 后机械限位 | KP 85, KD 3 |
| `waist_roll_joint` / 1 | **忽略并置 0** | 恒为 `0 rad` | KP 500, KD 2 |
| `waist_pitch_joint` / 2 | **忽略并置 0** | 恒为 `0 rad` | KP 500, KD 2 |

策略仍输出 31 维以兼容 checkpoint，但 roll/pitch 两列不得发送到执行器；部署
ActionAdapter 会将它们置零，并在下一周期 `last_action` 中反馈零。头部索引 3/4
也因 passive neck 置零，因此 `last_action[1:5]` 在本部署合同中恒为零。实测
`q/qd` 仍必须进入观测，不能伪造为零。

完整逐项计算和腰部迁移要求见 `docs/OBSERVATION_AND_WAIST.md`；planner、
状态机、正反手站位和实机接入见 `docs/DEPLOYMENT.md`。

## 快速验证

```bash
cd /home/fsr/HOPE/a3_deploy/model_72500_deploy_bundle
/home/fsr/miniconda3/envs/hope/bin/python scripts/verify_bundle.py
```

READY：

```bash
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_ready_eval.sh --view
```

普通连续 20 球：

```bash
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_normal_eval.sh --view
```

困难左右交替球：

```bash
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_hard_eval.sh --view
```

参考 runner：

```bash
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_reference_mujoco.sh --idle --max-ticks 500 --view
```

首次上机必须使用支撑/吊绳、低速与力矩限制、通信 watchdog、独立急停，并先
逐项比对 111 维观测；`config/runtime.yaml` 的 PD 是 MuJoCo 对齐值，不代表厂商
认证的真机增益。
