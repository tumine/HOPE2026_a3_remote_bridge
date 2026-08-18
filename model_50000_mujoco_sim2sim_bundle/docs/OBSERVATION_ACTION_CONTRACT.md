# 观测与动作合同

## 111 维 actor 观测

所有值为未归一化 `float32`，输入形状 `[1, 111]`。

| Slice | 名称 | 维度 | 语义 |
| --- | --- | ---: | --- |
| `[0:3]` | base_ang_vel | 3 | pelvis body frame 角速度，rad/s |
| `[3:34]` | joint_pos | 31 | `q - default_q`，canonical joint order |
| `[34:65]` | joint_vel | 31 | canonical joint order，rad/s |
| `[65:96]` | last_action | 31 | 上一 tick 实际应用的 clipped raw action |
| `[96:99]` | projected_gravity | 3 | 重力在 pelvis body frame 的方向 |
| `[99:101]` | base_forward_xy | 2 | pelvis 前向轴在 world xy 的投影 |
| `[101:103]` | fixed_station_error_xy | 2 | `base_target_xy - base_pos_w[:2]` |
| `[103:106]` | racket_target_rel_base | 3 | `target_pos_w - base_pos_w`，world 轴分量 |
| `[106:109]` | racket_target_vel_w | 3 | world frame，m/s |
| `[109:110]` | time_to_strike | 1 | 秒；击球后为负 |
| `[110:111]` | swing_side | 1 | 正手 `+1`，反手 `-1` |

`racket_target_rel_base` 只做位置相减，不按 pelvis yaw 旋转。`base_pos_w` 必须是可靠的世界坐标位置，不可以仅靠 IMU 速度无限积分。

## 31 维关节顺序

准确顺序记录在 `policy/policy_manifest.json` 和：

```text
a3_deploy/.../joint_order.py
```

顺序为腰 3、头 2、左臂 7、右臂 7、左腿 6、右腿 6。部署端必须按名字映射，不能假定硬件消息数组顺序相同。

## raw action 到关节目标

```text
raw_action = ONNX(obs)
applied_action = clip(raw_action, configured raw bounds)
applied_action[head_yaw, head_pitch] = 0
q_des = default_q + transform(applied_action) * action_scale
q_des = clip(q_des, software joint limits)
```

完整参数在 `config/action_adapter.yaml`。`last_action` 必须反馈 `applied_action`，不是未限幅的 ONNX 输出，也不是最终 `q_des`。

## MuJoCo PD

包内 MuJoCo 使用：

```text
tau = kp * (q_des - q) - kd * qd
```

每个物理子步重新计算，并受 MJCF actuator torque range 限制。`model_50000` 对齐配置中：

```text
waist_yaw_joint   kp=150, kd=3
waist_pitch_joint kp=150, kd=2
```

这些是 sim2sim 参数。真实机器必须由部署和安全负责人确定硬件控制增益，不能直接把 MuJoCo KP 当作硬件批准值。

## 必须对齐的状态来源

- `q`、`qd`：31 个关节的同一时刻状态
- pelvis quaternion：world 到 body 约定必须与 reference quaternion 代码一致
- pelvis gyro：body frame
- `base_pos_w[:2]`：与 planner target 相同 world frame
- `last_action`：上一 tick 实际应用动作
- planner target：position/velocity 使用同一 world 轴

任何一个坐标或时间源不一致，都可能让 ONNX 仍能运行但行为与 MuJoCo 不同。
