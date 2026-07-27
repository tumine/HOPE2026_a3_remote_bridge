# 观测与动作合同

## ONNX 签名

```text
observation: float32[batch, 111]
    ↓
hope_pingpong.onnx
    ↓
raw_action: float32[batch, 31]
```

- 控制周期：`0.02 s`，即 50 Hz。
- 不做 observation normalization。
- ONNX 只包含确定性 actor；PPO 的 critic、optimizer、探索噪声 `std` 都不在部署模型中。
- 训练阶段会给部分观测加噪声；部署阶段输入真实原始量，不主动加噪声。
- 所有关节数组必须严格使用 `config/joint_order_agibot_a3.yaml` 的 31 维顺序。

## 111 维观测

| 切片 | 名称 | 维度 | 坐标系/单位 | 定义 |
|---|---|---:|---|---|
| `[0:3]` | `base_ang_vel` | 3 | pelvis/body，rad/s | 基座 IMU 角速度 |
| `[3:34]` | `joint_pos` | 31 | rad | `q - default_q` |
| `[34:65]` | `joint_vel` | 31 | rad/s | 编码器关节速度 |
| `[65:96]` | `last_action` | 31 | 无量纲 | 上周期实际应用的 action |
| `[96:99]` | `projected_gravity` | 3 | base，单位向量 | 世界重力方向旋转到基座坐标系 |
| `[99:101]` | `base_forward_xy` | 2 | world XY，单位向量 | 基座前向轴在世界 XY 平面的投影 |
| `[101:103]` | `fixed_station_error_xy` | 2 | world，m | `startup_station_xy - base_pos_w[:2]` |
| `[103:106]` | `racket_target_rel_base` | 3 | **world 轴分量**，m | `target_pos_w - base_pos_w` |
| `[106:109]` | `racket_target_vel_w` | 3 | world，m/s | Planner 目标球拍速度 |
| `[109:110]` | `time_to_strike` | 1 | s | 距离触球的剩余时间 |
| `[110:111]` | `swing_side` | 1 | `+1/-1` | 正手 `+1`，反手 `-1` |

总维数：

```text
3 + 31 + 31 + 31 + 3 + 2 + 2 + 3 + 3 + 1 + 1 = 111
```

### 容易接错的地方

1. `racket_target_rel_base` 虽然名字包含 `rel_base`，但代码只是做位置相减，结果仍是
   世界坐标轴分量；不能再旋转到机身坐标系。
2. `fixed_station_xy` 在启动时捕获一次，之后保持不变。它不是每周期更新的机器人位置。
3. `base_forward_xy` 和目标向量必须来自同一个 world/table 标定坐标系。
4. 仅有关节编码器和陀螺仪不足以构造观测；还需要基座世界位置、基座姿态和 Planner
   目标。
5. 四元数进入参考观测构造器前使用 `(w,x,y,z)`；若厂商 SDK 输出 `(x,y,z,w)`，必须转换。
6. `swing_side` 对同一个 `task_id` 必须锁定，不能在挥拍过程中翻转。

## 31 维动作

网络输出 `raw_action[31]` 后按照以下顺序处理：

```python
applied_action = clip(raw_action, -100.0, 100.0)
applied_action[3] = 0.0  # head_yaw_joint
applied_action[4] = 0.0  # head_pitch_joint

q_des = default_q + applied_action * 0.25
q_des = clip(q_des, joint_lower, joint_upper)
last_action_next = applied_action
```

当前 ActionAdapter：

- `raw_action_transform = identity`，不做 `tanh` 或其他非线性压缩。
- `action_clip = [-100, 100]`，先限制 raw action，再处理被动 head。
- `action_scale = 0.25 rad`，31 个关节统一缩放。
- `default_q` 和关节角 clamp 见 `config/action_adapter.yaml`。
- head 两列仍保留在 ONNX 输入输出中，但部署时保持默认角度。
- ONNX 输出是位置残差，不是绝对关节角、速度、力矩或电流。
- `last_action` 必须是 clip 后且 head 清零的 `applied_action`；不能反馈 `raw_action`。
- 最后的关节角 clamp 与 raw-action clip 是两层不同的限制，不能删掉其中任意一层。

## 无命令 READY 观测

没有可用的 `RacketCommand` 时，不把目标相关观测手写成零，也不使用旧
`ready_reach_x/y/z`。当前 model_21500 使用：

```text
obs[103:106] racket_target_rel_base_w =
  [ 0.3201315701, -0.6952511668, -0.0577955581 ]
obs[106:109] racket_target_vel_w =
  [ 2.3803303242,  0.6271703243,  1.0374835730 ]
obs[109] time_to_strike = 1.0
obs[110] swing_side = +1
```

其中相对位置仍以 world 轴表达，不随 pelvis yaw 再旋转。非零目标速度只是策略条件输入，
不表示待机时真实球拍已经以该速度运动。该锚点的权威配置位于 `config/runtime.yaml`。

上面的 `obs[103:106]` 只描述仿真第一帧。运行时先把第一帧基座位置与该相对向量相加，
锁定一个世界坐标虚拟目标；后续每周期计算
`fixed_ready_target_w - current_base_pos_w`。因此机器人相对起始站位移动 `delta_p` 后，
这三维变为首帧锚点减 `delta_p`，不是永远保持这三个常数。

## 31 关节顺序

| 索引 | 关节 | 索引 | 关节 |
|---:|---|---:|---|
| 0 | waist_yaw_joint | 16 | right_wrist_roll_joint |
| 1 | waist_roll_joint | 17 | right_wrist_pitch_joint |
| 2 | waist_pitch_joint | 18 | right_wrist_yaw_joint |
| 3 | head_yaw_joint（被动） | 19 | left_hip_pitch_joint |
| 4 | head_pitch_joint（被动） | 20 | left_hip_roll_joint |
| 5 | left_shoulder_pitch_joint | 21 | left_hip_yaw_joint |
| 6 | left_shoulder_roll_joint | 22 | left_knee_joint |
| 7 | left_shoulder_yaw_joint | 23 | left_ankle_pitch_joint |
| 8 | left_elbow_joint | 24 | left_ankle_roll_joint |
| 9 | left_wrist_roll_joint | 25 | right_hip_pitch_joint |
| 10 | left_wrist_pitch_joint | 26 | right_hip_roll_joint |
| 11 | left_wrist_yaw_joint | 27 | right_hip_yaw_joint |
| 12 | right_shoulder_pitch_joint | 28 | right_knee_joint |
| 13 | right_shoulder_roll_joint | 29 | right_ankle_pitch_joint |
| 14 | right_shoulder_yaw_joint | 30 | right_ankle_roll_joint |
| 15 | right_elbow_joint |  |  |

球拍安装在右腕。部署前必须用厂商实时 joint-state 列顺序重新核对；不能只按数组长度判断。
