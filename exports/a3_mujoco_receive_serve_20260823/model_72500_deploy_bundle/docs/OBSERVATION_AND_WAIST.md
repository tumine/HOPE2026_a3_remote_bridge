# 完整 111 维观测计算与腰部锁定合同

## 1. 输入输出和坐标约定

```text
ONNX input : observation float32[batch,111]
ONNX output: raw_action  float32[batch,31]
dynamic ONNX batch, deploy batch = 1, control rate = 50 Hz, normalization = none
```

- `w`：与 planner/table 标定一致的世界坐标系。
- `b`：pelvis/base body 坐标系。
- 四元数：Hamilton scalar-first `(w,x,y,z)`，表示 body 到 world 的旋转。
- base 位姿：pelvis 位姿，不是 torso 位姿。
- 关节数组：严格采用 `config/joint_order_agibot_a3.yaml`。
- 所有观测先按下面公式用 `float64` 或等价精度计算，最终一次性转为连续
  `float32[111]`；不做 mean/std normalization。

## 2. 逐段布局和公式

| 切片 | 名称 | 计算 | 单位/坐标 |
|---|---|---|---|
| `[0:3]` | `base_ang_vel` | pelvis IMU/状态估计给出的 `omega_b` | rad/s，body |
| `[3:34]` | `joint_pos` | `q_canonical - default_q` | rad |
| `[34:65]` | `joint_vel` | `qd_canonical` | rad/s |
| `[65:96]` | `last_action` | 上一周期实际接受的 applied action | 无量纲 |
| `[96:99]` | `projected_gravity` | `R(q_wb)^T * [0,0,-1]` | body，单位向量 |
| `[99:101]` | `base_forward_xy` | 见下式 | world xy，近单位向量 |
| `[101:103]` | `fixed_station_error_xy` | `base_target_xy - base_pos_w[:2]` | world，m |
| `[103:106]` | `racket_target_rel_base` | `target_pos_w - base_pos_w` | **world 轴分量**，m |
| `[106:109]` | `racket_target_vel_w` | planner 目标拍速 `target_vel_w` | world，m/s |
| `[109]` | `time_to_strike` | lifecycle 当前未提前递减的 TTS | s |
| `[110]` | `swing_side` | 正手 `+1`，反手 `-1`，READY `0` | 标量 |

维度校验：

```text
3 + 31 + 31 + 31 + 3 + 2 + 2 + 3 + 3 + 1 + 1 = 111
```

base forward 的精确实现：

```text
fwd_w = R(q_wb) * [1,0,0]
base_forward_xy = fwd_w[:2] / (norm(fwd_w[:2]) + 1e-6)
```

注意 `[103:106]` 虽名为 relative-base，但只做位置相减，不乘 `R^T`；它保留
world 轴方向。把它错误旋转到 pelvis frame 会造成训练/部署严重不一致。

## 3. Canonical 关节索引

```text
 0 waist_yaw_joint                 16 right_wrist_roll_joint
 1 waist_roll_joint                17 right_wrist_pitch_joint
 2 waist_pitch_joint               18 right_wrist_yaw_joint
 3 head_yaw_joint                  19 left_hip_pitch_joint
 4 head_pitch_joint                20 left_hip_roll_joint
 5 left_shoulder_pitch_joint       21 left_hip_yaw_joint
 6 left_shoulder_roll_joint        22 left_knee_joint
 7 left_shoulder_yaw_joint         23 left_ankle_pitch_joint
 8 left_elbow_joint                24 left_ankle_roll_joint
 9 left_wrist_roll_joint           25 right_hip_pitch_joint
10 left_wrist_pitch_joint          26 right_hip_roll_joint
11 left_wrist_yaw_joint            27 right_hip_yaw_joint
12 right_shoulder_pitch_joint      28 right_knee_joint
13 right_shoulder_roll_joint       29 right_ankle_pitch_joint
14 right_shoulder_yaw_joint        30 right_ankle_roll_joint
15 right_elbow_joint
```

## 4. `last_action` 的精确定义

设 ONNX 输出为 `raw_action_t`：

```text
applied_t = clip(raw_action_t, -100, 100)
applied_t[1] = 0    # waist_roll locked
applied_t[2] = 0    # waist_pitch locked
applied_t[3] = 0    # passive head yaw
applied_t[4] = 0    # passive head pitch
observation_(t+1)[65:96] = applied_t
```

不能反馈 raw action、tanh action、`q_des` 或实测关节位置。第一个控制周期使用
全零 `last_action`。策略的原始 roll/pitch 输出可能非零，但必须被忽略；由于这两列
没有物理执行意义，部署日志应同时保留 raw 和 applied 以便诊断。

## 5. 动作解码和腰部变化

```text
adapter_action = applied_action             # identity transform
q_des = default_q + 0.25 * adapter_action
q_des = clip(q_des, mechanical_lower, mechanical_upper)
```

然后再次执行安全覆盖：

```text
q_des[waist_roll]  = 0 rad
q_des[waist_pitch] = 0 rad
q_des[head_yaw]    = default_q[head_yaw]
q_des[head_pitch]  = default_q[head_pitch]
```

`waist_yaw_joint` 没有锁定，仍由动作索引 0 控制。本节点相对 model_68000 的关键
变化是：waist roll/pitch 不再执行策略动作，训练正手、反手和 READY 参考中的这
两轴位置/速度也都为 0；MuJoCo nominal KP 从 50 提升到 500，KD 仍为 2。
训练保留 0.9--1.1 的 PD 随机化，因此训练 nominal 500 对应约 450--550 的随机
刚度；部署配置写的是 nominal 500。

正手参考保留 waist yaw 约 `-11.13..+6.90 deg`。反手在锁住 pitch/roll 后使用
waist yaw 和右臂补偿，yaw 约 `+5.63..+22.19 deg`；触球严格窗口相对旧轨迹误差
约 `0.63 mm / 0.216 deg`。因此部署不能顺带锁死 waist yaw。

高 KP 只保证目标更硬，不保证真实关节永远为零。`q[1:3]` 和 `qd[1:3]` 必须继续
使用实测值进入 `[3:34]`、`[34:65]`；若用伪零会切断策略对受力偏差的反馈。

## 6. READY 观测示例

没有有效球命令时：

```text
base_target_xy = startup nominal station
target_pos_w   = live_base_pos_w + [0.45,-0.25,0.08]
target_vel_w   = [0,0,0]
TTS            = 1.0
swing_side     = 0
```

因此 READY 的目标部分为：

```text
obs[101:103] = startup_xy - live_base_xy
obs[103:106] = [0.45,-0.25,0.08]
obs[106:109] = [0,0,0]
obs[109]     = 1.0
obs[110]     = 0.0
```

本体状态、关节实测、projected gravity 和 base heading 仍实时变化。

## 7. 部署侧逐拍断言

- `obs.shape == (111,)`，dtype 在送入 ONNX 前为 `float32`，所有元素有限。
- 四元数先归一化，顺序为 `(w,x,y,z)`。
- joint SDK order 已重排为 canonical order。
- `last_action[[1,2,3,4]] == 0`。
- `q_des[1] == q_des[2] == 0`。
- READY 时 `[103:111] == [0.45,-0.25,0.08,0,0,0,1,0]`。
- 真实球时 `swing_side` 只能为 `+1/-1`，同一 task 不改变。
- TTS 在执行完当前动作后才递减 0.02 s。
