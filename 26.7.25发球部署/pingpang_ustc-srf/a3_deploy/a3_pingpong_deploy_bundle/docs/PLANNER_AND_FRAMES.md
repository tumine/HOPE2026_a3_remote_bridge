# Planner、命令和坐标系

## 策略没有直接观察球

ONNX 输入中没有球的位置或速度。Planner 先把动捕来球转换为一次挥拍命令：

```text
ball mocap
  → 来球状态估计
  → 分别预测正手/反手击球平面
  → 检查对应击球窗口
  → 锁定 swing_side
  → 反解球拍目标速度
  → RacketCommand
  → 111 维策略观测
```

## RacketCommand

| 字段 | 类型 | 含义 |
|---|---|---|
| `header` | ROS Header | 动捕时间戳和坐标系 |
| `task_id` | uint64 | 每个新来球严格递增 |
| `task_revision` | uint32 | 同一来球触球前的目标更新版本 |
| `swing_side` | int8 | 正手 `+1`，反手 `-1` |
| `position` | Point | world/table 系目标球拍中心，m |
| `velocity` | Vector3 | world/table 系目标球拍速度，m/s |
| `time_to_strike` | float64 | 距离触球时间，s |

同一个 `task_id` 只挥一次；`swing_side` 一旦确定必须锁定。更高 revision 只允许在
`time_to_strike > 0` 时更新目标。

## 策略训练窗口

x/y 相对机器人启动时固定站位，z 是 floor world 高度：

| 动作 | x 相对站位 | y 相对站位 | z world 高度 |
|---|---|---|---|
| 正手 | `[0.18, 0.40] m` | `[-0.76, -0.52] m` | `[1.00, 1.21] m` |
| 反手 | `[0.45, 0.75] m` | `[-0.30, 0.10] m` | `[0.84, 1.10] m` |

目标球拍速度按 world 轴表达：

| 动作 | vx | vy | vz |
|---|---|---|---|
| 正手 | `[1.75, 3.40] m/s` | `[0.25, 1.10] m/s` | `[0.35, 1.45] m/s` |
| 反手 | `[1.05, 3.00] m/s` | `[-0.25, 0.50] m/s` | `[0.45, 1.45] m/s` |

训练中的球桌近端和桌面高度为：

```text
table_near_x = station_x + 0.50 m
table_surface_z = 0.76 m
```

完整动作、来球速度、课程学习和回球判定参数见
`config/policy_training_contract.yaml`。

## 当前 live Planner 已与窗口对齐

Planner 使用 canonical table-surface frame：近端左角为原点，`+x` 指向对方，`+y`
向左，`+z` 向上，桌面为 `z=0`。当前机器人启动站位在该坐标系中是：

```text
station_table = [-0.50, -0.7625, -0.76] m
```

因此从 canonical table 到机器人 floor world 的位置变换是：

```text
p_floor = p_table + [0.50, 0.7625, 0.76]
v_floor = v_table
```

训练窗口转换到 canonical table 后，与
`hope_ws/src/hope_planner/config/hope_planner.yaml` 一致：

| 动作 | x | y | z |
|---|---|---|---|
| 正手 | `[-0.32, -0.10]` | `[-1.5225, -1.2825]` | `[0.24, 0.45]` |
| 反手 | `[-0.05, 0.25]` | `[-1.0625, -0.6625]` | `[0.08, 0.34]` |

当前配置还固定：

- `use_side_aware_strike_regions: true`；
- 正反手分别使用各自 x 平面，不再共用单一 `x_hit`；
- `swing_side_split_y = -1.1725 m`，`hysteresis = 0.04 m`，滞回带完全位于两窗口间隙；
- 新任务只在 `time_to_strike ∈ [0.95, 1.0] s` 时接收；
- 球拍目标速度必须进入对应动作速度范围；
- 目标落点为对方半台中心 `(2.055, -0.7625, 0.02) m`；
- 位置、速度或时序不满足合同的来球不裁剪，Planner 不发布命令。

对应的一致性测试位于 `hope_ws/src/hope_planner/test/test_strike_selection.py` 和
`test_task_timing.py`。

## 现场仍需标定的坐标

代码中的数值对齐不等于现场标定完成。下面数据必须位于同一个经过测量的 world/table
frame：

- 动捕球位置和速度；
- Planner 的目标位置与速度；
- 机器人基座 `base_pos_w` 和基座前向方向；
- 启动时捕获的 `fixed_station_xy`；
- 球桌、球网和目标落点。

部署时至少要验证：

1. 静止球放在桌面四角时，动捕坐标与 canonical table 坐标一致。
2. 机器人站在标称位置时，`base_pos_w` 与
   `station_table = [-0.50,-0.7625,-0.76]` 一致。
3. `racket_target_rel_base = target_pos_w - base_pos_w` 只做位置相减，不再按 pelvis yaw
   旋转。
4. Planner 的 velocity 只做平移坐标系下的原样传递，不加位置偏置。

## 时间戳和延迟

Planner 的 `header.stamp` 表示球状态捕获/规划时刻。当前 ROS bridge 会减去 ROS 时钟域的
消息传输时间，并用单调时钟继续减去 callback 到控制周期的邮箱等待时间；无效、零或未来
时间戳不会增加 TTS。

实机还需要测量并处理 ONNX 推理和电机执行延迟：

```text
effective_time_to_strike
  = planner_time_to_strike
  - message_age
  - mailbox_age
  - measured_inference_and_actuation_latency
```

最后一项尚不能凭经验写死，必须用实机日志标定。还要对过期 task、乱序 revision、时钟
不同步和控制周期超时设置保护。
