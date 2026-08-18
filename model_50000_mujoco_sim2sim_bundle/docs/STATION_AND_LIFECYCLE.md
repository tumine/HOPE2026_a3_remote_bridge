# 站位计算与状态机

## 1. 谁计算站位

MuJoCo 和 reference deploy runner 都调用同一个函数：

```text
a3_deploy/a3_deploy_example/reference/
a3_deploy_onnx_ref_pingpong/station_target.py
```

入口是 `derive_lateral_base_target()`。planner 只提供击球点、球拍速度、TTS 和正反手；planner 不直接控制 base。

## 2. 站位公式

设：

- `target_y`：planner 给出的世界坐标击球点 y
- `nominal_y`：机器人启动时的中心站位 y
- `reach_y`：动作片段的舒适球拍侧向距离
- `[station_min, station_max]`：训练过的 base 相对中心站位范围

当前值：

```text
forehand_reach_y = -0.64 m
backhand_reach_y = -0.10 m
station range = [-0.35, 0.6625] m
```

计算：

```text
desired_offset_y = target_y - nominal_y - reach_y
clamped_offset_y = clip(desired_offset_y, station_min, station_max)
base_target_x = nominal_x
base_target_y = nominal_y + clamped_offset_y
```

因此只有 y 方向移动，x 始终锁定启动站位。训练和部署必须使用相同的世界轴，不得把 `target_y` 先按 pelvis yaw 再旋转。

## 3. 站位进入观测

站位不是直接的位置控制命令，而是：

```text
obs[101:103] = base_target_xy - current_base_xy
```

历史字段名仍为 `fixed_station_error_xy`，实际语义是 moving station error。策略通过腿部关节动作学习消除这个误差。

## 4. 状态机

```text
READY -> SWING -> FOLLOW_THROUGH -> RECOVERY -> READY
```

当前 `recovery_s=0`，所以实际通常为：

```text
READY -> SWING -> FOLLOW_THROUGH -> READY
```

| 状态 | 球拍目标 | base 目标 | TTS |
| --- | --- | --- | --- |
| READY | 固定正手 READY 锚点 | 启动中心站位 | 固定 1.0 s |
| SWING | active task 目标 | 根据击球点计算 | 每 tick 减 0.02 s |
| FOLLOW_THROUGH | 保留 active task 目标 | 保留击球站位 | 从 0 降至 -0.8 s |
| RECOVERY | READY 锚点 | 启动中心站位 | 固定 1.0 s |

## 5. 新任务规则

- 只有严格更大的 `task_id` 才是新球。
- 新球只能在 READY 或 RECOVERY 接入。
- 新任务 TTS 必须有限且 `>= 0`。
- 接入后 `swing_side` 锁定，不能由 revision 改变。
- active task 可在 SWING 且 TTS `> 0` 时接受更大的 `task_revision`。
- revision 可以更新 position、velocity 和 TTS。
- 每个 `task_id` 最多触发一次挥拍。

过期新任务的危险行为：负 TTS 的新 `task_id` 会被标记为已经处理但不进入 SWING；同一 task 后续 revision 不能重新触发挥拍。部署端必须在提交给 lifecycle 前记录 aged TTS。

## 6. 每个 20 ms tick 的顺序

```text
1. 读取机器人状态
2. poll 最新 RacketCommand
3. lifecycle.update(command, state)
4. 根据 phase 计算 base_target_xy
5. 构造 111 维观测
6. ONNX 推理
7. raw action 限幅和 passive neck 处理
8. ActionAdapter 解码为 31 个关节位置目标
9. 下发目标并推进物理/真实控制器
10. lifecycle.advance()，TTS 减少一个 20 ms tick
```

新命令必须先被策略观察一次，然后才减 TTS。部署端如果在观测前先减时钟，会与训练错一帧。

## 7. 部署日志最低要求

每个 tick 至少能导出：

```text
phase, task_id, task_revision, swing_side
command_tts_raw, command_age, observed_tts
target_pos_w, target_vel_w
nominal_station_xy, base_target_xy, current_base_xy
obs[101], obs[102], obs[109], obs[110]
```

偏球但不移动时，先看 `obs[102]`：非零而 base 不动属于策略/执行器问题；接近零属于坐标、站位或生命周期问题。
