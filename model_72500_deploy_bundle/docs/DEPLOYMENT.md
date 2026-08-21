# model_72500 部署、planner 与状态机说明

## 1. 每个 20 ms 控制周期

严格按以下顺序执行，不能把动作反馈或时钟提前一拍：

1. 读取同一时刻的 pelvis 位姿/角速度及 31 个关节 `q, qd`。
2. 按 canonical joint order 重排关节数据。
3. 获取 planner 最新 `RacketCommand`，用 `task_id/revision` 更新 lifecycle。
4. 根据 phase、击球点和正反手计算 `base_target_xy`。
5. 组装 raw `float32[111]` 观测，禁止额外归一化。
6. ONNX 推理得到 `raw_action[31]`。
7. 裁剪动作；waist roll/pitch 和 passive head 置零；保存为下一拍 `last_action`。
8. 解码 `q_des`，执行机械限位；roll/pitch 目标必须再次硬校验为 0。
9. 向后端发送目标和经过实机审核的增益，完成一个 20 ms 控制过渡。
10. 过渡完成后才将 lifecycle 的 TTS 减少 0.02 s。

完整观测公式见 `OBSERVATION_AND_WAIST.md`，可执行实现见
`reference/a3_deploy_onnx_ref_pingpong/{observation,action_adapter,runner}.py`。

## 2. Planner 输出合同

每条命令至少包含：

```text
task_id             int，下一球严格递增
task_revision       int，同一球预测更新时递增
swing_side          +1 正手 / -1 反手
position            世界系击球点 [x,y,z]，m
velocity            世界系目标球拍线速度 [vx,vy,vz]，m/s
time_to_strike      目标击球时刻 - 当前统一时钟 - 消息年龄，s
```

同一 `task_id` 的正反手一经接受即锁定；接触前允许更高 revision 修正击球点、
目标速度和 TTS。过期、非有限或 TTS<0 的新 task 不启动挥拍。planner、机器人
状态估计和 table 外参必须使用同一 world 坐标系和同一时钟源。

本策略训练挥拍时钟从 `TTS=1.0 s` 开始。首次有效命令明显低于 0.5 s 时，横向
移动时间不足；`TTS=0.25 s` 不属于本节点已验证训练分布。

## 3. 正反手选择与站位

击球窗口见 `config/strike_box.yaml`。横向切换点为 `y=-0.41 m`，无滞回空带：

```text
strike_y <= -0.41 -> forehand (+1)
strike_y >= -0.41 -> backhand (-1)
```

runner 从击球点反算站位：

```text
forehand: base_target_y = nominal_y + clamp(strike_y - nominal_y - (-0.64), -0.35, 0.6625)
backhand: base_target_y = nominal_y + clamp(strike_y - nominal_y - (-0.10), -0.35, 0.6625)
base_target_x = nominal startup x
```

这里的 `base_target_xy` 只进入观测 `[101:103]`；策略用腿部动作移动机器人。
部署侧不能直接修改 floating-base 位姿。

## 4. Lifecycle

```text
READY --接受新 task_id--> SWING --TTS 到 0--> FOLLOW_THROUGH
FOLLOW_THROUGH --0.8 s--> READY --下一 task_id--> SWING
```

- 每个 `task_id` 最多触发一次挥拍。
- 球间不重置机器人、关节状态、ONNX 或 `last_action` 历史。
- `RECOVERY` 配置为 0 s，因此 follow-through 后直接回 READY。
- READY 和 RECOVERY 的 base 目标是启动中央站位。
- 状态机实现：`reference/.../lifecycle.py`。
- 站位实现：`reference/.../station_target.py`。

## 5. 无球 READY

无有效命令时仍每 20 ms 推理，使用：

```text
base_target_xy          = nominal startup station
racket_target_rel_base  = [0.45, -0.25, 0.08] m（世界轴分量）
racket_target_vel_w     = [0, 0, 0] m/s
time_to_strike          = 1.0 s
swing_side              = 0
```

这是训练中 20% 独立无球 READY 环境的观测。包内 READY 参考动作的 waist
yaw/roll/pitch 为 `[0,0,0]`，roll/pitch 速度也为零；参考动作只用于监督/验证，
不直接拼进 actor 观测。

## 6. 球物理

planner 和 MuJoCo 都应加载 `config/ball_physics.yaml`：table restitution
`0.9448`，table tangential damping `0.2079`，quadratic drag `k=0.1261`。
替换配置后重启 planner，防止进程继续使用旧参数。

## 7. 真机接入与安全

`runtime.yaml` 中 KP/KD 用于 GJC MuJoCo 对齐。尤其 roll/pitch 的 KP=500 很高，
上线前必须由机器人/执行器负责人确认额定力矩、刚度单位和稳定性，不能直接把仿真
数字视为厂商推荐值。后端至少需要：

- position/velocity/effort/temperature 限制及独立急停；
- 观测和命令超时 watchdog；
- waist roll/pitch `q_des==0` 的发送前断言；
- canonical joint order 与 SDK order 显式映射；
- `(w,x,y,z)` 四元数和 body/world 角速度语义校验；
- 全量记录 obs、raw/applied action、q_des、q/qd、TTS、phase、task id/revision。

上线顺序：离线逐项观测比对；支撑下无球 READY；禁球低速横移；低速单侧球；
最后才开放连续正反手和完整速度。
