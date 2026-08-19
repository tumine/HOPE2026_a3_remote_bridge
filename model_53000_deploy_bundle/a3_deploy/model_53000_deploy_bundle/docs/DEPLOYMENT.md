# 部署说明

## 1. 固定策略契约

策略部署输入为 `float32[1,111]`，输出为 `float32[1,31]`，控制频率固定为 50 Hz，不使用观测归一化。ONNX 图本身把首维导出为动态 `batch`，部署时固定使用 `batch=1`。关节顺序必须严格等于 `config/joint_order_agibot_a3.yaml`，不能按 SDK 返回顺序直接拼接。

111 维观测布局：

| 切片 | 内容 | 单位/语义 |
|---|---|---|
| `[0:3]` | base angular velocity | pelvis body frame, rad/s |
| `[3:34]` | `q - default_q` | rad |
| `[34:65]` | joint velocity | rad/s |
| `[65:96]` | 上一周期 applied action | 裁剪后、被动头部置零后的动作 |
| `[96:99]` | projected gravity | base frame |
| `[99:101]` | base forward xy | world xy |
| `[101:103]` | `base_target_xy - base_xy` | m |
| `[103:106]` | `racket_target_w - base_pos_w` | m，仍沿 world 轴 |
| `[106:109]` | target racket velocity | world, m/s |
| `[109]` | time to strike | s |
| `[110]` | swing side | 正手 `+1`，反手 `-1` |

每个 20 ms 周期必须按如下顺序执行：读取机器人状态；消费最新 `RacketCommand`；更新 lifecycle 和 base 目标；组装观测；执行 ONNX；将 raw action 裁剪到 `[-100,100]`；将头部第 3、4 列置零；把这个 applied action 保存为下一周期的 `last_action`；最后按 `q_des = default_q + 0.25 * applied_action` 解码并执行机械限位裁剪。

`last_action` 不能保存 raw action，也不能保存最终 `q_des`。状态估计、目标点和 base 位置必须在同一个经过标定的 world/table 坐标系中。

## 2. planner、正反手与站位

planner 在球进入可预测轨迹后估计击球时刻、世界系击球点和目标拍速，并发布可修订的 `RacketCommand`。同一球使用相同 `task_id`，接触前的新预测递增 `task_revision`；下一球必须使用更大的 `task_id`。

正反手在横向击球点 `y=-0.41 m` 处闭合切换，没有滞回空带：右侧区域使用正手盒，左侧区域使用反手盒。精确窗口见 `config/strike_box.yaml`。

部署 runner 根据击球点反推横向站位：

```text
forehand: base_target_y = clamp(strike_y - (-0.64), -0.35, 0.6625)
backhand: base_target_y = clamp(strike_y - (-0.10), -0.35, 0.6625)
base_target_x = 启动时的中央站位 x
```

这是给观测 `[101:103]` 的闭环目标，不是直接修改 floating base。机器人通过腿部策略自行移动。READY/RECOVERY 时目标回到启动中央站位。

## 3. 状态切换

参考实现位于 `reference/a3_deploy_onnx_ref_pingpong/lifecycle.py` 和 `runner.py`：

```text
READY --新 task_id--> SWING --TTS 到 0 后--> FOLLOW_THROUGH
FOLLOW_THROUGH --0.8 s--> RECOVERY --0 s--> READY
```

同一球只触发一次挥拍；接触前允许 revision 更新击球点和 TTS。球间不重置机器人、关节状态或 policy history。无有效球命令时仍以 50 Hz 推理，使用 `runtime.yaml` 中的固定 READY 正手条件观测，同时 base 目标回中央。

TTS 应使用 planner 消息的目标时间减当前时间，并扣除消息传输/处理年龄。该模型按 `+1.0 s` 的动作片段起点训练；若首次可用命令已经小于 `0.5 s`，即使 planner 数学预测正确，腿部也通常没有足够时间完成横移。目前 `0.25 s` 首命令不在本模型已验证训练分布内。

## 4. 球物理一致性

planner 和验证必须加载 `config/ball_physics.yaml`。当前关键参数是：

- `contact.table.restitution = 0.9448`
- `contact.table.tangential_damping = 0.2079`
- `drag.k = 0.1261`
- `contact.paddle.restitution = 0.654`

替换配置后必须重启 planner 进程，避免继续使用内存中的旧参数。若 planner 支持环境变量，可将 `HOPE_BALL_PHYSICS_CONFIG` 指向本包的绝对路径；否则在 planner 参数中传入同一路径。

## 5. 真实机器人接入

参考 Python runner 是行为契约，不是完整的硬件安全控制器。真实后端至少需要提供：时间同步的 pelvis 位姿/角速度、31 个关节的位置与速度、关节顺序重排、50 Hz 位置目标发送、通讯超时、速度/力矩/温度保护和独立急停。

`runtime.yaml` 中 PD 为 GJC MuJoCo 仿真参数：waist yaw `85/3`、roll `50/2`、pitch `50/2` 等。它们不是实机推荐增益，不能未经低增益、限速、吊绳和急停验证直接写入机器人。

建议上线顺序：先离线校验观测逐项数值；再架空/支撑测试 READY；然后禁球低速测试左右站位；最后在安全区逐步放开挥拍。每一步记录 111 维观测、raw/applied action、q_des、实测 q、planner TTS、task id/revision 和状态机 phase。

## 6. 部署前硬检查

- ONNX SHA256 与 `bundle_manifest.json` 一致。
- 输入输出名称、shape、dtype 正确。
- joint order、四元数顺序 `(w,x,y,z)` 和 world 轴定义正确。
- `last_action` 反馈语义正确，头部保持默认位。
- table/world 外参经过实测，启动中央站位稳定。
- planner 正反手符号与 `strike_box.yaml` 一致。
- 新球 task id 单调递增，旧 revision 不会再次触发挥拍。
- TTS 使用统一时钟并补偿消息年龄。
- 机械限位、速度限制、通讯 watchdog 和急停独立于策略工作。
