# 本地 Planner 到 MDU 接收链路

这套链路把 Planner/动捕留在本地电脑，把 111 维 observation、ONNX actor、
动作适配和 50 Hz 控制循环放在 MDU。默认模式不发布关节命令；真实发布必须
经过手动状态机和显式安全解锁。

## 数据边界

```text
本地电脑
  /ppmocap/frame
    -> a3_receive_input_gateway
       -> /a3_mocap/ball_poses
       -> /a3_mocap/pelvis_pose
    -> hope_planner
       -> /racket/command

MDU
  /a3_mocap/pelvis_pose + /racket/command
    -> a3_mdu_planner_receiver
    -> PlannerInputMailbox

  /body_drive/*_joint_state + pelvis/torso IMU
    -> A3AimrtBackend (AimRT + iceoryx)
    -> synchronized robot_io::RobotState
    -> ReceiveController 50 Hz loop
    -> PingpongObservationBuilder (111-D)
    -> model_21500 ONNX -> 31-D RobotCommand
    -> RobotIOBackend::SendCommand()
    -> /body_drive/{waist,neck,arm,leg}_joint_command
```

`/racket/command` 使用 `hope_msgs/RacketCommand`，可靠、volatile、keep-last 10。
`/a3_mocap/pelvis_pose` 使用 `geometry_msgs/PoseStamped`，best-effort、volatile、
keep-last 5。两个消息的 `header.frame_id` 都必须是 `hope_table`。

MDU 接收器会校验：

- 位置、速度、四元数和 TTS 全部有限；
- `swing_side` 只能是 `+1` 或 `-1`；
- 新球的 `task_id` 单调递增；
- 同一球的 `task_revision` 单调递增，且挥拍侧不能改变；
- 从 `header.stamp` 扣除传输延迟，过期命令不进入 mailbox；
- command 和 base pose 分别执行超时检查。

## 构建与部署

当前现场采用 PC 与 MDU 的直连网段：

```text
PC enx6c1ff7ed3182: 192.168.1.11/24
MDU taixin_mdu:  192.168.1.100/24
ROS_DOMAIN_ID:   232
```

`run_receive_input_gateway.sh` 会检查到 MDU 的路由是否确实使用 PC 的有线地址。
如现场地址变化，可用 `A3_PC_WIRED_ADDRESS` 和 `A3_MDU_ADDRESS` 覆盖脚本默认值；
同时必须更新 `fastrtps_pc_planner_mdu.xml` 和 `fastrtps_mdu_planner.xml` 中的
接口白名单和 initial peer。旧的状态桥/PC推理DDS配置不受这条新链路影响。

在开发机生成 AArch64 包：

```bash
scripts/build_mdu_package.sh --jobs 8
```

把 `dist/a3_mdu_state_bridge/` 同步到 MDU 的独立探针目录：

```bash
rsync -av --progress \
  dist/a3_mdu_state_bridge/ \
  agi@192.168.1.100:/home/agi/a3_remote_bridge_probe/
```

MDU 上只启动接收探针：

```bash
cd /home/agi/a3_remote_bridge_probe
./scripts/run_mdu_planner_receiver.sh
```

这个默认模式完全不启动 RobotIO，适合先确认 PC 到 MDU 的 Fast DDS 链路。

确认 Planner 输入正常后，启动 RobotIO 只读探针：

```bash
cd /home/agi/a3_remote_bridge_probe
A3_ENABLE_ROBOT_IO_PROBE=1 ./scripts/run_mdu_planner_receiver.sh
```

确认状态同步后，启用 111 维 observation 探针：

```bash
cd /home/agi/a3_remote_bridge_probe
A3_ENABLE_OBSERVATION_PROBE=1 ./scripts/run_mdu_planner_receiver.sh
```

这个开关会自动启用 RobotIO 只读探针。它锁定首帧 pelvis XY 作为
`fixed_station_xy`，以 50 Hz 构造 observation，但不加载模型、不执行推理，且
`last_action` 固定为零。启动时还会逐项校验 RobotIO 的 31 DOF 顺序；顺序与
`model_21500` 合约不一致会直接拒绝启动。

## ONNX 只读推理探针

当前使用 `26.7.25发球部署` bundle 中的 `model_21500` actor：

```text
models/hope_pingpong.onnx
SHA256 6e2fcf9c9793a568f0583ae9b6e7b0439fb83df004c356e85af85841afc2d074
observation float32[1,111] -> raw_action float32[1,31]
```

MDU 上启动：

```bash
cd /home/agi/a3_remote_bridge_probe
A3_ENABLE_ONNX_PROBE=1 ./scripts/run_mdu_planner_receiver.sh
```

这个开关自动启用 RobotIO 和 observation probe，使用随包分发的 AArch64
ONNX Runtime CPU 库。模型加载时强制校验 input/output 名称、形状、类型、
checkpoint、策略 generation 和 31 DOF 元数据。推理输出只用于统计，并按合约
clip 后将被动 head 清零，反馈到下一帧 `last_action`；不会生成或发送
`RobotCommand`。

日志应包含：

```text
inference=(enabled=yes,runs=N,rejected=0,avg_ms=...,max_ms=...,raw_max_abs=...)
body_drive_publishers=0
```

## 完整 RobotCommand dry-run

ONNX 推理稳定后，在 MDU 上启动动作适配和命令结构校验：

```bash
cd /home/agi/a3_remote_bridge_probe
A3_ENABLE_ACTION_DRY_RUN=1 ./scripts/run_mdu_planner_receiver.sh
```

这个开关自动启用 RobotIO、observation 和 ONNX。适配器严格执行
`model_21500` 合约：raw action clip 到 `[-100, 100]`、head yaw/pitch 清零、
统一乘 `0.25` 后加默认关节角，并按 31 个机械限位裁剪。它构造的
`RobotCommand` 与真实发送包相同：`dq_des/tau_ff` 为零，策略态 Kp/Kd 来自
6/27 的冻结训练/部署配置；head 使用厂商 `ExpandToBackend()` 的 40/2 保持值。
backend 仍以 `publish_enabled=false` 启动，因此只返回 `dry_run`，不会调用
`SendCommand()`。

日志应包含：

```text
robot_io=(ticks=N,result=dry_run)
action=(output=dry_run,decoded=N,rejected=0,commands=N,raw_clips=0,...,gains=model_21500)
body_drive_publishers=0
```

只读探针严格使用 HOPE A3 部署约定：

- `CreateBackend("a3") -> Init() -> RegisterStateCallback() -> Start()`；
- MDU 使用 `config/a3_mdu_iceoryx.yaml` 和 `min_skew_pair`；
- backend 以 100 Hz 同步六路状态，控制线程以 50 Hz 读取最新快照；
- RobotIO callback 只缓存状态，不在 callback 线程执行策略；
- 默认 backend 使用 `publish_enabled=false`；只有下节的双重解锁才会改为 true。
- `observation=(enabled=yes,built=N,rejected=0,max_abs=...,tts=...)` 表示
  111 维向量正在持续构造；`built` 应随 50 Hz 控制循环增加。

本地电脑启动动捕 gateway 和 planner：

```bash
./scripts/run_local_receive_planner.sh
```

MDU 每秒输出一行 `planner_input` 状态。看到 `base_pose` 持续更新，并在出现合法
来球时看到新的 `task/revision`，说明链路成立。日志始终包含
`body_drive_publishers=0`，表示这个阶段不会控制机器人。

不依赖动捕的跨机冒烟测试可在PC另一个终端运行：

```bash
./scripts/run_planner_input_smoke_sender.sh
```

它只发送30组测试用 pelvis pose 和 `RacketCommand`，不创建关节命令 publisher。
普通 RobotIO/observation/ONNX probe 应短暂进入 `policy_unavailable`；action
dry-run 应短暂进入 `dry_run`。测试消息停止后都会回到 `planner_not_ready`。

RobotIO probe 的 `result` 可以按下面判断：

| result | 含义 |
| --- | --- |
| `no_state` | 尚未收到同步后的 RobotState，先检查 iceoryx/hal_ethercat。 |
| `state_stale` | 状态超过 50 ms，或六路状态不完整/未对齐。 |
| `planner_not_ready` | RobotState 正常，但 Planner command/base pose 尚未齐备或已超时。 |
| `observation_rejected` | 111 维输入存在非有限值、四元数无效或维度/挥拍侧不符合合约。 |
| `policy_unavailable` | RobotIO 和 Planner 输入都正常；当前阶段没有接入模型，这是预期终点。 |
| `dry_run` | ONNX、动作适配及完整命令结构/增益校验通过；命令没有发送。 |
| `policy_rejected` | action dry-run 没有生成可用命令，应检查 ONNX/action 拒绝计数。 |
| `command_invalid` | 31 维命令包含尺寸或有限值错误。 |

## 分步调试顺序

1. 普通 receiver：确认 `/racket/command` 和 pelvis pose 可到达 MDU。
2. RobotIO 只读 probe：确认六路 body-drive 状态、31 DOF 顺序和 100 Hz 同步。
3. 111 维 observation builder：已接入；golden test 与 Python `float32` 输出逐项比对。
4. ONNX actor dry-run：已通过 MDU 实测，50 Hz 推理稳定且 `rejected=0`。
5. action dry-run：检查 raw clip、head 清零、机械限位、完整增益命令和 watchdog。
6. 在机器人悬挂/可靠支撑、急停可用并停止厂商控制器后，才进入真实模式。

## 手动状态机与真实 RobotIO 接口

先用 shadow 模式统一验证按键流程，不创建 body-drive publisher：

```bash
A3_ENABLE_ACTION_DRY_RUN=1 A3_ENABLE_MANUAL_CONTROL=1 \
  ./scripts/run_mdu_planner_receiver.sh
```

- `P`：零增益 passive；
- `S`：从当前实测 q 用厂商生产 PD_STAND 增益插值到 model_21500 默认姿态，
  150 tick / 3 秒后 `pd_stand_ready=yes`；重复按 S 会从当前姿态重启插值；
- `M`：只有 PD_STAND ready 后才能进入策略，且 pelvis pose 必须新鲜；
- `X`：进入 halt，`Q` 仅允许在 passive 退出。

真实模式直接走官方 `RobotIOBackend::SendCommand()`。脚本同时要求两项解锁，
并检查 `agibot_pm` 已停止、`aimrt_main_hal` 与 RouDi 正在运行：

```bash
A3_ENABLE_COMMAND_PUBLISH=1 \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_ROBOT_SAFETY_READY=1 \
  ./scripts/run_mdu_planner_receiver.sh
```

这条命令会实际驱动机器人，必须先悬挂或可靠支撑机器人并确认实体急停。启动后
仍处于 `P`，需要人工依次按 `S`、等待 ready，再按 `M`。状态或 pelvis pose
过期、observation/policy/command 拒绝时，控制器通过同一官方接口发送零增益
safe-halt。
