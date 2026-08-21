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
    -> model_48000 ONNX -> 31-D RobotCommand
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

这个开关会自动启用 RobotIO 只读探针。进入 `M` 时把实时 pelvis XY 锁定为
`nominal_station_xy`；READY/RECOVERY 以此为 base target，SWING/FOLLOW_THROUGH
根据 active task 击球点和正反手 reach 每 tick 更新 moving base target。它以 50 Hz 构造 observation，但不加载模型、不执行推理，且
`last_action` 固定为零。启动时还会逐项校验 RobotIO 的 31 DOF 顺序；顺序与
`model_50000` 合约不一致会直接拒绝启动。

## model_50000 Planner 击球盒

真实部署使用 `model_50000_mujoco_sim2sim_bundle/config/strike_box.yaml` 的
`checkpoint_profile`，不使用同文件里的继续训练范围。Planner 使用 table frame，
冻结范围如下：

| 挥拍侧 | x (m) | y (m) | z (m) | 球拍速度 vx/vy/vz (m/s) |
| --- | --- | --- | --- | --- |
| 正手 | `[-0.32, -0.10]` | `[-1.5250, -1.1725]` | `[0.08, 0.45]` | `[1.75,3.40] / [0.20,1.10] / [0.35,1.60]` |
| 反手 | `[-0.05, 0.25]` | `[-1.1725, 0.0000]` | `[0.08, 0.45]` | `[1.05,3.10] / [-1.10,0.65] / [0.35,1.50]` |

站位横向目标范围为 `[-0.35, 0.6625] m`，checkpoint 的 floor/world 分侧线
是 `y=-0.41 m`，滞回半宽为 `0`。转换到当前 planner table frame 后，
分侧线是 `y=-1.1725 m`；两者不是同一个坐标原点，不能直接互换。MDU
在进入 `M` 时锁定启动中心站位。active task 期间使用
`desired_offset_y = target_y - nominal_y - reach_y` 反算 base target，并限制在
`[-0.35, 0.6625] m`；`obs[101:103] = base_target_xy - live_base_xy`。

真实硬件姿态源有一项明确的安全覆盖：`obs[96:99]` projected gravity 使用
MDU pelvis IMU 四元数，`obs[99:101]` world/table heading 仍使用标定后的
PPMocap pelvis 四元数。这样倾斜观测不依赖光学姿态，IMU yaw 漂移也不会
旋转 planner 的球台坐标。PPMocap 位姿超时门控仍然生效，丢失后不会继续
构造新的策略观测。
READY/RECOVERY 回启动站位，x 始终不变，不额外生成 base 速度命令。
`run_hope_planner.sh` 默认位置和速度 margin 均为 `0`，避免把未训练区域放进真机。
新任务 TTS 默认限制在 `[0.25, 1.00] s`；上限仍与 checkpoint 的
`lead_time_s=1.0` 对齐，下限恢复为现场调试时使用的窗口。可通过
`A3_NEW_TASK_TTS_MIN_S` 显式覆盖。

model_72500 策略模式的腰部增益与其冻结合同对齐：yaw/roll/pitch 的
`Kp=[85,500,500]`、`Kd=[3,2,2]`。roll/pitch 的 applied action 和
`q_des` 均强制为 0；yaw 仍由策略控制。这里的 500 是策略训练/MuJoCo nominal
值，不是厂商认证值，首次实机运行必须在支撑状态下检查跟踪误差、反馈力矩和振荡。
PD_STAND 仍使用独立的生产增益。为避免干扰 checkpoint 对 roll/pitch 的冻结合同，
独立的腰 pitch 正向虚拟墙默认关闭；如需旧版诊断行为，可显式设置
`A3_WAIST_PITCH_GUARD_ENABLED=1`。机械位置 clamp、腿部限位保护和通信 watchdog
不受此项修改影响。

## ONNX 只读推理探针

当前仅将 actor checkpoint 切换为 `model_48000_mujoco_deploy`，观测、状态机、
moving station、击球盒、实机增益和安全保护仍保持现有配置：

```text
models/hope_pingpong.onnx
SHA256 6e68f5ce582ac41856c23ac63c749b94c4ca7b96dbe041bdc0215332fbb6caba
observation float32[1,111] -> raw_action float32[1,31]
```

MDU 上启动：

```bash
cd /home/agi/a3_remote_bridge_probe
A3_ENABLE_ONNX_PROBE=1 ./scripts/run_mdu_planner_receiver.sh
```

这个开关自动启用 RobotIO 和 observation probe，使用随包分发的 AArch64
ONNX Runtime CPU 库。模型加载时强制校验 input/output 名称、形状、类型、
移动站位语义和 31 DOF 元数据；checkpoint 身份和哈希随包携带在
`models/model_48000/{policy_manifest,provenance}.json` 中供现场审计。推理输出只用于统计，并按合约
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
`model_50000` 合约：raw action clip 到 `[-100, 100]`、head yaw/pitch 清零、
统一乘 `0.25` 后加默认关节角，并按 31 个机械限位裁剪。它构造的
`RobotCommand` 与真实发送包相同：`dq_des/tau_ff` 为零，策略态 Kp/Kd 来自
6/27 的冻结训练/部署配置；head 使用厂商 `ExpandToBackend()` 的 40/2 保持值。
backend 仍以 `publish_enabled=false` 启动，因此只返回 `dry_run`，不会调用
`SendCommand()`。

日志应包含：

```text
robot_io=(ticks=N,result=dry_run)
action=(output=dry_run,decoded=N,rejected=0,commands=N,raw_clips=0,...,gains=model_50000)
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

### MuJoCo 对齐的 V→C→F 发球/接球闭环

发球直接修改 RobotIO 的同一个 31 维 `RobotCommand`，不存在第二个关节命令
publisher。轨迹和上肢增益来自
`Serve_A3_leg_model/config/a3_lower_body.yaml`；发球期间 model_72500 继续以
READY 观测控制腰部和双腿，头部保持，双臂由发球轨迹覆盖。

无执行器 dry-run：

```bash
A3_ENABLE_SERVE_VCF=1 A3_ENABLE_ACTION_DRY_RUN=1 \
  ./scripts/run_mdu_planner_receiver.sh
```

键盘顺序：

1. `S`：进入 PD_STAND，等待 `pd_stand_ready=yes`，再按 `M` 进入接球策略；
2. `V`：若有击球任务则等待 lifecycle 回 READY，再用 5 秒进入发球 Home；
3. READY 后按 `C` 关闭夹爪，真实模式须等待日志中的 `gripper=closed`；
4. READY 稳定至少 1 秒后按 `F`，执行 `windup -> swing -> settle -> return`；
5. 释放点异步打开夹爪，播放结束后用 0.6 秒把上肢融合回接球策略；
6. `G` 可手动打开夹爪；等待/归位/READY 时按 `M` 可取消发球。

31 维合成规则：

```text
[0..2]    腰：model_72500 READY 推理
[3..4]    颈：发球期间保持 V 时实测 q
[5..11]   左臂：上肢发球轨迹
[12..18]  右臂：上肢发球轨迹
[19..30]  腿：model_72500 READY 推理
```

日志应出现：

```text
manual=(enabled=yes,mode=motion,...)
serve=(enabled=yes,pending=no,phase=ready,gripper=closed,...,
       lower=model_72500_ready,gains=Serve_A3_leg_model)
observation=(...,tts=1.000) lifecycle=(phase=ready,active=no,...)
```

真实发布还要求 `A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER`。夹爪通过
MDU 本机 `10.42.10.12:56422` 的 HalHandService 异步调用；C 未成功时 F 会拒绝。
发送帧同时包含左右 `agi_claw_cmd`，并显式设置
`cmd=0, vel=20, force=20, clamp_method=2, finger_pos=0`；左侧位置仍为
打开 `4096`、关闭 `1200`，右侧固定为 `0`。


先用 shadow 模式统一验证按键流程，不创建 body-drive publisher：

```bash
A3_ENABLE_ACTION_DRY_RUN=1 A3_ENABLE_MANUAL_CONTROL=1 \
  ./scripts/run_mdu_planner_receiver.sh
```

- `P`：零增益 passive；
- `S`：从当前实测 q 用厂商生产 PD_STAND 增益插值到 model_50000 默认姿态，
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

状态汇总默认每 5 秒输出一次，不改变 50 Hz 控制频率。现场需要进一步降低
终端输出时可设置 `A3_STATUS_PERIOD_S=10`。

这条命令会实际驱动机器人，必须先悬挂或可靠支撑机器人并确认实体急停。启动后
仍处于 `P`，需要人工依次按 `S`、等待 ready，再按 `M`。状态或 pelvis pose
过期、observation/policy/command 拒绝时，控制器通过同一官方接口发送零增益
safe-halt。

腿部位置保护：

- 当前部署默认只对 A3 31 维槽位中的左右 `hip_pitch`（槽位 19、25）启用独立软限位；使用对应 URDF 有符号端点的 `90%`。
- 其他 `hip_roll/hip_yaw/knee/ankle` 关节当前不参与该限位检查，越界不会触发此处的 `leg_limit_damping`。
- 每个控制周期同时检查受保护关节的实测 `q` 和即将发送的 `q_des`。任一受保护关节越界后，控制器锁存 `leg_limit_damping`，不再发送策略目标。
- 阻尼命令保持当前实测 `q`，全身 `Kp=0`，腿部 `Kd` 默认 `2.0`，`dq_des/tau_ff=0`；锁存状态需要重启进程后重新走 `P -> S -> M` 才能重新使能。
- 现场应使用厂商确认的关节工作范围和阻尼增益覆盖默认值，例如：

```bash
./scripts/run_mdu_planner_receiver.sh \
  --leg-soft-scale 0.90 \
  --leg-soft-limit left_hip_pitch_joint:-1.00:0.80 \
  --leg-soft-limit right_hip_pitch_joint:-1.00:0.80 \
  --leg-damping-kd 2.0
```

`--leg-soft-limit` 的上下限必须位于 A3 机械范围内；同名参数重复时以后者为准。

状态汇总中的 `leg_damping=(active=yes,joint=...,commands=...)` 表示已触发；真实部署前应先在 dry-run、悬挂和实体急停条件下验证该路径。
