# A3 强化学习真机部署快速手册

本文保留的是 `model_21500` 电脑端推理流程。`model_48000` 真机部署不要执行本文
的 `start_rl_policy.sh`，请使用 [PLANNER_TO_MDU.md](PLANNER_TO_MDU.md) 中的
本机交叉编译、传输和 MDU 推理流程。
部署架构为电脑通过有线网络直达 MDU；HDU 只负责 IP 转发，不运行 ROS 2
话题中继。

更完整的观测合同、坐标系和安全限制见
[`RL_DEPLOYMENT.md`](RL_DEPLOYMENT.md)。

## 1. 数据链路

```text
FZMotion
  -> PPMocap /ppmocap/frame                 (电脑，ROS Domain 232)
  -> canonical table frame
  -> /a3_mocap/ball_poses
  -> HOPE planner
  -> /racket/command

MDU iceoryx /body_drive/* state
  -> a3_mdu_state_bridge
  -> /a3_internal/{joint_states,pelvis_imu} (电脑直接接收)
  -> 111-D observation
  -> model_21500 ONNX，50 Hz
  -> /a3_internal/joint_command_control
  -> MDU guarded bridge
  -> iceoryx /body_drive/*_joint_command
  -> HAL EtherCAT
```

没有合格真实来球时，策略使用仿真第一帧对应的固定世界虚拟目标保持 READY。
真实球只有在预测击球点、球拍速度和到达时间均落入训练范围时，planner 才发布
一次新的击球任务。正手/反手由 planner 自动选择，不需要人工按键触发。

## 2. 安全要求

首次运行和调试时必须满足：

- 机器人吊装或可靠支撑；
- 急停在操作员手中；
- 机器人周围、机械臂和腿部运动范围内无人；
- 网线、电源线和吊装带不会进入关节运动范围；
- 只有一个控制 bridge，且原厂 `motion_control` 已停止；
- 未看到所有门禁通过前，不按 `D`。

`--robot-suspended` 是操作员对物理安全条件的确认，不是用于绕过检查的普通参数。

## 3. 当前网络参数

```text
PC 有线地址：  192.168.123.123
HDU 网关：     192.168.123.1
MDU：          10.42.10.12
ROS_DOMAIN_ID：232
```

启动前检查：

```bash
ip route get 10.42.10.12
ping -c 3 10.42.10.12
```

正确路由应包含：

```text
via 192.168.123.1 dev enp129s0 src 192.168.123.123
```

电脑端 DDS 配置位于：

```text
config/fastrtps_pc_direct_mdu.xml
```

该配置同时包含本机 loopback/SHM、本机有线地址和 MDU 静态 peer。不要删除
`127.0.0.1`，否则 PPMocap、planner 和 runner 可能能被发现却无法交换数据。

## 4. 一次性准备

首次部署，或训练仓库、模型、planner 更新后执行：

```bash
cd /home/bth/workspace/a3_remote_bridge

./scripts/setup_rl_venv.sh
./scripts/setup_hope_planner.sh
./scripts/run_rl_preflight.sh
```

`run_rl_preflight.sh` 包含模型合同检查和一次 MuJoCo READY 稳定性验证。模型和
计算方式没有变化时，不需要每次真机启动都重复运行。

## 5. 启动 PPMocap

PPMocap 必须在 ROS Domain 232 启动。如果它此前运行在默认 Domain 0，先正常停止：

```bash
cd /home/bth/workspace/Mocap
./scripts/stop_ppmocap.sh
```

随后启动：

```bash
cd /home/bth/workspace/Mocap

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

./scripts/start_ppmocap.sh
```

检查接收状态：

```bash
timeout 5 ros2 topic echo /ppmocap/status --once
```

必须确认：

```text
receiving: true
socket_bound: true
table_tracked: true
robot_tracked: true
```

没有球时 `ball_visible: false` 是正常 READY 状态。正常频率约为 120 Hz。

## 6. 可选：只读完整链路检查

第一次使用新配置时，可以运行不会创建关节 command publisher 的 dry-run。该模式不会
通过 SSH 启动 MDU，因此要求 MDU 上已经有只读 `a3_mdu_state_bridge` 在发布状态：

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/start_rl_policy.sh --dry-run
```

按 `I` 查看状态。dry-run 不会让机器人进入默认位姿，也不会驱动关节。如果
`state_age=inf`，先启动 MDU 只读状态桥，或直接按后文真实流程由总脚本管理 MDU。
检查结束后，在 `OBSERVE` 按 `Q` 退出。

## 7. 启动真实控制

保持 PPMocap 运行，在新的电脑终端执行：

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/start_rl_policy.sh --robot-suspended
```

需要同时查看与真实位置、关节动作对齐的机器人模型和地面时，增加 `--rviz`：

```bash
./scripts/start_rl_policy.sh --robot-suspended --rviz
```

输入 MDU SSH 密码。脚本会自动：

1. 启动电脑端 HOPE planner；
2. 连接 MDU；
3. 停止 `agibot_pm` 和冲突的原厂 `motion_control`；
4. 启动或复用 RouDi 和 HAL EtherCAT；
5. 启动唯一的真实控制 bridge；
6. 启动电脑端 ONNX runner；
7. 停留在 `OBSERVE`，等待人工按键。

成功日志应包含：

```text
REAL CONTROL ARMED
kp<=250
kd<=8
MDU armed. PC remains in OBSERVE until P, then D.
model_21500 runner in OBSERVE
PPMOCAP CALIBRATED
```

启动不会自动进入位姿或策略。只有按 `P` 后才开始发送有增益的位置指令。

### 7.1 单独启动 RViz（只读）

如果控制程序已经运行，也可以在另一个电脑终端单独启动可视化：

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/run_a3_rviz.sh
```

该脚本不发布关节命令，只执行以下可视化数据融合：

```text
/ppmocap/frame
  -> 与策略相同的球桌标定和 BotA3 marker-to-pelvis 外参
  -> world -> hope_table -> a3/pelvis_link

/a3_internal/joint_states
  -> robot_state_publisher
  -> a3/pelvis_link -> A3 全身和右手球拍各 link

hope_table -> hope_ground (z=-0.76 m)
  -> RViz Physical ground 网格

/ppmocap/markers + /ppmocap/ball/path + /ppmocap/prediction/path
  -> 原 PPMocap 球桌、球、marker、实测轨迹和预测轨迹场景
```

RViz 的 Fixed Frame 与原 PPMocap 场景一致，使用 `world`。其中 `hope_table` 的
`z=0` 是桌面而不是地面；真实/仿真地面位于 `hope_table z=-0.76 m`。机器人所有
TF 使用 `a3/` 前缀，避免与 PPMocap 的 `bot_a3` marker 和机器人原厂 TF 重名。

第一次运行或更换模型后可以先做完全离线检查：

```bash
./scripts/run_a3_rviz.sh --check
```

若只需要发布机器人 TF 而不打开图形界面：

```bash
./scripts/run_a3_rviz.sh --no-rviz
```

关闭 RViz 窗口或在该终端按 `Ctrl+C`，脚本会同时停止其创建的 pose bridge 和
`robot_state_publisher`；不会影响控制 runner、planner 或 MDU bridge。

## 8. 控制按键

按键为单键操作，不需要按 Enter。

| 按键 | 作用 | 允许状态 |
|---|---|---|
| `I` | 输出一次完整状态报告 | 任意运行状态 |
| `P` | 用 5 秒平滑进入训练默认位姿并保持 | 仅 `OBSERVE` |
| `D` | 通过全部门禁后进入策略 | 仅 `DEFAULT_HOLD` 且 `pose_ready=yes` |
| `C` | 清空并重新锁定桌面标定 | 仅 `OBSERVE` |
| `S` | 正常停止：屏蔽新球、READY 2 秒、零增益并退出 | 默认位姿或策略状态 |
| `X` | 立即进入软件 HALT、发送零增益并退出 | 除已退出状态外 |
| `Q` | 不进入控制，直接退出 | 仅 `OBSERVE` |
| `H` | 打印按键帮助 | 任意运行状态 |

没有 `F/B` 人工正反手按键。真实球进入训练区域后，planner 自动选择正手或反手。

## 9. 正确操作顺序

### 9.1 OBSERVE 检查

启动后先按：

```text
I
```

理想状态应接近：

```text
mode=OBSERVE
state_age < 80 ms
mocap_age < 80 ms
base ~= [-0.500, -0.7625, 0.3064]
forward ~= [1.000, 0.000]
gate=ready
invalid=0/0/0
```

如果 `state_age=inf`，说明没有收到 MDU 状态；如果 `mocap_age=inf` 或桌面标定一直
是 `0/20`，说明没有收到 PPMocap。任何一种情况都不能继续。

### 9.2 进入默认位姿

确认机器人受到可靠支撑后按：

```text
P
```

等待 5 秒平滑过渡和姿态稳定，直到看到：

```text
DEFAULT POSE READY: pose_ready=yes; mocap/planner_ready=yes
```

然后再次按 `I`，确认：

```text
mode=DEFAULT_HOLD
pose_ready=yes
gate=ready
```

### 9.3 进入策略

只有以上条件全部满足后按：

```text
D
```

策略先执行 3 秒 READY warmup，此时不会接受真实球任务。随后应看到：

```text
POLICY_ACTIVE: validated READY is running; live /racket/command enabled
```

进入 `POLICY_ACTIVE` 后：

- 无合格来球：持续使用虚拟目标保持 READY；
- 合格来球：日志出现 `LIVE BALL ENGAGED`，自动挥拍；
- 挥拍完成：日志出现 `BALL COMPLETE`，自动返回 READY；
- 同一个 planner `task_id` 只执行一次。

每次实测球从对手侧向机器人方向穿过反手 `x=0.10 m` 或正手
`x=-0.21 m` 平面时，控制终端还会输出一次事件诊断：

```text
STRIKE_PLANE_CROSS ball=... plane=BACKHAND ... result=REJECT: ...
STRIKE_PLANE_CROSS ball=... plane=FOREHAND ... result=ACCEPT: task_id=...
```

`measured_y/measured_z` 是相邻两帧动捕在该平面的插值位置。`result` 会说明
位置越界、正反手分类、TTS 太早/太晚、球拍速度越界，或者已经接受的
`task_id/side/TTS`。该日志只做诊断，不参与控制判定；完整 planner 日志仍位于
`log/hope_planner.log`。

完整状态机：

```text
OBSERVE
  --P--> DEFAULT_RAMP
  -----> DEFAULT_HOLD / pose_ready=yes
  --D--> POLICY_WARMUP
  -----> POLICY_ACTIVE
             | 合格真实球：READY -> SWING -> FOLLOW-THROUGH -> READY
             | 无合格球：固定世界虚拟目标 READY
  --S--> STOP_READY -> ZERO_GAIN -> EXIT
```

## 10. 正常停止和紧急停止

### 正常停止

实验结束按：

```text
S
```

等待：

```text
STOP: planner commands blocked
zero-gain stop complete
RL control bridge stopped
```

不要把直接关闭终端或断开网络当成正常停止方式。

### 紧急情况

程序仍响应时按：

```text
X
```

人进入工作区、机器人失控、线缆缠绕或姿态危险时，直接使用物理急停，不等待软件
处理。`Ctrl+C` 也会尝试发送紧急零增益，但优先级低于物理急停。

### OBSERVE 中退出

如果尚未按 `P`，可以按：

```text
Q
```

这会退出 runner 并停止 RL bridge，不会进入默认位姿。

## 11. 停止后的 MDU 状态

`start_rl_policy.sh` 退出时会停止真实控制 bridge 和电脑端 planner，但为了避免自动
切回另一个控制器，默认不会自动重启 `agibot_pm`。HAL/RouDi 可能继续运行，便于下一次
实验直接复用。

如果需要恢复原厂控制栈，机器人仍须可靠支撑，然后执行：

```bash
ssh agi@10.42.10.12

/agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh shutdown-owned
sudo systemctl start agibot_pm
systemctl is-active agibot_pm
```

启动 `agibot_pm` 可能重新启用原厂控制器。确认 RL bridge 已停止且现场安全后才能执行。

停止 PPMocap：

```bash
cd /home/bth/workspace/Mocap
./scripts/stop_ppmocap.sh
```

## 12. 常见问题

### `sequence size exceeds remaining buffer`

当前 Humble 电脑与 Jazzy MDU 发现时可能偶发出现一两行该提示。不能仅根据这行判断
链路失败，应按 `I` 检查 `state_age`、`mocap_age`，并确认 planner 端点。

如果该提示持续刷屏，同时 `mocap_age=inf` 或标定为 `0/20`，检查是否误删了
`fastrtps_pc_direct_mdu.xml` 中的 loopback、SHM 或本机 initial peer。

### `D rejected: pose_ready=no`

等待默认位姿稳定，直到出现 `pose_ready=yes`。不要通过继续放宽阈值绕过明显的姿态或
速度问题。

### `D rejected: PPMocap/planner gate`

根据日志修正真实原因：

- `station mismatch`：机器人相对球桌站位或 marker-to-pelvis 平移不正确；
- `base height outside`：BotA3 marker 原点不是 pelvis，需要标定平移；
- `heading error`：BotA3 marker 到 pelvis 的旋转外参不正确；
- `pose stale/no tracked BotA3`：动捕跟踪丢失；
- planner publisher/subscriber absent：planner 或 DDS 本机发现异常。

外参配置位于：

```text
config/a3_rl_deploy.yaml
  mocap.robot_marker_to_pelvis.translation_m
  mocap.robot_marker_to_pelvis.rpy_rad
```

### 当前现场标定状态

最近一次只读实测为：

```text
base=[-0.479, -0.865, 0.163]
forward=[0.924, -0.381]
station mismatch=0.105 m
```

训练参考约为：

```text
base=[-0.500, -0.7625, 0.3064]
forward=[1.000, 0.000]
```

因此当前启动链已经打通，但 BotA3 刚体到 pelvis 的外参仍需现场标定。在修正并看到
`gate=ready` 前，禁止按 `D`。
