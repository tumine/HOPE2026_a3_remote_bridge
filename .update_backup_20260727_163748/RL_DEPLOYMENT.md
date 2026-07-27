# A3 无动捕强化学习部署

这套流程在电脑上执行 `balance_v2/model_7000` ONNX 策略，电脑通过有线 DDS
直接和 MDU 通信，不经过 HDU ROS 话题中继：

```text
MDU /body_drive 状态 (iceoryx)
  -> a3_mdu_state_bridge
  -> /a3_internal/joint_states + /a3_internal/pelvis_imu (DDS)
  -> PC 111-D observation -> ONNX -> raw_action[31]
  -> tanh + per-joint scale + clamp
  -> /a3_internal/joint_command_control (DDS)
  -> MDU guarded gateway
  -> /body_drive/*_joint_command (iceoryx)
  -> hal_ethercat
```

HDU 只负责 `192.168.123.0/24 -> 10.42.10.0/24` 的 Linux 路由。电脑的有线地址
为 `192.168.123.123`，MDU 为 `10.42.10.12`。

## 固定策略合同

程序只接受外层部署包中的 `balance_v2/model_7000`，不接受训练仓库内另一个
`legacy model_29000`。启动时会检查：

- ONNX、manifest、配置和训练快照 SHA256；
- `observation[111] -> raw_action[31]`、50 Hz、无 observation normalization；
- joint order、默认位姿、`tanh`、逐关节 action scale 和机械角度 clamp；
- checkpoint `params/env.yaml`、`training_actuator_gains.yaml` 与 MuJoCo runtime 的
  31 关节 Kp/Kd 完全相等；
- 当前训练源码的 target box 和两份动作 sidecar。

无动捕节律严格采用训练合同：

```text
冻结准备：50 tick，time_to_strike 固定 1.0 s
完整动作：91 tick，strike frame=50
TTS：+1.0 s -> 0.0 s -> -0.8 s
总计：141 tick = 2.82 s
```

训练和 sim2sim 的 Kp/Kd 会逐关节写入真机命令。范围为 Kp `20..250`、Kd
`2..8`。这满足本项目的对齐要求，但这些数值仍是训练动力学参数，不等于厂商对实机
安全性的认证。

## 0. 一次性准备和离线自检

在电脑运行：

```bash
cd ~/workspace/a3_remote_bridge
./scripts/setup_rl_venv.sh
./scripts/run_rl_preflight.sh
```

必须看到：

```text
"status": "ok"
"policy_generation": "balance_v2"
"checkpoint_iteration": 7000
"training_equals_checkpoint_snapshot": true
"training_equals_sim2sim": true
"total_ticks": 141
"sequence": "WAIT->F->WAIT->B->WAIT"
"total_duration_s": 25.64
```

两个自检都不加载ROS、不创建command publisher，也不会控制机器人。第二个自检使用
实际 `balance_v2/model_7000`、训练/sim2sim Kp/Kd和相同MJCF做无界面动态仿真；
这两项门禁已经对当前模型和配置验证通过，不放进每次真机启动路径。只有 ONNX、训练
源码、runtime、Kp/Kd、观测或动作节律发生改变后才重新执行；运行程序本身仍会加载并
校验固定策略合同。

推荐的完整启动器会自动复制两个 MDU 管理脚本，已有的 aarch64 bridge 二进制无需
重新编译。需要手工复制时执行：

```bash
cd ~/workspace/a3_remote_bridge
scp scripts/run_mdu_rl_control.sh scripts/mdu_rl_stack.sh \
  agi@10.42.10.12:/agibot/a3_remote_bridge/scripts/
ssh agi@10.42.10.12 \
  'chmod +x /agibot/a3_remote_bridge/scripts/run_mdu_rl_control.sh /agibot/a3_remote_bridge/scripts/mdu_rl_stack.sh'
```

## 1. 先做只读/不下发推理

保持 MDU 的只读状态桥运行。在电脑执行：

```bash
cd ~/workspace/a3_remote_bridge
./scripts/start_rl_policy.sh --dry-run
```

程序启动在 `OBSERVE`，且不创建真实 command publisher。默认是无需回车的单键控制：

```text
P = 计算默认位姿渐入/保持（dry-run 不会向机器人发送）
D = 仅在 pose_ready=yes 后启动无球 READY 策略
F = 注入一颗正手球的观测
B = 注入一颗反手球的观测
I = 状态
Q = 仅在 OBSERVE 退出
```

此模式会构造实时 111 维观测并推理，但所有命令只在电脑内计算，不发送给机器人。

## 2. 推荐：一条命令完成 MDU 切换和电脑策略启动

以下命令会修改 MDU 运行状态，只能在机器人已吊装/可靠支撑、急停在操作员手中、
周围无人且两个电源均已接通后执行：

```bash
cd ~/workspace/a3_remote_bridge
./scripts/start_rl_policy.sh --robot-suspended
```

脚本建立一条复用 SSH 连接，SSH 密码只输入一次且不会保存；如果 `agibot_pm` 仍在
运行，MDU 的 `sudo` 可能再询问一次密码。它依次完成：

1. 复制最新 MDU 管理脚本；
2. 停止 `agibot_pm`；
3. 停止旧的只读 bridge，并拒绝任何残留 `motion_control`；
4. 复用或用 MDU `tmux` 启动 RouDi、HAL；
5. 启动唯一的真实控制 bridge，并检查 `--command-enable`；
6. 在电脑运行策略状态机。

电脑节点始终先停在 `OBSERVE`，不会自动移动、进入策略或注入球。确认吊装/支撑和现场
安全后先按 `P`，程序用5秒 smoothstep 从实测姿态渐入训练 `default_q`，随后持续保持。
最大位置误差不超过0.15 rad、最大关节速度不超过0.25 rad/s连续1秒后，日志才会显示
`pose_ready=yes`。此时按 `D`，再连续运行3秒后 `ready=yes`。单键如下：

```text
P = 5秒平滑进入并保持训练默认位姿
D = 仅在 pose_ready=yes 后进入无球 READY 策略
F = 注入一颗正手球观测；完成后返回无球等待
B = 注入一颗反手球观测；完成后返回无球等待
I = 立即打印状态
S = 用 READY 策略保持 2 秒，零增益并退出
X = 立即零增益并退出
H = 打印按键帮助
```

程序不再每秒自动打印状态，终端只显示状态切换、任务完成和故障等关键报告；需要查看
`default_err/pose_ready/ready/ACK` 等完整快照时按一次 `I`。

电脑程序退出后，完整启动器会自动关闭 MDU 的真实 bridge；HAL/RouDi 保持运行，
`agibot_pm` 保持停止，避免自动恢复原厂力矩。恢复原厂系统仍需按“停止”章节人工检查。

## 3. 手工方式：MDU 切换到唯一真实控制网关

以下步骤会让机器人失去原厂控制器的站立力矩。机器人必须可靠吊装/支撑，急停在操作员
手中，周围无人，并确认两个电源均已接通。

先开两个直连 MDU 的终端：

```bash
ssh agi@10.42.10.12
```

在 MDU 检查并停止原厂进程：

```bash
sudo systemctl stop agibot_pm

pgrep -af 'start_motion_control|./motion_control --cfg_file_path|a3_mdu_state_bridge'
```

如果只读 bridge 还在其原终端中，优先在原终端按 `Ctrl-C`。然后在第一个 MDU 终端只启动
HAL；该终端保持运行：

```bash
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_ethercat/start_hal_ethercat.sh
```

在第二个 MDU 终端确认只有 HAL、RouDi，没有原厂 motion_control 或旧 bridge：

```bash
pgrep -af 'iox-roudi|aimrt_main_hal|start_motion_control|./motion_control|a3_mdu_state_bridge'
```

如果这里没有 `iox-roudi`，在另一个 MDU 终端启动并保持运行：

```bash
/usr/local/bin/roudi/iox-roudi \
  --config-file=/usr/local/bin/roudi/cfg/roudi_config.toml -m off
```

然后启动唯一真实网关：

```bash
cd /agibot/a3_remote_bridge
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
  ./scripts/run_mdu_rl_control.sh
```

正确日志应包含 `REAL CONTROL ARMED`、`kp<=250`、`kd<=8`。网关尚未收到第一条电脑命令
之前不会启动 100 ms command watchdog。

## 4. 电脑启动真实控制

真实模式固定使用 `policy_blend=1.0`，与训练/sim2sim动作映射一致：

```bash
cd ~/workspace/a3_remote_bridge
./scripts/run_pc_rl.sh \
  --enable-actuation \
  --i-understand-this-can-move-the-robot \
  --policy-blend 1.0 \
  --allow-full-policy
```

程序停在 `OBSERVE`，只接收状态。必须按以下两步操作。

### 进入训练默认位姿

按一次：

```text
P
```

程序经过5秒平滑插值进入 `default_q`，然后保持。默认位姿阶段将训练Kp/Kd放大2倍，
并分别限制在MDU既有上限Kp 250、Kd 8以内；按下
`D` 的第一帧起恢复训练/sim2sim原始逐关节Kp/Kd。这只是关节位姿控制，不是平衡策略；
从 `P` 到按下 `D` 期间必须保持可靠吊装/支撑。
等待出现：

```text
mode=DEFAULT_HOLD ... pose_ready=yes ready=no
```

### 进入无球策略等待

确认 `pose_ready=yes` 后按一次：

```text
D
```

如果尚未按 `P` 或位姿尚未稳定，`D` 会被拒绝。接受 `D` 后程序从第一帧开始构造
sim2sim 的无球 READY 观测并执行
ONNX，按训练/sim2sim 的动作映射直接生成目标关节位置。无球观测为相对基座
`[0.40, +0.20, -0.05]`、目标速度 `[0,0,0]`、`TTS=1.0`、站位误差 `[0,0]`。
连续安全运行 3 秒后出现：

```text
mode=POLICY_WAIT ... ready=yes
```

`ready=no` 时 `F/B` 会被拒绝。`POLICY_WAIT` 中 ONNX 始终以 50 Hz 运行，不是静态
关节保持。

### 注入一颗球的观测

正手：

```text
F
```

反手：

```text
B
```

每次按键只注入一个严格递增的任务语义：50 tick 训练中位数预备段，加完整91帧动作，
共2.82秒。完成后的下一个50 Hz周期直接恢复 `POLICY_WAIT`；不回静态默认位姿，不清空
`last_action`，关节状态、速度和策略历史连续。再次看到 `ready=yes` 后可以按下一颗球。

真实模式会拒绝0.20等降额值：相同MuJoCo READY动态测试中，0.20在1.70秒超过
0.35 rad倾角阈值，而1.0通过了完整25.64秒 `WAIT -> F -> WAIT -> B -> WAIT`。
降额并不会让平衡策略更安全，只会削弱它的稳定修正。

## 5. 停止

正常停止，在电脑按一次：

```text
S
```

已经进入策略时，程序先用无球 READY 策略保持2秒，再连续发送15帧当前位置零增益命令
并退出。仍处于 `DEFAULT_RAMP/DEFAULT_HOLD` 时，`S` 不会进入策略，而是直接发送15帧
当前位置零增益命令并退出。随后MDU watchdog会锁存，必须重启网关才能再次控制。

需要跳过 READY 保持并立即撤销软件增益时输入：

```text
X
```

`Ctrl-C` 也会尝试发送 5 帧零增益，但优先使用 `S` 或 `X`。机器人姿态失控、发生
碰撞或软件停止无效时，直接使用物理急停，不要等待终端命令。

使用 `start_rl_policy.sh` 时，电脑程序退出后会自动停止 MDU 真实 bridge。使用手工方式
时，电脑程序退出后：

1. 在 MDU 网关终端按 `Ctrl-C`；
2. 在 HAL 终端按 `Ctrl-C`；
3. 确认手工 HAL、bridge、motion_control 都已退出；
4. 需要恢复原厂系统时运行 `sudo systemctl start agibot_pm`。

不要在真实网关、手工 HAL 尚未退出时启动 `agibot_pm`。

## 自动保护

电脑端会在以下情况进入零增益 halt：状态超过 80 ms、基座倾角超过 0.35 rad、ACK
积压超过 10 帧、ONNX 单次推理超过 15 ms、raw action 超过 20，或输入/输出出现
NaN/Inf。电脑端只保留训练合同中的机械关节角度 clamp，不再增加会削弱平衡策略的
每周期步长或跟踪误差削顶。

MDU 再次检查 31 关节名称/顺序、sequence、单一 session、状态新鲜度、相对实测位置
不超过 1.50 rad（当前完整动态门禁实测峰值约 1.408 rad）、会话
行程、Kp/Kd、速度和力矩；命令中断超过 100 ms 后锁存零增益。

## 无动捕限制

当前没有基座世界平移和实时 Planner，因此 `[101:103]` 固定为零，目标位置、目标速度
使用训练盒中点。关节位置/速度、pelvis IMU、上一帧 action 仍全部使用真机实时数据。
MuJoCo初始时机器人面向世界 `+X`。真机没有table/world定位时，IMU yaw零点是任意的，
所以按 `D` 时把当前朝向标定为MuJoCo `+X`；roll/pitch和
`projected_gravity[96:99]`保持原始值，`base_forward_xy[99:101]`则使用这个启动对齐坐标系。

这种模式不能感知来球、不能修正机器人在地面的 XY 漂移，也不能用于自主接球。它只用于
吊装/系绳、无球、有人值守的策略站立和单次来球观测链路测试。
