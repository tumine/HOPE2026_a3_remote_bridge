# A3 Remote Bridge

独立的 A3 远程部署通信工程，不依赖原 `a3_deploy_onnx_ref` 的 ONNX、
RKNN、动作库或策略循环。

无动捕 `balance_v2/model_7000` 强化学习部署已经作为独立的电脑端状态机接入，完整的
准备、无球策略站立、单次正/反手来球观测和停止流程见 [RL_DEPLOYMENT.md](RL_DEPLOYMENT.md)。
该流程不会复用旧的 930/1570 维 observation 探针。

当前部署使用电脑直达 MDU 的单套 ROS 2 话题：

```text
MDU iceoryx -> a3_mdu_state_bridge <- routed DDS -> laptop
```

网线物理接入 HDU 的外部 USB 网卡。HDU 只做 Linux IP 转发，不运行 ROS 2
状态或命令中继。电脑通过静态路由访问 MDU `10.42.10.12`。

## 组件

- `a3_robot_io`：从原项目复制并去除 teleop/策略依赖的目标库。
- `a3_mdu_state_bridge`：默认只读的 MDU 网关。只有双重显式解锁时才允许
  注册真实 command publisher。
- `hdu_relay/a3_state_relay.py`：旧的 HDU 双网口中继，仅保留作回退，
  直连模式不运行。
- `pc_tools/a3_state_monitor.py`：电脑端状态频率和完整性检查。
- `pc_tools/a3_observation.py`：与参考部署一致的 29-DOF、10 帧
  proprioception observation 构造器。
- `pc_tools/a3_observation_probe.py`：电脑端只读 observation 探针；按
  header 时间戳匹配关节状态和 pelvis IMU，不创建 command publisher。
- `pc_tools/a3_rl_contract.py`：验证 `balance_v2/model_7000`、训练快照、
  sim2sim、111/31 维接口和无动捕动作节律。
- `pc_tools/a3_rl_deploy.py`：电脑端交互式 `OBSERVE -> DEFAULT_RAMP ->
  DEFAULT_HOLD -> POLICY_WARMUP -> POLICY_WAIT -> BALL -> POLICY_WAIT` 状态机。
  操作员先按 `P` 平滑进入并保持训练默认位姿，看到 `pose_ready=yes` 后按 `D` 才进入
  无球策略闭环；`F/B` 各注入一颗正/反手球的观测，
  完成后自动回到 `POLICY_WAIT`，没有自动发球或交替模式。
- `pc_tools/a3_rl_mujoco_check.py`：策略或配置变化后，用实际ONNX一次性动态验证
  `WAIT -> F -> WAIT -> B -> WAIT`；它不在每次真机启动时重复运行。
- `scripts/start_rl_policy.sh`：复制 MDU 管理脚本、切换唯一真实网关并启动电脑
  策略；机器人未可靠支撑时拒绝运行。MuJoCo 动态门禁是策略或配置变化后的一次性检查，
  不在每次真机启动时重复。
- `scripts/mdu_rl_stack.sh`：在 MDU 用独立 `tmux` 会话管理 RouDi、HAL 和真实
  bridge，清理旧只读 bridge 并拒绝残留的厂商控制器。

## 话题

电脑直接订阅 MDU：

```text
/a3_internal/joint_states
/a3_internal/pelvis_imu
/a3_internal/torso_imu
```

状态话题使用标准 `sensor_msgs`，QoS 为 reliable、depth 16。MDU内部按
55 Hz 调度发布，在当前受限 CPU 集合上电脑实测三路均不低于50 Hz。
dry-run命令和 ACK 使用 reliable、depth 64。

## 安全边界

MDU bridge 的安全边界：

- 默认不调用 `SendCommand()`，并以 `publish_enabled=false` 初始化后端；
- 仅发布完整、对齐且有限值的 31 关节状态和两路 IMU。
- 只有显式传入 `--command-dry-run` 才订阅隔离命令话题；该订阅只校验并
  返回 ACK，不驱动关节。

默认启动仍处于上述只读模式。真实模式另有双重解锁、限幅、会话锁和 watchdog；
强化学习策略只能按 [RL_DEPLOYMENT.md](RL_DEPLOYMENT.md) 的显式状态机进入。

## 电脑端命令构造检查（不发布）

先生成本机 `joint_msgs` Python type support：

```bash
source /opt/ros/humble/setup.bash
cmake --build build --target joint_msgs__rosidl_typesupport_c__pyext -j2
source scripts/setup_pc_joint_msgs.bash
```

从远程状态构造一帧完整的 31-DOF `joint_msgs/JointCommand`：

```bash
python3 pc_tools/a3_command_prepare.py --topic-prefix /a3_internal
```

该程序只创建消息对象并检查关节顺序。目标位置取当前实测位置，目标速度、
前馈力矩、stiffness 和 damping 全部为零，而且程序没有创建 publisher。

## 跨板命令干跑与 ACK

直连模式使用隔离话题传输标准 `joint_msgs/JointCommand`：

```text
PC  /a3_internal/joint_command_dry_run
 -> MDU direct
 -> validate only, no SendCommand()
 -> /a3_internal/joint_command_ack
```

MDU 状态桥必须显式增加 `--command-dry-run` 才注册干跑 subscriber。即使
启用该参数，backend 仍固定使用 `publish_enabled=false`，只接受当前位置附近
0.05 rad 内且 velocity、effort、stiffness、damping 全为零的完整31关节命令。

电脑端直连环境：

```bash
source /opt/ros/humble/setup.bash
source scripts/setup_pc_joint_msgs.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastrtps_pc_direct_mdu.xml"
unset FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
```

50 Hz dry-run验收：

```bash
python3 pc_tools/a3_command_dry_sender.py \
  --topic-prefix /a3_internal --hz 50 --count 1000 --warmup-sec 3
```

## 无动作指令链路 RTT 测试

在接入真实 `/body_drive/*_joint_command` 前，可用隔离的 `/a3_test/*`
话题验证电脑到 MDU 的双向通信。测试包默认包含 156 个 float64：一个序号
和 31 关节乘 5 个控制字段等量的占位数据。测试在 DDS 发现后预热 3 秒，
并使用 best-effort、depth 1，避免把已经过期的命令积压在队列中。MDU 只原样回显 ROS 2 消息，
不加载 iceoryx backend，不调用 `SendCommand()`。

链路为：

```text
PC /a3_test/pc_ping
 -> HDU /a3_test/mdu_ping
 -> MDU echo /a3_test/mdu_pong
 -> HDU /a3_test/pc_pong
 -> PC RTT statistics
```

## 电脑端 observation 探针

参考策略不是只使用关节状态。它从远程状态中使用：

- 31 DOF 中跳过 head/neck 后的 29 维关节位置和速度；
- pelvis IMU quaternion 计算的 projected gravity；
- pelvis IMU angular velocity；
- 上一帧 29 维 raw policy action（当前探针固定为零）；
- 上述 proprioception 的 10 帧历史。

`effort`、线加速度和 torso IMU 不进入参考策略 observation。MDU 状态桥
仍保留并转发它们，便于健康监控或后续自定义策略使用。

电脑端运行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

cd ~/workspace/a3_remote_bridge
python3 pc_tools/a3_observation_probe.py \
  --topic-prefix /a3_internal --duration-sec 20
```

探针以 50 Hz 构造 930 维 proprioception，并预留 640 维 tokenizer 为零，
因此同时检查完整的 1570 维 `obs_dict` 结构。全程不会执行模型推理，也不会
发布控制命令。零 tokenizer 只用于接口检查，不能用于真实策略控制。

当前直连实测结果：同步状态约 `51.9 Hz`，策略观测构造约 `50.0 Hz`，
`buffered=10`、`proprio=930`、`obs=1570`、`stale=0`、`invalid=0`。

## 真实控制网关（默认锁定）

真实链路为：

```text
PC /a3_internal/joint_command_control
 -> MDU guarded validator
 -> RobotIOBackend::SendCommand()
 -> iceoryx /body_drive/{waist,leg,arm,neck}_joint_command
 -> hal_ethercat
```

MDU 网关只有同时设置以下两项才会使用 `publish_enabled=true`：

```text
--command-enable
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION
```

保护条件包括：31 关节规范名称和顺序、有限值、严格递增 sequence、单一会话、
状态新鲜度、相对当前位置限幅、相对会话初始位置限幅、速度/力矩/Kp/Kd 上限。
默认所有速度、力矩和增益上限均为 0。命令断流 100 ms 后，MDU 会锁存为
零增益 safe-halt；此后即使通信恢复也拒绝控制，必须重启网关才能重新解锁。

机器人系统自带的 `motion_control` 也会向同一组 body-drive command 话题发布。
真实控制前必须按原项目 README 的上机流程停止 `agibot_pm`，再只启动
`hal_ethercat`。不能让两个控制器同时运行：

```bash
sudo systemctl stop agibot_pm
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_ethercat/start_hal_ethercat.sh
```

只杀掉 `motion_control` 子进程不够；如果 `start_motion_control.sh` 外层脚本仍在，
它会再次拉起原厂控制器。真实模式启动脚本会检查这两个进程以及已有的
`a3_mdu_state_bridge`，发现任一冲突就以退出码 73 拒绝启动。启动前应确认：

```bash
pgrep -af 'start_motion_control|./motion_control --cfg_file_path|a3_mdu_state_bridge'
```

需要切换只读桥到真实桥时，先正常停止只读桥，再启动带
`--command-enable` 的唯一实例。

停止原控制器可能让机器人失去站立力矩，只能在机器人已吊装/可靠支撑、急停
在手边且周围无人时执行。

第一阶段只允许零增益真实出口测试。MDU 启动参数保持默认上限为零：

```bash
cd /agibot/a3_remote_bridge
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
  ./scripts/run_mdu_state_bridge.sh --command-enable
```

电脑端显式确认后，以 50 Hz 发送 5 秒当前位置/零增益命令：

```bash
python3 pc_tools/a3_control_sender.py \
  --mode zero-gain --hz 50 --duration-sec 5 \
  --i-understand-this-can-move-the-robot
```

单关节微动是下一阶段，必须重启 MDU 网关并显式给出很低的增益上限，例如
`--max-kp 2 --max-kd 0.2`。电脑端使用 `joint-sine`、`0.005 rad` 做一次回零
轨迹。首次测试不要选择腰腿关节；机器人必须保持吊装。

2026-07-25 真机验收结果：

- 零增益真实出口：50 Hz、5 秒，`250/250 ACK`，0 丢包、0 拒绝；
- `right_wrist_yaw_joint` 微动：命令幅度 `0.005 rad`、`Kp=2`、`Kd=0.2`，
  `250/250 ACK`，0 丢包、0 拒绝；
- 关节实测初始 `-0.114690189 rad`，范围
  `[-0.116826420, -0.113469486] rad`，响应跨度 `0.003356934 rad`，最终
  `-0.115300541 rad`；
- 两次测试停止发送后 watchdog 均正常锁存为零增益 safe-halt，随后人工退出
  真实模式并恢复 `publish_enabled=false`。

同日右髋排障补充：

- 原厂 `motion_control` 的腿部 command 实测约 500 Hz；HAL 会把低频输入保持并
  以 500 Hz 写入 EtherCAT，因此电脑侧 50 Hz 不是不动作的原因；
- HAL 的 `/hal_debug/leg_joint_command` 已确认能采用远程目标和 PD 增益；
- 前几次不动作的主要原因是另一个 SSH 终端残留的
  `start_motion_control.sh` 自动重启原厂控制器，两个发布者反复覆盖同一 command；
- 清除原厂外层脚本、子进程及重复状态桥后，`right_hip_pitch_joint` 使用
  5° 目标、`Kp=300`、`Kd=5`、50 Hz、12 秒余弦往返轨迹，获得
  `600/600 ACK`、0 丢包、0 拒绝；实测位移 `0.076600075 rad`（4.39°），
  峰值实测力矩约 `4.47 N·m`；
- `Kp=300` 仍低于原厂 PD_STAND 右髋 `Kp=1500`，5° 目标与实测值之间的差值
  是负载下的 PD 跟踪误差。测试后已停止真实网关并恢复只读桥。

## 本机构建检查

```bash
source /opt/ros/humble/setup.bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target a3_mdu_state_bridge -j2
```

首次配置会获取 AimRT 及其依赖。构建产物位于 `build/dist/`。

当前本机构建只验证 x86_64 代码和链接，不能上传到 MDU 运行。

## MDU交叉编译

交叉编译复用原 A3 项目的构建方式：x86_64 Docker builder、
`aarch64-linux-gnu` 工具链以及从 Rockchip 镜像提取的目标 sysroot。

先生成一次 sysroot：

```bash
bash scripts/export_rockchip_sysroot.sh
```

也可以继续使用原仓库的导出命令：

```bash
cd ~/workspace/a3_deploy_example
bash scripts/export_rockchip_sysroot.sh
```

`build_mdu_package.sh` 会在新仓库没有 sysroot 时自动查找上述原仓库产物。

默认从以下镜像提取：

```text
registry.agibot.com/agibot-tech/rockchip:1.0
```

随后构建 MDU 包：

```bash
bash scripts/build_mdu_package.sh --jobs 20
```

产物位于：

```text
dist/a3_mdu_state_bridge/
  dist/a3_mdu_state_bridge
  dist/libaimrt_iceoryx_plugin.so
  dist/libjoint_msgs__*.so
  config/a3_mdu_iceoryx.yaml
  config/fastrtps_mdu.xml
  scripts/run_mdu_state_bridge.sh
```

构建脚本最后会用 `aarch64-linux-gnu-readelf` 检查 ELF 的目标架构。
