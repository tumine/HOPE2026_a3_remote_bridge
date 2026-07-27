# A3 model_21500 + PPMocap 真机部署

这套部署在电脑运行当前 `model_21500` ONNX 策略，使用 MDU 的关节/IMU 状态和
PPMocap 的桌面、球、BotA3 位姿。球经过官方 HOPE planner 变成
`hope_msgs/RacketCommand`，策略以 50 Hz 输出 31 关节位置目标。

## 当前固定合同

- policy generation：`full_body_uniform_action_return_physics_v1`
- checkpoint：`model_21500.pt`
- observation：111 维、无 normalization
- action：31 维
- 动作计算：`clip(raw,-100,100)`，头部 3/4 清零，统一乘 `0.25 rad`，最后机械限位
- `last_action`：上一周期 clip 后、头部清零后的 applied action
- 无球 READY：用 bundle 锚点和仿真第一帧基座构造一个固定世界虚拟目标；
  每周期仍以实时动捕 pelvis 重新计算 `target-base`，不能把整个切片固定或置零
- 挥拍时钟：50 Hz，91 个动作采样，`+1.0 s -> 0 -> -0.8 s`
- Kp/Kd：训练与 sim2sim 数值完全一致；它们仍不是厂商出具的真机安全认证参数

权威 bundle 位于：

```text
26.7.25发球部署/pingpang_ustc-srf/a3_deploy/a3_pingpong_deploy_bundle
```

## 数据流

```text
PPMocap /ppmocap/frame
  ├─ table + BotA3 -> 锁定 canonical table frame -> base_pos/base_quat
  └─ ball -> /a3_mocap/ball_poses (PoseArray, hope_table)
                  -> hope_planner
                  -> /racket/command (RacketCommand)

MDU iceoryx joint/IMU
  -> /a3_internal/{joint_states,pelvis_imu}
  -> 111-D observation + model_21500 ONNX
  -> /a3_internal/joint_command_control
  -> MDU guarded bridge -> iceoryx -> HAL
```

姿态源刻意拆分：`projected_gravity` 和倾倒保护使用 MDU pelvis IMU 四元数；
`base_forward_xy` 使用标定后的 PPMocap pelvis 四元数，因为它必须与球桌/planner 的
全局 XY 航向一致。站位误差和目标相对位置使用 PPMocap pelvis 位置。不要用光学姿态
替代高频重力观测，也不要用未对齐的 IMU yaw 替代桌面世界航向。

无球 READY 的位置也不是每周期执行 `live_base + ready_rel`。部署固定使用仿真
第一帧 pelvis 在 canonical table frame 中的位置
`[-0.5,-0.7625,0.3064]`，加上 bundle 的 `ready_target_rel_base_w`，得到固定虚拟目标：

```text
ready_target_table = [-0.1798684299, -1.4577511668, 0.2486044419]
obs[103:106] = ready_target_table - live_mocap_pelvis_table
```

所以机器人站在仿真起始位姿时，前三维恰好等于 bundle 锚点；机器人漂移后，这三维会
反向变化，为策略保留训练时的世界目标/站位反馈。速度、TTS 和 swing side 在 READY
期间仍使用 bundle 的固定首帧值。

桌面 canonical frame 定义固定为：近端左角原点，`+X` 从机器人指向对手，`+Y`
为机器人左侧，`+Z` 向上，桌面高度为 `Z=0`。代码用桌面长轴和 BotA3 所在一侧自动
消除桌面矩形的 180° 歧义。球、机器人和 planner 命令共享这个坐标系。

## 一次性准备

### 1. Python 环境

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/setup_rl_venv.sh
```

### 2. 构建当前 HOPE planner/message

训练仓库更新后执行一次；每次启动真机不用重复：

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/setup_hope_planner.sh
```

输出默认位于 `.hope_ros2/install`。这里只构建 `hope_msgs` 和 `hope_planner` 依赖链，
不会启动 VRPN 或机器人。

### 3. 运行一次离线门禁

策略、bundle、动作计算或 READY 观测变化后执行一次：

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/run_rl_preflight.sh
```

它先检查 hash、111/31 维接口、identity/clip/0.25、91 tick lifecycle 和纯数值动捕
坐标变换，然后运行 10 秒 MuJoCo READY。成功后不需要每次真机启动重复。

## 启动 PPMocap

PPMocap、planner、策略和 MDU 必须使用同一个 ROS domain `232`。如果动捕此前在默认
domain 0 启动，需要停止后按下面方式重启：

```bash
cd /home/bth/workspace/Mocap
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 launch ppmocap_driver ppmocap.launch.py
```

至少确认：

```bash
ros2 topic hz /ppmocap/frame
ros2 topic echo /ppmocap/status --once
```

`receiving`、`table_tracked`、`robot_tracked` 应为 true。球不在视野时
`ball_visible=false` 是正常 READY 状态。

## 先做不驱动真机的完整链路检查

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/start_rl_policy.sh --dry-run
```

该命令会启动官方 HOPE planner 和电脑 runner，但不会创建关节 command publisher。
按 `I` 查看状态，重点检查：

```text
base ~= [-0.500, -0.7625, 0.30]
forward ~= [1.000, 0.000]
gate=ready
planner subscription/publisher present
```

若 `D rejected` 显示站位、朝向或高度错误，不要放宽门禁。先修改：

```text
config/a3_rl_deploy.yaml
  mocap.robot_marker_to_pelvis.translation_m
  mocap.robot_marker_to_pelvis.rpy_rad
```

这里的 translation/RPY 是“pelvis 原点在 BotA3 刚体坐标中的固定姿态”。默认全零只在
FZMotion 的 BotA3 原点和 pelvis 完全重合时成立。桌子或刚体定义移动后，在 OBSERVE
按 `C` 重新锁定桌面。

把一个静止球依次放到近端左/右、远端左/右四角，检查 runner 发布的
`/a3_mocap/ball_poses`：canonical 坐标应分别接近
`[0,0,0]`、`[0,-1.525,0]`、`[2.74,0,0]`、`[2.74,-1.525,0]`。

## 真机启动

机器人必须吊装或可靠支撑，急停在手中，周围无人。随后在电脑只运行：

```bash
cd /home/bth/workspace/a3_remote_bridge
./scripts/start_rl_policy.sh --robot-suspended
```

需要同步显示动捕基座、编码器关节动作、右手球拍和物理地面时使用：

```bash
./scripts/start_rl_policy.sh --robot-suspended --rviz
```

也可在第二个终端独立运行只读可视化：

```bash
./scripts/run_a3_rviz.sh
```

可视化以原 PPMocap RViz 场景为底稿，保留球桌、球、marker、实测/预测球轨迹和
可选相机。它复用策略的 canonical-table 标定和 marker-to-pelvis 外参，发布
`world -> hope_table -> a3/pelvis_link`；`robot_state_publisher` 订阅
`/a3_internal/joint_states` 并载入带右手球拍的 A3 URDF，地面为 `hope_ground`
（桌面下方 0.76 m）。

脚本会：

1. 启动本机 HOPE planner；
2. 通过 SSH 进入 MDU；
3. 停止原厂控制栈并启动唯一 HAL/RouDi/真实 bridge；
4. 启动电脑 runner，但停在 `OBSERVE`，不会自动发命令。

## 键盘状态机

按键不需要 Enter：

| 按键 | 功能 |
|---|---|
| `P` | 5 秒平滑进入训练 default pose，然后保持 |
| `D` | 只有 pose、动捕、站位、朝向、planner 全部通过后才进入策略 |
| `I` | 打印一次完整状态快照 |
| `C` | 仅 OBSERVE 清空并重新做桌面标定 |
| `S` | 屏蔽新球命令，READY 保持 2 秒，发送零增益并退出 |
| `X` | 立即进入零增益 halt 并退出 |
| `H` | 帮助 |
| `Q` | 仅 OBSERVE 直接退出 |

安全顺序：

```text
启动 -> OBSERVE -> I确认 -> P -> 等待 pose_ready=yes -> I确认 gate=ready -> D
```

`D` 后先进行 3 秒 READY warmup，期间 planner 命令被屏蔽；随后进入
`POLICY_ACTIVE`。不再提供 `F/B` 人工球观测。每颗真实来球由 planner 产生递增
`task_id`，策略对同一 task 只挥一次并自动回到 READY。

## 正常停止与故障

- 正常结束按 `S`。
- 人、线缆或机器人状态危险时按 `X` 并使用物理急停。
- 关节/IMU 超时、BotA3 丢失、桌面/站位门禁丢失、倾角越界、推理超时或 ACK backlog
  会自动 HALT，发送零增益后退出。
- 退出后 MDU watchdog 会锁存，下一次控制需重新启动 bridge；启动脚本的 cleanup 会处理。

## 当前仍需现场完成的项目

代码更新不能替代以下实测：

1. 标定 BotA3 刚体到 pelvis 的固定平移和旋转；
2. 四角确认 PPMocap canonical table 坐标；
3. 确认实际站位为 `[-0.50,-0.7625] m` 且机器人朝 `+X`；
4. 测量动捕、planner、推理、DDS 和执行器总延迟；
5. 从软球、低速、单次正手/反手开始，最后才连续来球。

bundle 自己标记为 `current_sim_candidate_not_hardware_certified`。真机测试必须保留 MDU
限位、watchdog、急停和物理支撑，不能因为 MuJoCo 通过就视为真机安全认证。
