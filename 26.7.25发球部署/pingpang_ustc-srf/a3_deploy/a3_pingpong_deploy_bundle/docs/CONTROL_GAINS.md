# 控制频率、KP/KD 与限制

## 三类参数不能混用

本项目中存在三套不同含义的控制参数：

1. **Isaac 训练增益**：策略训练时的隐式位置 PD，记录在
   `config/training_actuator_gains.yaml`。
2. **MuJoCo 参考增益**：参考播放器在 MuJoCo 中实现位置目标用的示例 PD，记录在
   `config/runtime.yaml`。
3. **A3 实机增益**：应由智元厂商后端和机器人负责人给出；本仓库没有权威值。

前两套都是仿真参数，不能直接标成实机参数。

## 控制时序

Isaac 训练：

```text
physics dt = 0.005 s（200 Hz）
policy decimation = 4
policy/action update = 0.020 s（50 Hz）
```

一个 `q_des` 在四个物理子步中保持。Isaac 使用隐式 actuator PD。
本策略的训练快照中 `events.pd_gains=null`，没有对 KP/KD 做 domain randomization；
这里列出的就是该 run 使用的名义增益。

MuJoCo 参考运行器：

```text
MuJoCo physics dt = 0.001 s（1000 Hz）
policy/action update = 0.020 s（50 Hz）
每个物理子步重新计算 tau
tau = kp * (q_des - q) - kd * qd
tau 再按 actuator ctrlrange 限幅
```

实机建议保持策略推理和 `q_des` 更新为严格 50 Hz；更高频的电机伺服、插值和安全限幅
由厂商实时后端负责。

## Isaac 训练 KP/KD

| 关节组 | KP | KD |
|---|---:|---:|
| waist_yaw | 85 | 3 |
| waist_roll / waist_pitch | 50 | 2 |
| head_yaw / head_pitch | 40 | 2 |
| shoulder_pitch / shoulder_roll | 40 | 3 |
| shoulder_yaw / elbow / wrist_roll | 30 | 2 |
| wrist_pitch / wrist_yaw | 20 | 2 |
| hip_yaw / hip_pitch | 80 | 3 |
| hip_roll | 120 | 4 |
| knee | 250 | 8 |
| ankle_pitch / ankle_roll | 50 | 2 |

具体力矩与速度上限：

| 关节组 | effort limit | velocity limit |
|---|---:|---:|
| waist_yaw | 220 Nm | 12.0 rad/s |
| waist_roll | 46 Nm | 22.7 rad/s |
| waist_pitch | 118 Nm | 9.2 rad/s |
| head | 6 Nm | 12.7 rad/s |
| shoulder_pitch / shoulder_roll | 60 Nm | 13.6 rad/s |
| shoulder_yaw / elbow / wrist_roll | 24 Nm | 15.7 rad/s |
| wrist_pitch / wrist_yaw | 6 Nm | 12.7 rad/s |
| hip_yaw / hip_roll / hip_pitch | 220 Nm | 12.0 rad/s |
| knee | 320 Nm | 14.6 rad/s |
| ankle_pitch | 118 Nm | 10.8 rad/s |
| ankle_roll | 55 Nm | 19.3 rad/s |

这些值描述训练仿真的策略动力学环境，不代表厂商允许的实机连续力矩或安全速度。

## MuJoCo 参考 KP/KD

当前 `config/runtime.yaml` 按 31 个关节逐项列出与上表相同的名义 KP/KD。MuJoCo
reference bridge 在 1 kHz 显式计算 PD，而 Isaac/PhysX 在 200 Hz 使用隐式 PD，因此数值
相同不代表两个积分器和接触动力学完全等价。它们仍只用于 sim-to-sim 对齐，不是实机参数。

## 实机参数要求

`config/hardware_gains_TEMPLATE.yaml` 故意保留 `null`。在以下事项完成前不要发送电机使能：

- 厂商确认每个关节的控制模式是位置、混合或力矩模式。
- 填写每个关节的 KP、KD、最大力矩/电流、最大速度和最大单周期位置变化。
- 确认厂商 SDK 是否已经在下层应用 PD；避免在两层重复施加高增益。
- 设置通信 watchdog、姿态/倾角保护、脚底接触保护和物理急停。
- 从经过厂商批准的低增益和限速配置开始无球测试。`action_scale=0.25` 是训练合同，不能在
  生产策略上临时缩小后仍宣称与训练一致；若需减小动作，应使用单独的调试策略或后端明确
  标注的安全测试模式。

若为了接近训练动力学而调整实机增益，应由机器人控制负责人根据频响和跟踪误差标定，
而不是简单令实机 KP/KD 等于 Isaac 数字。
