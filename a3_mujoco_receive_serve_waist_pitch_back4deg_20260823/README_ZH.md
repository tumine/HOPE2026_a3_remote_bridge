# A3 MuJoCo 接球后发球与腰 pitch 后移补偿包

本包冻结了经过 MuJoCo 验证的完整流程：机器人先使用 `model_72500` 完成一次接球，回到 READY 并稳定，然后执行与按键 `V → C → F` 一致的 1 号发球流程。进入发球 Home 时，`waist_pitch_joint` 用 1.35 s 平滑移动到 −4°目标，以减小双臂前移造成的躯干前倾；发球结束后再用 0.20 s 平滑回到接球策略的 0°锁定目标。

这是独立仿真包，不会导入训练代码，也不会连接 RobotIO、EtherCAT、夹爪 HTTP 或真实机器人。

## 1. 冻结内容

- 接球策略：`model_72500`，111 维观测、31 维动作、50 Hz。
- ONNX SHA256：`6ab2e061062455a997afb7c630599d1eafd0a66f90dfcfb5c1da8177c8093ed0`。
- 动力学模型：GJC A3 MJCF，包含完整网格与碰撞。
- 球台参数：`restitution=0.9448`、`tangential_damping=0.2079`。
- 球拍参数：`restitution=0.654`、`tangential_damping=0.52`。
- 发球轨迹：实机调试后的 1–5 号双臂 YAML。
- 默认腰 pitch 发球目标：−4°，KP=500、KD=2。
- 发球 Home：1.35 s smoothstep。
- 发球后随挥：0.05 s；回接球控制：0.20 s smoothstep。

机器可读的冻结合同见 `PACKAGE_INFO.json`。

## 2. 控制权与状态切换

| 状态 | 双臂 | waist yaw / 双腿 | waist roll | waist pitch |
|---|---|---|---|---|
| RECEIVE READY/击球 | model_72500 | model_72500 | 0°锁定 | 0°锁定 |
| SERVE_HOMING | 发球 Home smoothstep | model_72500 READY 推理 | 0°锁定 | 从当前目标 smoothstep 到 −4° |
| SERVE_READY | 发球 Home | model_72500 READY 推理 | 0°锁定 | 保持 −4° |
| SERVE_WINDUP/SWING/SETTLE | 发球 YAML | model_72500 READY 推理 | 0°锁定 | 保持 −4° |
| TO_RECEIVE | 平滑交还 model_72500 | model_72500 READY 推理 | 0°锁定 | 0.20 s 平滑回 0° |

状态顺序：

```text
RECEIVE
  → 注入一球
  → SWING / FOLLOW_THROUGH
  → 回到 READY 并稳定 0.5 s
  → V: SERVE_HOMING（双臂 Home + 腰 pitch 同步后移）
  → SERVE_READY
  → C
  → F: WINDUP → SWING → SETTLE
  → TO_RECEIVE
  → RECEIVE
```

自动流程中，1 号轨迹通过与按 `V` 相同的 `request_serve_mode()` 入口进入，不使用另一套简化状态机。

## 3. 腰 pitch 计算

面向球台为世界 `+X`，腰 pitch 正角会让躯干向球台前倾，负角为向后。默认目标：

```text
q_waist_pitch_serve = -4° = -0.06981317 rad
```

进入 Home 时使用 smoothstep：

```text
r = clamp(t / 1.35, 0, 1)
a = r²(3 - 2r)
q_des = q_start + a(-4° - q_start)
```

从发球切回接球时：

```text
r = clamp(t / 0.20, 0, 1)
a = r²(3 - 2r)
q_des = -4° + a(0° - (-4°))
```

CLI 将腰 pitch 补偿限制在 `[-8°, +5°]`。默认 −4°是扫参后选择的保守值；不要在不了解真实机器人力矩、限位和状态机同步的情况下扩大范围。

## 4. 发球期间的策略观测

`model_72500` 在所有状态下仍以 50 Hz 推理。发球阶段：

- 关节位置、关节速度、IMU、projected gravity 等本体观测保持实时。
- 任务字段固定为无球 READY：`station_error=[0,0]`、`racket_rel_base=[0.45,-0.25,0.08]`、目标速度 `[0,0,0]`、`TTS=1.0`、`side=0`。
- 策略原始动作和 `last_action` 合同保持 31 维。
- 双臂由发球轨迹硬覆盖；腰 pitch 仅在发球状态由本补偿器覆盖。
- 腰 yaw 和十二个腿关节始终来自策略；腰 roll 始终保持 0°。

因此策略能观测到双臂和腰 pitch 的真实反馈，但不会与发球控制器争夺这十五个关节的最终目标所有权。

## 5. 环境安装

建议 Ubuntu 22.04/24.04、x86_64、Python 3.10 或 3.11。可视化需要桌面环境和 OpenGL。

```bash
sudo apt update
sudo apt install -y python3-venv libgl1 libglfw3 libxrandr2 libxinerama1 libxcursor1 libxi6

cd a3_mujoco_receive_serve_waist_pitch_back4deg_20260823
./setup.sh
./verify_bundle.sh
```

`setup.sh` 只创建包内 `.venv`，依赖版本冻结在 `requirements-lock.txt`。

已有兼容环境时也可直接指定：

```bash
export MUJOCO_PYTHON=/path/to/python
```

## 6. 推荐可视化：先接球再发球

```bash
./run_receive_then_serve_viewer.sh
```

默认运行三轮，每轮严格执行：

1. 注入一颗合法单跳来球。
2. 等待物理球拍碰撞、过网和接球生命周期完成。
3. READY 下腰腿速度小于 0.15 rad/s 并连续稳定 0.5 s。
4. 自动执行 V、等待发球 READY、执行 C/F。
5. 返回接球模式后再开始下一轮。

改变轮数：

```bash
CYCLES=5 DURATION=180 ./run_receive_then_serve_viewer.sh
```

## 7. 交互式完整闭环

```bash
./run_viewer.sh
```

按键：

- `N`：在接球模式注入真实物理来球。
- `V`：固定选择 1 号轨迹并进入发球 Home，同时启动 −4°腰 pitch 补偿。
- `1–5`：显式选择相应发球轨迹并进入 Home。
- `C`：模拟关闭夹爪。
- `F`：播放发球轨迹；必须已进入 SERVE_READY 且先按 C。
- `G`：模拟打开夹爪。
- `M`：从发球 Home/READY 取消并返回接球。
- `I`：打印状态和策略任务观测。

推荐人工顺序：`N → 等待接球完成 → V → 等待发球就绪 → C → F`。

临时覆盖补偿角度：

```bash
./run_viewer.sh --serve-waist-pitch-deg=-3
```

## 8. 无窗口验证与 0°基线

运行默认 −4°三轮闭环：

```bash
./run_headless_check.sh
```

运行相同顺序的 0°基线：

```bash
./run_zero_pitch_baseline.sh
```

主程序结束时输出 JSON，包括每次进入 SERVE_READY 时的：

- base 高度；
- pelvis/torso roll、pitch 和总倾角；
- 腰 pitch 目标与反馈；
- 腰腿最大速度和跟踪误差；
- 球拍碰撞、过网和对方台面落球次数。

可使用 `--output-json /path/result.json` 保存新结果。

## 9. 已验证结果

相同初始条件下三轮“接一球 → V/C/F”结果：

| 指标 | 0°基线 | −4°补偿 |
|---|---:|---:|
| 完成闭环 | 3/3 | 3/3 |
| 球拍接触/过网/对方落台 | 3/3/3 | 3/3/3 |
| SERVE_READY 躯干 pitch | +7.96° 到 +9.11° | +5.21° 到 +5.95° |
| SERVE_READY 实际腰 pitch | +3.31° 到 +3.54° | −0.72° 到 −0.57° |
| 最低 base 高度 | 0.9876 m | 0.9869 m |
| 全局最大 pelvis 倾角 | 7.92° | 9.49° |

结论：−4°能把胸腔拉回约 3°，并保持三轮接发球成功；pelvis 会多承担约 1°前倾，因此该方法是躯干姿态补偿，不等价于整个机器人重心后移。−7°和 −8°会进一步增加 pelvis 前倾，不作为默认值。

验证文件：

```text
validation/three_cycles_waist_pitch_0deg.json
validation/three_cycles_waist_pitch_-4deg.json
validation/waist_pitch_-4deg_v_exact.json
validation/waist_pitch_sweep/waist_pitch_0deg.json
validation/waist_pitch_sweep/waist_pitch_-1deg.json ... -8deg.json
```

## 10. 配置与源码

```text
pc_tools/a3_serve_receive_mujoco.py       完整接发球状态机、策略推理、补偿与指标
PACKAGE_INFO.json                         机器可读冻结合同
config/ball_physics.yaml                  本次验证使用的球/球台/球拍物理
config/serve_tracks/1.yaml ... 5.yaml     真机热加载轨迹备份
Serve_A3_leg_model/tracks/1.yaml ... 5.yaml  MuJoCo 使用的发球轨迹
model_72500_deploy_bundle/                 ONNX、观测、ActionAdapter、MJCF与网格
yfr_hope_pingpong_deploy_20500/sim/       球台、球网和动态球场景
scripts/verify_waist_pitch_contract.py     策略/物理/状态机/验证结果合同检查
```

## 11. 完整性校验

```bash
./verify_bundle.sh
```

它会执行两层检查：

1. `SHA256SUMS`：检查包内所有冻结文件。
2. `verify_waist_pitch_contract.py`：检查 model_72500 SHA、GJC MJCF、球物理、−4°默认状态机源码和三轮验证结果。

若替换策略、MJCF、球物理或状态机源码，校验应当失败；不要只更新哈希而不重新完成物理验证。

## 12. 真机迁移注意事项

本包没有授权或实现真实机器人控制。部署侧若要复现，至少必须同时对齐：

- V/C/F 状态机和控制周期；
- Home 的 1.35 s smoothstep 与回接球的 0.20 s smoothstep；
- 腰 pitch 的符号、弧度单位、−4°限幅、KP/KD 和执行器力矩限制；
- 双臂/腰 pitch 的单一最终命令所有权；
- 策略实时本体观测和 READY 任务字段；
- 急停、碰撞、倾角、足底支撑和通信超时保护。

在真机启用前，应先从更小角度和更慢 Home 开始，并记录 torso/pelvis pitch、腰 pitch 反馈、足底受力和关节力矩。
