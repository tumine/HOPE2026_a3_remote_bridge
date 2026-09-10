# A3 MuJoCo 完整接发球闭环包

本包冻结了 2026-08-23 的完整 MuJoCo 接发球实现：

- 接球：`model_72500`，111 维观测、31 维动作、50 Hz。
- 发球：实机调试后的 1–5 号双臂轨迹。
- 发球阶段：腰部和双腿持续由 `model_72500` 的 READY 推理控制，仅覆盖头部/双臂。
- 发球完成：随挥保持 0.05 s，随后用 0.20 s 切回接球全身策略。
- 场景：A3 完整 MJCF/网格、标准球台、球网、40 mm 动态乒乓球和球拍碰撞。
- 发球流程只模拟夹爪状态，不生成发球球；接球测试球由 `N` 注入并参与真实 MuJoCo 碰撞。

策略文件 SHA256：

```text
6ab2e061062455a997afb7c630599d1eafd0a66f90dfcfb5c1da8177c8093ed0
```

## 1. 支持环境

建议使用 Ubuntu 22.04、x86_64、Python 3.10 或 3.11。可视化需要桌面环境和 OpenGL。

Ubuntu 常用系统依赖：

```bash
sudo apt update
sudo apt install -y python3-venv libgl1 libglfw3 libxrandr2 libxinerama1 libxcursor1 libxi6
```

## 2. 解压与安装

```bash
tar -xzf a3_mujoco_receive_serve_20260823.tar.gz
cd a3_mujoco_receive_serve_20260823
./setup.sh
./verify_bundle.sh
```

`setup.sh` 只创建包内 `.venv`，不会修改系统 Python。

## 3. 可视化运行

```bash
./run_viewer.sh
```

按键：

- `1–5`：选择相应实机发球轨迹并进入 Home。
- `V`：固定选择 1 号轨迹并进入发球状态。
- `C`：模拟关闭夹爪。
- `F`：播放当前发球轨迹；必须先到 READY、再按 C。
- `G`：模拟打开夹爪。
- `M`：从发球 Home/READY 返回接球。
- `N`：在接球模式注入一个真实物理来球。
- `I`：打印当前状态。
- 关闭窗口：退出。

选择 4 号轨迹启动：

```bash
./run_viewer.sh --serve-track 4
```

三倍慢放观察发球：

```bash
./run_viewer.sh --serve-time-scale 3
```

只观察无来球 READY/发球流程：

```bash
./run_viewer.sh --no-mocap-serve-test
```

## 4. 自动闭环验证

下面命令自动执行一次“发球 → 切回接球 → 接一球 → READY 稳定采样”：

```bash
./run_headless_check.sh
```

自定义循环次数和每轮接球数：

```bash
CYCLES=3 RECEIVE_BALLS=2 ./run_headless_check.sh
```

完成时输出 JSON，其中包括状态访问序列、球拍接触、过网、基座姿态、最大跟踪误差，以及接完球稳定后腰部/下肢反馈位置。

## 5. 配置位置

MuJoCo 发球关键帧：

```text
Serve_A3_leg_model/tracks/1.yaml
Serve_A3_leg_model/tracks/2.yaml
Serve_A3_leg_model/tracks/3.yaml
Serve_A3_leg_model/tracks/4.yaml
Serve_A3_leg_model/tracks/5.yaml
```

MuJoCo 完整闭环默认将 Home 加速为 1.35 s、随挥缩短为 0.05 s、接球切换设为 0.20 s；可分别使用：

```text
--serve-home-s
--post-serve-settle-s
--receive-transition-s
```

当前真机热加载版 YAML 也随包保存在：

```text
config/serve_tracks/1.yaml ... 5.yaml
```

它们用于核对真机轨迹、双臂增益和夹爪位置；MuJoCo 不会发送任何夹爪 HTTP 或机器人控制命令。

接球模型合同和球物理：

```text
model_72500_deploy_bundle/config/runtime.yaml
model_72500_deploy_bundle/config/action_adapter.yaml
model_72500_deploy_bundle/config/ball_physics.yaml
```

## 6. 常见问题

- `MuJoCo Python 环境不存在`：先运行 `./setup.sh`。
- 无法打开窗口：确认当前终端存在 `DISPLAY`；远程机器需要桌面转发。
- ONNX 哈希不一致：运行 `./verify_bundle.sh`，不要替换策略文件。
- 修改 YAML 后维度错误：`home` 必须 14 维，`windup_right` 和 `hit_through_right` 必须各 7 维。
- 自动测试超时：增加 `DURATION`，例如 `DURATION=60 ./run_headless_check.sh`。

## 7. 目录说明

```text
model_72500_deploy_bundle/       接球 ONNX、合同、参考实现、MJCF 和网格
Serve_A3_leg_model/              发球基础增益及 1–5 号轨迹
yfr_hope_pingpong_deploy_20500/  MuJoCo 3.11 兼容的球台/球碰撞场景
pc_tools/                        完整闭环入口
config/serve_tracks/             当前真机热加载发球配置备份
```

本包仅用于仿真，不包含 RobotIO、EtherCAT、夹爪 HTTP 执行或真机上传功能。
