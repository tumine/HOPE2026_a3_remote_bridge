# A3 IK 路径记录

本目录使用 `HOPE2026_serve_deploy` 构建出的两个 C++ 示教程序：

- `hope_a3_teach_keyframes`：记录蓄力位和击球位两个关键帧；
- `hope_a3_teach_trajectory`：连续记录 `initial -> windup -> contact -> initial` 右臂路径。

## 构建

在本目录运行：

```bash
./compile.sh
```

默认优先使用 `hope_deploy` 嵌套源码；未初始化时使用工作区中的
`HOPE2026_serve_deploy`。也可通过 `HOPE_DEPLOY_DIR` 指定源码位置。

## 主机一键采集

```bash
./run_a3_teach_and_fetch.sh trajectory
```

脚本默认连接 `agi@192.168.1.101`，先在 MDU 上执行只读 `--check`，再进入
交互式采集。按键流程为：

1. `E`：左臂对齐、右臂进入低刚度示教；
2. `R`：开始记录；
3. 手动拖动右臂走完 `initial -> windup -> contact -> initial`；
4. `R`：停止记录；
5. `Q`：保存并退出。

记录器输出同名 YAML 和 CSV。CSV 包含源时间戳、右臂七轴位置/速度/力矩，
以及左臂七轴位置；主机脚本会下载到：

```text
artifacts/a3_teach_downloads/trajectory/<MMDD_HHMM>/
```

关键帧模式使用：

```bash
./run_a3_teach_and_fetch.sh keyframes
```

## 机器人端直接运行

只读检查：

```bash
./run_hope_a3_teach_trajectory.sh --check
```

实机记录需要显式双重使能：

```bash
A3_TRANSPORT=iceoryx \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_ROBOT_SAFETY_READY=1 \
./run_hope_a3_teach_trajectory.sh \
  --enable-command \
  --align-duration-s 8 \
  --right-damping 4 \
  --max-duration-s 120 \
  --output artifacts/a3_teach/trajectory_<RUN_ID>.yaml
```

启动器会检查 `agibot_pm` 已停止、HAL/RouDi 已运行，并阻止多个 A3 命令
进程并发。默认输出目录及主机下载目录已由仓库忽略，不会误提交实机记录。
