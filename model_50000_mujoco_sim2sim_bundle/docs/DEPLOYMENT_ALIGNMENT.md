# 部署对齐说明

## 架构边界

```text
动捕 PoseArray
  -> BallStateEstimator
  -> BallTrajectoryPredictor
  -> side/position/velocity/TTS admission
  -> RacketCommand
  -> SwingLifecycle
  -> lateral station target
  -> 111-D observation
  -> ONNX actor
  -> ActionAdapter
  -> 31 joint position targets
  -> MuJoCo PD 或真实机器人控制器
```

planner 不生成腿部轨迹，也不直接发送 base 速度。base 移动来自策略根据 `obs[101:103]` 输出的腿部关节动作。

## 坐标系

planner canonical table frame：

```text
origin: 近侧左角球台表面
+x: 指向对手
+y: 左
+z: 上
table x: [0, 2.74]
table y: [-1.525, 0]
table surface z: 0
```

MuJoCo world 是 floor frame。两者只有平移：

```text
p_mujoco = p_table + [station_x + 0.5,
                       station_y + 1.525/2,
                       0.76]
```

速度向量不平移。真实部署必须保证 planner 的 world target 与 robot `base_pos_w` 在同一坐标系。

## planner 命令合同

```text
uint64 task_id
uint32 task_revision
int8 swing_side
Point position
Vector3 velocity
float64 time_to_strike
header.stamp
```

`time_to_strike` 定义在 `header.stamp` 对应的球观测时刻。接收端应扣除传输和队列时间，不得把过期命令伪装成 TTS=0。

当前训练时钟从 `+1.0 s` 到 `-0.8 s`。策略在训练中见过 0.25 s，但那是从 1.0 s 连续执行后的中间状态，不等价于 READY 姿态第一次收到 TTS=0.25 s。

## planner 源代码

包内 `hope_ws/src/hope_planner/hope_planner/` 包含：

- `ball_state_estimator.py`：位置窗口和速度拟合
- `ball_trajectory_predictor.py`：阻力、重力和球台反弹预测
- `racket_target_planner.py`：目标落点反解球拍速度
- `strike_selection.py`：正反手位置/速度盒准入
- `side_selection.py`：正反手分界和滞回
- `task_timing.py`：新任务 TTS 窗口
- `node.py`：ROS2 topic 和 task/revision 生命周期

## 部署侧对齐清单

1. ONNX SHA-256 与 `MANIFEST.sha256` 一致。
2. 运行时输入维度 111、输出维度 31、频率 50 Hz。
3. 无观测归一化。
4. canonical joint name/order 完全一致。
5. passive neck 两列在反馈和下发前都清零。
6. `last_action` 使用 clipped applied action。
7. planner target 与 `base_pos_w` 使用同一 world frame。
8. `fixed_station_error_xy` 使用 moving base target，不是永远固定中心。
9. READY/RECOVERY 回中心；SWING/FOLLOW_THROUGH 使用击球站位。
10. `task_id` 每球递增；同球只增加 revision；side 一旦锁定不变。
11. 新命令在观测后才开始减少一个控制 tick。
12. 消息时间戳和接收时钟同步，记录 raw/aged TTS。
13. 使用 `config/action_adapter.yaml` 的 scale、default pose 和软件限位。
14. 真实机器人硬件增益、扭矩、速度和急停由硬件侧单独审核。

## 推荐验收顺序

```text
1. 包内无窗口 10 球测试
2. 包内可视化连续交替测试
3. 左右边缘压力测试
4. 1.0/0.25 TTS 无球对照
5. 部署端离线回放同一批 111-D 观测，比较 ONNX raw_action
6. 部署端仅下发上电安全姿态
7. 无球 READY 和回中心
8. 低速、中心球
9. 逐步扩大左右和高度范围
```

部署端若能导出每 tick 的 111 维观测，可以直接与包内 `build_observation()` 逐字段比较。优先检查 `[101:103]`、`[103:106]`、`[109]` 和 `last_action`。
