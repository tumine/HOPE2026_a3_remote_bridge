# 部署与安全检查清单

## 当前结论

| 项目 | 状态 |
|---|---|
| model_21500 ONNX actor | 已生成并校验 |
| 111 维观测/31 维动作合同 | 已明确 |
| identity/clip/统一 scale ActionAdapter | 已对齐训练 |
| 训练分布 READY 输入 | 已做 10 秒 MuJoCo 稳定性验证 |
| Planner 正反手击球区域 | 已与训练窗口做代码级对齐 |
| MuJoCo 回球 smoke | 8/8 成功、0 跌倒 |
| 真机厂商 KP/KD | **未提供，阻塞实机使能** |
| 基座 world/table 现场标定 | **未完成，阻塞实机击球** |
| 推理/执行延迟标定 | **未完成** |
| 真机安全认证 | **未完成** |

## A. 离线文件检查

- [ ] `tools/verify_bundle.py` 全部通过。
- [ ] ONNX 输入为 `float32[batch,111]`，输出为 `float32[batch,31]`。
- [ ] ONNX metadata 中的 31 关节顺序与 A3 SDK 完全一致。
- [ ] `observation_normalization=none`。
- [ ] 策略推理固定为 50 Hz，超时有 watchdog。
- [ ] raw action 先 clip 到 `[-100,100]`。
- [ ] head 索引 3、4 在 clip 后清零。
- [ ] 下一周期 `last_action` 使用 clip 后且 head 清零的 applied action。
- [ ] 没有额外 `tanh`，统一 `action_scale=0.25`。
- [ ] READY 的后 8 维目标观测与 `config/runtime.yaml` 完全一致。

## B. 传感器和坐标检查

- [ ] 所有关节角为 rad，关节速度为 rad/s。
- [ ] IMU 角速度已经转换到 pelvis/body frame。
- [ ] 四元数顺序和左右手坐标系转换正确。
- [ ] 有可靠的 `base_pos_w` 和 base yaw，不是只依赖陀螺仪无限积分。
- [ ] 动捕、球桌、Planner、机器人基座共享同一个标定 world/table frame。
- [ ] 机器人站位在 canonical table frame 中实测为预期的
  `[-0.50,-0.7625,-0.76] m`，或已同步修改所有相关变换。
- [ ] 启动时只捕获一次 `fixed_station_xy`。
- [ ] `racket_target_rel_base` 只做 world 位置相减，不再按 pelvis yaw 旋转。
- [ ] 静止机器人时逐项打印 111 维观测，确认量级、符号和切片。

## C. Planner 检查

- [ ] `use_side_aware_strike_regions=true`。
- [ ] 正手目标进入正手位置/速度窗口，反手目标进入反手窗口。
- [ ] 正反手分界 `-1.1725 m` 和 `0.04 m` 滞回带位于两窗口间隙。
- [ ] 不满足窗口的来球被拒绝，而不是裁剪到边界后强行挥拍。
- [ ] 新任务只在 TTS `[0.95,1.0] s` 内接收。
- [ ] `task_id` 每个新球递增，同一球 revision 递增。
- [ ] 同一 task 的 `swing_side` 不翻转。
- [ ] `time_to_strike` 已减去消息和邮箱年龄。
- [ ] 已另外测量并补偿推理/执行延迟。
- [ ] 丢包、过期消息、乱序 revision 和时钟跳变时不触发挥拍。

## D. 实机控制参数

- [ ] 由厂商/控制负责人填写并审核 `hardware_gains_TEMPLATE.yaml`。
- [ ] 确认厂商底层是否已经执行 PD，避免双层高增益。
- [ ] 设置电流、力矩、速度、位置变化率和机械关节限制。
- [ ] 设置通信超时后的安全保持/卸力策略。
- [ ] READY 条件下先保持策略推理但不使能电机，检查输出稳定性。
- [ ] 急停、系绳、保护架和安全观察员到位。
- [ ] 球拍周围和机器人可达空间无人。

## E. 逐级测试

按顺序进行，任何一级异常都退回上一层：

1. ONNX 零输入、READY 输入和录制观测离线回放。
2. MuJoCo `--idle` 保持 READY，确认 10 秒不跌倒。
3. MuJoCo 正手、反手、连续切换和真实来球碰撞。
4. 仿真记录 `raw_action/applied_action/q_des/q/qd/tau`、推理时延和各层饱和率。
5. 真机电机不使能，只读取并打印观测与目标。
6. 真机悬空或保护架，使用厂商批准的低增益/限速模式核对关节方向。
7. 系绳站立，无球 READY 和单次挥拍。
8. 正手和反手分别进行低速软球测试并分别统计结果。
9. 最后才允许连续来球和完整速度范围。

不要为了“慢一点”临时改变 `action_scale=0.25` 或加入 `tanh`；这会改变策略合同。调试限速
应由厂商安全后端显式实现并单独标注，正式验收仍需恢复训练合同。

## F. 当前性能快照

checkpoint 21500 的训练快照：

```text
mean_reward          14.1434
mean_episode_length  495.28 / 500
racket_pos_error       0.2804 m
racket_vel_error       0.6912 m/s
contact_rate           0.9963
net_clear_rate         0.9605
return_success         0.9474
both_feet_contact      0.9698
base_tilt              0.0686 rad
station_error          0.0501 m
```

MuJoCo current-training-planner smoke 为 8/8 回球成功、0 跌倒；READY 10 秒最低 pelvis
`1.0653 m`、最大站位漂移 `0.0212 m`。这些结果说明当前仿真链路已对齐，不能替代现场
标定、真机安全审批或更大样本量的正反手分侧评估。
