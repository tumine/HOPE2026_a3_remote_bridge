# 部署包验证记录

验证日期：2026-07-27

## 策略来源

- run：`2026-07-26_01-47-04`
- checkpoint：`model_21500.pt`
- checkpoint iteration：21500
- checkpoint SHA256：
  `7939b3a2e4716e764271126668602b0826f6f4fc9615a74ccdb7a0761ee19987`
- 原始训练导出 ONNX SHA256：
  `6def579da350d75d53f733ad6ba742336738b4d18267f111a2d33d2d8dcadf09`
- bundle ONNX SHA256：
  `6e2fcf9c9793a568f0583ae9b6e7b0439fb83df004c356e85af85841afc2d074`

bundle ONNX 在相同 actor 权重上补充了 bundle-relative ActionAdapter、joint-order、
policy generation 和 checkpoint metadata，因此文件哈希与训练目录原始导出不同。

## 已通过

1. checkpoint actor 结构严格为 `111→512→256→128→31`，隐藏层为 ELU。
2. bundle ONNX 由 `model_21500.pt` 的 8 个 actor 参数直接导出。
3. 随机 8 条观测的 PyTorch/ONNX 最大绝对输出误差：
   `1.430511474609375e-06`。
4. `onnx.checker.check_model` 通过。
5. 动态 batch inference 通过：
   - `[1,111] → [1,31]`
   - `[4,111] → [4,31]`
6. ONNX metadata、manifest、31 关节 YAML、ActionAdapter 和 READY 锚点一致。
7. 动作合同固定为：
   `identity → clip[-100,100] → head[3,4]=0 → scale=0.25 → joint clamp`。
8. 训练来源快照哈希：
   - `training_env.yaml`：
     `b53ce46b5dd241cd01bc46ee73c5230f407c1cd85115550631d32bf7aad1c70c`
   - `training_agent.yaml`：
     `c6a3ad503602f71704fe447f5c4e5432b6adba3ff3b6e8c7bcc403a8797f43bc`
   - `training_worktree.diff`：
     `8c129c02e7bc134da7c3eb1cfd3afba4c2e5652d4975143a83e62feea7ab77d3`
9. `hope_training/whole_body_tracking/tests`：`139 passed, 1 skipped`。
10. `hope_ws/src/hope_planner/test`：`64 passed`。
11. bundle runtime `--idle --duration 2` 和默认 example runtime
    `--idle --duration 2` 均正常结束；默认正手 feed `--duration 4` 正常进入
    swing/follow-through。

## 训练保存点指标

TensorBoard step 21500：

```text
mean_reward             14.143363
mean_episode_length    495.280 / 500
racket_pos_error         0.280438 m
racket_vel_error         0.691190 m/s
racket_normal_error      0.312212 rad
contact_rate             0.996255
net_clear_rate           0.960510
return_success           0.947412
both_feet_contact        0.969765
base_tilt                0.068573 rad
station_error            0.050099 m
curriculum_progress      1.0
```

`return_success` 是训练仿真 exact-strike 尝试中，满足触球、过网和对方台首跳的累计比例；
它不是实机成功率，也没有分别拆分正手和反手。

## MuJoCo 记录

使用原始训练导出 ONNX、当前 ActionAdapter/PD 和 READY：

- 10 秒无命令 READY：
  - 500 control ticks；
  - 最低 pelvis 高度 `1.065332 m`；
  - 最大站位漂移 `0.021174 m`；
  - 未低于 `0.5 m`，未跌倒。
- current-training-planner smoke：
  - 8/8 合法来球；
  - 8/8 MuJoCo 实际触球；
  - 8/8 过网并在对方台首跳；
  - 0 次跌倒；
  - raw action 最大绝对值 `7.685287`，0 个 control tick 触发 `[-100,100]` raw clip。

原始报告保存在 `validation/`。

## 尚未作为通过项

- 未完成真机执行、厂商 KP/KD 审核、现场 world/table 标定、执行延迟标定或安全认证。
- 训练/MuJoCo 成功率不能替代实机软球测试。
- 当前总成功率没有分别报告正手和反手，部署验收应按动作侧分别统计。

可重复执行离线验证：

```bash
cd /home/fsr/HOPE
/home/fsr/miniconda3/envs/hope/bin/python \
  a3_deploy/a3_pingpong_deploy_bundle/tools/verify_bundle.py
```
