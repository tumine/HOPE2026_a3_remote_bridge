# HOPE model_53000 部署包

这是 `model_53000.pt` 的可部署 ONNX 包，包含策略、精确观测/动作契约、参考状态机、横向站位计算、GJC MuJoCo 模型、球物理参数和验证记录。

## 核心产物

- `policy/hope_pingpong.onnx`：部署策略，SHA256 `8f87d7d9...e080057`
- `policy/policy_manifest.json`：ONNX 输入输出和观测布局
- `config/runtime.yaml`：50 Hz 运行配置、READY 条件、横移站位与仿真 PD
- `config/action_adapter.yaml`：31 维策略输出到关节目标角的唯一映射
- `config/strike_box.yaml`：正反手击球盒、切换边界和 base 站位范围
- `config/ball_physics.yaml`：当前 planner/仿真球物理；桌面恢复系数 `0.9448`，切向阻尼 `0.2079`
- `reference/a3_deploy_onnx_ref_pingpong/`：可执行参考控制循环和状态切换代码
- `docs/DEPLOYMENT.md`：部署侧必须实现的接口与安全检查
- `validation/hard_alternating_12_new_table_physics.json`：困难左右交替球验证原始结果

## 快速校验

```bash
cd /home/fsr/HOPE/a3_deploy/model_53000_deploy_bundle
/home/fsr/miniconda3/envs/hope/bin/python scripts/verify_bundle.py
```

## MuJoCo 参考控制循环

该命令展示策略控制和状态切换，不生成完整物理发球：

```bash
cd /home/fsr/HOPE/a3_deploy/model_53000_deploy_bundle
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_reference_mujoco.sh --view --realtime
```

完整困难球复验（依赖相邻的 HOPE 源码仓库）：

```bash
cd /home/fsr/HOPE/a3_deploy/model_53000_deploy_bundle
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_hard_eval.sh --view
```

## 策略状态

`model_53000` 已完成本轮击球位置、击球高度和横向站位窗口课程。在导出时，push 课程约完成 55%，因此这是已通过 MuJoCo 困难球测试的部署候选，不代表已经完成最终推力鲁棒性训练或真实机器人安全认证。

策略的动作片段从 `time_to_strike=+1.0 s` 开始，经过击球时刻 `0`，跟随至约 `-0.8 s`。当前策略没有专门训练“首次收到命令时 TTS 只有 0.25 s”的分布；planner 应尽可能提前预测并持续修订，部署侧不得把短 TTS 表现视为已验证能力。

开始真实机器人测试前请完整阅读 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。
