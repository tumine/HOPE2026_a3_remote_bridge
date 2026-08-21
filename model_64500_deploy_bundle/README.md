# HOPE Ping-Pong model_64500 部署候选包

本包冻结训练节点 `model_64500.pt`，包含 ONNX 策略、GJC 精确 MuJoCo 模型、部署配置、站位计算与状态机参考实现、独立 READY 参考动作、验证脚本及训练溯源文件。

## 策略契约

- 输入：`float32[1,111]`
- 输出：`float32[1,31]`
- 控制频率：`50 Hz`
- 动作解码：`q_des = default_q + 0.25 * applied_action`
- 站位：根据正/反手击球盒计算横向 `base_target_y`，READY/RECOVERY 回到启动中央站位
- 训练命令时间：动作片段以 `TTS=1.0 s` 起始；首次命令 `TTS=0.25 s` 不在本策略已验证分布内

策略对应当前完整横向、纵向和高度击球盒，以及降低 2 cm、平脚处理后的正反手参考动作；READY 训练使用独立抬高手臂的中立参考动作。桌面碰撞参数为 restitution `0.9448`、tangential damping `0.2079`。

## 当前验证状态

10 秒无球 READY MuJoCo 测试中策略没有跌倒，最大站位漂移 `2.81 cm`，手臂在球台上方的最小保守余量约 `3.90 cm`。但姿势与中立 READY 参考仍有明显差异：腰最终为 yaw/roll/pitch `-11.16/+4.71/-2.91 deg`，左右脚倾角为 `11.76/16.34 deg`，全关节最终 RMS 误差 `0.378 rad`。

结论：这是一个 **MuJoCo READY 稳定的部署候选包**，不是已通过真机安全认证的包。详细数据见 `docs/VALIDATION.md`。

## 使用

校验所有关键文件和 ONNX 接口：

```bash
cd /home/fsr/HOPE/a3_deploy/model_64500_deploy_bundle
/home/fsr/miniconda3/envs/hope/bin/python scripts/verify_bundle.py
```

从包内复现 10 秒无球 READY 测试：

```bash
cd /home/fsr/HOPE/a3_deploy/model_64500_deploy_bundle
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_ready_eval.sh
```

增加可视化窗口：

```bash
HOPE_DEPLOY_PYTHON=/home/fsr/miniconda3/envs/hope/bin/python \
  scripts/run_ready_eval.sh --view
```

完整观测、planner、正反手站位、状态切换和实机接入要求见 `docs/DEPLOYMENT.md`。参考实现位于 `reference/a3_deploy_onnx_ref_pingpong/`，其中 `station_target.py` 负责站位计算，`lifecycle.py` 负责状态切换，`runner.py` 负责 50 Hz 策略闭环。
