# 数字发球轨迹

`run_hope_lower_body.sh` 默认把本目录传给
`hope_lower_body_deploy --serve-tracks-dir`。程序运行后按数字键会读取同名
YAML，校验成功后从实测关节位置平滑移动到该轨迹的 `serve.home`；看到
`arm: ready/home` 后可继续使用 `C/F` 等原有命令。

| 数字 | 原始轨迹 | 说明 |
| --- | --- | --- |
| `1` | `192.168.1.100/track-r-1f-y-10.yaml` | 实机最终轨迹；swing 0.12 s，release -0.15 s |
| `2` | `192.168.1.100/track-r-1f-f.yaml` | 实机最终轨迹；swing 0.06 s，release -0.18 s |
| `3` | `192.168.1.101/track-r-1f.yaml` | 实机最终轨迹；swing 0.10 s，release -0.17 s |
| `4` | `192.168.1.101/track-r-1-wind-right-enhanced1.yaml` | 实机最终增强引拍轨迹；swing 0.13 s，release -0.15 s |
| `5` | `192.168.1.101/track-r-1-wind-right-enhanced4.yaml` | 实机最终增强轨迹；swing 0.16 s，release -0.15 s |
| `6` | `track-r-1-z40.yaml` | 拍面朝 world +Z 40° |

原始文件中明确标注“不可用于发球”的 `track-r-1.yaml` 和
`track-r-1-z30.yaml` 未纳入数字轨迹。`7–9` 和 `0` 目前没有对应文件，
按下后只会报告加载失败，不会改变当前轨迹或触发手臂动作。

可用 `A3_SERVE_TRACKS_DIR=/path/to/tracks` 覆盖默认目录。目录内的数字
YAML 必须是可由 `LoadServeConfig` 完整校验的 `serve:` 配置。

完整接发球 MuJoCo 闭环绑定实机最终确认的 `1`～`5`。在接球 READY
状态直接按数字会选择对应 YAML 并自动回 Home，随后使用 `C/F` 发球；`G`
仍可随时请求夹爪打开。发球结束后自动平滑恢复 model_72500 全身接球控制。
