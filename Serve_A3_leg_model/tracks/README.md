# 数字发球轨迹

`run_hope_lower_body.sh` 默认把本目录传给
`hope_lower_body_deploy --serve-tracks-dir`。程序运行后按数字键会读取同名
YAML，校验成功后从实测关节位置平滑移动到该轨迹的 `serve.home`；看到
`arm: ready/home` 后可继续使用 `C/F` 等原有命令。

| 数字 | 原始轨迹 | 说明 |
| --- | --- | --- |
| `1` | `track-r-1f.yaml` | 基准可行长球 |
| `2` | `track-r-1f-y-5.yaml` | 拍面朝 world -Y 5° |
| `3` | `track-r-1f-y-10.yaml` | 拍面朝 world -Y 10° |
| `4` | `track-r-1f-y-15.yaml` | 拍面朝 world -Y 15° |
| `5` | `track-r-1-z20.yaml` | 拍面朝 world +Z 20° |
| `6` | `track-r-1-z40.yaml` | 拍面朝 world +Z 40° |

原始文件中明确标注“不可用于发球”的 `track-r-1.yaml` 和
`track-r-1-z30.yaml` 未纳入数字轨迹。`7–9` 和 `0` 目前没有对应文件，
按下后只会报告加载失败，不会改变当前轨迹或触发手臂动作。

可用 `A3_SERVE_TRACKS_DIR=/path/to/tracks` 覆盖默认目录。目录内的数字
YAML 必须是可由 `LoadServeConfig` 完整校验的 `serve:` 配置。
