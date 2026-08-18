# A3 Planner UDP 协议

跨 PC/MDU 的 planner 数据面默认使用 A3PP v1。ROS2 仅在 PC 本机用于动捕、
gateway 和 HOPE planner 之间的通信。

## 网络参数

- PC 源地址：`192.168.1.11`
- MDU 目标地址：`192.168.1.100`
- UDP 端口：`15001`
- 发送频率：`50 Hz`
- 固定 UDP payload：`96 bytes`
- 多字节字段：little-endian；浮点数：IEEE-754 `float32`

96 字节包由以下字段顺序组成：

| 偏移 | 类型 | 字段 |
| ---: | --- | --- |
| 0 | `uint32` | magic，线上字节为 `A3PP` |
| 4 | `uint8` | version，当前为 1 |
| 5 | `uint8` | flags：bit0 pose valid，bit1 command valid |
| 6 | `uint16` | packet length，固定 96 |
| 8 | `uint32` | sender session ID |
| 12 | `uint32` | packet sequence |
| 16 | `uint32` | pose sequence |
| 20 | `float32[3]` | pelvis position XYZ |
| 32 | `float32[4]` | pelvis quaternion WXYZ |
| 48 | `uint64` | task ID |
| 56 | `uint32` | task revision |
| 60 | `int8` | swing side，-1 或 +1 |
| 61 | 3 bytes | reserved，必须为 0 |
| 64 | `int32` | time to strike，微秒 |
| 68 | `float32[3]` | racket target position XYZ |
| 80 | `float32[3]` | racket target velocity XYZ |
| 92 | `uint32` | 前 92 字节的 CRC32 |

协议版本固定坐标系为 `hope_table`。PC 在每次发包前重新计算剩余 TTS，MDU
从收到包的本地单调时钟继续扣减，因此两台机器不需要进行墙上时钟同步。

## 启动

PC：

```bash
./scripts/run_local_receive_planner.sh
```

MDU：

```bash
cd /home/agi/a3_remote_bridge_probe
./scripts/run_mdu_planner_receiver.sh
```

脚本默认选择 UDP。临时回退到旧 ROS2 跨机输入：

```bash
A3_PLANNER_INPUT_TRANSPORT=ros2 ./scripts/run_mdu_planner_receiver.sh
```

地址、端口和频率可以通过 `A3_PC_WIRED_ADDRESS`、`A3_MDU_ADDRESS`、
`A3_PLANNER_UDP_PORT` 和 `A3_PLANNER_UDP_HZ` 覆盖。频率上限为 100 Hz，默认
50 Hz 的估算以太网线速为 `0.065 Mbps`。

MDU 只接受配置的 PC 源 IP，并检查包长、版本、CRC、session/sequence、有限值、
四元数、task/revision 和 TTS。pose sequence 停止时不会刷新 mailbox，因此动捕
冻结仍会触发原来的位姿 watchdog。
