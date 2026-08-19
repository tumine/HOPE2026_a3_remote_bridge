#!/usr/bin/env python3
"""Print concise Chinese ball-admission events from HOPE planner logs."""

from __future__ import annotations

import argparse
import re
import sys


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PUBLISHED_RE = re.compile(
    r"PLANNER_TASK_PUBLISHED ball=(?P<ball>\d+) task=(?P<task>\d+) "
    r"side=(?P<side>\w+) tts=(?P<tts>-?[\d.]+)s "
    r"target=(?P<target>\[[^]]+\]) racket_v=(?P<velocity>\[[^]]+\])"
)
DROP_RE = re.compile(
    r"TTS_GATE ball=(?P<ball>\d+) side=(?P<side>\w+) decision=DROP "
    r"tts=(?P<tts>-?[\d.]+)s window=\[(?P<minimum>-?[\d.]+), "
    r"(?P<maximum>-?[\d.]+)\]s"
)
CROSS_RE = re.compile(
    r"STRIKE_PLANE_CROSS ball=(?P<ball>\d+) plane=(?P<side>\w+) "
    r"x=(?P<x>-?[\d.]+) measured_y=(?P<y>-?[\d.]+) "
    r"measured_z=(?P<z>-?[\d.]+) result=(?P<result>.+)$"
)


def side_cn(side: str) -> str:
    return {"FOREHAND": "正手", "BACKHAND": "反手"}.get(side, side)


def reason_cn(reason: str) -> str:
    text = reason.strip()
    prefixes = ("REJECT: ", "WAIT: ", "ACCEPT: ")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break

    replacements = (
        ("estimated ball velocity is not incoming (vx >= 0)", "球速方向无效（vx>=0，球未朝机器人运动）"),
        ("no usable future crossing", "预测时间内没有可用的击球面交点"),
        ("plane not reached or dead-ball crossing", "未到达击球面或已成为死球"),
        ("predicted position", "预测击球位置"),
        ("predicted y", "预测横向位置 y"),
        ("classified as", "被分到"),
        ("active task is locked to", "当前任务已锁定为"),
        ("time_to_strike", "剩余击球时间TTS"),
        ("is too late for admission window", "低于新任务允许窗口"),
        ("physical ball dropped", "本次来球已丢弃"),
        ("outside", "超出允许范围"),
        ("FOREHAND", "正手"),
        ("BACKHAND", "反手"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


class PlannerEventFormatter:
    """Translate planner decisions while avoiding rejection-after-accept noise."""

    def __init__(self) -> None:
        self._allowed_balls: set[str] = set()

    def event_for_line(self, line: str) -> str | None:
        clean = ANSI_ESCAPE_RE.sub("", line).strip()

        match = PUBLISHED_RE.search(clean)
        if match:
            values = match.groupdict()
            self._allowed_balls.add(values["ball"])
            return (
                f"[球事件] 球ID={values['ball']} 结果=允许 "
                f"侧别={side_cn(values['side'])} 任务ID={values['task']} "
                f"TTS={values['tts']}s 原因=击球位置、目标拍速和TTS均通过；"
                f"目标位置={values['target']} 目标拍速={values['velocity']}"
            )

        match = DROP_RE.search(clean)
        if match:
            values = match.groupdict()
            if values["ball"] in self._allowed_balls:
                return None
            return (
                f"[球事件] 球ID={values['ball']} 结果=拒绝 "
                f"侧别={side_cn(values['side'])} 原因=TTS={values['tts']}s，"
                f"低于允许窗口[{values['minimum']},{values['maximum']}]s"
            )

        match = CROSS_RE.search(clean)
        if match:
            values = match.groupdict()
            result = values["result"]
            if (
                values["ball"] in self._allowed_balls
                or not result.startswith("REJECT:")
            ):
                return None
            return (
                f"[球事件] 球ID={values['ball']} 结果=拒绝 "
                f"检测面={side_cn(values['side'])} 实测位置="
                f"[x={values['x']},y={values['y']},z={values['z']}] "
                f"原因={reason_cn(result)}"
            )

        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="print each original planner line before its Chinese event",
    )
    args = parser.parse_args()
    formatter = PlannerEventFormatter()

    try:
        for line in sys.stdin:
            if args.passthrough:
                print(line, end="", flush=True)
            event = formatter.event_for_line(line)
            if event is not None:
                print(event, flush=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
