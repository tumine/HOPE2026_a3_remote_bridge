#!/usr/bin/env python3
"""Parse waist_diag logs and render tracking-error plots without Matplotlib."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
LINE_RE = re.compile(
    r"\[INFO\] \[(?P<stamp>[0-9.]+)\].*?waist_diag "
    r"mode=(?P<mode>\S+) result=(?P<result>\S+) active=(?P<active>yes|no).*?"
    r"policy_raw=\[(?P<policy>[^]]+)\] "
    r"q_exec_rad=\[(?P<q_exec>[^]]+)\] "
    r"q_feedback_rad=\[(?P<q_feedback>[^]]+)\] "
    r"tau_theoretical_nm=\[(?P<tau_theoretical>[^]]+)\] "
    r"tau_feedback_nm=\[(?P<tau_feedback>[^]]+)\]"
)


def parse_vector(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid three-element vector: {text}")
    return values  # type: ignore[return-value]


def parse_log(path: Path) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    parsed_total = 0
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = LINE_RE.search(line)
        if not match:
            continue
        parsed_total += 1
        row: dict[str, object] = {
            "line": line_number,
            "stamp": float(match.group("stamp")),
            "mode": match.group("mode"),
            "result": match.group("result"),
            "active": match.group("active") == "yes",
        }
        for key in ("policy", "q_exec", "q_feedback", "tau_theoretical", "tau_feedback"):
            row[key] = parse_vector(match.group(key))
        if row["mode"] == "motion" and row["active"] and row["result"] == "command_sent":
            rows.append(row)
    if not rows:
        raise RuntimeError("no active motion command_sent waist_diag samples found")
    first_stamp = float(rows[0]["stamp"])
    for row in rows:
        row["time_s"] = float(row["stamp"]) - first_stamp
    return rows, parsed_total


def percentile_abs(values: list[float], percentile: float) -> float:
    ordered = sorted(abs(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def metrics(values: list[float], times: list[float]) -> dict[str, float]:
    absolute = [abs(value) for value in values]
    max_index = max(range(len(values)), key=lambda index: absolute[index])
    return {
        "bias": statistics.fmean(values),
        "mae": statistics.fmean(absolute),
        "rmse": math.sqrt(statistics.fmean(value * value for value in values)),
        "p95_abs": percentile_abs(values, 0.95),
        "max_abs": absolute[max_index],
        "max_time_s": times[max_index],
    }


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT = load_font(20)
SMALL_FONT = load_font(16)
TITLE_FONT = load_font(29, bold=True)
PANEL_FONT = load_font(19, bold=True)


def nice_range(series: list[list[float]]) -> tuple[float, float]:
    values = [value for one_series in series for value in one_series]
    low, high = min(values), max(values)
    if low == high:
        padding = max(abs(low) * 0.1, 1.0)
    else:
        padding = (high - low) * 0.12
    return low - padding, high + padding


def draw_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    times: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    ylabel: str,
    zero_line: bool = False,
    reference_lines: list[tuple[str, float, tuple[int, int, int]]] | None = None,
) -> None:
    left, top, right, bottom = bounds
    plot_left, plot_top = left + 88, top + (76 if len(series) > 1 else 48)
    plot_right, plot_bottom = right - 24, bottom - 55
    range_series = [values for _, values, _ in series]
    if reference_lines:
        range_series.append([value for _, value, _ in reference_lines])
    y_min, y_max = nice_range(range_series)
    x_min, x_max = times[0], times[-1]
    if x_max <= x_min:
        x_max = x_min + 1.0

    def x_px(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def y_px(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    draw.rectangle((left, top, right, bottom), fill=(250, 251, 253), outline=(205, 210, 218), width=1)
    draw.text((left + 14, top + 10), title, font=PANEL_FONT, fill=(28, 35, 48))
    for index in range(5):
        fraction = index / 4
        y_value = y_min + fraction * (y_max - y_min)
        y = y_px(y_value)
        draw.line((plot_left, y, plot_right, y), fill=(222, 226, 232), width=1)
        label = f"{y_value:.2f}"
        draw.text((plot_left - 10, y), label, font=SMALL_FONT, fill=(75, 82, 94), anchor="rm")
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = x_px(x_value)
        draw.line((x, plot_top, x, plot_bottom), fill=(235, 237, 241), width=1)
        draw.text((x, plot_bottom + 8), f"{x_value:.0f}", font=SMALL_FONT,
                  fill=(75, 82, 94), anchor="ma")
    if zero_line and y_min <= 0.0 <= y_max:
        y = y_px(0.0)
        draw.line((plot_left, y, plot_right, y), fill=(80, 85, 92), width=2)
    if reference_lines:
        for label, value, color in reference_lines:
            if y_min <= value <= y_max:
                y = y_px(value)
                draw.line((plot_left, y, plot_right, y), fill=color, width=2)
                draw.text((plot_right - 4, y - 4), label, font=SMALL_FONT,
                          fill=color, anchor="rs")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(80, 85, 92), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(80, 85, 92), width=1)
    draw.text((left + 18, (plot_top + plot_bottom) / 2), ylabel, font=SMALL_FONT,
              fill=(60, 67, 78), anchor="mm")
    draw.text(((plot_left + plot_right) / 2, bottom - 17), "time from first active sample (s)",
              font=SMALL_FONT, fill=(60, 67, 78), anchor="mm")
    if len(series) > 1:
        legend_x = plot_left
        for label, _, color in series:
            text_width = draw.textbbox((0, 0), label, font=SMALL_FONT)[2]
            draw.line((legend_x, top + 54, legend_x + 20, top + 54), fill=color, width=4)
            draw.text((legend_x + 25, top + 54), label, font=SMALL_FONT,
                      fill=(45, 52, 63), anchor="lm")
            legend_x += text_width + 55
    for _, values, color in series:
        points = [(x_px(time), y_px(value)) for time, value in zip(times, values)]
        intervals = [right - left for left, right in zip(times, times[1:])]
        typical_interval = statistics.median(intervals) if intervals else 0.0
        segment: list[tuple[float, float]] = []
        for index, point in enumerate(points):
            if (index > 0 and typical_interval > 0.0 and
                    times[index] - times[index - 1] > 2.5 * typical_interval):
                if len(segment) > 1:
                    draw.line(segment, fill=color, width=3, joint="curve")
                segment = []
            segment.append(point)
        if len(segment) > 1:
            draw.line(segment, fill=color, width=3, joint="curve")
        for x, y in points:
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=color)


def render_overview(rows: list[dict[str, object]], output: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    image = Image.new("RGB", (1900, 1450), (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.text((950, 34), "A3 Waist Command vs Feedback (active motion samples)",
              font=TITLE_FONT, fill=(22, 30, 44), anchor="ma")
    q_exec = [[row["q_exec"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    q_feedback = [[row["q_feedback"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    tau_theoretical = [[row["tau_theoretical"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    tau_feedback = [[row["tau_feedback"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    for index, joint in enumerate(JOINTS):
        top = 90 + index * 440
        draw_panel(draw, (30, top, 935, top + 410), times,
                   [("command", q_exec[index], (30, 104, 211)),
                    ("feedback", q_feedback[index], (226, 84, 62))],
                   f"{joint}: position", "rad")
        draw_panel(draw, (965, top, 1870, top + 410), times,
                   [("theoretical", tau_theoretical[index], (112, 70, 190)),
                    ("feedback", tau_feedback[index], (25, 145, 108))],
                   f"{joint}: torque", "N m")
    image.save(output, optimize=True)


def render_policy_command_feedback(rows: list[dict[str, object]], output: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    image = Image.new("RGB", (1900, 1450), (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.text((950, 34), "A3 Waist Policy Output, Position Command, and Feedback",
              font=TITLE_FONT, fill=(22, 30, 44), anchor="ma")
    policy = [[row["policy"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    q_exec = [[row["q_exec"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    q_feedback = [[row["q_feedback"][i] for row in rows] for i in range(3)]  # type: ignore[index]
    for index, joint in enumerate(JOINTS):
        top = 90 + index * 440
        draw_panel(draw, (30, top, 935, top + 410), times,
                   [("policy_raw", policy[index], (112, 70, 190))],
                   f"{joint}: raw policy output", "raw", zero_line=True)
        draw_panel(draw, (965, top, 1870, top + 410), times,
                   [("q_exec target", q_exec[index], (30, 104, 211)),
                    ("q_feedback", q_feedback[index], (226, 84, 62))],
                   f"{joint}: commanded target vs measured position", "rad")
    image.save(output, optimize=True)


def render_pitch_clip_check(rows: list[dict[str, object]], output: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    raw_target = [row["policy"][2] * 0.25 for row in rows]  # type: ignore[index,operator]
    q_exec = [row["q_exec"][2] for row in rows]  # type: ignore[index]
    q_feedback = [row["q_feedback"][2] for row in rows]  # type: ignore[index]
    clip_removed = [max(raw - executed, 0.0) for raw, executed in zip(raw_target, q_exec)]
    tracking_error = [executed - feedback for executed, feedback in zip(q_exec, q_feedback)]

    image = Image.new("RGB", (1900, 980), (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.text((950, 34), "Waist Pitch 0.29 rad Upper-Clip Verification",
              font=TITLE_FONT, fill=(22, 30, 44), anchor="ma")
    draw_panel(
        draw, (30, 90, 1870, 500), times,
        [("raw * 0.25 (unclipped)", raw_target, (112, 70, 190)),
         ("q_exec target", q_exec, (30, 104, 211)),
         ("q_feedback", q_feedback, (226, 84, 62))],
        "Policy-derived target, executed target, and measured position",
        "rad", zero_line=True,
        reference_lines=[("upper clip = 0.290 rad", 0.29, (190, 45, 55))],
    )
    draw_panel(
        draw, (30, 530, 1870, 940), times,
        [("amount removed by clip", clip_removed, (112, 70, 190)),
         ("q_exec - q_feedback", tracking_error, (211, 69, 90))],
        "Clip intervention and position tracking error",
        "rad", zero_line=True,
    )
    image.save(output, optimize=True)


def render_errors(rows: list[dict[str, object]], summary: dict[str, dict[str, dict[str, float]]], output: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    image = Image.new("RGB", (1900, 1450), (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.text((950, 34), "A3 Waist Tracking Errors (command-feedback)",
              font=TITLE_FONT, fill=(22, 30, 44), anchor="ma")
    for index, joint in enumerate(JOINTS):
        position_error_deg = [
            math.degrees(row["q_exec"][index] - row["q_feedback"][index])  # type: ignore[index]
            for row in rows
        ]
        torque_error = [
            row["tau_theoretical"][index] - row["tau_feedback"][index]  # type: ignore[index]
            for row in rows
        ]
        top = 90 + index * 440
        pm = summary[joint]["position_deg"]
        tm = summary[joint]["torque_nm"]
        draw_panel(draw, (30, top, 935, top + 410), times,
                   [("q_cmd - q_fb", position_error_deg, (211, 69, 90))],
                   f"{joint}: position error | MAE {pm['mae']:.2f} deg | RMSE {pm['rmse']:.2f} | max {pm['max_abs']:.2f}",
                   "deg", zero_line=True)
        draw_panel(draw, (965, top, 1870, top + 410), times,
                   [("tau_pd - tau_fb", torque_error, (230, 126, 34))],
                   f"{joint}: torque discrepancy | MAE {tm['mae']:.2f} N m | RMSE {tm['rmse']:.2f} | max {tm['max_abs']:.2f}",
                   "N m", zero_line=True)
    image.save(output, optimize=True)


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    fields = ["source_line", "stamp", "time_s", "mode", "result", "active"]
    for joint in JOINTS:
        fields.extend((f"{joint}_policy_raw", f"{joint}_q_exec_rad", f"{joint}_q_feedback_rad",
                       f"{joint}_position_error_rad", f"{joint}_position_error_deg",
                       f"{joint}_tau_theoretical_nm", f"{joint}_tau_feedback_nm",
                       f"{joint}_torque_discrepancy_nm"))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat: dict[str, object] = {
                "source_line": row["line"], "stamp": row["stamp"], "time_s": row["time_s"],
                "mode": row["mode"], "result": row["result"], "active": row["active"],
            }
            for index, joint in enumerate(JOINTS):
                position_error = row["q_exec"][index] - row["q_feedback"][index]  # type: ignore[index]
                torque_error = row["tau_theoretical"][index] - row["tau_feedback"][index]  # type: ignore[index]
                flat.update({
                    f"{joint}_policy_raw": row["policy"][index],  # type: ignore[index]
                    f"{joint}_q_exec_rad": row["q_exec"][index],  # type: ignore[index]
                    f"{joint}_q_feedback_rad": row["q_feedback"][index],  # type: ignore[index]
                    f"{joint}_position_error_rad": position_error,
                    f"{joint}_position_error_deg": math.degrees(position_error),
                    f"{joint}_tau_theoretical_nm": row["tau_theoretical"][index],  # type: ignore[index]
                    f"{joint}_tau_feedback_nm": row["tau_feedback"][index],  # type: ignore[index]
                    f"{joint}_torque_discrepancy_nm": torque_error,
                })
            writer.writerow(flat)


def build_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    times = [float(row["time_s"]) for row in rows]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for index, joint in enumerate(JOINTS):
        position_rad = [row["q_exec"][index] - row["q_feedback"][index] for row in rows]  # type: ignore[index]
        position_deg = [math.degrees(value) for value in position_rad]
        torque_nm = [row["tau_theoretical"][index] - row["tau_feedback"][index] for row in rows]  # type: ignore[index]
        summary[joint] = {
            "position_rad": metrics(position_rad, times),
            "position_deg": metrics(position_deg, times),
            "torque_nm": metrics(torque_nm, times),
        }
    return summary


def write_summary(
    rows: list[dict[str, object]], parsed_total: int,
    summary: dict[str, dict[str, dict[str, float]]], output: Path,
) -> None:
    times = [float(row["time_s"]) for row in rows]
    intervals = [right - left for left, right in zip(times, times[1:])]
    lines = [
        "# Waist log tracking-error summary", "",
        f"- Parsed waist_diag samples: {parsed_total}",
        f"- Active motion samples used: {len(rows)}",
        f"- Active data span: {times[-1]:.3f} s",
        f"- Median log interval: {statistics.median(intervals):.3f} s" if intervals else "- Median log interval: n/a",
        "- Position error definition: q_exec - q_feedback",
        "- Torque discrepancy definition: tau_theoretical - tau_feedback", "",
        "| Joint | Pos MAE (deg) | Pos RMSE (deg) | Pos P95 (deg) | Pos max (deg @ s) | Torque MAE (N m) | Torque RMSE (N m) | Torque P95 (N m) | Torque max (N m @ s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for joint in JOINTS:
        position = summary[joint]["position_deg"]
        torque = summary[joint]["torque_nm"]
        lines.append(
            f"| {joint} | {position['mae']:.3f} | {position['rmse']:.3f} | "
            f"{position['p95_abs']:.3f} | {position['max_abs']:.3f} @ {position['max_time_s']:.1f} | "
            f"{torque['mae']:.3f} | {torque['rmse']:.3f} | {torque['p95_abs']:.3f} | "
            f"{torque['max_abs']:.3f} @ {torque['max_time_s']:.1f} |"
        )
    lines.extend(("", "The 1 Hz status log is a downsampled snapshot of the 50 Hz control loop; transient peaks may be missed."))
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, parsed_total = parse_log(args.log)
    summary = build_summary(rows)
    write_csv(rows, args.output_dir / "waist_active_samples.csv")
    write_summary(rows, parsed_total, summary, args.output_dir / "waist_error_summary.md")
    render_overview(rows, args.output_dir / "waist_tracking_overview.png")
    render_policy_command_feedback(
        rows, args.output_dir / "waist_policy_command_feedback.png")
    render_pitch_clip_check(rows, args.output_dir / "waist_pitch_clip_check.png")
    render_errors(rows, summary, args.output_dir / "waist_error_dashboard.png")
    print((args.output_dir / "waist_error_summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
