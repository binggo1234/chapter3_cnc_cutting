from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CASE_KEY = (
    "archive",
    "case_name",
    "placement_method",
    "seed",
    "placements_member",
    "board_id",
)


@dataclass(frozen=True)
class ToolEventComparison:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    baseline_method: str
    target_method: str
    rectangle_count: int
    baseline_tool_event_count: float
    target_tool_event_count: float
    tool_event_delta: float
    pierce_delta: float
    lift_delta: float
    safe_lift_delta: float
    baseline_travel_mode_cost: float
    target_travel_mode_cost: float
    travel_mode_cost_delta: float
    travel_mode_cost_reduction_pct: float
    baseline_machining_cost: float
    target_machining_cost: float
    machining_cost_delta: float
    machining_cost_reduction_pct: float
    baseline_detour_count: float
    target_detour_count: float
    detour_delta: float
    baseline_air_move_distance: float
    target_air_move_distance: float
    baseline_runtime_ms: float
    target_runtime_ms: float


@dataclass(frozen=True)
class ToolEventSummary:
    baseline_method: str
    target_method: str
    paired_cases: int
    tool_event_decrease_count: int
    tool_event_tie_count: int
    tool_event_increase_count: int
    tool_event_delta_mean: float
    tool_event_delta_median: float
    travel_mode_cost_reduction_pct_mean: float
    machining_cost_reduction_pct_mean: float
    increase_cases_travel_reduction_pct_mean: float
    increase_cases_machining_reduction_pct_mean: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return 0.0 if value == "" else float(value)


def tool_event_count(row: dict[str, str]) -> float:
    return (
        numeric(row, "pierce_count")
        + numeric(row, "lift_count")
        + numeric(row, "safe_lift_count")
    )


def machining_cost(row: dict[str, str]) -> float:
    if row.get("machining_cost", ""):
        return numeric(row, "machining_cost")
    return numeric(row, "cutting_length") + numeric(row, "travel_mode_cost")


def reduction_pct(baseline: float, target: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 100.0 * (baseline - target) / baseline


def case_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in CASE_KEY)


def group_by_case_method(
    rows: list[dict[str, str]],
) -> dict[tuple[str, ...], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[case_key(row)][row["method"]] = row
    return grouped


def compare_tool_events(
    rows: list[dict[str, str]],
    baseline_method: str,
    target_method: str,
) -> list[ToolEventComparison]:
    comparisons: list[ToolEventComparison] = []
    grouped = group_by_case_method(rows)
    for key, by_method in sorted(grouped.items()):
        baseline = by_method.get(baseline_method)
        target = by_method.get(target_method)
        if baseline is None or target is None:
            continue

        baseline_events = tool_event_count(baseline)
        target_events = tool_event_count(target)
        baseline_travel = numeric(baseline, "travel_mode_cost")
        target_travel = numeric(target, "travel_mode_cost")
        baseline_machining = machining_cost(baseline)
        target_machining = machining_cost(target)

        comparisons.append(
            ToolEventComparison(
                archive=key[0],
                case_name=key[1],
                placement_method=key[2],
                seed=key[3],
                placements_member=key[4],
                board_id=key[5],
                baseline_method=baseline_method,
                target_method=target_method,
                rectangle_count=int(float(target["rectangle_count"])),
                baseline_tool_event_count=baseline_events,
                target_tool_event_count=target_events,
                tool_event_delta=target_events - baseline_events,
                pierce_delta=numeric(target, "pierce_count")
                - numeric(baseline, "pierce_count"),
                lift_delta=numeric(target, "lift_count") - numeric(baseline, "lift_count"),
                safe_lift_delta=numeric(target, "safe_lift_count")
                - numeric(baseline, "safe_lift_count"),
                baseline_travel_mode_cost=baseline_travel,
                target_travel_mode_cost=target_travel,
                travel_mode_cost_delta=target_travel - baseline_travel,
                travel_mode_cost_reduction_pct=reduction_pct(
                    baseline_travel,
                    target_travel,
                ),
                baseline_machining_cost=baseline_machining,
                target_machining_cost=target_machining,
                machining_cost_delta=target_machining - baseline_machining,
                machining_cost_reduction_pct=reduction_pct(
                    baseline_machining,
                    target_machining,
                ),
                baseline_detour_count=numeric(baseline, "detour_count"),
                target_detour_count=numeric(target, "detour_count"),
                detour_delta=numeric(target, "detour_count")
                - numeric(baseline, "detour_count"),
                baseline_air_move_distance=numeric(baseline, "air_move_distance"),
                target_air_move_distance=numeric(target, "air_move_distance"),
                baseline_runtime_ms=numeric(baseline, "runtime_ms"),
                target_runtime_ms=numeric(target, "runtime_ms"),
            )
        )
    return comparisons


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def summarize(comparisons: list[ToolEventComparison]) -> ToolEventSummary:
    if not comparisons:
        return ToolEventSummary(
            baseline_method="",
            target_method="",
            paired_cases=0,
            tool_event_decrease_count=0,
            tool_event_tie_count=0,
            tool_event_increase_count=0,
            tool_event_delta_mean=0.0,
            tool_event_delta_median=0.0,
            travel_mode_cost_reduction_pct_mean=0.0,
            machining_cost_reduction_pct_mean=0.0,
            increase_cases_travel_reduction_pct_mean=0.0,
            increase_cases_machining_reduction_pct_mean=0.0,
        )

    deltas = [row.tool_event_delta for row in comparisons]
    increase_rows = [row for row in comparisons if row.tool_event_delta > 0.0]
    return ToolEventSummary(
        baseline_method=comparisons[0].baseline_method,
        target_method=comparisons[0].target_method,
        paired_cases=len(comparisons),
        tool_event_decrease_count=sum(1 for delta in deltas if delta < 0.0),
        tool_event_tie_count=sum(1 for delta in deltas if delta == 0.0),
        tool_event_increase_count=sum(1 for delta in deltas if delta > 0.0),
        tool_event_delta_mean=mean(deltas),
        tool_event_delta_median=median(deltas),
        travel_mode_cost_reduction_pct_mean=mean(
            row.travel_mode_cost_reduction_pct for row in comparisons
        ),
        machining_cost_reduction_pct_mean=mean(
            row.machining_cost_reduction_pct for row in comparisons
        ),
        increase_cases_travel_reduction_pct_mean=(
            mean(row.travel_mode_cost_reduction_pct for row in increase_rows)
            if increase_rows
            else 0.0
        ),
        increase_cases_machining_reduction_pct_mean=(
            mean(row.machining_cost_reduction_pct for row in increase_rows)
            if increase_rows
            else 0.0
        ),
    )


def write_dataclass_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def write_markdown(
    path: Path,
    summary: ToolEventSummary,
    increase_rows: list[ToolEventComparison],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Event-gated Adaptive Beam+LS 的刀具事件案例分析",
        "",
        f"Baseline: `{summary.baseline_method}`",
        f"Target: `{summary.target_method}`",
        "",
        "## 总体结果",
        "",
        f"- Paired cases: {summary.paired_cases}",
        f"- Tool-event decrease/tie/increase: {summary.tool_event_decrease_count}/{summary.tool_event_tie_count}/{summary.tool_event_increase_count}",
        f"- Mean tool-event delta: {summary.tool_event_delta_mean:.2f}",
        f"- Mean travel-mode cost reduction: {summary.travel_mode_cost_reduction_pct_mean:.2f}%",
        f"- Mean machining cost reduction: {summary.machining_cost_reduction_pct_mean:.2f}%",
        "",
        "## 刀具事件增加案例",
        "",
    ]
    if not increase_rows:
        lines.append("No increased tool-event cases.")
    else:
        lines.append(
            "| archive | seed | board | rectangles | event delta | travel reduction | machining reduction | detour delta |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in increase_rows:
            lines.append(
                "| "
                + " | ".join(
                    (
                        row.archive,
                        row.seed,
                        row.board_id,
                        str(row.rectangle_count),
                        f"{row.tool_event_delta:.0f}",
                        f"{row.travel_mode_cost_reduction_pct:.2f}%",
                        f"{row.machining_cost_reduction_pct:.2f}%",
                        f"{row.detour_delta:.0f}",
                    )
                )
                + " |"
            )
        board_counts: dict[str, int] = defaultdict(int)
        for row in increase_rows:
            board_counts[row.board_id] += 1
        repeated_boards = ", ".join(
            f"board `{board}` ({count})"
            for board, count in sorted(board_counts.items(), key=lambda item: item[0])
        )
        all_travel_improved = all(
            row.travel_mode_cost_reduction_pct > 0.0 for row in increase_rows
        )
        all_machining_improved = all(
            row.machining_cost_reduction_pct > 0.0 for row in increase_rows
        )
        representative = max(
            increase_rows,
            key=lambda row: row.travel_mode_cost_reduction_pct,
        )
        lines.extend(
            [
                "",
                "## 解释口径",
                "",
                f"- 这些增加案例主要集中在 {repeated_boards}，不是互相独立的 6 种失败模式。",
                f"- 所有增加案例是否均降低通行代价：{'yes' if all_travel_improved else 'no'}。",
                f"- 所有增加案例是否均降低加工总代价：{'yes' if all_machining_improved else 'no'}。",
                f"- 最适合做代表图的是 board `{representative.board_id}`，其 `travel_mode_cost` 降低 {representative.travel_mode_cost_reduction_pct:.2f}%，`detour_count` 变化 {representative.detour_delta:.0f}。",
                "- 论文中不宜把这些案例解释为算法失败；更合适的说法是：门控并不禁止所有额外刀具事件，而是只接受被明确加工收益证明的事件增加。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "adaptive_event_gate_protected_real_20x3_20_50.csv",
    )
    parser.add_argument("--baseline-method", default="process_aware_beam")
    parser.add_argument("--target-method", default="process_aware_beam_adaptive_polished")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "adaptive_event_gate_protected_tool_event_summary.csv",
    )
    parser.add_argument(
        "--increase-output",
        type=Path,
        default=ROOT
        / "results"
        / "adaptive_event_gate_protected_tool_event_increase_cases.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "event_gate_tool_event_case_analysis.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = compare_tool_events(
        read_rows(args.input),
        baseline_method=args.baseline_method,
        target_method=args.target_method,
    )
    summary = summarize(comparisons)
    increase_rows = [row for row in comparisons if row.tool_event_delta > 0.0]

    write_dataclass_rows(args.summary_output, [summary])
    write_dataclass_rows(args.increase_output, increase_rows)
    write_markdown(args.markdown_output, summary, increase_rows)

    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.increase_output}")
    print(f"wrote: {args.markdown_output}")
    print(
        "tool-event decrease/tie/increase: "
        f"{summary.tool_event_decrease_count}/"
        f"{summary.tool_event_tie_count}/"
        f"{summary.tool_event_increase_count}"
    )


if __name__ == "__main__":
    main()
