from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = (
    "greedy",
    "path_distance_local_search",
    "topology",
    "topology_process_aware",
    "process_local_search_multistart",
    "process_aware_beam",
    "process_aware_beam_adaptive",
    "process_aware_beam_adaptive_polished",
    "process_aware_beam_polished",
    "topology_local_search",
    "topology_local_search_process_aware",
)
METHOD_LABELS = {
    "greedy": "Greedy",
    "path_distance_local_search": "Path-LS",
    "topology": "Topology",
    "topology_process_aware": "Process-aware",
    "process_local_search_multistart": "Multi-start process LS",
    "process_aware_beam": "Process-aware beam",
    "process_aware_beam_adaptive": "Adaptive beam portfolio",
    "process_aware_beam_adaptive_polished": "Event-gated adaptive beam+LS",
    "process_aware_beam_polished": "Beam+process LS",
    "topology_local_search": "Topology+LS",
    "topology_local_search_process_aware": "Process-aware+LS",
}
METHOD_COLORS = {
    "greedy": "#7B8794",
    "path_distance_local_search": "#CC79A7",
    "topology": "#2A9D8F",
    "topology_process_aware": "#264653",
    "process_local_search_multistart": "#8E6C8A",
    "process_aware_beam": "#E69F00",
    "process_aware_beam_adaptive": "#0072B2",
    "process_aware_beam_adaptive_polished": "#009E73",
    "process_aware_beam_polished": "#A6761D",
    "topology_local_search": "#E9C46A",
    "topology_local_search_process_aware": "#E76F51",
}
CASE_KEY_CANDIDATES = (
    ("archive", "placements_member", "board_id"),
    ("scenario", "size", "repeat"),
    ("case_name", "placement_method", "seed", "board_id"),
)
SUMMARY_METRICS = (
    "runtime_ms",
    "air_move_distance",
    "cutting_length",
    "machining_cost",
    "travel_mode_cost",
    "hard_penalty",
    "stability_penalty",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
    "rectangle_count",
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    }
)


@dataclass(frozen=True)
class ResultRow:
    values: dict[str, str]
    case_id: str
    method: str

    def number(self, field: str, default: float = 0.0) -> float:
        if field == "machining_cost" and field not in self.values:
            return self.number("cutting_length", default) + self.number(
                "travel_mode_cost",
                default,
            )
        value = self.values.get(field, "")
        if value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default


def load_rows(path: Path) -> tuple[ResultRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        key_fields = infer_case_key_fields(reader.fieldnames)
        rows: list[ResultRow] = []
        for row in reader:
            method = row.get("method", "")
            if not method:
                continue
            rows.append(
                ResultRow(
                    values=row,
                    case_id=build_case_id(row, key_fields),
                    method=method,
                )
            )
    if not rows:
        raise ValueError(f"no method rows found in {path}")
    return tuple(rows)


def infer_case_key_fields(fieldnames: Iterable[str]) -> tuple[str, ...]:
    available = set(fieldnames)
    for candidate in CASE_KEY_CANDIDATES:
        if all(field in available for field in candidate):
            return candidate
    fallback = tuple(
        field
        for field in ("case_id", "board_id", "size", "repeat")
        if field in available
    )
    if fallback:
        return fallback
    raise ValueError(
        "cannot infer case identity fields; expected archive/placements_member/board_id "
        "or scenario/size/repeat"
    )


def build_case_id(row: dict[str, str], key_fields: tuple[str, ...]) -> str:
    return "|".join(row.get(field, "") for field in key_fields)


def method_sort_key(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def by_method(rows: tuple[ResultRow, ...]) -> dict[str, list[ResultRow]]:
    grouped: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
    return grouped


def by_case(rows: tuple[ResultRow, ...]) -> dict[str, dict[str, ResultRow]]:
    grouped: dict[str, dict[str, ResultRow]] = defaultdict(dict)
    for row in rows:
        grouped[row.case_id][row.method] = row
    return grouped


def finite_mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return mean(clean)


def finite_median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return median(clean)


def relative_reduction_pct(baseline: float, target: float) -> float | None:
    if abs(baseline) < 1e-12:
        return None
    return (baseline - target) / abs(baseline) * 100.0


def lower_better_winner(target: ResultRow, baseline: ResultRow) -> str:
    target_tool_events = (
        target.number("pierce_count")
        + target.number("lift_count")
        + target.number("safe_lift_count")
    )
    baseline_tool_events = (
        baseline.number("pierce_count")
        + baseline.number("lift_count")
        + baseline.number("safe_lift_count")
    )
    target_key = (
        target.number("hard_penalty"),
        target.number("stability_penalty"),
        target.number("machining_cost"),
        target_tool_events,
        target.number("travel_mode_cost"),
        target.number("detour_distance"),
        target.number("air_move_distance"),
    )
    baseline_key = (
        baseline.number("hard_penalty"),
        baseline.number("stability_penalty"),
        baseline.number("machining_cost"),
        baseline_tool_events,
        baseline.number("travel_mode_cost"),
        baseline.number("detour_distance"),
        baseline.number("air_move_distance"),
    )
    if target_key < baseline_key:
        return "target"
    if target_key > baseline_key:
        return "baseline"
    return "tie"


def summarize_methods(rows: tuple[ResultRow, ...]) -> tuple[dict[str, str], ...]:
    summary: list[dict[str, str]] = []
    for method, items in sorted(by_method(rows).items(), key=lambda item: method_sort_key(item[0])):
        out: dict[str, str] = {"method": method, "n": str(len(items))}
        for metric in SUMMARY_METRICS:
            values = [
                row.number(metric)
                for row in items
                if metric in row.values or metric == "machining_cost"
            ]
            if not values:
                continue
            out[f"{metric}_mean"] = f"{mean(values):.6f}"
            out[f"{metric}_median"] = f"{median(values):.6f}"
            out[f"{metric}_std"] = f"{pstdev(values):.6f}" if len(values) > 1 else "0.000000"
        summary.append(out)
    return tuple(summary)


def paired_rows(
    rows: tuple[ResultRow, ...],
    target_method: str,
    baseline_methods: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    grouped = by_case(rows)
    comparisons: list[dict[str, str]] = []
    for case_id, case_rows in sorted(grouped.items()):
        target = case_rows.get(target_method)
        if target is None:
            continue
        for baseline_method in baseline_methods:
            baseline = case_rows.get(baseline_method)
            if baseline is None:
                continue
            travel_improvement = relative_reduction_pct(
                baseline.number("travel_mode_cost"),
                target.number("travel_mode_cost"),
            )
            air_improvement = relative_reduction_pct(
                baseline.number("air_move_distance"),
                target.number("air_move_distance"),
            )
            runtime_ratio = None
            if baseline.number("runtime_ms") > 1e-12:
                runtime_ratio = target.number("runtime_ms") / baseline.number("runtime_ms")
            machining_improvement = relative_reduction_pct(
                baseline.number("machining_cost"),
                target.number("machining_cost"),
            )
            comparisons.append(
                {
                    "case_id": case_id,
                    "baseline_method": baseline_method,
                    "target_method": target_method,
                    "rectangle_count": str(int(target.number("rectangle_count"))),
                    "baseline_travel_mode_cost": f"{baseline.number('travel_mode_cost'):.6f}",
                    "target_travel_mode_cost": f"{target.number('travel_mode_cost'):.6f}",
                    "travel_mode_cost_delta": (
                        f"{target.number('travel_mode_cost') - baseline.number('travel_mode_cost'):.6f}"
                    ),
                    "travel_mode_cost_reduction_pct": format_optional(travel_improvement),
                    "baseline_air_move_distance": f"{baseline.number('air_move_distance'):.6f}",
                    "target_air_move_distance": f"{target.number('air_move_distance'):.6f}",
                    "air_move_distance_reduction_pct": format_optional(air_improvement),
                    "baseline_cutting_length": f"{baseline.number('cutting_length'):.6f}",
                    "target_cutting_length": f"{target.number('cutting_length'):.6f}",
                    "baseline_machining_cost": f"{baseline.number('machining_cost'):.6f}",
                    "target_machining_cost": f"{target.number('machining_cost'):.6f}",
                    "machining_cost_reduction_pct": format_optional(
                        machining_improvement
                    ),
                    "baseline_hard_penalty": f"{baseline.number('hard_penalty'):.6f}",
                    "target_hard_penalty": f"{target.number('hard_penalty'):.6f}",
                    "hard_penalty_delta": (
                        f"{target.number('hard_penalty') - baseline.number('hard_penalty'):.6f}"
                    ),
                    "baseline_stability_penalty": f"{baseline.number('stability_penalty'):.6f}",
                    "target_stability_penalty": f"{target.number('stability_penalty'):.6f}",
                    "stability_penalty_delta": (
                        f"{target.number('stability_penalty') - baseline.number('stability_penalty'):.6f}"
                    ),
                    "stability_penalty_reduction": (
                        f"{baseline.number('stability_penalty') - target.number('stability_penalty'):.6f}"
                    ),
                    "baseline_tool_event_count": (
                        f"{baseline.number('pierce_count') + baseline.number('lift_count') + baseline.number('safe_lift_count'):.6f}"
                    ),
                    "target_tool_event_count": (
                        f"{target.number('pierce_count') + target.number('lift_count') + target.number('safe_lift_count'):.6f}"
                    ),
                    "tool_event_count_reduction": (
                        f"{baseline.number('pierce_count') + baseline.number('lift_count') + baseline.number('safe_lift_count') - target.number('pierce_count') - target.number('lift_count') - target.number('safe_lift_count'):.6f}"
                    ),
                    "baseline_runtime_ms": f"{baseline.number('runtime_ms'):.6f}",
                    "target_runtime_ms": f"{target.number('runtime_ms'):.6f}",
                    "runtime_ratio": format_optional(runtime_ratio),
                    "process_key_winner": lower_better_winner(target, baseline),
                }
            )
    return tuple(comparisons)


def format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def parse_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    return float(value)


def summarize_pairs(rows: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["baseline_method"]].append(row)

    summary: list[dict[str, str]] = []
    for baseline_method, items in sorted(grouped.items(), key=lambda item: method_sort_key(item[0])):
        travel_reductions = [
            parse_float(row, "travel_mode_cost_reduction_pct") for row in items
        ]
        machining_reductions = [
            parse_float(row, "machining_cost_reduction_pct") for row in items
        ]
        air_reductions = [parse_float(row, "air_move_distance_reduction_pct") for row in items]
        runtime_ratios = [parse_float(row, "runtime_ratio") for row in items]
        stability_reductions = [
            parse_float(row, "stability_penalty_reduction") or 0.0 for row in items
        ]
        tool_event_reductions = [
            parse_float(row, "tool_event_count_reduction") or 0.0 for row in items
        ]
        hard_deltas = [parse_float(row, "hard_penalty_delta") or 0.0 for row in items]
        target_wins = sum(1 for row in items if row["process_key_winner"] == "target")
        target_ties = sum(1 for row in items if row["process_key_winner"] == "tie")
        travel_wins = sum(
            1
            for row in items
            if float(row["target_travel_mode_cost"]) < float(row["baseline_travel_mode_cost"])
        )
        travel_losses = sum(
            1
            for row in items
            if float(row["target_travel_mode_cost"]) > float(row["baseline_travel_mode_cost"])
        )
        machining_wins = sum(
            1
            for row in items
            if float(row["target_machining_cost"]) < float(row["baseline_machining_cost"])
        )
        machining_losses = sum(
            1
            for row in items
            if float(row["target_machining_cost"]) > float(row["baseline_machining_cost"])
        )
        stability_wins = sum(
            1
            for row in items
            if float(row["target_stability_penalty"]) < float(row["baseline_stability_penalty"])
        )
        stability_losses = sum(
            1
            for row in items
            if float(row["target_stability_penalty"]) > float(row["baseline_stability_penalty"])
        )
        tool_event_wins = sum(
            1
            for row in items
            if float(row["target_tool_event_count"]) < float(row["baseline_tool_event_count"])
        )
        tool_event_losses = sum(
            1
            for row in items
            if float(row["target_tool_event_count"]) > float(row["baseline_tool_event_count"])
        )
        process_losses = sum(1 for row in items if row["process_key_winner"] == "baseline")
        summary.append(
            {
                "baseline_method": baseline_method,
                "paired_cases": str(len(items)),
                "travel_mode_cost_reduction_pct_mean": format_optional(
                    finite_mean(travel_reductions)
                ),
                "travel_mode_cost_reduction_pct_median": format_optional(
                    finite_median(travel_reductions)
                ),
                "machining_cost_reduction_pct_mean": format_optional(
                    finite_mean(machining_reductions)
                ),
                "machining_cost_reduction_pct_median": format_optional(
                    finite_median(machining_reductions)
                ),
                "air_move_distance_reduction_pct_mean": format_optional(
                    finite_mean(air_reductions)
                ),
                "stability_penalty_reduction_mean": f"{mean(stability_reductions):.6f}",
                "tool_event_count_reduction_mean": f"{mean(tool_event_reductions):.6f}",
                "hard_penalty_delta_mean": f"{mean(hard_deltas):.6f}",
                "runtime_ratio_mean": format_optional(finite_mean(runtime_ratios)),
                "process_key_win_rate": f"{target_wins / len(items):.6f}",
                "process_key_tie_rate": f"{target_ties / len(items):.6f}",
                "process_key_sign_test_p": format_optional(
                    exact_sign_test_p_value(target_wins, process_losses)
                ),
                "travel_cost_win_rate": f"{travel_wins / len(items):.6f}",
                "travel_cost_sign_test_p": format_optional(
                    exact_sign_test_p_value(travel_wins, travel_losses)
                ),
                "machining_cost_win_rate": f"{machining_wins / len(items):.6f}",
                "machining_cost_sign_test_p": format_optional(
                    exact_sign_test_p_value(machining_wins, machining_losses)
                ),
                "tool_event_win_rate": f"{tool_event_wins / len(items):.6f}",
                "tool_event_sign_test_p": format_optional(
                    exact_sign_test_p_value(tool_event_wins, tool_event_losses)
                ),
                "stability_win_rate": f"{stability_wins / len(items):.6f}",
                "stability_sign_test_p": format_optional(
                    exact_sign_test_p_value(stability_wins, stability_losses)
                ),
            }
        )
    return tuple(summary)


def exact_sign_test_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    lower_tail = sum(comb(trials, k) for k in range(observed + 1)) / (2**trials)
    return min(1.0, 2.0 * lower_tail)


def write_csv(rows: tuple[dict[str, str], ...], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    method_summary: tuple[dict[str, str], ...],
    pair_summary: tuple[dict[str, str], ...],
    output_path: Path,
    target_method: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Target method: {target_method}", ""]
    lines.append("Method summary")
    for row in method_summary:
        lines.append(
            "  "
            f"{row['method']}: n={row['n']}, "
            f"cost_mean={row.get('travel_mode_cost_mean', 'NA')}, "
            f"machining_mean={row.get('machining_cost_mean', 'NA')}, "
            f"stability_mean={row.get('stability_penalty_mean', 'NA')}, "
            f"runtime_mean={row.get('runtime_ms_mean', 'NA')}"
        )
    lines.append("")
    lines.append("Paired comparison summary")
    for row in pair_summary:
        lines.append(
            "  "
            f"vs {row['baseline_method']}: cases={row['paired_cases']}, "
            f"cost_reduction_mean={row['travel_mode_cost_reduction_pct_mean']}%, "
            f"machining_reduction_mean={row['machining_cost_reduction_pct_mean']}%, "
            f"stability_reduction_mean={row['stability_penalty_reduction_mean']}, "
            f"tool_event_reduction_mean={row['tool_event_count_reduction_mean']}, "
            f"runtime_ratio_mean={row['runtime_ratio_mean']}, "
            f"process_win_rate={row['process_key_win_rate']}, "
            f"process_sign_test_p={row['process_key_sign_test_p']}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_method_summary(
    method_summary: tuple[dict[str, str], ...],
    figures_dir: Path,
) -> None:
    rows = [
        row
        for row in method_summary
        if "travel_mode_cost_mean" in row and "stability_penalty_mean" in row
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: method_sort_key(row["method"]))
    labels = [METHOD_LABELS.get(row["method"], row["method"]) for row in rows]
    colors = [METHOD_COLORS.get(row["method"], "#999999") for row in rows]
    x = range(len(rows))

    specs = (
        ("machining_cost_mean", "Machining cost"),
        ("travel_mode_cost_mean", "Travel-mode cost"),
        ("stability_penalty_mean", "Stability penalty"),
        ("runtime_ms_mean", "Runtime (ms)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0))
    for ax, (field, ylabel) in zip(axes, specs):
        values = [float(row[field]) for row in rows]
        ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.4)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=24, ha="right")
        if field == "runtime_ms_mean":
            ax.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, figures_dir, "fig_result_method_summary")


def plot_pair_summary(
    pair_summary: tuple[dict[str, str], ...],
    figures_dir: Path,
) -> None:
    if not pair_summary:
        return
    rows = sorted(pair_summary, key=lambda row: method_sort_key(row["baseline_method"]))
    labels = [METHOD_LABELS.get(row["baseline_method"], row["baseline_method"]) for row in rows]
    cost_reduction = [
        float(row["machining_cost_reduction_pct_mean"])
        if row["machining_cost_reduction_pct_mean"] != ""
        else 0.0
        for row in rows
    ]
    travel_reduction = [
        float(row["travel_mode_cost_reduction_pct_mean"])
        if row["travel_mode_cost_reduction_pct_mean"] != ""
        else 0.0
        for row in rows
    ]
    stability_reduction = [
        float(row["stability_penalty_reduction_mean"]) for row in rows
    ]
    runtime_ratio = [
        float(row["runtime_ratio_mean"]) if row["runtime_ratio_mean"] != "" else 0.0
        for row in rows
    ]
    x = range(len(rows))

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0))
    axes[0].bar(x, cost_reduction, color="#E69F00", edgecolor="#333333", linewidth=0.4)
    axes[0].axhline(0.0, color="#111827", linewidth=0.6)
    axes[0].set_title("Machining reduction")
    axes[0].set_ylabel("Reduction (%)")
    axes[1].bar(x, travel_reduction, color="#D55E00", edgecolor="#333333", linewidth=0.4)
    axes[1].axhline(0.0, color="#111827", linewidth=0.6)
    axes[1].set_title("Travel reduction")
    axes[1].set_ylabel("Reduction (%)")
    axes[2].bar(x, stability_reduction, color="#009E73", edgecolor="#333333", linewidth=0.4)
    axes[2].axhline(0.0, color="#111827", linewidth=0.6)
    axes[2].set_title("Stability improvement")
    axes[2].set_ylabel("Penalty reduction")
    axes[3].bar(x, runtime_ratio, color="#56B4E9", edgecolor="#333333", linewidth=0.4)
    axes[3].axhline(1.0, color="#111827", linewidth=0.6)
    axes[3].set_title("Runtime ratio")
    axes[3].set_ylabel("Target / baseline")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=24, ha="right")
    fig.tight_layout()
    save_figure(fig, figures_dir, "fig_result_paired_summary")


def plot_scaling(rows: tuple[ResultRow, ...], figures_dir: Path) -> None:
    if not rows or "rectangle_count" not in rows[0].values:
        return
    grouped: dict[tuple[str, int], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method, int(row.number("rectangle_count")))].append(row)
    methods = sorted({method for method, _ in grouped}, key=method_sort_key)
    if len({count for _, count in grouped}) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for method in methods:
        counts = sorted(count for m, count in grouped if m == method)
        cost_values = [
            mean(item.number("travel_mode_cost") for item in grouped[(method, count)])
            for count in counts
        ]
        runtime_values = [
            mean(item.number("runtime_ms") for item in grouped[(method, count)])
            for count in counts
        ]
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "#999999")
        axes[0].plot(counts, cost_values, marker="o", label=label, color=color)
        axes[1].plot(counts, runtime_values, marker="o", label=label, color=color)
    axes[0].set_xlabel("Rectangle count")
    axes[0].set_ylabel("Travel-mode cost")
    axes[0].set_title("Cost scaling")
    axes[1].set_xlabel("Rectangle count")
    axes[1].set_ylabel("Runtime (ms)")
    axes[1].set_yscale("log")
    axes[1].set_title("Runtime scaling")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save_figure(fig, figures_dir, "fig_result_scaling")


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf")
    fig.savefig(figures_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "results"
        / "chapter2_batch_expanded_3x3_upto50_multi_unstable_retained.csv",
    )
    parser.add_argument("--target-method", default="process_aware_beam")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=(
            "greedy",
            "path_distance_local_search",
            "topology_process_aware",
            "process_local_search_multistart",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "analysis_multi_unstable_retained",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "analysis_multi_unstable_retained",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    method_summary = summarize_methods(rows)
    pair_rows = paired_rows(rows, args.target_method, tuple(args.baseline_methods))
    pair_summary = summarize_pairs(pair_rows)

    write_csv(method_summary, args.output_dir / "method_summary.csv")
    write_csv(pair_rows, args.output_dir / "paired_comparison.csv")
    write_csv(pair_summary, args.output_dir / "paired_summary.csv")
    write_report(
        method_summary,
        pair_summary,
        args.output_dir / "analysis_report.txt",
        args.target_method,
    )

    if not args.no_figures:
        plot_method_summary(method_summary, args.figures_dir)
        plot_pair_summary(pair_summary, args.figures_dir)
        plot_scaling(rows, args.figures_dir)

    print(f"input: {args.input}")
    print(f"rows: {len(rows)}")
    print(f"method_summary: {args.output_dir / 'method_summary.csv'}")
    print(f"paired_comparison: {args.output_dir / 'paired_comparison.csv'}")
    print(f"paired_summary: {args.output_dir / 'paired_summary.csv'}")
    print(f"report: {args.output_dir / 'analysis_report.txt'}")
    if not args.no_figures:
        print(f"figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
