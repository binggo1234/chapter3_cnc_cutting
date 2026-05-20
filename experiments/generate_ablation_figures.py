from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_METHOD = "full_process_aware_beam"
VARIANT_ORDER = (
    "full_process_aware_beam",
    "single_edges_only",
    "no_stability_guidance",
    "no_adjacency_support_guidance",
    "topology_no_beam",
    "path_distance_baseline",
    "no_detour_operator",
    "no_safe_travel_modes",
)
VARIANT_LABELS = {
    "full_process_aware_beam": "Full",
    "single_edges_only": "Single edges",
    "no_stability_guidance": "No stability",
    "no_adjacency_support_guidance": "No adjacency",
    "topology_no_beam": "No beam",
    "path_distance_baseline": "Path-LS",
    "no_detour_operator": "No detour",
    "no_safe_travel_modes": "No safe travel",
}
VARIANT_COLORS = {
    "full_process_aware_beam": "#E69F00",
    "single_edges_only": "#7B8794",
    "no_stability_guidance": "#D55E00",
    "no_adjacency_support_guidance": "#CC79A7",
    "topology_no_beam": "#264653",
    "path_distance_baseline": "#56B4E9",
    "no_detour_operator": "#009E73",
    "no_safe_travel_modes": "#BE123C",
}
UNIT_COLORS = {
    "selected_single_edge_count_mean": "#7B8794",
    "selected_shared_edge_count_mean": "#0072B2",
    "selected_near_shared_channel_count_mean": "#009E73",
    "selected_collinear_chain_count_mean": "#E69F00",
}
UNIT_LABELS = {
    "selected_single_edge_count_mean": "Single edge",
    "selected_shared_edge_count_mean": "Shared edge",
    "selected_near_shared_channel_count_mean": "Near-shared",
    "selected_collinear_chain_count_mean": "Collinear chain",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
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
class SummaryRow:
    method: str
    n: int
    runtime_ms_mean: float
    machining_cost_mean: float
    travel_mode_cost_mean: float
    cutting_length_mean: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    safe_lift_count_mean: float
    selected_single_edge_count_mean: float
    selected_shared_edge_count_mean: float
    selected_near_shared_channel_count_mean: float
    selected_collinear_chain_count_mean: float


@dataclass(frozen=True)
class PairSummaryRow:
    baseline_method: str
    paired_cases: int
    machining_cost_reduction_pct_mean: float
    travel_mode_cost_reduction_pct_mean: float
    stability_penalty_reduction_mean: float
    tool_event_count_reduction_mean: float
    process_key_win_rate: float
    machining_cost_win_rate: float
    process_key_sign_test_p: float | None


def parse_float(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    return float(value)


def load_summary(path: Path) -> tuple[SummaryRow, ...]:
    rows: list[SummaryRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            cutting_length_mean = parse_float(raw.get("cutting_length_mean", ""))
            if not cutting_length_mean:
                cutting_length_mean = (
                    parse_float(raw.get("machining_cost_mean", ""))
                    - parse_float(raw.get("travel_mode_cost_mean", ""))
                )
            rows.append(
                SummaryRow(
                    method=raw["method"],
                    n=int(raw["n"]),
                    runtime_ms_mean=parse_float(raw["runtime_ms_mean"]),
                    machining_cost_mean=parse_float(raw.get("machining_cost_mean", "")),
                    travel_mode_cost_mean=parse_float(raw["travel_mode_cost_mean"]),
                    cutting_length_mean=cutting_length_mean,
                    hard_penalty_mean=parse_float(raw["hard_penalty_mean"]),
                    stability_penalty_mean=parse_float(raw["stability_penalty_mean"]),
                    safe_lift_count_mean=parse_float(raw["safe_lift_count_mean"]),
                    selected_single_edge_count_mean=parse_float(
                        raw.get("selected_single_edge_count_mean", "")
                    ),
                    selected_shared_edge_count_mean=parse_float(
                        raw.get("selected_shared_edge_count_mean", "")
                    ),
                    selected_near_shared_channel_count_mean=parse_float(
                        raw.get("selected_near_shared_channel_count_mean", "")
                    ),
                    selected_collinear_chain_count_mean=parse_float(
                        raw.get("selected_collinear_chain_count_mean", "")
                    ),
                )
            )
    return tuple(sorted(rows, key=lambda row: variant_sort_key(row.method)))


def load_pair_summary(path: Path) -> tuple[PairSummaryRow, ...]:
    rows: list[PairSummaryRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                PairSummaryRow(
                    baseline_method=raw["baseline_method"],
                    paired_cases=int(raw["paired_cases"]),
                    machining_cost_reduction_pct_mean=parse_float(
                        raw["machining_cost_reduction_pct_mean"]
                    ),
                    travel_mode_cost_reduction_pct_mean=parse_float(
                        raw["travel_mode_cost_reduction_pct_mean"]
                    ),
                    stability_penalty_reduction_mean=parse_float(
                        raw["stability_penalty_reduction_mean"]
                    ),
                    tool_event_count_reduction_mean=parse_float(
                        raw["tool_event_count_reduction_mean"]
                    ),
                    process_key_win_rate=parse_float(raw["process_key_win_rate"]),
                    machining_cost_win_rate=parse_float(raw["machining_cost_win_rate"]),
                    process_key_sign_test_p=(
                        parse_float(raw["process_key_sign_test_p"])
                        if raw["process_key_sign_test_p"]
                        else None
                    ),
                )
            )
    return tuple(sorted(rows, key=lambda row: variant_sort_key(row.baseline_method)))


def variant_sort_key(method: str) -> tuple[int, str]:
    if method in VARIANT_ORDER:
        return (VARIANT_ORDER.index(method), method)
    return (len(VARIANT_ORDER), method)


def ordered_summary(rows: tuple[SummaryRow, ...]) -> tuple[SummaryRow, ...]:
    by_method = {row.method: row for row in rows}
    return tuple(by_method[method] for method in VARIANT_ORDER if method in by_method)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_ablation_overview(rows: tuple[SummaryRow, ...], output_dir: Path) -> None:
    rows = ordered_summary(rows)
    x = np.arange(len(rows))
    labels = [VARIANT_LABELS.get(row.method, row.method) for row in rows]
    colors = [VARIANT_COLORS.get(row.method, "#999999") for row in rows]
    full = next(row for row in rows if row.method == TARGET_METHOD)

    specs = (
        (
            "Machining cost",
            [row.machining_cost_mean / full.machining_cost_mean for row in rows],
            "Relative to full",
            False,
        ),
        (
            "Travel-mode cost",
            [row.travel_mode_cost_mean / full.travel_mode_cost_mean for row in rows],
            "Relative to full",
            False,
        ),
        (
            "Stability penalty",
            [row.stability_penalty_mean for row in rows],
            "Mean penalty",
            False,
        ),
        (
            "Runtime",
            [row.runtime_ms_mean for row in rows],
            "Runtime (ms)",
            True,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.0))
    for ax, (title, values, ylabel, log_scale) in zip(axes.flat, specs):
        ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.4)
        if "Relative" in ylabel:
            ax.axhline(1.0, color="#111827", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        if log_scale:
            ax.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_ablation_overview")


def plot_unit_composition(rows: tuple[SummaryRow, ...], output_dir: Path) -> None:
    rows = ordered_summary(rows)
    x = np.arange(len(rows))
    labels = [VARIANT_LABELS.get(row.method, row.method) for row in rows]
    bottoms = np.zeros(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for field, color in UNIT_COLORS.items():
        values = np.array([getattr(row, field) for row in rows])
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=color,
            edgecolor="#333333",
            linewidth=0.25,
            label=UNIT_LABELS[field],
        )
        bottoms += values
    ax.set_ylabel("Selected cutting units")
    ax.set_title("Cutting-unit composition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_ablation_unit_composition")


def plot_paired_improvements(
    rows: tuple[PairSummaryRow, ...],
    output_dir: Path,
) -> None:
    rows = tuple(row for row in rows if row.baseline_method != TARGET_METHOD)
    labels = [VARIANT_LABELS.get(row.baseline_method, row.baseline_method) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    axes[0].bar(
        x,
        [row.machining_cost_reduction_pct_mean for row in rows],
        color="#E69F00",
        edgecolor="#333333",
        linewidth=0.4,
    )
    axes[0].axhline(0.0, color="#111827", linewidth=0.6)
    axes[0].set_title("Machining reduction")
    axes[0].set_ylabel("Full vs. ablation (%)")

    axes[1].bar(
        x,
        [row.stability_penalty_reduction_mean for row in rows],
        color="#009E73",
        edgecolor="#333333",
        linewidth=0.4,
    )
    axes[1].axhline(0.0, color="#111827", linewidth=0.6)
    axes[1].set_title("Stability improvement")
    axes[1].set_ylabel("Penalty reduction")

    axes[2].bar(
        x,
        [row.process_key_win_rate for row in rows],
        color="#56B4E9",
        edgecolor="#333333",
        linewidth=0.4,
    )
    axes[2].axhline(0.5, color="#111827", linewidth=0.6)
    axes[2].set_title("Process-key win rate")
    axes[2].set_ylabel("Win rate")
    axes[2].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_ablation_paired_improvements")


def write_key_table(rows: tuple[PairSummaryRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline_method",
                "paired_cases",
                "machining_cost_reduction_pct_mean",
                "stability_penalty_reduction_mean",
                "process_key_win_rate",
                "machining_cost_win_rate",
                "process_key_sign_test_p",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "baseline_method": row.baseline_method,
                    "paired_cases": row.paired_cases,
                    "machining_cost_reduction_pct_mean": (
                        f"{row.machining_cost_reduction_pct_mean:.6f}"
                    ),
                    "stability_penalty_reduction_mean": (
                        f"{row.stability_penalty_reduction_mean:.6f}"
                    ),
                    "process_key_win_rate": f"{row.process_key_win_rate:.6f}",
                    "machining_cost_win_rate": f"{row.machining_cost_win_rate:.6f}",
                    "process_key_sign_test_p": (
                        "" if row.process_key_sign_test_p is None else f"{row.process_key_sign_test_p:.6f}"
                    ),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "ablation_real_3x3_upto50_summary.csv",
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=ROOT / "results" / "analysis_ablation_real_3x3_upto50" / "paired_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "ablation_real_3x3_upto50",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=ROOT / "results" / "ablation_real_3x3_upto50_key_table.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    paired = load_pair_summary(args.paired_summary)
    plot_ablation_overview(summary, args.output_dir)
    plot_unit_composition(summary, args.output_dir)
    plot_paired_improvements(paired, args.output_dir)
    write_key_table(paired, args.table_output)
    print(f"summary: {args.summary}")
    print(f"paired_summary: {args.paired_summary}")
    print(f"figures: {args.output_dir}")
    print(f"key_table: {args.table_output}")


if __name__ == "__main__":
    main()
