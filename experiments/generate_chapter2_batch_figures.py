from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = (
    "greedy",
    "path_distance_local_search",
    "topology",
    "topology_local_search",
    "topology_process_aware",
    "process_local_search_multistart",
    "process_aware_beam",
    "process_aware_beam_adaptive",
    "process_aware_beam_polished",
    "process_aware_beam_adaptive_polished",
    "topology_local_search_process_aware",
)
METHOD_LABELS = {
    "greedy": "Greedy",
    "path_distance_local_search": "Path-LS",
    "topology": "Topology",
    "topology_local_search": "Topology+LS",
    "topology_process_aware": "Process-aware",
    "process_local_search_multistart": "Multi-start process LS",
    "process_aware_beam": "Process-aware beam",
    "process_aware_beam_adaptive": "Adaptive beam",
    "process_aware_beam_polished": "Beam+process LS",
    "process_aware_beam_adaptive_polished": "Adaptive beam+LS",
    "topology_local_search_process_aware": "Process-aware+LS",
}
METHOD_COLORS = {
    "greedy": "#7B8794",
    "path_distance_local_search": "#CC79A7",
    "topology": "#2A9D8F",
    "topology_local_search": "#E9C46A",
    "topology_process_aware": "#264653",
    "process_local_search_multistart": "#8E6C8A",
    "process_aware_beam": "#E69F00",
    "process_aware_beam_adaptive": "#0072B2",
    "process_aware_beam_polished": "#A6761D",
    "process_aware_beam_adaptive_polished": "#009E73",
    "topology_local_search_process_aware": "#E76F51",
}
METHOD_MARKERS = {
    "greedy": "o",
    "path_distance_local_search": "v",
    "topology": "s",
    "topology_local_search": "^",
    "topology_process_aware": "D",
    "process_local_search_multistart": "h",
    "process_aware_beam": "X",
    "process_aware_beam_adaptive": "P",
    "process_aware_beam_polished": "*",
    "process_aware_beam_adaptive_polished": "D",
    "topology_local_search_process_aware": "P",
}

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
class BatchSummaryRow:
    method: str
    n: int
    runtime_ms_mean: float
    runtime_ms_std: float
    air_move_distance_mean: float
    air_move_distance_std: float
    travel_mode_cost_mean: float
    travel_mode_cost_std: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    safe_lift_count_mean: float
    detour_count_mean: float
    rectangle_count_mean: float


@dataclass(frozen=True)
class BatchRouteRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    method: str
    runtime_ms: float
    rectangle_count: int
    air_move_distance: float
    travel_mode_cost: float
    hard_penalty: float
    stability_penalty: float

    @property
    def case_id(self) -> str:
        return "|".join(
            (
                self.archive,
                self.placements_member,
                self.board_id,
            )
        )


def load_summary(path: Path) -> tuple[BatchSummaryRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            BatchSummaryRow(
                method=row["method"],
                n=int(row["n"]),
                runtime_ms_mean=float(row["runtime_ms_mean"]),
                runtime_ms_std=float(row["runtime_ms_std"]),
                air_move_distance_mean=float(row["air_move_distance_mean"]),
                air_move_distance_std=float(row["air_move_distance_std"]),
                travel_mode_cost_mean=float(row["travel_mode_cost_mean"]),
                travel_mode_cost_std=float(row["travel_mode_cost_std"]),
                hard_penalty_mean=float(row["hard_penalty_mean"]),
                stability_penalty_mean=float(row["stability_penalty_mean"]),
                safe_lift_count_mean=float(row["safe_lift_count_mean"]),
                detour_count_mean=float(row["detour_count_mean"]),
                rectangle_count_mean=float(row["rectangle_count_mean"]),
            )
            for row in csv.DictReader(handle)
        )


def load_routes(path: Path) -> tuple[BatchRouteRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            BatchRouteRow(
                archive=row["archive"],
                case_name=row["case_name"],
                placement_method=row["placement_method"],
                seed=row["seed"],
                placements_member=row["placements_member"],
                board_id=row["board_id"],
                method=row["method"],
                runtime_ms=float(row["runtime_ms"]),
                rectangle_count=int(row["rectangle_count"]),
                air_move_distance=float(row["air_move_distance"]),
                travel_mode_cost=float(row["travel_mode_cost"]),
                hard_penalty=float(row["hard_penalty"]),
                stability_penalty=float(row["stability_penalty"]),
            )
            for row in csv.DictReader(handle)
        )


def method_rows(summary: tuple[BatchSummaryRow, ...]) -> tuple[BatchSummaryRow, ...]:
    by_method = {row.method: row for row in summary}
    return tuple(by_method[method] for method in METHOD_ORDER if method in by_method)


def sem(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return std / sqrt(n)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_method_comparison(
    summary: tuple[BatchSummaryRow, ...],
    output_dir: Path,
) -> None:
    rows = method_rows(summary)
    x = np.arange(len(rows))
    labels = [METHOD_LABELS[row.method] for row in rows]
    colors = [METHOD_COLORS[row.method] for row in rows]

    metric_specs = (
        ("runtime_ms_mean", "runtime_ms_std", "Runtime (ms)", True),
        ("air_move_distance_mean", "air_move_distance_std", "Air move distance (mm)", False),
        ("travel_mode_cost_mean", "travel_mode_cost_std", "Travel-mode cost", False),
        ("stability_penalty_mean", None, "Stability penalty", False),
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (mean_field, std_field, ylabel, log_scale) in zip(axes.flat, metric_specs):
        means = [getattr(row, mean_field) for row in rows]
        yerr = (
            [sem(getattr(row, std_field), row.n) for row in rows]
            if std_field is not None
            else None
        )
        ax.bar(
            x,
            means,
            yerr=yerr,
            capsize=2.5 if yerr is not None else 0,
            color=colors,
            edgecolor="#333333",
            linewidth=0.4,
            alpha=0.92,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if log_scale:
            ax.set_yscale("log")

    fig.suptitle("Real nesting layouts: method-level comparison", y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_chapter2_batch_method_comparison")


def plot_tradeoff_scatter(
    rows: tuple[BatchRouteRow, ...],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.1, 3.05))
    for method in METHOD_ORDER:
        items = [row for row in rows if row.method == method]
        if not items:
            continue
        ax.scatter(
            [row.travel_mode_cost for row in items],
            [row.stability_penalty for row in items],
            s=34,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            alpha=0.82,
            label=METHOD_LABELS[method],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("Travel-mode cost")
    ax.set_ylabel("Stability penalty")
    ax.set_title("Path-efficiency vs. process-stability tradeoff")
    ax.legend(loc="upper right", ncol=1)
    save_figure(fig, output_dir, "fig_chapter2_batch_tradeoff_scatter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "chapter2_batch_summary.csv",
    )
    parser.add_argument(
        "--routes",
        type=Path,
        default=ROOT / "results" / "chapter2_batch_routes.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    routes = load_routes(args.routes)
    plot_method_comparison(summary, args.figure_dir)
    plot_tradeoff_scatter(routes, args.figure_dir)
    print(f"wrote figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
