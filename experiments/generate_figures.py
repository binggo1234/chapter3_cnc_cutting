from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "results" / "scalability_results.csv",
    ROOT / "results" / "scalability_clustered_results.csv",
    ROOT / "results" / "scalability_clustered_process_aware.csv",
)

METHOD_LABELS = {
    "greedy": "Greedy",
    "path_distance_local_search": "Path-LS",
    "topology": "Topology",
    "topology_process_aware": "Process-aware",
    "process_local_search_multistart": "Multi-start process LS",
    "process_aware_beam": "Process-aware beam",
    "process_aware_beam_adaptive": "Adaptive beam",
    "process_aware_beam_polished": "Beam+process LS",
    "process_aware_beam_adaptive_polished": "Adaptive beam+LS",
    "topology_local_search": "Topology+LS",
}
METHOD_ORDER = (
    "greedy",
    "path_distance_local_search",
    "topology",
    "topology_process_aware",
    "process_local_search_multistart",
    "process_aware_beam",
    "process_aware_beam_adaptive",
    "process_aware_beam_polished",
    "process_aware_beam_adaptive_polished",
    "topology_local_search",
)
COLORS = {
    "greedy": "#7B8794",
    "path_distance_local_search": "#CC79A7",
    "topology": "#2A9D8F",
    "topology_process_aware": "#264653",
    "process_local_search_multistart": "#8E6C8A",
    "process_aware_beam": "#E69F00",
    "process_aware_beam_adaptive": "#0072B2",
    "process_aware_beam_polished": "#A6761D",
    "process_aware_beam_adaptive_polished": "#009E73",
    "topology_local_search": "#E76F51",
}
MARKERS = {
    "greedy": "o",
    "path_distance_local_search": "v",
    "topology": "s",
    "topology_process_aware": "D",
    "process_local_search_multistart": "h",
    "process_aware_beam": "X",
    "process_aware_beam_adaptive": "P",
    "process_aware_beam_polished": "*",
    "process_aware_beam_adaptive_polished": "D",
    "topology_local_search": "^",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)


@dataclass(frozen=True)
class ResultRow:
    variant: str
    scenario: str
    size: int
    method: str
    repeat: int
    runtime_ms: float
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    cutting_length: float
    pierce_count: int
    lift_count: int
    stability_penalty: float
    hard_penalty: float


@dataclass(frozen=True)
class SummaryRow:
    variant: str
    scenario: str
    size: int
    method: str
    repeats: int
    runtime_ms_mean: float
    air_move_distance_mean: float
    stability_penalty_mean: float
    hard_penalty_mean: float
    candidate_unit_count_mean: float
    selected_unit_count_mean: float
    action_count_mean: float


def infer_variant(path: Path) -> str:
    stem = path.stem
    if "process_aware" in stem:
        return "process_aware"
    return "fast"


def load_rows(paths: tuple[Path, ...]) -> tuple[ResultRow, ...]:
    rows: list[ResultRow] = []
    for path in paths:
        variant = infer_variant(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                rows.append(
                    ResultRow(
                        variant=variant,
                        scenario=raw["scenario"],
                        size=int(raw["size"]),
                        method=raw["method"],
                        repeat=int(raw["repeat"]),
                        runtime_ms=float(raw["runtime_ms"]),
                        candidate_unit_count=int(raw["candidate_unit_count"]),
                        selected_unit_count=int(raw["selected_unit_count"]),
                        action_count=int(raw["action_count"]),
                        air_move_distance=float(raw["air_move_distance"]),
                        cutting_length=float(raw["cutting_length"]),
                        pierce_count=int(raw["pierce_count"]),
                        lift_count=int(raw["lift_count"]),
                        stability_penalty=float(raw["stability_penalty"]),
                        hard_penalty=float(raw["hard_penalty"]),
                    )
                )
    return tuple(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: tuple[ResultRow, ...]) -> tuple[SummaryRow, ...]:
    groups: dict[tuple[str, str, int, str], list[ResultRow]] = defaultdict(list)
    for row in rows:
        groups[(row.variant, row.scenario, row.size, row.method)].append(row)

    summaries: list[SummaryRow] = []
    for (variant, scenario, size, method), items in groups.items():
        summaries.append(
            SummaryRow(
                variant=variant,
                scenario=scenario,
                size=size,
                method=method,
                repeats=len(items),
                runtime_ms_mean=mean([item.runtime_ms for item in items]),
                air_move_distance_mean=mean([item.air_move_distance for item in items]),
                stability_penalty_mean=mean([item.stability_penalty for item in items]),
                hard_penalty_mean=mean([item.hard_penalty for item in items]),
                candidate_unit_count_mean=mean(
                    [float(item.candidate_unit_count) for item in items]
                ),
                selected_unit_count_mean=mean(
                    [float(item.selected_unit_count) for item in items]
                ),
                action_count_mean=mean([float(item.action_count) for item in items]),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda row: (row.variant, row.scenario, row.size, METHOD_ORDER.index(row.method)),
        )
    )


def write_summary(summary: tuple[SummaryRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SummaryRow.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def subset(
    summary: tuple[SummaryRow, ...],
    variant: str,
    scenario: str,
) -> tuple[SummaryRow, ...]:
    return tuple(
        row for row in summary if row.variant == variant and row.scenario == scenario
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_runtime_scalability(
    summary: tuple[SummaryRow, ...],
    variant: str,
    scenario: str,
    output_dir: Path,
) -> None:
    data = subset(summary, variant=variant, scenario=scenario)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    for method in METHOD_ORDER:
        rows = sorted([row for row in data if row.method == method], key=lambda row: row.size)
        if not rows:
            continue
        ax.plot(
            [row.size for row in rows],
            [row.runtime_ms_mean for row in rows],
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
        )
    ax.set_xlabel("Number of rectangles")
    ax.set_ylabel("Runtime (ms)")
    ax.set_yscale("log")
    ax.set_title(f"Scalability on {scenario} layouts")
    ax.legend(loc="upper left")
    save_figure(fig, output_dir, f"fig_runtime_{scenario}_{variant}")


def plot_clustered_process_tradeoff(
    summary: tuple[SummaryRow, ...],
    output_dir: Path,
) -> None:
    data = subset(summary, variant="process_aware", scenario="clustered")
    if not data:
        return

    sizes = sorted({row.size for row in data})
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75), sharex=True)
    metrics = (
        ("stability_penalty_mean", "Stability penalty"),
        ("hard_penalty_mean", "Hard penalty"),
    )
    width = 0.22
    x = np.arange(len(sizes))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, method in enumerate(METHOD_ORDER):
            rows = {
                row.size: row
                for row in data
                if row.method == method
            }
            values = [getattr(rows[size], metric) for size in sizes]
            offset = (index - 1) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                alpha=0.92,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel("Number of rectangles")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    axes[0].legend(loc="upper left")
    save_figure(fig, output_dir, "fig_clustered_process_aware_tradeoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUTS),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "scalability_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(tuple(args.inputs))
    summary = summarize(rows)
    write_summary(summary, args.summary_output)
    plot_runtime_scalability(summary, "fast", "grid", args.figure_dir)
    plot_runtime_scalability(summary, "fast", "clustered", args.figure_dir)
    plot_clustered_process_tradeoff(summary, args.figure_dir)
    print(f"wrote: {args.summary_output}")
    print(f"wrote figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
