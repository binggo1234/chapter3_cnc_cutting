from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "topology_process_aware": "Process-aware topology",
    "process_aware_beam": "Process-aware beam",
    "process_aware_beam_adaptive_polished": "Adaptive beam+LS",
}
METHOD_COLORS = {
    "topology_process_aware": "#264653",
    "process_aware_beam": "#E69F00",
    "process_aware_beam_adaptive_polished": "#009E73",
}


@dataclass(frozen=True)
class ScalabilitySummaryRow:
    scenario: str
    size: int
    method: str
    n: int
    runtime_ms_mean: float
    runtime_ms_std: float
    travel_mode_cost_mean: float
    travel_mode_cost_std: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    candidate_unit_count_mean: float
    selected_unit_count_mean: float
    action_count_mean: float


def read_rows(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return 0.0 if value == "" else float(value)


def summarize(rows: list[dict[str, str]]) -> list[ScalabilitySummaryRow]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], int(row["size"]), row["method"])].append(row)

    summaries: list[ScalabilitySummaryRow] = []
    for (scenario, size, method), items in sorted(grouped.items()):
        summaries.append(
            ScalabilitySummaryRow(
                scenario=scenario,
                size=size,
                method=method,
                n=len(items),
                runtime_ms_mean=mean(number(row, "runtime_ms") for row in items),
                runtime_ms_std=(
                    pstdev(number(row, "runtime_ms") for row in items)
                    if len(items) > 1
                    else 0.0
                ),
                travel_mode_cost_mean=mean(
                    number(row, "travel_mode_cost") for row in items
                ),
                travel_mode_cost_std=(
                    pstdev(number(row, "travel_mode_cost") for row in items)
                    if len(items) > 1
                    else 0.0
                ),
                hard_penalty_mean=mean(number(row, "hard_penalty") for row in items),
                stability_penalty_mean=mean(
                    number(row, "stability_penalty") for row in items
                ),
                candidate_unit_count_mean=mean(
                    number(row, "candidate_unit_count") for row in items
                ),
                selected_unit_count_mean=mean(
                    number(row, "selected_unit_count") for row in items
                ),
                action_count_mean=mean(number(row, "action_count") for row in items),
            )
        )
    return summaries


def write_summary(path: Path, rows: list[ScalabilitySummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [field.name for field in fields(rows[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def plot_summary(rows: list[ScalabilitySummaryRow], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = tuple(dict.fromkeys(row.scenario for row in rows))
    methods = tuple(dict.fromkeys(row.method for row in rows))
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(9.2, 3.1 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]

    for row_axes, scenario in zip(axes, scenarios):
        scenario_rows = [row for row in rows if row.scenario == scenario]
        for method in methods:
            method_rows = sorted(
                [row for row in scenario_rows if row.method == method],
                key=lambda row: row.size,
            )
            if not method_rows:
                continue
            x = [row.size for row in method_rows]
            row_axes[0].plot(
                x,
                [row.travel_mode_cost_mean for row in method_rows],
                marker="o",
                color=METHOD_COLORS.get(method, "#777777"),
                label=METHOD_LABELS.get(method, method),
            )
            row_axes[1].plot(
                x,
                [row.runtime_ms_mean for row in method_rows],
                marker="o",
                color=METHOD_COLORS.get(method, "#777777"),
                label=METHOD_LABELS.get(method, method),
            )
        row_axes[0].set_title(f"{scenario}: travel-mode cost")
        row_axes[0].set_ylabel("Mean cost")
        row_axes[1].set_title(f"{scenario}: runtime")
        row_axes[1].set_ylabel("Mean runtime (ms)")
        for ax in row_axes:
            ax.set_xlabel("Rectangles")
            ax.grid(alpha=0.2)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_final_scalability_summary.pdf")
    fig.savefig(output_dir / "fig_final_scalability_summary.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=(ROOT / "results" / "scalability_final_clustered_50_100.csv",),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "scalability_final_50_100_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures" / "scalability_final_50_100",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(tuple(args.inputs))
    summary = summarize(rows)
    write_summary(args.summary_output, summary)
    plot_summary(summary, args.figure_dir)
    print(f"wrote: {args.summary_output}")
    print(f"figures: {args.figure_dir}")
    for row in summary:
        print(
            f"{row.scenario:<9} size={row.size:<4d} {row.method:<38} "
            f"n={row.n:<2d} cost={row.travel_mode_cost_mean:10.3f} "
            f"runtime={row.runtime_ms_mean:10.3f} ms "
            f"hard={row.hard_penalty_mean:5.2f} "
            f"stability={row.stability_penalty_mean:5.2f}"
        )


if __name__ == "__main__":
    main()
