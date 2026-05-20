from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import load_chapter2_config_from_zip, load_chapter2_layouts_from_zip
from cnc_cutting.io import tool_config_from_chapter2_config
from cnc_cutting.models import Panel
from cnc_cutting.optimizer import plan_local_search_route, plan_topology_route
from process_options import add_stability_model_args, build_process_model_from_args
from run_chapter2_batch import (
    DEFAULT_ARCHIVES,
    board_counts_from_zip,
    compact_local_search_config,
    parse_member_metadata,
    sample_placement_members,
    select_board_ids,
    topology_pool_size,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COSTS = (0.0, 100.0, 250.0, 500.0, 1000.0)
DEFAULT_METHODS = (
    "topology_process_aware",
    "topology_local_search_process_aware",
)
METHOD_LABELS = {
    "topology_process_aware": "Process-aware",
    "topology_local_search_process_aware": "Process-aware+LS",
}
METHOD_COLORS = {
    "topology_process_aware": "#264653",
    "topology_local_search_process_aware": "#E76F51",
}
METHOD_MARKERS = {
    "topology_process_aware": "D",
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
    }
)


@dataclass(frozen=True)
class SensitivityRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    method: str
    safe_lift_fixed_cost: float
    runtime_ms: float
    rectangle_count: int
    air_move_distance: float
    travel_mode_cost: float
    hard_penalty: float
    stability_penalty: float
    safe_lift_count: int
    safe_lift_distance: float
    detour_count: int
    detour_distance: float


@dataclass(frozen=True)
class SensitivitySummaryRow:
    method: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    safe_lift_fixed_cost: float
    n: int
    runtime_ms_mean: float
    air_move_distance_mean: float
    travel_mode_cost_mean: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    safe_lift_count_mean: float
    safe_lift_distance_mean: float
    detour_count_mean: float
    detour_distance_mean: float


def run_method(
    archive: Path,
    placements_member: str,
    board_id: str,
    method: str,
    safe_lift_fixed_cost: float,
    stability_args: argparse.Namespace,
) -> SensitivityRow:
    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    base_tool = tool_config_from_chapter2_config(cfg)
    tool = replace(base_tool, safe_lift_fixed_cost=safe_lift_fixed_cost)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=placements_member,
        board_ids=(board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, stability_args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    size = len(layout.rectangles)

    start = perf_counter()
    if method == "topology_process_aware":
        plan = plan_topology_route(
            units,
            panel,
            tool,
            process_model=process_model,
            candidate_pool_size=topology_pool_size(size),
            process_aware=True,
        )
    elif method == "topology_local_search_process_aware":
        plan = plan_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    else:
        raise ValueError(f"unsupported method: {method}")
    runtime_ms = (perf_counter() - start) * 1000.0

    case_name, placement_method, seed = parse_member_metadata(placements_member)
    return SensitivityRow(
        archive=archive.name,
        case_name=case_name,
        placement_method=placement_method,
        seed=seed,
        placements_member=placements_member,
        board_id=board_id,
        support_policy=stability_args.support_policy,
        min_support_count=stability_args.min_support_count,
        min_support_ratio=stability_args.min_support_ratio,
        min_area_normalized_support=stability_args.min_area_normalized_support,
        adjacency_support_weight=stability_args.adjacency_support_weight,
        method=method,
        safe_lift_fixed_cost=safe_lift_fixed_cost,
        runtime_ms=runtime_ms,
        rectangle_count=len(layout.rectangles),
        air_move_distance=plan.metrics.air_move_distance,
        travel_mode_cost=plan.metrics.travel_mode_cost,
        hard_penalty=plan.metrics.hard_penalty,
        stability_penalty=plan.metrics.stability_penalty,
        safe_lift_count=plan.metrics.safe_lift_count,
        safe_lift_distance=plan.metrics.safe_lift_distance,
        detour_count=plan.metrics.detour_count,
        detour_distance=plan.metrics.detour_distance,
    )


def sample_cases(
    archives: tuple[Path, ...],
    max_members_per_archive: int,
    boards_per_member: int,
    min_rectangles: int,
    max_rectangles: int | None,
) -> tuple[tuple[Path, str, str], ...]:
    cases: list[tuple[Path, str, str]] = []
    for archive in archives:
        for member in sample_placement_members(archive, max_members_per_archive):
            counts = board_counts_from_zip(archive, member)
            board_ids = select_board_ids(
                counts,
                boards_per_member=boards_per_member,
                min_rectangles=min_rectangles,
                max_rectangles=max_rectangles,
            )
            for board_id in board_ids:
                cases.append((archive, member, board_id))
    return tuple(cases)


def summarize(rows: tuple[SensitivityRow, ...]) -> tuple[SensitivitySummaryRow, ...]:
    groups: dict[tuple[str, float], list[SensitivityRow]] = defaultdict(list)
    for row in rows:
        groups[(row.method, row.safe_lift_fixed_cost)].append(row)

    summary: list[SensitivitySummaryRow] = []
    for (method, cost), items in sorted(groups.items()):
        summary.append(
            SensitivitySummaryRow(
                method=method,
                support_policy=items[0].support_policy,
                min_support_count=items[0].min_support_count,
                min_support_ratio=items[0].min_support_ratio,
                min_area_normalized_support=items[0].min_area_normalized_support,
                adjacency_support_weight=items[0].adjacency_support_weight,
                safe_lift_fixed_cost=cost,
                n=len(items),
                runtime_ms_mean=mean(row.runtime_ms for row in items),
                air_move_distance_mean=mean(row.air_move_distance for row in items),
                travel_mode_cost_mean=mean(row.travel_mode_cost for row in items),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(row.stability_penalty for row in items),
                safe_lift_count_mean=mean(row.safe_lift_count for row in items),
                safe_lift_distance_mean=mean(row.safe_lift_distance for row in items),
                detour_count_mean=mean(row.detour_count for row in items),
                detour_distance_mean=mean(row.detour_distance for row in items),
            )
        )
    return tuple(summary)


def write_dataclass_rows(rows: tuple, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_sensitivity(
    summary: tuple[SensitivitySummaryRow, ...],
    output_dir: Path,
) -> None:
    metric_specs = (
        ("safe_lift_count_mean", "Safe-lift count"),
        ("detour_count_mean", "Detour count"),
        ("travel_mode_cost_mean", "Travel-mode cost"),
        ("runtime_ms_mean", "Runtime (ms)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    for ax, (metric, ylabel) in zip(axes.flat, metric_specs):
        for method in DEFAULT_METHODS:
            rows = sorted(
                [row for row in summary if row.method == method],
                key=lambda row: row.safe_lift_fixed_cost,
            )
            if not rows:
                continue
            ax.plot(
                [row.safe_lift_fixed_cost for row in rows],
                [getattr(row, metric) for row in rows],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.7,
                markersize=5,
            )
        ax.set_xlabel("Safe-lift fixed cost")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Sensitivity to safe-lift fixed cost", y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_safe_lift_fixed_cost_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, action="append", dest="zip_paths")
    parser.add_argument("--max-members-per-archive", type=int, default=1)
    parser.add_argument("--boards-per-member", type=int, default=1)
    parser.add_argument("--min-rectangles", type=int, default=2)
    parser.add_argument("--max-rectangles", type=int, default=20)
    parser.add_argument("--costs", nargs="+", type=float, default=list(DEFAULT_COSTS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "safe_lift_sensitivity.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "safe_lift_sensitivity_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures",
    )
    add_stability_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archives = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    methods = tuple(args.methods)
    cases = sample_cases(
        archives,
        max_members_per_archive=args.max_members_per_archive,
        boards_per_member=args.boards_per_member,
        min_rectangles=args.min_rectangles,
        max_rectangles=args.max_rectangles,
    )
    if not cases:
        raise ValueError("no sensitivity cases produced; relax filters or check archives")

    rows: list[SensitivityRow] = []
    print(f"sampled_cases: {len(cases)}")
    for archive, member, board_id in cases:
        print(f"case: {archive.name} board={board_id} member={member}")
        for cost in args.costs:
            for method in methods:
                rows.append(run_method(archive, member, board_id, method, cost, args))

    summary = summarize(tuple(rows))
    write_dataclass_rows(tuple(rows), args.output)
    write_dataclass_rows(summary, args.summary_output)
    plot_sensitivity(summary, args.figure_dir)

    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote figures to: {args.figure_dir}")
    for row in summary:
        print(
            f"{row.method:<36} cost={row.safe_lift_fixed_cost:>7.1f} "
            f"safe_lift={row.safe_lift_count_mean:>6.3f} "
            f"detour={row.detour_count_mean:>6.3f} "
            f"mode_cost={row.travel_mode_cost_mean:>10.3f}"
        )


if __name__ == "__main__":
    main()
