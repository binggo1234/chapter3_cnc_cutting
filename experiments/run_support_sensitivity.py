from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from types import SimpleNamespace

import matplotlib.pyplot as plt

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import load_chapter2_config_from_zip, load_chapter2_layouts_from_zip
from cnc_cutting.io import tool_config_from_chapter2_config
from cnc_cutting.models import Layout, Panel
from process_options import SUPPORT_POLICIES, build_process_model_from_args
from run_chapter2_batch import (
    DEFAULT_ARCHIVES,
    DEFAULT_METHODS,
    board_counts_from_zip,
    build_planners,
    parse_member_metadata,
    sample_placement_members,
    select_board_ids,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_ADJACENCY_WEIGHTS = (0.0,)
DEFAULT_SUPPORT_POLICIES = ("all_edges",)
DEFAULT_SENSITIVITY_METHODS = (
    "path_distance_local_search",
    "topology",
    "topology_process_aware",
    "process_aware_beam",
    "topology_local_search_process_aware",
)
METHOD_LABELS = {
    "greedy": "Greedy",
    "path_distance_local_search": "Path-LS",
    "topology": "Topology",
    "topology_process_aware": "Process-aware",
    "process_aware_beam": "Process-aware beam",
    "topology_local_search": "Topology+LS",
    "topology_local_search_process_aware": "Process-aware+LS",
}
METHOD_COLORS = {
    "greedy": "#999999",
    "path_distance_local_search": "#CC79A7",
    "topology": "#0072B2",
    "topology_process_aware": "#D55E00",
    "process_aware_beam": "#E69F00",
    "topology_local_search": "#56B4E9",
    "topology_local_search_process_aware": "#009E73",
}
METHOD_MARKERS = {
    "greedy": "o",
    "path_distance_local_search": "v",
    "topology": "s",
    "topology_process_aware": "D",
    "process_aware_beam": "X",
    "topology_local_search": "^",
    "topology_local_search_process_aware": "P",
}
ADJACENCY_LINESTYLES = {
    0.0: "--",
    1.0: "-",
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
class SupportSensitivityRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    method: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    runtime_ms: float
    rectangle_count: int
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    travel_mode_cost: float
    hard_penalty: float
    stability_penalty: float
    safe_lift_count: int
    detour_count: int


@dataclass(frozen=True)
class SupportSensitivitySummaryRow:
    method: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    n: int
    runtime_ms_mean: float
    runtime_ms_std: float
    air_move_distance_mean: float
    air_move_distance_std: float
    travel_mode_cost_mean: float
    travel_mode_cost_std: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    stability_penalty_std: float
    safe_lift_count_mean: float
    detour_count_mean: float
    rectangle_count_mean: float


def process_args(
    support_policy: str,
    min_support_count: int,
    min_support_ratio: float,
    min_area_normalized_support: float,
    adjacency_support_weight: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        support_policy=support_policy,
        min_support_count=min_support_count,
        min_support_ratio=min_support_ratio,
        min_area_normalized_support=min_area_normalized_support,
        adjacency_support_weight=adjacency_support_weight,
    )


def run_case(
    archive: Path,
    placements_member: str,
    board_id: str,
    methods: tuple[str, ...],
    support_policy: str,
    min_support_count: int,
    min_support_ratio: float,
    min_area_normalized_support: float,
    adjacency_support_weight: float,
) -> tuple[SupportSensitivityRow, ...]:
    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=placements_member,
        board_ids=(board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(
        layout,
        process_args(
            support_policy,
            min_support_count,
            min_support_ratio,
            min_area_normalized_support,
            adjacency_support_weight,
        ),
    )
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    rows: list[SupportSensitivityRow] = []

    for method, planner in build_planners(
        methods,
        units,
        panel,
        tool,
        process_model,
        len(layout.rectangles),
    ):
        start = perf_counter()
        plan = planner()
        runtime_ms = (perf_counter() - start) * 1000.0
        rows.append(
            SupportSensitivityRow(
                archive=archive.name,
                case_name=case_name,
                placement_method=placement_method,
                seed=seed,
                placements_member=placements_member,
                board_id=board_id,
                method=method,
                support_policy=support_policy,
                min_support_count=min_support_count,
                min_support_ratio=min_support_ratio,
                min_area_normalized_support=min_area_normalized_support,
                adjacency_support_weight=adjacency_support_weight,
                runtime_ms=runtime_ms,
                rectangle_count=len(layout.rectangles),
                candidate_unit_count=len(units),
                selected_unit_count=len(plan.selected_units),
                action_count=len(plan.actions),
                air_move_distance=plan.metrics.air_move_distance,
                travel_mode_cost=plan.metrics.travel_mode_cost,
                hard_penalty=plan.metrics.hard_penalty,
                stability_penalty=plan.metrics.stability_penalty,
                safe_lift_count=plan.metrics.safe_lift_count,
                detour_count=plan.metrics.detour_count,
            )
        )
    return tuple(rows)


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


def summarize(
    rows: tuple[SupportSensitivityRow, ...],
) -> tuple[SupportSensitivitySummaryRow, ...]:
    groups: dict[tuple[str, str, int, float, float, float], list[SupportSensitivityRow]] = (
        defaultdict(list)
    )
    for row in rows:
        groups[
            (
                row.method,
                row.support_policy,
                row.min_support_count,
                row.min_support_ratio,
                row.min_area_normalized_support,
                row.adjacency_support_weight,
            )
        ].append(row)

    summary: list[SupportSensitivitySummaryRow] = []
    for (
        method,
        support_policy,
        min_support_count,
        min_support_ratio,
        min_area_normalized_support,
        adjacency_support_weight,
    ), items in sorted(groups.items()):
        runtime_values = [row.runtime_ms for row in items]
        air_values = [row.air_move_distance for row in items]
        mode_values = [row.travel_mode_cost for row in items]
        stability_values = [row.stability_penalty for row in items]
        summary.append(
            SupportSensitivitySummaryRow(
                method=method,
                support_policy=support_policy,
                min_support_count=min_support_count,
                min_support_ratio=min_support_ratio,
                min_area_normalized_support=min_area_normalized_support,
                adjacency_support_weight=adjacency_support_weight,
                n=len(items),
                runtime_ms_mean=mean(runtime_values),
                runtime_ms_std=pstdev(runtime_values),
                air_move_distance_mean=mean(air_values),
                air_move_distance_std=pstdev(air_values),
                travel_mode_cost_mean=mean(mode_values),
                travel_mode_cost_std=pstdev(mode_values),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(stability_values),
                stability_penalty_std=pstdev(stability_values),
                safe_lift_count_mean=mean(row.safe_lift_count for row in items),
                detour_count_mean=mean(row.detour_count for row in items),
                rectangle_count_mean=mean(row.rectangle_count for row in items),
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


def plot_support_sensitivity(
    summary: tuple[SupportSensitivitySummaryRow, ...],
    output_dir: Path,
) -> None:
    settings = sorted(
        {
            (row.support_policy, row.adjacency_support_weight)
            for row in summary
        }
    )
    for policy, adjacency_weight in settings:
        rows_for_policy = [
            row
            for row in summary
            if row.support_policy == policy
            and row.adjacency_support_weight == adjacency_weight
        ]
        metric_specs = (
            ("stability_penalty_mean", "Stability penalty"),
            ("air_move_distance_mean", "Air-move distance"),
            ("travel_mode_cost_mean", "Travel-mode cost"),
            ("runtime_ms_mean", "Runtime (ms)"),
        )
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))
        for ax, (metric, ylabel) in zip(axes.flat, metric_specs):
            for method in sorted({row.method for row in rows_for_policy}):
                method_rows = sorted(
                    [row for row in rows_for_policy if row.method == method],
                    key=lambda row: row.min_support_ratio,
                )
                if not method_rows:
                    continue
                ax.plot(
                    [row.min_support_ratio for row in method_rows],
                    [getattr(row, metric) for row in method_rows],
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "#333333"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linewidth=1.7,
                    markersize=5,
                )
            ax.set_xlabel("Minimum remaining support ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
        axes[0, 0].legend(loc="best")
        fig.suptitle(
            (
                f"Support-constraint sensitivity ({policy}, "
                f"adj. weight={adjacency_weight:g})"
            ),
            y=1.02,
            fontsize=11,
        )
        fig.tight_layout()
        save_figure(
            fig,
            output_dir,
            f"fig_support_constraint_sensitivity_{policy}_adj{adjacency_weight:g}",
        )


def adjacency_linestyle(weight: float) -> str:
    return ADJACENCY_LINESTYLES.get(weight, ":")


def plot_adjacency_support_ablation(
    summary: tuple[SupportSensitivitySummaryRow, ...],
    output_dir: Path,
) -> None:
    policies = sorted({row.support_policy for row in summary})
    for policy in policies:
        rows_for_policy = [row for row in summary if row.support_policy == policy]
        adjacency_weights = sorted({row.adjacency_support_weight for row in rows_for_policy})
        if len(adjacency_weights) < 2:
            continue

        metric_specs = (
            ("stability_penalty_mean", "Stability penalty"),
            ("air_move_distance_mean", "Air-move distance"),
            ("travel_mode_cost_mean", "Travel-mode cost"),
            ("runtime_ms_mean", "Runtime (ms)"),
        )
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
        for ax, (metric, ylabel) in zip(axes.flat, metric_specs):
            for method in sorted({row.method for row in rows_for_policy}):
                for weight in adjacency_weights:
                    method_rows = sorted(
                        [
                            row
                            for row in rows_for_policy
                            if row.method == method
                            and row.adjacency_support_weight == weight
                        ],
                        key=lambda row: row.min_support_ratio,
                    )
                    if not method_rows:
                        continue
                    ax.plot(
                        [row.min_support_ratio for row in method_rows],
                        [getattr(row, metric) for row in method_rows],
                        label=(
                            f"{METHOD_LABELS.get(method, method)}, "
                            f"adj={weight:g}"
                        ),
                        color=METHOD_COLORS.get(method, "#333333"),
                        marker=METHOD_MARKERS.get(method, "o"),
                        linestyle=adjacency_linestyle(weight),
                        linewidth=1.7,
                        markersize=5,
                        alpha=0.95 if weight > 0 else 0.72,
                    )
            ax.set_xlabel("Minimum remaining support ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
        axes[0, 0].legend(loc="best", ncol=1)
        fig.suptitle(
            f"Adjacency-support ablation ({policy})",
            y=1.02,
            fontsize=11,
        )
        fig.tight_layout()
        save_figure(
            fig,
            output_dir,
            f"fig_adjacency_support_ablation_{policy}",
        )


def plot_adjacency_support_ablation_compact(
    summary: tuple[SupportSensitivitySummaryRow, ...],
    output_dir: Path,
) -> None:
    policies = sorted({row.support_policy for row in summary})
    for policy in policies:
        rows_for_policy = [row for row in summary if row.support_policy == policy]
        adjacency_weights = sorted({row.adjacency_support_weight for row in rows_for_policy})
        if len(adjacency_weights) < 2:
            continue

        metric_specs = (
            ("stability_penalty_mean", "Stability penalty"),
            ("air_move_distance_mean", "Air-move distance"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
        for ax, (metric, ylabel) in zip(axes, metric_specs):
            for method in sorted({row.method for row in rows_for_policy}):
                for weight in adjacency_weights:
                    method_rows = sorted(
                        [
                            row
                            for row in rows_for_policy
                            if row.method == method
                            and row.adjacency_support_weight == weight
                        ],
                        key=lambda row: row.min_support_ratio,
                    )
                    if not method_rows:
                        continue
                    ax.plot(
                        [row.min_support_ratio for row in method_rows],
                        [getattr(row, metric) for row in method_rows],
                        label=(
                            f"{METHOD_LABELS.get(method, method)}, "
                            f"adj={weight:g}"
                        ),
                        color=METHOD_COLORS.get(method, "#333333"),
                        marker=METHOD_MARKERS.get(method, "o"),
                        linestyle=adjacency_linestyle(weight),
                        linewidth=1.7,
                        markersize=5,
                        alpha=0.96 if weight > 0 else 0.72,
                    )
            ax.set_xlabel("Minimum remaining support ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
        axes[0].legend(loc="best", fontsize=7.2)
        fig.suptitle(
            f"Adjacency-support ablation ({policy})",
            y=1.04,
            fontsize=11,
        )
        fig.tight_layout()
        save_figure(
            fig,
            output_dir,
            f"fig_adjacency_support_ablation_compact_{policy}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, action="append", dest="zip_paths")
    parser.add_argument("--max-members-per-archive", type=int, default=1)
    parser.add_argument("--boards-per-member", type=int, default=1)
    parser.add_argument("--min-rectangles", type=int, default=2)
    parser.add_argument("--max-rectangles", type=int, default=20)
    parser.add_argument(
        "--support-policy",
        nargs="+",
        choices=SUPPORT_POLICIES,
        default=list(DEFAULT_SUPPORT_POLICIES),
    )
    parser.add_argument("--min-support-count", type=int, default=1)
    parser.add_argument(
        "--min-support-ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_RATIOS),
    )
    parser.add_argument(
        "--adjacency-support-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_ADJACENCY_WEIGHTS),
    )
    parser.add_argument("--min-area-normalized-support", type=float, default=0.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_SENSITIVITY_METHODS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "support_sensitivity.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "support_sensitivity_summary.csv",
    )
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archives = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    methods = tuple(args.methods)
    support_policies = tuple(args.support_policy)
    support_ratios = tuple(args.min_support_ratios)
    adjacency_weights = tuple(args.adjacency_support_weights)
    cases = sample_cases(
        archives,
        max_members_per_archive=args.max_members_per_archive,
        boards_per_member=args.boards_per_member,
        min_rectangles=args.min_rectangles,
        max_rectangles=args.max_rectangles,
    )
    if not cases:
        raise ValueError("no support-sensitivity cases produced; relax filters")

    rows: list[SupportSensitivityRow] = []
    print(f"sampled_cases: {len(cases)}")
    for archive, member, board_id in cases:
        print(f"case: {archive.name} board={board_id} member={member}")
        for support_policy in support_policies:
            for ratio in support_ratios:
                for adjacency_weight in adjacency_weights:
                    rows.extend(
                        run_case(
                            archive,
                            member,
                            board_id,
                            methods,
                            support_policy=support_policy,
                            min_support_count=args.min_support_count,
                            min_support_ratio=ratio,
                            min_area_normalized_support=args.min_area_normalized_support,
                            adjacency_support_weight=adjacency_weight,
                        )
                    )

    result_rows = tuple(rows)
    summary = summarize(result_rows)
    write_dataclass_rows(result_rows, args.output)
    write_dataclass_rows(summary, args.summary_output)
    plot_support_sensitivity(summary, args.figure_dir)
    plot_adjacency_support_ablation(summary, args.figure_dir)
    plot_adjacency_support_ablation_compact(summary, args.figure_dir)

    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote figures to: {args.figure_dir}")
    for row in summary:
        print(
            f"{row.method:<36} policy={row.support_policy:<9} "
            f"ratio={row.min_support_ratio:>4.2f} "
            f"adj={row.adjacency_support_weight:>4.2f} "
            f"stability={row.stability_penalty_mean:>6.3f} "
            f"air={row.air_move_distance_mean:>10.3f} "
            f"runtime={row.runtime_ms_mean:>9.3f} ms"
        )


if __name__ == "__main__":
    main()
