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
from cnc_cutting.local_search import BeamSearchConfig, process_aware_beam_search_order
from cnc_cutting.models import Panel
from cnc_cutting.optimizer import select_coverage_units
from cnc_cutting.travel import clear_detour_cache
from process_options import build_process_model_from_args
from run_chapter2_batch import (
    DEFAULT_ARCHIVES,
    board_counts_from_zip,
    parse_member_metadata,
    sample_placement_members,
    select_board_ids,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEAM_WIDTHS = (1, 2, 4, 8)
DEFAULT_LAYER_MULTIPLIERS = (6, 7, 12)
DEFAULT_DIVERSITY_LIMITS = (0, 1, 2)
DEFAULT_UNSTABLE_LAYER_MULTIPLIERS = (1.0, 1.5)

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
class BeamSensitivityRow:
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
    beam_width: int
    candidate_pool_size: int
    max_expansions_per_node: int
    max_layer_expansions: int
    layer_multiplier: int
    unstable_layer_expansion_multiplier: float
    unstable_layer_expansion_bonus: int
    diversity_bucket_limit: int
    min_expansions_per_parent: int
    unstable_min_expansions_per_parent: int
    runtime_ms: float
    expanded_nodes: int
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
class BeamSensitivitySummaryRow:
    beam_width: int
    candidate_pool_size: int
    max_expansions_per_node: int
    max_layer_expansions: int
    layer_multiplier: int
    unstable_layer_expansion_multiplier: float
    unstable_layer_expansion_bonus: int
    diversity_bucket_limit: int
    min_expansions_per_parent: int
    unstable_min_expansions_per_parent: int
    n: int
    runtime_ms_mean: float
    runtime_ms_std: float
    expanded_nodes_mean: float
    air_move_distance_mean: float
    travel_mode_cost_mean: float
    hard_penalty_mean: float
    stability_penalty_mean: float
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


def candidate_pool_size(size: int) -> int:
    if size <= 20:
        return 24
    if size <= 75:
        return 18
    return 12


def node_expansion_limit(size: int) -> int:
    if size <= 20:
        return 48
    if size <= 75:
        return 36
    return 24


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


def run_case(
    archive: Path,
    placements_member: str,
    board_id: str,
    stability_args: argparse.Namespace,
    beam_width: int,
    layer_multiplier: int,
    unstable_layer_expansion_multiplier: float,
    unstable_layer_expansion_bonus: int,
    diversity_bucket_limit: int,
    min_expansions_per_parent: int,
    unstable_min_expansions_per_parent: int,
) -> BeamSensitivityRow:
    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    tool = tool_config_from_chapter2_config(cfg)
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
    selected_units = select_coverage_units(units)
    pool_size = candidate_pool_size(len(layout.rectangles))
    max_per_node = node_expansion_limit(len(layout.rectangles))
    max_layer = max(beam_width, beam_width * layer_multiplier)
    config = BeamSearchConfig(
        beam_width=beam_width,
        candidate_pool_size=pool_size,
        max_expansions_per_node=max_per_node,
        max_layer_expansions=max_layer,
        unstable_layer_expansion_multiplier=unstable_layer_expansion_multiplier,
        unstable_layer_expansion_bonus=unstable_layer_expansion_bonus,
        diversity_bucket_limit=(
            diversity_bucket_limit if diversity_bucket_limit > 0 else None
        ),
        min_expansions_per_parent=min_expansions_per_parent,
        unstable_min_expansions_per_parent=unstable_min_expansions_per_parent,
    )

    clear_detour_cache()
    start = perf_counter()
    result = process_aware_beam_search_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=config,
    )
    runtime_ms = (perf_counter() - start) * 1000.0
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    return BeamSensitivityRow(
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
        beam_width=beam_width,
        candidate_pool_size=pool_size,
        max_expansions_per_node=max_per_node,
        max_layer_expansions=max_layer,
        layer_multiplier=layer_multiplier,
        unstable_layer_expansion_multiplier=unstable_layer_expansion_multiplier,
        unstable_layer_expansion_bonus=unstable_layer_expansion_bonus,
        diversity_bucket_limit=diversity_bucket_limit,
        min_expansions_per_parent=min_expansions_per_parent,
        unstable_min_expansions_per_parent=unstable_min_expansions_per_parent,
        runtime_ms=runtime_ms,
        expanded_nodes=result.expanded_nodes,
        rectangle_count=len(layout.rectangles),
        candidate_unit_count=len(units),
        selected_unit_count=len(selected_units),
        action_count=len(result.actions),
        air_move_distance=result.metrics.air_move_distance,
        travel_mode_cost=result.metrics.travel_mode_cost,
        hard_penalty=result.metrics.hard_penalty,
        stability_penalty=result.metrics.stability_penalty,
        safe_lift_count=result.metrics.safe_lift_count,
        detour_count=result.metrics.detour_count,
    )


def summarize(
    rows: tuple[BeamSensitivityRow, ...],
) -> tuple[BeamSensitivitySummaryRow, ...]:
    groups: dict[
        tuple[int, int, int, int, int, float, int, int, int, int],
        list[BeamSensitivityRow],
    ] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.beam_width,
                row.candidate_pool_size,
                row.max_expansions_per_node,
                row.max_layer_expansions,
                row.layer_multiplier,
                row.unstable_layer_expansion_multiplier,
                row.unstable_layer_expansion_bonus,
                row.diversity_bucket_limit,
                row.min_expansions_per_parent,
                row.unstable_min_expansions_per_parent,
            )
        ].append(row)

    summary: list[BeamSensitivitySummaryRow] = []
    for (
        beam_width,
        pool_size,
        max_per_node,
        max_layer,
        layer_multiplier,
        unstable_layer_expansion_multiplier,
        unstable_layer_expansion_bonus,
        diversity_bucket_limit,
        min_expansions_per_parent,
        unstable_min_expansions_per_parent,
    ), items in sorted(groups.items()):
        runtime_values = [row.runtime_ms for row in items]
        summary.append(
            BeamSensitivitySummaryRow(
                beam_width=beam_width,
                candidate_pool_size=pool_size,
                max_expansions_per_node=max_per_node,
                max_layer_expansions=max_layer,
                layer_multiplier=layer_multiplier,
                unstable_layer_expansion_multiplier=unstable_layer_expansion_multiplier,
                unstable_layer_expansion_bonus=unstable_layer_expansion_bonus,
                diversity_bucket_limit=diversity_bucket_limit,
                min_expansions_per_parent=min_expansions_per_parent,
                unstable_min_expansions_per_parent=unstable_min_expansions_per_parent,
                n=len(items),
                runtime_ms_mean=mean(runtime_values),
                runtime_ms_std=pstdev(runtime_values),
                expanded_nodes_mean=mean(row.expanded_nodes for row in items),
                air_move_distance_mean=mean(row.air_move_distance for row in items),
                travel_mode_cost_mean=mean(row.travel_mode_cost for row in items),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(row.stability_penalty for row in items),
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


def plot_summary(
    summary: tuple[BeamSensitivitySummaryRow, ...],
    output_dir: Path,
) -> None:
    metric_specs = (
        ("travel_mode_cost_mean", "Travel-mode cost"),
        ("stability_penalty_mean", "Stability penalty"),
        ("runtime_ms_mean", "Runtime (ms)"),
        ("expanded_nodes_mean", "Expanded nodes"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))
    series_keys = sorted(
        {
            (
                row.layer_multiplier,
                row.unstable_layer_expansion_multiplier,
                row.unstable_layer_expansion_bonus,
                row.diversity_bucket_limit,
            )
            for row in summary
        }
    )
    color_cycle = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    markers = {0: "o", 1: "s", 2: "D", 3: "^"}
    for ax, (metric, ylabel) in zip(axes.flat, metric_specs):
        for series_index, (
            multiplier,
            unstable_multiplier,
            unstable_bonus,
            diversity_limit,
        ) in enumerate(series_keys):
            rows = sorted(
                [
                    row
                    for row in summary
                    if row.layer_multiplier == multiplier
                    and row.unstable_layer_expansion_multiplier == unstable_multiplier
                    and row.unstable_layer_expansion_bonus == unstable_bonus
                    and row.diversity_bucket_limit == diversity_limit
                ],
                key=lambda row: row.beam_width,
            )
            ax.plot(
                [row.beam_width for row in rows],
                [getattr(row, metric) for row in rows],
                color=color_cycle[series_index % len(color_cycle)],
                marker=markers.get(series_index % len(markers), "o"),
                linewidth=1.7,
                markersize=5,
                label=(
                    f"layer={multiplier}x, div="
                    f"{diversity_limit if diversity_limit > 0 else 'off'}, "
                    f"u={unstable_multiplier:g}+{unstable_bonus}"
                ),
            )
        ax.set_xlabel("Beam width")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Process-aware beam search sensitivity", y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_process_aware_beam_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, action="append", dest="zip_paths")
    parser.add_argument("--max-members-per-archive", type=int, default=1)
    parser.add_argument("--boards-per-member", type=int, default=1)
    parser.add_argument("--min-rectangles", type=int, default=2)
    parser.add_argument("--max-rectangles", type=int, default=20)
    parser.add_argument("--support-policy", default="all_edges")
    parser.add_argument("--min-support-count", type=int, default=1)
    parser.add_argument("--min-support-ratio", type=float, default=0.75)
    parser.add_argument("--min-area-normalized-support", type=float, default=0.0)
    parser.add_argument("--adjacency-support-weight", type=float, default=1.0)
    parser.add_argument("--beam-widths", nargs="+", type=int, default=list(DEFAULT_BEAM_WIDTHS))
    parser.add_argument(
        "--layer-multipliers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYER_MULTIPLIERS),
    )
    parser.add_argument(
        "--unstable-layer-expansion-multipliers",
        nargs="+",
        type=float,
        default=list(DEFAULT_UNSTABLE_LAYER_MULTIPLIERS),
    )
    parser.add_argument("--unstable-layer-expansion-bonus", type=int, default=0)
    parser.add_argument(
        "--diversity-bucket-limits",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIVERSITY_LIMITS),
    )
    parser.add_argument("--min-expansions-per-parent", type=int, default=0)
    parser.add_argument("--unstable-min-expansions-per-parent", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "beam_sensitivity.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "beam_sensitivity_summary.csv",
    )
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archives = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    cases = sample_cases(
        archives,
        max_members_per_archive=args.max_members_per_archive,
        boards_per_member=args.boards_per_member,
        min_rectangles=args.min_rectangles,
        max_rectangles=args.max_rectangles,
    )
    if not cases:
        raise ValueError("no beam-sensitivity cases produced; relax filters")

    rows: list[BeamSensitivityRow] = []
    print(f"sampled_cases: {len(cases)}")
    for archive, member, board_id in cases:
        print(f"case: {archive.name} board={board_id} member={member}")
        stability_args = process_args(
            args.support_policy,
            args.min_support_count,
            args.min_support_ratio,
            args.min_area_normalized_support,
            args.adjacency_support_weight,
        )
        for beam_width in args.beam_widths:
            for layer_multiplier in args.layer_multipliers:
                for unstable_multiplier in args.unstable_layer_expansion_multipliers:
                    for diversity_limit in args.diversity_bucket_limits:
                        rows.append(
                            run_case(
                                archive,
                                member,
                                board_id,
                                stability_args,
                                beam_width=beam_width,
                                layer_multiplier=layer_multiplier,
                                unstable_layer_expansion_multiplier=unstable_multiplier,
                                unstable_layer_expansion_bonus=(
                                    args.unstable_layer_expansion_bonus
                                ),
                                diversity_bucket_limit=diversity_limit,
                                min_expansions_per_parent=args.min_expansions_per_parent,
                                unstable_min_expansions_per_parent=(
                                    args.unstable_min_expansions_per_parent
                                ),
                            )
                        )

    result_rows = tuple(rows)
    summary = summarize(result_rows)
    write_dataclass_rows(result_rows, args.output)
    write_dataclass_rows(summary, args.summary_output)
    plot_summary(summary, args.figure_dir)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote figures to: {args.figure_dir}")
    for row in summary:
        print(
            f"beam={row.beam_width:<2} layer={row.layer_multiplier:<2} "
            f"unstable_layer={row.unstable_layer_expansion_multiplier:<3.1f}+"
            f"{row.unstable_layer_expansion_bonus:<2} "
            f"div={row.diversity_bucket_limit:<2} "
            f"parent={row.min_expansions_per_parent}/"
            f"{row.unstable_min_expansions_per_parent} "
            f"n={row.n:<3} runtime={row.runtime_ms_mean:>9.3f} ms "
            f"expanded={row.expanded_nodes_mean:>8.1f} "
            f"mode_cost={row.travel_mode_cost_mean:>10.3f} "
            f"stability={row.stability_penalty_mean:>6.3f}"
        )


if __name__ == "__main__":
    main()
