from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.exact_dp import ExactDPConfig, exact_process_dp_order
from cnc_cutting.io import (
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import process_metric_key
from cnc_cutting.models import Layout, Panel, PathMetrics
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_process_aware_beam_adaptive_polished_route,
    plan_process_aware_beam_adaptive_route,
    plan_process_aware_beam_polished_route,
    plan_process_aware_beam_route,
    plan_process_local_search_multistart_route,
    plan_topology_route,
    select_coverage_units,
)
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)
from run_chapter2_batch import (
    DEFAULT_ARCHIVES,
    board_counts_from_zip,
    compact_beam_search_config,
    compact_local_search_config,
    parse_member_metadata,
    sample_placement_members,
    select_board_ids,
    topology_pool_size,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "exact_process_dp",
    "process_aware_beam",
    "process_aware_beam_adaptive",
    "process_aware_beam_polished",
    "process_aware_beam_adaptive_polished",
    "process_local_search_multistart",
    "topology_process_aware",
)


@dataclass(frozen=True)
class ExactGapRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    method: str
    rectangle_count: int
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    runtime_ms: float
    exact_expanded_nodes: int | None
    exact_retained_states: int | None
    hard_penalty: float
    stability_penalty: float
    travel_mode_cost: float
    machining_cost: float
    air_move_distance: float
    safe_lift_count: int
    detour_count: int
    tool_event_count: int
    exact_hard_penalty: float
    exact_stability_penalty: float
    exact_travel_mode_cost: float
    exact_machining_cost: float
    travel_mode_cost_gap: float
    machining_cost_gap: float
    travel_mode_cost_gap_ratio: float
    process_key_winner: str


@dataclass(frozen=True)
class ExactGapSummaryRow:
    method: str
    n: int
    runtime_ms_mean: float
    runtime_ms_std: float
    travel_mode_cost_mean: float
    travel_mode_cost_gap_mean: float
    travel_mode_cost_gap_ratio_mean: float
    machining_cost_gap_mean: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    process_key_win_count: int
    process_key_loss_count: int


def tool_event_count(metrics: PathMetrics) -> int:
    return metrics.pierce_count + metrics.lift_count + metrics.safe_lift_count


def build_nonexact_plan(
    method: str,
    units,
    panel: Panel,
    tool,
    process_model,
    size: int,
) -> RoutePlan:
    if method == "process_aware_beam":
        return plan_process_aware_beam_route(
            units,
            panel,
            tool,
            config=compact_beam_search_config(size),
            process_model=process_model,
        )
    if method == "process_aware_beam_adaptive":
        return plan_process_aware_beam_adaptive_route(
            units,
            panel,
            tool,
            beam_config=compact_beam_search_config(size),
            topology_candidate_pool_size=topology_pool_size(size),
            fallback_margin=1000.0,
            process_model=process_model,
        )
    if method == "process_aware_beam_polished":
        return plan_process_aware_beam_polished_route(
            units,
            panel,
            tool,
            beam_config=compact_beam_search_config(size),
            polish_config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    if method == "process_aware_beam_adaptive_polished":
        return plan_process_aware_beam_adaptive_polished_route(
            units,
            panel,
            tool,
            beam_config=compact_beam_search_config(size),
            polish_config=compact_local_search_config(size, True),
            topology_candidate_pool_size=topology_pool_size(size),
            fallback_margin=1000.0,
            process_model=process_model,
        )
    if method == "process_local_search_multistart":
        return plan_process_local_search_multistart_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    if method == "topology_process_aware":
        return plan_topology_route(
            units,
            panel,
            tool,
            candidate_pool_size=topology_pool_size(size),
            process_aware=True,
            process_model=process_model,
        )
    raise ValueError(f"unsupported method: {method}")


def process_key_winner(metrics: PathMetrics, exact_metrics: PathMetrics) -> str:
    key = process_metric_key(metrics)
    exact_key = process_metric_key(exact_metrics)
    if key < exact_key:
        return "target"
    if key > exact_key:
        return "exact"
    return "tie"


def row_from_metrics(
    *,
    archive: Path,
    placements_member: str,
    board_id: str,
    method: str,
    layout: Layout,
    candidate_unit_count: int,
    selected_unit_count: int,
    action_count: int,
    runtime_ms: float,
    metrics: PathMetrics,
    exact_metrics: PathMetrics,
    exact_expanded_nodes: int | None = None,
    exact_retained_states: int | None = None,
) -> ExactGapRow:
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    travel_gap = metrics.travel_mode_cost - exact_metrics.travel_mode_cost
    machining_gap = metrics.machining_cost - exact_metrics.machining_cost
    travel_gap_ratio = (
        travel_gap / exact_metrics.travel_mode_cost
        if exact_metrics.travel_mode_cost > 1e-9
        else 0.0
    )
    return ExactGapRow(
        archive=archive.name,
        case_name=case_name,
        placement_method=placement_method,
        seed=seed,
        placements_member=placements_member,
        board_id=board_id,
        method=method,
        rectangle_count=len(layout.rectangles),
        candidate_unit_count=candidate_unit_count,
        selected_unit_count=selected_unit_count,
        action_count=action_count,
        runtime_ms=runtime_ms,
        exact_expanded_nodes=exact_expanded_nodes,
        exact_retained_states=exact_retained_states,
        hard_penalty=metrics.hard_penalty,
        stability_penalty=metrics.stability_penalty,
        travel_mode_cost=metrics.travel_mode_cost,
        machining_cost=metrics.machining_cost,
        air_move_distance=metrics.air_move_distance,
        safe_lift_count=metrics.safe_lift_count,
        detour_count=metrics.detour_count,
        tool_event_count=tool_event_count(metrics),
        exact_hard_penalty=exact_metrics.hard_penalty,
        exact_stability_penalty=exact_metrics.stability_penalty,
        exact_travel_mode_cost=exact_metrics.travel_mode_cost,
        exact_machining_cost=exact_metrics.machining_cost,
        travel_mode_cost_gap=travel_gap,
        machining_cost_gap=machining_gap,
        travel_mode_cost_gap_ratio=travel_gap_ratio,
        process_key_winner=process_key_winner(metrics, exact_metrics),
    )


def run_case(
    archive: Path,
    placements_member: str,
    board_id: str,
    methods: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[ExactGapRow, ...]:
    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=placements_member,
        board_ids=(board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    selected_units = select_coverage_units(units)
    if len(selected_units) > args.max_exact_units:
        print(
            f"    skip exact: board={board_id} rectangles={len(layout.rectangles)} "
            f"selected_units={len(selected_units)} max_exact_units={args.max_exact_units}"
        )
        return ()

    start = perf_counter()
    exact = exact_process_dp_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=ExactDPConfig(max_units=args.max_exact_units),
    )
    exact_runtime_ms = (perf_counter() - start) * 1000.0
    rows = [
        row_from_metrics(
            archive=archive,
            placements_member=placements_member,
            board_id=board_id,
            method="exact_process_dp",
            layout=layout,
            candidate_unit_count=len(units),
            selected_unit_count=len(selected_units),
            action_count=len(exact.actions),
            runtime_ms=exact_runtime_ms,
            metrics=exact.metrics,
            exact_metrics=exact.metrics,
            exact_expanded_nodes=exact.expanded_nodes,
            exact_retained_states=exact.retained_states,
        )
    ]

    for method in methods:
        if method == "exact_process_dp":
            continue
        start = perf_counter()
        plan = build_nonexact_plan(
            method,
            units,
            panel,
            tool,
            process_model,
            len(layout.rectangles),
        )
        runtime_ms = (perf_counter() - start) * 1000.0
        rows.append(
            row_from_metrics(
                archive=archive,
                placements_member=placements_member,
                board_id=board_id,
                method=method,
                layout=layout,
                candidate_unit_count=len(units),
                selected_unit_count=len(plan.selected_units),
                action_count=len(plan.actions),
                runtime_ms=runtime_ms,
                metrics=plan.metrics,
                exact_metrics=exact.metrics,
            )
        )
    return tuple(rows)


def write_rows(rows: tuple[ExactGapRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_rows(rows: tuple[ExactGapRow, ...]) -> tuple[ExactGapSummaryRow, ...]:
    grouped: dict[str, list[ExactGapRow]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)

    summary: list[ExactGapSummaryRow] = []
    for method, items in sorted(grouped.items()):
        summary.append(
            ExactGapSummaryRow(
                method=method,
                n=len(items),
                runtime_ms_mean=mean(row.runtime_ms for row in items),
                runtime_ms_std=(
                    pstdev(row.runtime_ms for row in items)
                    if len(items) > 1
                    else 0.0
                ),
                travel_mode_cost_mean=mean(row.travel_mode_cost for row in items),
                travel_mode_cost_gap_mean=mean(
                    row.travel_mode_cost_gap for row in items
                ),
                travel_mode_cost_gap_ratio_mean=mean(
                    row.travel_mode_cost_gap_ratio for row in items
                ),
                machining_cost_gap_mean=mean(row.machining_cost_gap for row in items),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(row.stability_penalty for row in items),
                process_key_win_count=sum(
                    1 for row in items if row.process_key_winner == "target"
                ),
                process_key_loss_count=sum(
                    1 for row in items if row.process_key_winner == "exact"
                ),
            )
        )
    return tuple(summary)


def write_summary(rows: tuple[ExactGapSummaryRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, action="append", dest="zip_paths")
    parser.add_argument("--max-members-per-archive", type=int, default=2)
    parser.add_argument("--boards-per-member", type=int, default=2)
    parser.add_argument("--min-rectangles", type=int, default=1)
    parser.add_argument("--max-rectangles", type=int, default=3)
    parser.add_argument("--max-exact-units", type=int, default=12)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "exact_gap.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "exact_gap_summary.csv",
    )
    add_experiment_preset_arg(parser)
    add_stability_model_args(parser)
    return apply_experiment_preset(parser.parse_args())


def main() -> None:
    args = parse_args()
    archives = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    methods = tuple(args.methods)
    rows: list[ExactGapRow] = []

    for archive in archives:
        members = sample_placement_members(archive, args.max_members_per_archive)
        print(f"archive: {archive.name}, sampled_members={len(members)}")
        for member in members:
            counts = board_counts_from_zip(archive, member)
            board_ids = select_board_ids(
                counts,
                boards_per_member=args.boards_per_member,
                min_rectangles=args.min_rectangles,
                max_rectangles=args.max_rectangles,
            )
            print(
                f"  member: {member} boards={','.join(board_ids) if board_ids else 'none'}"
            )
            for board_id in board_ids:
                case_rows = run_case(archive, member, board_id, methods, args)
                rows.extend(case_rows)
                for row in case_rows:
                    print(
                        f"    {row.method:<32} board={board_id} "
                        f"runtime={row.runtime_ms:>8.3f} ms "
                        f"gap={row.travel_mode_cost_gap:>9.3f} "
                        f"stability={row.stability_penalty:>6.3f}"
                    )

    if not rows:
        raise ValueError("no exact-gap rows produced; relax filters or increase max units")

    output_rows = tuple(rows)
    summary_rows = summarize_rows(output_rows)
    write_rows(output_rows, args.output)
    write_summary(summary_rows, args.summary_output)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")


if __name__ == "__main__":
    main()
