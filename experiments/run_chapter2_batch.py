from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable, Iterator

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import (
    discover_chapter2_placement_members,
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import BeamSearchConfig, LocalSearchConfig
from cnc_cutting.models import Layout, Panel
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_greedy_route,
    plan_path_distance_local_search_route,
    plan_process_aware_beam_route,
    plan_local_search_route,
    plan_topology_route,
)
from experiment_manifest import default_manifest_path, write_experiment_manifest
from progress_log import (
    append_progress_event,
    default_progress_log_path,
    new_progress_event,
    prepare_progress_log,
)
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)
from task_timeout import TaskTimeoutError, task_timeout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVES = (
    Path(
        "/Users/binggo/Desktop/codex_handoff_20260502/data/"
        "paper_extended_suite_20260321_continuous_run_only_20260324_0001.zip"
    ),
    Path("/Users/binggo/Desktop/codex_handoff_20260502/data/review_overnight_20260422.zip"),
    Path(
        "/Users/binggo/Desktop/codex_handoff_20260502/data/"
        "strong_baseline_overnight_20260407_001243.zip"
    ),
)
DEFAULT_METHODS = (
    "greedy",
    "path_distance_local_search",
    "topology",
    "topology_process_aware",
    "process_aware_beam",
    "topology_local_search",
    "topology_local_search_process_aware",
)


@dataclass(frozen=True)
class BatchRouteRow:
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
    topology_candidate_pool_size: int | None
    beam_width: int | None
    beam_candidate_pool_size: int | None
    beam_max_expansions_per_node: int | None
    beam_max_layer_expansions: int | None
    beam_diversity_bucket_limit: int | None
    beam_min_expansions_per_parent: int | None
    beam_unstable_min_expansions_per_parent: int | None
    beam_unstable_layer_expansion_multiplier: float | None
    beam_unstable_layer_expansion_bonus: int | None
    runtime_ms: float
    rectangle_count: int
    rectangle_count_bin: str
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    cutting_length: float
    pierce_count: int
    lift_count: int
    collision_penalty: float
    boundary_penalty: float
    stability_penalty: float
    safe_lift_count: int
    safe_lift_distance: float
    detour_count: int
    detour_distance: float
    travel_mode_cost: float
    hard_penalty: float


@dataclass(frozen=True)
class BatchSummaryRow:
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
    safe_lift_count_mean: float
    detour_count_mean: float
    rectangle_count_mean: float


@dataclass(frozen=True)
class BatchBinSummaryRow:
    rectangle_count_bin: str
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
    safe_lift_count_mean: float
    detour_count_mean: float
    rectangle_count_mean: float


@dataclass(frozen=True)
class PlannerSpec:
    method: str
    planner: Callable[[], RoutePlan]
    topology_candidate_pool_size: int | None = None
    beam_width: int | None = None
    beam_candidate_pool_size: int | None = None
    beam_max_expansions_per_node: int | None = None
    beam_max_layer_expansions: int | None = None
    beam_diversity_bucket_limit: int | None = None
    beam_min_expansions_per_parent: int | None = None
    beam_unstable_min_expansions_per_parent: int | None = None
    beam_unstable_layer_expansion_multiplier: float | None = None
    beam_unstable_layer_expansion_bonus: int | None = None


def compact_local_search_config(size: int, process_aware_initial_order: bool) -> LocalSearchConfig:
    if size <= 50:
        return LocalSearchConfig(
            max_iterations=2,
            max_swap_span=6,
            max_relocate_span=6,
            max_two_opt_span=6,
            max_neighbors_per_iteration=160,
            first_improvement=True,
            topology_candidate_pool_size=96,
            process_aware_initial_order=process_aware_initial_order,
        )
    return LocalSearchConfig(
        max_iterations=1,
        max_swap_span=5,
        max_relocate_span=5,
        max_two_opt_span=5,
        max_neighbors_per_iteration=120,
        first_improvement=True,
        topology_candidate_pool_size=64,
        process_aware_initial_order=process_aware_initial_order,
    )


def topology_pool_size(size: int) -> int:
    if size <= 50:
        return 96
    if size <= 200:
        return 64
    return 48


def compact_beam_search_config(size: int) -> BeamSearchConfig:
    if size <= 20:
        return BeamSearchConfig(
            beam_width=8,
            candidate_pool_size=24,
            max_expansions_per_node=48,
            max_layer_expansions=56,
            diversity_bucket_limit=1,
            min_expansions_per_parent=0,
            unstable_min_expansions_per_parent=2,
            unstable_layer_expansion_multiplier=1.0,
        )
    if size <= 75:
        return BeamSearchConfig(
            beam_width=6,
            candidate_pool_size=18,
            max_expansions_per_node=36,
            max_layer_expansions=42,
            diversity_bucket_limit=1,
            min_expansions_per_parent=0,
            unstable_min_expansions_per_parent=2,
            unstable_layer_expansion_multiplier=1.0,
        )
    return BeamSearchConfig(
        beam_width=4,
        candidate_pool_size=12,
        max_expansions_per_node=24,
        max_layer_expansions=28,
        diversity_bucket_limit=1,
        min_expansions_per_parent=0,
        unstable_min_expansions_per_parent=2,
        unstable_layer_expansion_multiplier=1.0,
    )


def rectangle_count_bin(count: int) -> str:
    if count <= 20:
        return "000_020"
    if count <= 50:
        return "021_050"
    if count <= 75:
        return "051_075"
    return "076_plus"


def parse_member_metadata(member: str) -> tuple[str, str, str]:
    parts = member.replace("\\", "/").split("/")
    seed = ""
    placement_method = ""
    case_name = ""
    for index, part in enumerate(parts):
        if part.startswith("seed_"):
            seed = part
            if index >= 1:
                placement_method = parts[index - 1]
            if index >= 2:
                case_name = parts[index - 2]
            break
    return case_name, placement_method, seed


def sample_placement_members(zip_path: Path, max_members: int) -> tuple[str, ...]:
    members = discover_chapter2_placement_members(zip_path)
    grouped: dict[str, list[str]] = defaultdict(list)
    for member in members:
        _, placement_method, _ = parse_member_metadata(member)
        grouped[placement_method or "unknown"].append(member)

    selected: list[str] = []
    while len(selected) < max_members and any(grouped.values()):
        for method in sorted(grouped):
            if not grouped[method]:
                continue
            selected.append(grouped[method].pop(0))
            if len(selected) >= max_members:
                break
    return tuple(selected)


def board_counts_from_zip(zip_path: Path, placements_member: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(placements_member) as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_handle)
            for row in reader:
                counts[_canonical_board_id(row["board"])] += 1
    return counts


def select_board_ids(
    counts: Counter[str],
    boards_per_member: int,
    min_rectangles: int,
    max_rectangles: int | None,
) -> tuple[str, ...]:
    candidates = [
        (board_id, count)
        for board_id, count in counts.items()
        if count >= min_rectangles
        and (max_rectangles is None or count <= max_rectangles)
    ]
    candidates.sort(key=lambda item: (-item[1], _board_sort_key(item[0])))
    return tuple(board_id for board_id, _ in candidates[:boards_per_member])


def build_planners(
    methods: tuple[str, ...],
    units,
    panel: Panel,
    tool,
    process_model,
    size: int,
) -> tuple[PlannerSpec, ...]:
    pool_size = topology_pool_size(size)
    beam_config = compact_beam_search_config(size)
    planners: dict[str, PlannerSpec] = {
        "greedy": PlannerSpec(
            method="greedy",
            planner=lambda: plan_greedy_route(
                units,
                panel,
                tool,
                process_model=process_model,
            ),
        ),
        "path_distance_local_search": PlannerSpec(
            method="path_distance_local_search",
            planner=lambda: plan_path_distance_local_search_route(
                units,
                panel,
                tool,
                config=compact_local_search_config(size, False),
                process_model=process_model,
            ),
        ),
        "topology": PlannerSpec(
            method="topology",
            planner=lambda: plan_topology_route(
                units,
                panel,
                tool,
                process_model=process_model,
                candidate_pool_size=pool_size,
            ),
            topology_candidate_pool_size=pool_size,
        ),
        "topology_process_aware": PlannerSpec(
            method="topology_process_aware",
            planner=lambda: plan_topology_route(
                units,
                panel,
                tool,
                process_model=process_model,
                candidate_pool_size=pool_size,
                process_aware=True,
            ),
            topology_candidate_pool_size=pool_size,
        ),
        "process_aware_beam": PlannerSpec(
            method="process_aware_beam",
            planner=lambda: plan_process_aware_beam_route(
                units,
                panel,
                tool,
                config=beam_config,
                process_model=process_model,
            ),
            beam_width=beam_config.beam_width,
            beam_candidate_pool_size=beam_config.candidate_pool_size,
            beam_max_expansions_per_node=beam_config.max_expansions_per_node,
            beam_max_layer_expansions=beam_config.max_layer_expansions,
            beam_diversity_bucket_limit=beam_config.diversity_bucket_limit,
            beam_min_expansions_per_parent=beam_config.min_expansions_per_parent,
            beam_unstable_min_expansions_per_parent=(
                beam_config.unstable_min_expansions_per_parent
            ),
            beam_unstable_layer_expansion_multiplier=(
                beam_config.unstable_layer_expansion_multiplier
            ),
            beam_unstable_layer_expansion_bonus=(
                beam_config.unstable_layer_expansion_bonus
            ),
        ),
        "topology_local_search": PlannerSpec(
            method="topology_local_search",
            planner=lambda: plan_local_search_route(
                units,
                panel,
                tool,
                config=compact_local_search_config(size, False),
                process_model=process_model,
            ),
        ),
        "topology_local_search_process_aware": PlannerSpec(
            method="topology_local_search_process_aware",
            planner=lambda: plan_local_search_route(
                units,
                panel,
                tool,
                config=compact_local_search_config(size, True),
                process_model=process_model,
            ),
        ),
    }
    return tuple(planners[method] for method in methods)


def run_plan(
    archive: Path,
    placements_member: str,
    board_id: str,
    support_policy: str,
    min_support_count: int,
    min_support_ratio: float,
    min_area_normalized_support: float,
    adjacency_support_weight: float,
    spec: PlannerSpec,
    layout: Layout,
    candidate_unit_count: int,
) -> BatchRouteRow:
    start = perf_counter()
    plan = spec.planner()
    runtime_ms = (perf_counter() - start) * 1000.0
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    return BatchRouteRow(
        archive=archive.name,
        case_name=case_name,
        placement_method=placement_method,
        seed=seed,
        placements_member=placements_member,
        board_id=board_id,
        support_policy=support_policy,
        min_support_count=min_support_count,
        min_support_ratio=min_support_ratio,
        min_area_normalized_support=min_area_normalized_support,
        adjacency_support_weight=adjacency_support_weight,
        method=spec.method,
        topology_candidate_pool_size=spec.topology_candidate_pool_size,
        beam_width=spec.beam_width,
        beam_candidate_pool_size=spec.beam_candidate_pool_size,
        beam_max_expansions_per_node=spec.beam_max_expansions_per_node,
        beam_max_layer_expansions=spec.beam_max_layer_expansions,
        beam_diversity_bucket_limit=spec.beam_diversity_bucket_limit,
        beam_min_expansions_per_parent=spec.beam_min_expansions_per_parent,
        beam_unstable_min_expansions_per_parent=(
            spec.beam_unstable_min_expansions_per_parent
        ),
        beam_unstable_layer_expansion_multiplier=(
            spec.beam_unstable_layer_expansion_multiplier
        ),
        beam_unstable_layer_expansion_bonus=spec.beam_unstable_layer_expansion_bonus,
        runtime_ms=runtime_ms,
        rectangle_count=len(layout.rectangles),
        rectangle_count_bin=rectangle_count_bin(len(layout.rectangles)),
        candidate_unit_count=candidate_unit_count,
        selected_unit_count=len(plan.selected_units),
        action_count=len(plan.actions),
        air_move_distance=plan.metrics.air_move_distance,
        cutting_length=plan.metrics.cutting_length,
        pierce_count=plan.metrics.pierce_count,
        lift_count=plan.metrics.lift_count,
        collision_penalty=plan.metrics.collision_penalty,
        boundary_penalty=plan.metrics.boundary_penalty,
        stability_penalty=plan.metrics.stability_penalty,
        safe_lift_count=plan.metrics.safe_lift_count,
        safe_lift_distance=plan.metrics.safe_lift_distance,
        detour_count=plan.metrics.detour_count,
        detour_distance=plan.metrics.detour_distance,
        travel_mode_cost=plan.metrics.travel_mode_cost,
        hard_penalty=plan.metrics.hard_penalty,
    )


def run_case(
    archive: Path,
    placements_member: str,
    board_id: str,
    methods: tuple[str, ...],
    stability_args: argparse.Namespace,
    progress_output: Path | None = None,
    task_timeout_seconds: float = 0.0,
) -> tuple[BatchRouteRow, ...]:
    return tuple(
        iter_case_rows(
            archive,
            placements_member,
            board_id,
            methods,
            stability_args,
            progress_output=progress_output,
            task_timeout_seconds=task_timeout_seconds,
        )
    )


def iter_case_rows(
    archive: Path,
    placements_member: str,
    board_id: str,
    methods: tuple[str, ...],
    stability_args: argparse.Namespace,
    completed_keys: set[tuple[str, ...]] | None = None,
    progress_output: Path | None = None,
    task_timeout_seconds: float = 0.0,
) -> Iterator[BatchRouteRow]:
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
    for spec in build_planners(
        methods,
        units,
        panel,
        tool,
        process_model,
        len(layout.rectangles),
    ):
        key = batch_key_from_parts(
            archive.name,
            placements_member,
            board_id,
            spec.method,
            stability_args,
        )
        if completed_keys is not None and key in completed_keys:
            if progress_output is not None:
                append_progress_event(
                    progress_output,
                    new_progress_event(
                        event="skipped",
                        archive=archive.name,
                        placements_member=placements_member,
                        board_id=board_id,
                        method=spec.method,
                        rectangle_count=len(layout.rectangles),
                        candidate_unit_count=len(units),
                        message="completed row already present in output CSV",
                    ),
                )
            print(
                f"    skip completed: board={board_id} method={spec.method} "
                f"rectangles={len(layout.rectangles)} units={len(units)}"
            )
            continue
        if progress_output is not None:
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="started",
                    archive=archive.name,
                    placements_member=placements_member,
                    board_id=board_id,
                    method=spec.method,
                    rectangle_count=len(layout.rectangles),
                    candidate_unit_count=len(units),
                ),
            )
        print(
            f"    run: board={board_id} method={spec.method} "
            f"rectangles={len(layout.rectangles)} units={len(units)}"
        )
        task_start = perf_counter()
        try:
            with task_timeout(task_timeout_seconds):
                row = run_plan(
                    archive,
                    placements_member,
                    board_id,
                    stability_args.support_policy,
                    stability_args.min_support_count,
                    stability_args.min_support_ratio,
                    stability_args.min_area_normalized_support,
                    stability_args.adjacency_support_weight,
                    spec,
                    layout,
                    len(units),
                )
        except TaskTimeoutError:
            elapsed_ms = (perf_counter() - task_start) * 1000.0
            if progress_output is not None:
                append_progress_event(
                    progress_output,
                    new_progress_event(
                        event="timed_out",
                        archive=archive.name,
                        placements_member=placements_member,
                        board_id=board_id,
                        method=spec.method,
                        rectangle_count=len(layout.rectangles),
                        candidate_unit_count=len(units),
                        elapsed_ms=elapsed_ms,
                        message=f"timeout after {task_timeout_seconds:g} seconds",
                    ),
                )
            print(
                f"    timed out: board={board_id} method={spec.method} "
                f"elapsed={elapsed_ms:.3f} ms limit={task_timeout_seconds:g} s"
            )
            continue
        if progress_output is not None:
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="completed",
                    archive=archive.name,
                    placements_member=placements_member,
                    board_id=board_id,
                    method=spec.method,
                    rectangle_count=row.rectangle_count,
                    candidate_unit_count=row.candidate_unit_count,
                    elapsed_ms=row.runtime_ms,
                ),
            )
        print(
            f"    done: board={board_id} method={spec.method} "
            f"runtime={row.runtime_ms:.3f} ms"
        )
        yield row


def write_rows(rows: tuple[BatchRouteRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def append_rows(rows: tuple[BatchRouteRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def prepare_stream_output(output_path: Path, resume: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")


def load_existing_rows(output_path: Path) -> tuple[BatchRouteRow, ...]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return ()
    with output_path.open(newline="", encoding="utf-8") as handle:
        return tuple(batch_row_from_dict(row) for row in csv.DictReader(handle))


def batch_row_from_dict(raw: dict[str, str]) -> BatchRouteRow:
    return BatchRouteRow(
        **{
            field.name: coerce_batch_value(field.name, raw.get(field.name, ""))
            for field in fields(BatchRouteRow)
        }
    )


def coerce_batch_value(name: str, value: str):
    if value == "":
        return None
    if name in BATCH_INT_FIELDS:
        return int(value)
    if name in BATCH_FLOAT_FIELDS:
        return float(value)
    return value


BATCH_INT_FIELDS = {
    "min_support_count",
    "topology_candidate_pool_size",
    "beam_width",
    "beam_candidate_pool_size",
    "beam_max_expansions_per_node",
    "beam_max_layer_expansions",
    "beam_diversity_bucket_limit",
    "beam_min_expansions_per_parent",
    "beam_unstable_min_expansions_per_parent",
    "beam_unstable_layer_expansion_bonus",
    "rectangle_count",
    "candidate_unit_count",
    "selected_unit_count",
    "action_count",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
}
BATCH_FLOAT_FIELDS = {
    "min_support_ratio",
    "min_area_normalized_support",
    "adjacency_support_weight",
    "beam_unstable_layer_expansion_multiplier",
    "runtime_ms",
    "air_move_distance",
    "cutting_length",
    "collision_penalty",
    "boundary_penalty",
    "stability_penalty",
    "safe_lift_distance",
    "detour_distance",
    "travel_mode_cost",
    "hard_penalty",
}


def batch_key(row: BatchRouteRow) -> tuple[str, ...]:
    return (
        row.archive,
        row.placements_member,
        row.board_id,
        row.method,
        row.support_policy,
        str(row.min_support_count),
        f"{row.min_support_ratio:.12g}",
        f"{row.min_area_normalized_support:.12g}",
        f"{row.adjacency_support_weight:.12g}",
    )


def batch_key_from_parts(
    archive_name: str,
    placements_member: str,
    board_id: str,
    method: str,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    return (
        archive_name,
        placements_member,
        board_id,
        method,
        args.support_policy,
        str(args.min_support_count),
        f"{args.min_support_ratio:.12g}",
        f"{args.min_area_normalized_support:.12g}",
        f"{args.adjacency_support_weight:.12g}",
    )


def summarize_rows(rows: tuple[BatchRouteRow, ...]) -> tuple[BatchSummaryRow, ...]:
    grouped: dict[str, list[BatchRouteRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    summary: list[BatchSummaryRow] = []
    for method in sorted(grouped):
        items = grouped[method]
        runtime_values = [row.runtime_ms for row in items]
        air_values = [row.air_move_distance for row in items]
        mode_cost_values = [row.travel_mode_cost for row in items]
        summary.append(
            BatchSummaryRow(
                method=method,
                support_policy=items[0].support_policy,
                min_support_count=items[0].min_support_count,
                min_support_ratio=items[0].min_support_ratio,
                min_area_normalized_support=items[0].min_area_normalized_support,
                adjacency_support_weight=items[0].adjacency_support_weight,
                n=len(items),
                runtime_ms_mean=mean(runtime_values),
                runtime_ms_std=pstdev(runtime_values),
                air_move_distance_mean=mean(air_values),
                air_move_distance_std=pstdev(air_values),
                travel_mode_cost_mean=mean(mode_cost_values),
                travel_mode_cost_std=pstdev(mode_cost_values),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(row.stability_penalty for row in items),
                safe_lift_count_mean=mean(row.safe_lift_count for row in items),
                detour_count_mean=mean(row.detour_count for row in items),
                rectangle_count_mean=mean(row.rectangle_count for row in items),
            )
        )
    return tuple(summary)


def summarize_bin_rows(rows: tuple[BatchRouteRow, ...]) -> tuple[BatchBinSummaryRow, ...]:
    grouped: dict[tuple[str, str], list[BatchRouteRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.rectangle_count_bin, row.method)].append(row)

    summary: list[BatchBinSummaryRow] = []
    for rectangle_bin, method in sorted(grouped):
        items = grouped[(rectangle_bin, method)]
        runtime_values = [row.runtime_ms for row in items]
        air_values = [row.air_move_distance for row in items]
        mode_cost_values = [row.travel_mode_cost for row in items]
        summary.append(
            BatchBinSummaryRow(
                rectangle_count_bin=rectangle_bin,
                method=method,
                support_policy=items[0].support_policy,
                min_support_count=items[0].min_support_count,
                min_support_ratio=items[0].min_support_ratio,
                min_area_normalized_support=items[0].min_area_normalized_support,
                adjacency_support_weight=items[0].adjacency_support_weight,
                n=len(items),
                runtime_ms_mean=mean(runtime_values),
                runtime_ms_std=pstdev(runtime_values),
                air_move_distance_mean=mean(air_values),
                air_move_distance_std=pstdev(air_values),
                travel_mode_cost_mean=mean(mode_cost_values),
                travel_mode_cost_std=pstdev(mode_cost_values),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(row.stability_penalty for row in items),
                safe_lift_count_mean=mean(row.safe_lift_count for row in items),
                detour_count_mean=mean(row.detour_count for row in items),
                rectangle_count_mean=mean(row.rectangle_count for row in items),
            )
        )
    return tuple(summary)


def write_summary(rows: tuple[BatchSummaryRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_bin_summary(rows: tuple[BatchBinSummaryRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_method_summary(rows: tuple[BatchRouteRow, ...]) -> None:
    for row in summarize_rows(rows):
        print(
            f"{row.method:<36} n={row.n:<3} runtime={row.runtime_ms_mean:>9.3f} ms "
            f"air={row.air_move_distance_mean:>10.3f} "
            f"mode_cost={row.travel_mode_cost_mean:>10.3f} "
            f"hard={row.hard_penalty_mean:>6.3f} "
            f"stability={row.stability_penalty_mean:>6.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, action="append", dest="zip_paths")
    parser.add_argument("--max-members-per-archive", type=int, default=2)
    parser.add_argument("--boards-per-member", type=int, default=1)
    parser.add_argument("--min-rectangles", type=int, default=2)
    parser.add_argument("--max-rectangles", type=int, default=20)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "chapter2_batch_routes.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "chapter2_batch_summary.csv",
    )
    parser.add_argument(
        "--bin-summary-output",
        type=Path,
        default=ROOT / "results" / "chapter2_batch_bin_summary.csv",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=0.0,
        help="Skip one method run if it exceeds this many seconds; 0 disables timeout.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output CSV and skip rows already present in it.",
    )
    add_experiment_preset_arg(parser)
    add_stability_model_args(parser)
    args = parser.parse_args()
    return apply_experiment_preset(args)


def main() -> None:
    args = parse_args()
    archives = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    methods = tuple(args.methods)
    progress_output = args.progress_output or default_progress_log_path(args.output)
    args.progress_output = progress_output
    existing_rows = load_existing_rows(args.output) if args.resume else ()
    completed_keys = {batch_key(row) for row in existing_rows}
    prepare_stream_output(args.output, resume=args.resume)
    prepare_progress_log(progress_output, resume=args.resume)
    rows: list[BatchRouteRow] = list(existing_rows)
    if args.resume and existing_rows:
        print(f"resume: loaded {len(existing_rows)} existing rows from {args.output}")

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
                for row in iter_case_rows(
                    archive,
                    member,
                    board_id,
                    methods,
                    args,
                    completed_keys=completed_keys,
                    progress_output=progress_output,
                    task_timeout_seconds=args.task_timeout_seconds,
                ):
                    append_rows((row,), args.output)
                    rows.append(row)
                    completed_keys.add(batch_key(row))

    if not rows:
        raise ValueError("no batch rows produced; relax filters or check archives")

    summary_rows = summarize_rows(tuple(rows))
    write_summary(summary_rows, args.summary_output)
    bin_summary_rows = summarize_bin_rows(tuple(rows))
    write_bin_summary(bin_summary_rows, args.bin_summary_output)
    manifest_output = args.manifest_output or default_manifest_path(args.output)
    write_experiment_manifest(
        manifest_output,
        experiment_name="chapter2_real_layout_batch",
        args=args,
        archives=archives,
        outputs=(
            args.output,
            args.summary_output,
            args.bin_summary_output,
            progress_output,
        ),
        root=ROOT,
        extra={
            "row_count": len(rows),
            "summary_row_count": len(summary_rows),
            "bin_summary_row_count": len(bin_summary_rows),
            "methods": methods,
        },
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.bin_summary_output}")
    print(f"wrote: {progress_output}")
    print(f"wrote: {manifest_output}")
    print_method_summary(tuple(rows))


def _canonical_board_id(value: int | str) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _board_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()
