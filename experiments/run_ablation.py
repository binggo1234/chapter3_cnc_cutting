from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable, Iterator

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.geometry import rectangle_edges
from cnc_cutting.io import (
    discover_chapter2_placement_members,
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import BeamSearchConfig, LocalSearchConfig
from cnc_cutting.metrics import evaluate_actions, materialize_action_clearance
from cnc_cutting.models import CuttingProcessModel, CuttingUnit, CuttingUnitType, Layout, Panel
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_greedy_route,
    plan_local_search_route,
    plan_path_distance_local_search_route,
    plan_process_aware_beam_route,
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
DEFAULT_VARIANTS = (
    "full_process_aware_beam",
    "single_edges_only",
    "no_stability_guidance",
    "no_adjacency_support_guidance",
    "topology_no_beam",
    "path_distance_baseline",
    "no_detour_operator",
    "no_safe_travel_modes",
)
VARIANT_CHOICES = DEFAULT_VARIANTS + (
    "greedy_baseline",
    "topology_local_search",
    "topology_local_search_process_aware",
)


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    planner_kind: str
    use_single_edge_units_only: bool = False
    disable_stability_guidance: bool = False
    disable_adjacency_guidance: bool = False
    disable_detour_operator: bool = False
    disable_safe_travel_modes: bool = False
    evaluate_with_planning_tool: bool = False


@dataclass(frozen=True)
class AblationRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    method: str
    variant: str
    planner_kind: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    use_single_edge_units_only: bool
    disable_stability_guidance: bool
    disable_adjacency_guidance: bool
    disable_detour_operator: bool
    disable_safe_travel_modes: bool
    evaluate_with_planning_tool: bool
    runtime_ms: float
    rectangle_count: int
    rectangle_count_bin: str
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    primitive_unit_count: int
    relation_unit_count: int
    selected_single_edge_count: int
    selected_shared_edge_count: int
    selected_near_shared_channel_count: int
    selected_collinear_chain_count: int
    air_move_distance: float
    cutting_length: float
    pierce_count: int
    lift_count: int
    collision_penalty: float
    boundary_penalty: float
    hard_penalty: float
    stability_penalty: float
    safe_lift_count: int
    safe_lift_distance: float
    detour_count: int
    detour_distance: float
    travel_mode_cost: float
    machining_cost: float
    planning_air_move_distance: float
    planning_travel_mode_cost: float
    planning_machining_cost: float
    planning_hard_penalty: float
    planning_stability_penalty: float
    planning_safe_lift_count: int
    planning_detour_count: int


@dataclass(frozen=True)
class AblationSummaryRow:
    method: str
    variant: str
    planner_kind: str
    n: int
    runtime_ms_mean: float
    runtime_ms_std: float
    travel_mode_cost_mean: float
    travel_mode_cost_std: float
    machining_cost_mean: float
    machining_cost_std: float
    air_move_distance_mean: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    safe_lift_count_mean: float
    detour_count_mean: float
    selected_shared_edge_count_mean: float
    selected_near_shared_channel_count_mean: float
    selected_collinear_chain_count_mean: float
    rectangle_count_mean: float


def ablation_spec(variant: str) -> AblationSpec:
    specs = {
        "full_process_aware_beam": AblationSpec(
            variant=variant,
            planner_kind="process_aware_beam",
        ),
        "single_edges_only": AblationSpec(
            variant=variant,
            planner_kind="process_aware_beam",
            use_single_edge_units_only=True,
        ),
        "no_stability_guidance": AblationSpec(
            variant=variant,
            planner_kind="process_aware_beam",
            disable_stability_guidance=True,
        ),
        "no_adjacency_support_guidance": AblationSpec(
            variant=variant,
            planner_kind="process_aware_beam",
            disable_adjacency_guidance=True,
        ),
        "topology_no_beam": AblationSpec(
            variant=variant,
            planner_kind="topology_process_aware",
        ),
        "path_distance_baseline": AblationSpec(
            variant=variant,
            planner_kind="path_distance_local_search",
        ),
        "no_detour_operator": AblationSpec(
            variant=variant,
            planner_kind="process_aware_beam",
            disable_detour_operator=True,
            evaluate_with_planning_tool=True,
        ),
        "no_safe_travel_modes": AblationSpec(
            variant=variant,
            planner_kind="process_aware_beam",
            disable_detour_operator=True,
            disable_safe_travel_modes=True,
            evaluate_with_planning_tool=True,
        ),
        "greedy_baseline": AblationSpec(
            variant=variant,
            planner_kind="greedy",
        ),
        "topology_local_search": AblationSpec(
            variant=variant,
            planner_kind="topology_local_search",
        ),
        "topology_local_search_process_aware": AblationSpec(
            variant=variant,
            planner_kind="topology_local_search_process_aware",
        ),
    }
    return specs[variant]


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


def topology_pool_size(size: int) -> int:
    if size <= 50:
        return 96
    if size <= 200:
        return 64
    return 48


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


def single_edge_units(layout: Layout) -> tuple[CuttingUnit, ...]:
    return tuple(
        CuttingUnit(
            unit_id=f"single:{segment.segment_id}",
            unit_type=CuttingUnitType.SINGLE_EDGE,
            segments=(segment,),
            start=segment.start,
            end=segment.end,
            covered_segment_ids=(segment.segment_id,),
        )
        for rectangle in layout.rectangles
        for segment in rectangle_edges(rectangle)
    )


def planning_process_args(
    args: argparse.Namespace,
    spec: AblationSpec,
) -> argparse.Namespace:
    support_policy = "none" if spec.disable_stability_guidance else args.support_policy
    adjacency_support_weight = (
        0.0 if spec.disable_adjacency_guidance else args.adjacency_support_weight
    )
    return argparse.Namespace(
        support_policy=support_policy,
        min_support_count=args.min_support_count,
        min_support_ratio=args.min_support_ratio,
        min_area_normalized_support=args.min_area_normalized_support,
        adjacency_support_weight=adjacency_support_weight,
    )


def planning_tool(base_tool, spec: AblationSpec):
    tool = base_tool
    if spec.disable_detour_operator:
        tool = replace(tool, allow_low_clearance_detour=False)
    if spec.disable_safe_travel_modes:
        tool = replace(tool, allow_safe_lift_over_released_parts=False)
    return tool


def build_plan(
    spec: AblationSpec,
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool,
    process_model: CuttingProcessModel,
    size: int,
) -> RoutePlan:
    if spec.planner_kind == "greedy":
        return plan_greedy_route(units, panel, tool, process_model=process_model)
    if spec.planner_kind == "path_distance_local_search":
        return plan_path_distance_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, False),
            process_model=process_model,
        )
    if spec.planner_kind == "topology_process_aware":
        return plan_topology_route(
            units,
            panel,
            tool,
            process_model=process_model,
            candidate_pool_size=topology_pool_size(size),
            process_aware=True,
        )
    if spec.planner_kind == "topology_local_search":
        return plan_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, False),
            process_model=process_model,
        )
    if spec.planner_kind == "topology_local_search_process_aware":
        return plan_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    if spec.planner_kind == "process_aware_beam":
        return plan_process_aware_beam_route(
            units,
            panel,
            tool,
            config=compact_beam_search_config(size),
            process_model=process_model,
        )
    raise ValueError(f"unsupported planner_kind: {spec.planner_kind}")


def evaluated_metrics_and_actions(
    plan: RoutePlan,
    panel: Panel,
    tool,
    process_model: CuttingProcessModel,
) -> tuple[object, tuple]:
    actions = materialize_action_clearance(
        plan.actions,
        panel,
        tool,
        process_model=process_model,
    )
    return evaluate_actions(actions, panel, tool, process_model=process_model), actions


def unit_type_counts(units: tuple[CuttingUnit, ...]) -> Counter[str]:
    return Counter(unit.unit_type.value for unit in units)


def run_variant(
    archive: Path,
    placements_member: str,
    board_id: str,
    layout: Layout,
    base_units: tuple[CuttingUnit, ...],
    panel: Panel,
    base_tool,
    evaluation_process_model: CuttingProcessModel,
    args: argparse.Namespace,
    variant: str,
) -> AblationRow:
    spec = ablation_spec(variant)
    units = single_edge_units(layout) if spec.use_single_edge_units_only else base_units
    plan_args = planning_process_args(args, spec)
    plan_process_model = build_process_model_from_args(layout, plan_args)
    plan_tool = planning_tool(base_tool, spec)
    eval_tool = plan_tool if spec.evaluate_with_planning_tool else base_tool

    start = perf_counter()
    plan = build_plan(
        spec,
        units,
        panel,
        plan_tool,
        plan_process_model,
        len(layout.rectangles),
    )
    runtime_ms = (perf_counter() - start) * 1000.0
    eval_metrics, eval_actions = evaluated_metrics_and_actions(
        plan,
        panel,
        eval_tool,
        evaluation_process_model,
    )

    case_name, placement_method, seed = parse_member_metadata(placements_member)
    selected_counts = unit_type_counts(plan.selected_units)
    return AblationRow(
        archive=archive.name,
        case_name=case_name,
        placement_method=placement_method,
        seed=seed,
        placements_member=placements_member,
        board_id=board_id,
        method=variant,
        variant=variant,
        planner_kind=spec.planner_kind,
        support_policy=args.support_policy,
        min_support_count=args.min_support_count,
        min_support_ratio=args.min_support_ratio,
        min_area_normalized_support=args.min_area_normalized_support,
        adjacency_support_weight=args.adjacency_support_weight,
        use_single_edge_units_only=spec.use_single_edge_units_only,
        disable_stability_guidance=spec.disable_stability_guidance,
        disable_adjacency_guidance=spec.disable_adjacency_guidance,
        disable_detour_operator=spec.disable_detour_operator,
        disable_safe_travel_modes=spec.disable_safe_travel_modes,
        evaluate_with_planning_tool=spec.evaluate_with_planning_tool,
        runtime_ms=runtime_ms,
        rectangle_count=len(layout.rectangles),
        rectangle_count_bin=rectangle_count_bin(len(layout.rectangles)),
        candidate_unit_count=len(units),
        selected_unit_count=len(plan.selected_units),
        action_count=len(eval_actions),
        primitive_unit_count=sum(1 for unit in units if unit.unit_type == CuttingUnitType.SINGLE_EDGE),
        relation_unit_count=sum(1 for unit in units if unit.unit_type != CuttingUnitType.SINGLE_EDGE),
        selected_single_edge_count=selected_counts[CuttingUnitType.SINGLE_EDGE.value],
        selected_shared_edge_count=selected_counts[CuttingUnitType.SHARED_EDGE.value],
        selected_near_shared_channel_count=selected_counts[
            CuttingUnitType.NEAR_SHARED_CHANNEL.value
        ],
        selected_collinear_chain_count=selected_counts[CuttingUnitType.COLLINEAR_CHAIN.value],
        air_move_distance=eval_metrics.air_move_distance,
        cutting_length=eval_metrics.cutting_length,
        pierce_count=eval_metrics.pierce_count,
        lift_count=eval_metrics.lift_count,
        collision_penalty=eval_metrics.collision_penalty,
        boundary_penalty=eval_metrics.boundary_penalty,
        hard_penalty=eval_metrics.hard_penalty,
        stability_penalty=eval_metrics.stability_penalty,
        safe_lift_count=eval_metrics.safe_lift_count,
        safe_lift_distance=eval_metrics.safe_lift_distance,
        detour_count=eval_metrics.detour_count,
        detour_distance=eval_metrics.detour_distance,
        travel_mode_cost=eval_metrics.travel_mode_cost,
        machining_cost=eval_metrics.machining_cost,
        planning_air_move_distance=plan.metrics.air_move_distance,
        planning_travel_mode_cost=plan.metrics.travel_mode_cost,
        planning_machining_cost=plan.metrics.machining_cost,
        planning_hard_penalty=plan.metrics.hard_penalty,
        planning_stability_penalty=plan.metrics.stability_penalty,
        planning_safe_lift_count=plan.metrics.safe_lift_count,
        planning_detour_count=plan.metrics.detour_count,
    )


def run_case(
    archive: Path,
    placements_member: str,
    board_id: str,
    variants: tuple[str, ...],
    args: argparse.Namespace,
    progress_output: Path | None = None,
    task_timeout_seconds: float = 0.0,
) -> tuple[AblationRow, ...]:
    return tuple(
        iter_case_rows(
            archive,
            placements_member,
            board_id,
            variants,
            args,
            progress_output=progress_output,
            task_timeout_seconds=task_timeout_seconds,
        )
    )


def iter_case_rows(
    archive: Path,
    placements_member: str,
    board_id: str,
    variants: tuple[str, ...],
    args: argparse.Namespace,
    completed_keys: set[tuple[str, ...]] | None = None,
    progress_output: Path | None = None,
    task_timeout_seconds: float = 0.0,
) -> Iterator[AblationRow]:
    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    base_tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=placements_member,
        board_ids=(board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    evaluation_process_model = build_process_model_from_args(layout, args)
    base_units = build_candidate_cutting_units(
        layout,
        base_tool,
        max_collinear_gap=base_tool.min_channel_width,
    )
    for variant in variants:
        spec = ablation_spec(variant)
        candidate_unit_count = (
            4 * len(layout.rectangles)
            if spec.use_single_edge_units_only
            else len(base_units)
        )
        key = ablation_key_from_parts(
            archive.name,
            placements_member,
            board_id,
            variant,
            args,
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
                        method=variant,
                        rectangle_count=len(layout.rectangles),
                        candidate_unit_count=candidate_unit_count,
                        message="completed row already present in output CSV",
                    ),
                )
            print(
                f"    skip completed: board={board_id} variant={variant} "
                f"rectangles={len(layout.rectangles)} units={candidate_unit_count}"
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
                    method=variant,
                    rectangle_count=len(layout.rectangles),
                    candidate_unit_count=candidate_unit_count,
                ),
            )
        print(
            f"    run: board={board_id} variant={variant} "
            f"rectangles={len(layout.rectangles)} units={candidate_unit_count}"
        )
        task_start = perf_counter()
        try:
            with task_timeout(task_timeout_seconds):
                row = run_variant(
                    archive,
                    placements_member,
                    board_id,
                    layout,
                    base_units,
                    panel,
                    base_tool,
                    evaluation_process_model,
                    args,
                    variant,
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
                        method=variant,
                        rectangle_count=len(layout.rectangles),
                        candidate_unit_count=candidate_unit_count,
                        elapsed_ms=elapsed_ms,
                        message=f"timeout after {task_timeout_seconds:g} seconds",
                    ),
                )
            print(
                f"    timed out: board={board_id} variant={variant} "
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
                    method=variant,
                    rectangle_count=row.rectangle_count,
                    candidate_unit_count=row.candidate_unit_count,
                    elapsed_ms=row.runtime_ms,
                ),
            )
        print(
            f"    done: board={board_id} variant={variant} "
            f"runtime={row.runtime_ms:.3f} ms"
        )
        yield row


def write_rows(rows: tuple[AblationRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def append_rows(rows: tuple[AblationRow, ...], output_path: Path) -> None:
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


def load_existing_rows(output_path: Path) -> tuple[AblationRow, ...]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return ()
    with output_path.open(newline="", encoding="utf-8") as handle:
        return tuple(ablation_row_from_dict(row) for row in csv.DictReader(handle))


def ablation_row_from_dict(raw: dict[str, str]) -> AblationRow:
    return AblationRow(
        **{
            field.name: coerce_ablation_value(field.name, raw.get(field.name, ""))
            for field in fields(AblationRow)
        }
    )


def coerce_ablation_value(name: str, value: str):
    if value == "":
        return None
    if name in ABLATION_BOOL_FIELDS:
        return value in {"True", "true", "1"}
    if name in ABLATION_INT_FIELDS:
        return int(value)
    if name in ABLATION_FLOAT_FIELDS:
        return float(value)
    return value


ABLATION_BOOL_FIELDS = {
    "use_single_edge_units_only",
    "disable_stability_guidance",
    "disable_adjacency_guidance",
    "disable_detour_operator",
    "disable_safe_travel_modes",
    "evaluate_with_planning_tool",
}
ABLATION_INT_FIELDS = {
    "min_support_count",
    "rectangle_count",
    "candidate_unit_count",
    "selected_unit_count",
    "action_count",
    "primitive_unit_count",
    "relation_unit_count",
    "selected_single_edge_count",
    "selected_shared_edge_count",
    "selected_near_shared_channel_count",
    "selected_collinear_chain_count",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
    "planning_safe_lift_count",
    "planning_detour_count",
}
ABLATION_FLOAT_FIELDS = {
    "min_support_ratio",
    "min_area_normalized_support",
    "adjacency_support_weight",
    "runtime_ms",
    "air_move_distance",
    "cutting_length",
    "collision_penalty",
    "boundary_penalty",
    "hard_penalty",
    "stability_penalty",
    "safe_lift_distance",
    "detour_distance",
    "travel_mode_cost",
    "machining_cost",
    "planning_air_move_distance",
    "planning_travel_mode_cost",
    "planning_machining_cost",
    "planning_hard_penalty",
    "planning_stability_penalty",
}


def ablation_key(row: AblationRow) -> tuple[str, ...]:
    return (
        row.archive,
        row.placements_member,
        row.board_id,
        row.variant,
        row.support_policy,
        str(row.min_support_count),
        f"{row.min_support_ratio:.12g}",
        f"{row.min_area_normalized_support:.12g}",
        f"{row.adjacency_support_weight:.12g}",
    )


def ablation_key_from_parts(
    archive_name: str,
    placements_member: str,
    board_id: str,
    variant: str,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    return (
        archive_name,
        placements_member,
        board_id,
        variant,
        args.support_policy,
        str(args.min_support_count),
        f"{args.min_support_ratio:.12g}",
        f"{args.min_area_normalized_support:.12g}",
        f"{args.adjacency_support_weight:.12g}",
    )


def summarize_rows(rows: tuple[AblationRow, ...]) -> tuple[AblationSummaryRow, ...]:
    grouped: dict[str, list[AblationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.variant].append(row)

    summary: list[AblationSummaryRow] = []
    for variant in sorted(grouped, key=lambda value: VARIANT_CHOICES.index(value)):
        items = grouped[variant]
        runtime_values = [row.runtime_ms for row in items]
        cost_values = [row.travel_mode_cost for row in items]
        machining_values = [row.machining_cost for row in items]
        summary.append(
            AblationSummaryRow(
                method=variant,
                variant=variant,
                planner_kind=items[0].planner_kind,
                n=len(items),
                runtime_ms_mean=mean(runtime_values),
                runtime_ms_std=pstdev(runtime_values),
                travel_mode_cost_mean=mean(cost_values),
                travel_mode_cost_std=pstdev(cost_values),
                machining_cost_mean=mean(machining_values),
                machining_cost_std=pstdev(machining_values),
                air_move_distance_mean=mean(row.air_move_distance for row in items),
                hard_penalty_mean=mean(row.hard_penalty for row in items),
                stability_penalty_mean=mean(row.stability_penalty for row in items),
                safe_lift_count_mean=mean(row.safe_lift_count for row in items),
                detour_count_mean=mean(row.detour_count for row in items),
                selected_shared_edge_count_mean=mean(
                    row.selected_shared_edge_count for row in items
                ),
                selected_near_shared_channel_count_mean=mean(
                    row.selected_near_shared_channel_count for row in items
                ),
                selected_collinear_chain_count_mean=mean(
                    row.selected_collinear_chain_count for row in items
                ),
                rectangle_count_mean=mean(row.rectangle_count for row in items),
            )
        )
    return tuple(summary)


def write_summary(rows: tuple[AblationSummaryRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_summary(rows: tuple[AblationRow, ...]) -> None:
    for row in summarize_rows(rows):
        print(
            f"{row.variant:<34} n={row.n:<3} "
            f"runtime={row.runtime_ms_mean:>9.3f} ms "
            f"cost={row.travel_mode_cost_mean:>10.3f} "
            f"machining={row.machining_cost_mean:>10.3f} "
            f"hard={row.hard_penalty_mean:>6.3f} "
            f"stability={row.stability_penalty_mean:>6.3f} "
            f"safe_lift={row.safe_lift_count_mean:>6.3f} "
            f"detour={row.detour_count_mean:>6.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, action="append", dest="zip_paths")
    parser.add_argument("--max-members-per-archive", type=int, default=2)
    parser.add_argument("--boards-per-member", type=int, default=1)
    parser.add_argument("--min-rectangles", type=int, default=2)
    parser.add_argument("--max-rectangles", type=int, default=25)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_CHOICES,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "ablation_routes.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "ablation_summary.csv",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=0.0,
        help="Skip one variant run if it exceeds this many seconds; 0 disables timeout.",
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
    variants = tuple(args.variants)
    progress_output = args.progress_output or default_progress_log_path(args.output)
    args.progress_output = progress_output
    existing_rows = load_existing_rows(args.output) if args.resume else ()
    completed_keys = {ablation_key(row) for row in existing_rows}
    prepare_stream_output(args.output, resume=args.resume)
    prepare_progress_log(progress_output, resume=args.resume)
    rows: list[AblationRow] = list(existing_rows)
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
                    variants,
                    args,
                    completed_keys=completed_keys,
                    progress_output=progress_output,
                    task_timeout_seconds=args.task_timeout_seconds,
                ):
                    append_rows((row,), args.output)
                    rows.append(row)
                    completed_keys.add(ablation_key(row))

    if not rows:
        raise ValueError("no ablation rows produced; relax filters or check archives")

    result_rows = tuple(rows)
    summary_rows = summarize_rows(result_rows)
    write_summary(summary_rows, args.summary_output)
    manifest_output = args.manifest_output or default_manifest_path(args.output)
    write_experiment_manifest(
        manifest_output,
        experiment_name="chapter3_ablation",
        args=args,
        archives=archives,
        outputs=(args.output, args.summary_output, progress_output),
        root=ROOT,
        extra={
            "row_count": len(result_rows),
            "summary_row_count": len(summary_rows),
            "variants": variants,
        },
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {progress_output}")
    print(f"wrote: {manifest_output}")
    print_summary(result_rows)


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
