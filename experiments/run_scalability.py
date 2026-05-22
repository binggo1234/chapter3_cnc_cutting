from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.layout_generator import (
    ClusteredLayoutConfig,
    SyntheticLayoutConfig,
    generate_clustered_channel_layout,
    generate_synthetic_layout,
)
from cnc_cutting.local_search import BeamSearchConfig, LocalSearchConfig
from cnc_cutting.models import Layout, Panel, Point, ToolConfig
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_greedy_route,
    plan_local_search_route,
    plan_path_distance_local_search_route,
    plan_process_local_search_multistart_route,
    plan_process_aware_beam_adaptive_polished_route,
    plan_process_aware_beam_adaptive_route,
    plan_process_aware_beam_polished_route,
    plan_process_aware_beam_route,
    plan_topology_route,
    wider_beam_search_config,
)
from experiment_manifest import default_manifest_path, write_experiment_manifest
from progress_log import (
    append_progress_event,
    default_progress_log_path,
    new_progress_event,
    prepare_progress_log,
)
from progress_bar import TerminalProgressBar
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)
from task_timeout import TaskTimeoutError, task_timeout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (10, 50, 100, 200, 500)
DEFAULT_METHODS = (
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


@dataclass(frozen=True)
class ScalabilityRow:
    scenario: str
    size: int
    method: str
    repeat: int
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
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
    beam_completion_aware_prerank: bool | None
    beam_unstable_completion_focus_count: int | None
    runtime_ms: float
    rectangle_count: int
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    cutting_length: float
    pierce_count: int
    lift_count: int
    safe_lift_count: int
    safe_lift_distance: float
    detour_count: int
    detour_distance: float
    travel_mode_cost: float
    stability_penalty: float
    hard_penalty: float


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
    beam_completion_aware_prerank: bool | None = None
    beam_unstable_completion_focus_count: int | None = None


def compact_local_search_config(
    size: int,
    process_aware_initial_order: bool = False,
) -> LocalSearchConfig:
    if size <= 50:
        return LocalSearchConfig(
            max_iterations=2,
            max_swap_span=6,
            max_relocate_span=6,
            max_two_opt_span=6,
            max_neighbors_per_iteration=150,
            first_improvement=True,
            topology_candidate_pool_size=96,
            process_aware_initial_order=process_aware_initial_order,
        )
    if size <= 200:
        return LocalSearchConfig(
            max_iterations=1,
            max_swap_span=5,
            max_relocate_span=5,
            max_two_opt_span=5,
            max_neighbors_per_iteration=100,
            first_improvement=True,
            topology_candidate_pool_size=64,
            process_aware_initial_order=process_aware_initial_order,
        )
    return LocalSearchConfig(
        max_iterations=1,
        max_swap_span=4,
        max_relocate_span=4,
        max_two_opt_span=4,
        max_neighbors_per_iteration=60,
        first_improvement=True,
        topology_candidate_pool_size=48,
        process_aware_initial_order=process_aware_initial_order,
    )


def topology_candidate_pool_size(size: int) -> int:
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
        )
    return BeamSearchConfig(
        beam_width=4,
        candidate_pool_size=12,
        max_expansions_per_node=24,
        max_layer_expansions=28,
        diversity_bucket_limit=1,
        min_expansions_per_parent=0,
        unstable_min_expansions_per_parent=2,
    )


def beam_search_config_from_args(size: int, args: argparse.Namespace) -> BeamSearchConfig:
    config = compact_beam_search_config(size)
    overrides = {}
    for arg_name, field_name in (
        ("beam_width", "beam_width"),
        ("beam_candidate_pool_size", "candidate_pool_size"),
        ("beam_max_expansions_per_node", "max_expansions_per_node"),
        ("beam_max_layer_expansions", "max_layer_expansions"),
        ("beam_diversity_bucket_limit", "diversity_bucket_limit"),
        ("beam_min_expansions_per_parent", "min_expansions_per_parent"),
        (
            "beam_unstable_min_expansions_per_parent",
            "unstable_min_expansions_per_parent",
        ),
        (
            "beam_unstable_layer_expansion_multiplier",
            "unstable_layer_expansion_multiplier",
        ),
        ("beam_unstable_layer_expansion_bonus", "unstable_layer_expansion_bonus"),
        ("beam_completion_aware_prerank", "completion_aware_prerank"),
        ("beam_unstable_completion_focus_count", "unstable_completion_focus_count"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value
    if not overrides:
        return config
    return replace(config, **overrides)


def run_method(
    scenario: str,
    spec: PlannerSpec,
    layout: Layout,
    candidate_unit_count: int,
    repeat: int,
    stability_args: argparse.Namespace,
) -> ScalabilityRow:
    start = perf_counter()
    plan = spec.planner()
    runtime_ms = (perf_counter() - start) * 1000.0
    return ScalabilityRow(
        scenario=scenario,
        size=len(layout.rectangles),
        method=spec.method,
        repeat=repeat,
        support_policy=stability_args.support_policy,
        min_support_count=stability_args.min_support_count,
        min_support_ratio=stability_args.min_support_ratio,
        min_area_normalized_support=stability_args.min_area_normalized_support,
        adjacency_support_weight=stability_args.adjacency_support_weight,
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
        beam_completion_aware_prerank=spec.beam_completion_aware_prerank,
        beam_unstable_completion_focus_count=(
            spec.beam_unstable_completion_focus_count
        ),
        runtime_ms=runtime_ms,
        rectangle_count=len(layout.rectangles),
        candidate_unit_count=candidate_unit_count,
        selected_unit_count=len(plan.selected_units),
        action_count=len(plan.actions),
        air_move_distance=plan.metrics.air_move_distance,
        cutting_length=plan.metrics.cutting_length,
        pierce_count=plan.metrics.pierce_count,
        lift_count=plan.metrics.lift_count,
        safe_lift_count=plan.metrics.safe_lift_count,
        safe_lift_distance=plan.metrics.safe_lift_distance,
        detour_count=plan.metrics.detour_count,
        detour_distance=plan.metrics.detour_distance,
        travel_mode_cost=plan.metrics.travel_mode_cost,
        stability_penalty=plan.metrics.stability_penalty,
        hard_penalty=plan.metrics.hard_penalty,
    )


def generate_layout_for_scenario(
    scenario: str,
    size: int,
    tool: ToolConfig,
    seed: int,
) -> Layout:
    if scenario == "grid":
        return generate_synthetic_layout(
            size,
            tool,
            SyntheticLayoutConfig(seed=seed),
        )
    if scenario == "clustered":
        return generate_clustered_channel_layout(
            size,
            tool,
            ClusteredLayoutConfig(seed=seed),
        )
    raise ValueError(f"unsupported scenario: {scenario}")


def run_size(
    size: int,
    repeat: int,
    seed: int,
    scenario: str,
    methods: tuple[str, ...],
    stability_args: argparse.Namespace,
    process_aware_topology: bool = False,
) -> tuple[ScalabilityRow, ...]:
    return tuple(
        iter_size_rows(
            size,
            repeat=repeat,
            seed=seed,
            scenario=scenario,
            methods=methods,
            stability_args=stability_args,
            process_aware_topology=process_aware_topology,
        )
    )


def iter_size_rows(
    size: int,
    repeat: int,
    seed: int,
    scenario: str,
    methods: tuple[str, ...],
    stability_args: argparse.Namespace,
    process_aware_topology: bool = False,
    completed_keys: set[tuple[str, ...]] | None = None,
    progress_output: Path | None = None,
    task_timeout_seconds: float = 0.0,
    progress_bar: TerminalProgressBar | None = None,
) -> Iterator[ScalabilityRow]:
    effective_margin = 8.0
    tool = ToolConfig(
        trim_margin=5,
        tool_diameter=6,
        centerline_boundary_margin=3,
        allow_safe_lift_over_released_parts=True,
        allow_low_clearance_detour=True,
        start_point=Point(effective_margin, effective_margin),
    )
    layout = generate_layout_for_scenario(scenario, size, tool, seed=seed + repeat)
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, stability_args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    local_config = compact_local_search_config(
        size,
        process_aware_initial_order=process_aware_topology,
    )
    topology_pool_size = topology_candidate_pool_size(size)
    beam_config = beam_search_config_from_args(size, stability_args)
    fallback_beam_config = wider_beam_search_config(beam_config)

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
                candidate_pool_size=topology_pool_size,
                process_aware=process_aware_topology,
            ),
            topology_candidate_pool_size=topology_pool_size,
        ),
        "topology_process_aware": PlannerSpec(
            method="topology_process_aware",
            planner=lambda: plan_topology_route(
                units,
                panel,
                tool,
                process_model=process_model,
                candidate_pool_size=topology_pool_size,
                process_aware=True,
            ),
            topology_candidate_pool_size=topology_pool_size,
        ),
        "process_local_search_multistart": PlannerSpec(
            method="process_local_search_multistart",
            planner=lambda: plan_process_local_search_multistart_route(
                units,
                panel,
                tool,
                config=compact_local_search_config(size, True),
                process_model=process_model,
            ),
            topology_candidate_pool_size=topology_pool_size,
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
            beam_completion_aware_prerank=beam_config.completion_aware_prerank,
            beam_unstable_completion_focus_count=(
                beam_config.unstable_completion_focus_count
            ),
        ),
        "process_aware_beam_adaptive": PlannerSpec(
            method="process_aware_beam_adaptive",
            planner=lambda: plan_process_aware_beam_adaptive_route(
                units,
                panel,
                tool,
                beam_config=beam_config,
                fallback_beam_config=fallback_beam_config,
                topology_candidate_pool_size=topology_pool_size,
                fallback_margin=1000.0,
                process_model=process_model,
            ),
            topology_candidate_pool_size=topology_pool_size,
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
            beam_completion_aware_prerank=beam_config.completion_aware_prerank,
            beam_unstable_completion_focus_count=(
                beam_config.unstable_completion_focus_count
            ),
        ),
        "process_aware_beam_polished": PlannerSpec(
            method="process_aware_beam_polished",
            planner=lambda: plan_process_aware_beam_polished_route(
                units,
                panel,
                tool,
                beam_config=beam_config,
                polish_config=local_config,
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
            beam_completion_aware_prerank=beam_config.completion_aware_prerank,
            beam_unstable_completion_focus_count=(
                beam_config.unstable_completion_focus_count
            ),
        ),
        "process_aware_beam_adaptive_polished": PlannerSpec(
            method="process_aware_beam_adaptive_polished",
            planner=lambda: plan_process_aware_beam_adaptive_polished_route(
                units,
                panel,
                tool,
                beam_config=beam_config,
                fallback_beam_config=fallback_beam_config,
                polish_config=local_config,
                topology_candidate_pool_size=topology_pool_size,
                fallback_margin=1000.0,
                process_model=process_model,
            ),
            topology_candidate_pool_size=topology_pool_size,
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
            beam_completion_aware_prerank=beam_config.completion_aware_prerank,
            beam_unstable_completion_focus_count=(
                beam_config.unstable_completion_focus_count
            ),
        ),
        "topology_local_search": PlannerSpec(
            method="topology_local_search",
            planner=lambda: plan_local_search_route(
                units,
                panel,
                tool,
                config=local_config,
                process_model=process_model,
            ),
        ),
    }
    for method in methods:
        spec = planners[method]
        key = scalability_key_from_parts(
            scenario,
            size,
            repeat,
            spec.method,
            stability_args,
        )
        if completed_keys is not None and key in completed_keys:
            if progress_output is not None:
                append_progress_event(
                    progress_output,
                    new_progress_event(
                        event="skipped",
                        archive=f"synthetic:{scenario}",
                        placements_member=f"repeat:{repeat}",
                        board_id=str(size),
                        method=spec.method,
                        rectangle_count=len(layout.rectangles),
                        candidate_unit_count=len(units),
                        message="completed row already present in output CSV",
                    ),
                )
            print(
                f"skip completed: scenario={scenario} size={size} "
                f"repeat={repeat} method={spec.method}"
            )
            if progress_bar is not None:
                progress_bar.advance(
                    "skipped",
                    f"{scenario} n={size} r={repeat} {spec.method}",
                )
            continue

        if progress_output is not None:
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="started",
                    archive=f"synthetic:{scenario}",
                    placements_member=f"repeat:{repeat}",
                    board_id=str(size),
                    method=spec.method,
                    rectangle_count=len(layout.rectangles),
                    candidate_unit_count=len(units),
                ),
            )
        print(
            f"run: scenario={scenario} size={size} repeat={repeat} "
            f"method={spec.method} units={len(units)}"
        )
        task_start = perf_counter()
        try:
            with task_timeout(task_timeout_seconds):
                row = run_method(
                    scenario,
                    spec,
                    layout=layout,
                    candidate_unit_count=len(units),
                    repeat=repeat,
                    stability_args=stability_args,
                )
        except TaskTimeoutError:
            elapsed_ms = (perf_counter() - task_start) * 1000.0
            if progress_output is not None:
                append_progress_event(
                    progress_output,
                    new_progress_event(
                        event="timed_out",
                        archive=f"synthetic:{scenario}",
                        placements_member=f"repeat:{repeat}",
                        board_id=str(size),
                        method=spec.method,
                        rectangle_count=len(layout.rectangles),
                        candidate_unit_count=len(units),
                        elapsed_ms=elapsed_ms,
                        message=f"timeout after {task_timeout_seconds:g} seconds",
                    ),
                )
            print(
                f"timed out: scenario={scenario} size={size} repeat={repeat} "
                f"method={spec.method} elapsed={elapsed_ms:.3f} ms "
                f"limit={task_timeout_seconds:g} s"
            )
            if progress_bar is not None:
                progress_bar.advance(
                    "timed_out",
                    f"{scenario} n={size} r={repeat} {spec.method}",
                )
            continue

        if progress_output is not None:
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="completed",
                    archive=f"synthetic:{scenario}",
                    placements_member=f"repeat:{repeat}",
                    board_id=str(size),
                    method=spec.method,
                    rectangle_count=row.rectangle_count,
                    candidate_unit_count=row.candidate_unit_count,
                    elapsed_ms=row.runtime_ms,
                ),
            )
        print(
            f"done: scenario={scenario} size={size} repeat={repeat} "
            f"method={spec.method} runtime={row.runtime_ms:.3f} ms"
        )
        if progress_bar is not None:
            progress_bar.advance(
                "completed",
                f"{scenario} n={size} r={repeat} {spec.method}",
            )
        yield row


def write_rows(rows: tuple[ScalabilityRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def append_rows(rows: tuple[ScalabilityRow, ...], output_path: Path) -> None:
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


def load_existing_rows(output_path: Path) -> tuple[ScalabilityRow, ...]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return ()
    with output_path.open(newline="", encoding="utf-8") as handle:
        return tuple(scalability_row_from_dict(row) for row in csv.DictReader(handle))


def scalability_row_from_dict(raw: dict[str, str]) -> ScalabilityRow:
    return ScalabilityRow(
        **{
            field.name: coerce_scalability_value(field.name, raw.get(field.name, ""))
            for field in fields(ScalabilityRow)
        }
    )


def coerce_scalability_value(name: str, value: str):
    if value == "":
        return None
    if name in SCALABILITY_BOOL_FIELDS:
        return value in {"True", "true", "1"}
    if name in SCALABILITY_INT_FIELDS:
        return int(value)
    if name in SCALABILITY_FLOAT_FIELDS:
        return float(value)
    return value


SCALABILITY_BOOL_FIELDS = {"beam_completion_aware_prerank"}
SCALABILITY_INT_FIELDS = {
    "size",
    "repeat",
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
    "beam_unstable_completion_focus_count",
    "rectangle_count",
    "candidate_unit_count",
    "selected_unit_count",
    "action_count",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
}
SCALABILITY_FLOAT_FIELDS = {
    "min_support_ratio",
    "min_area_normalized_support",
    "adjacency_support_weight",
    "beam_unstable_layer_expansion_multiplier",
    "runtime_ms",
    "air_move_distance",
    "cutting_length",
    "safe_lift_distance",
    "detour_distance",
    "travel_mode_cost",
    "stability_penalty",
    "hard_penalty",
}


def scalability_key(row: ScalabilityRow) -> tuple[str, ...]:
    return (
        row.scenario,
        str(row.size),
        str(row.repeat),
        row.method,
        row.support_policy,
        str(row.min_support_count),
        f"{row.min_support_ratio:.12g}",
        f"{row.min_area_normalized_support:.12g}",
        f"{row.adjacency_support_weight:.12g}",
        str(row.beam_width),
        str(row.beam_candidate_pool_size),
        str(row.beam_max_expansions_per_node),
        str(row.beam_max_layer_expansions),
        str(row.beam_diversity_bucket_limit),
        str(row.beam_min_expansions_per_parent),
        str(row.beam_unstable_min_expansions_per_parent),
        f"{row.beam_unstable_layer_expansion_multiplier}",
        str(row.beam_unstable_layer_expansion_bonus),
        str(row.beam_completion_aware_prerank),
        str(row.beam_unstable_completion_focus_count),
    )


def scalability_key_from_parts(
    scenario: str,
    size: int,
    repeat: int,
    method: str,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    config = beam_search_config_from_args(size, args)
    beam_methods = {
        "process_aware_beam",
        "process_aware_beam_adaptive",
        "process_aware_beam_polished",
        "process_aware_beam_adaptive_polished",
    }
    has_beam_fields = method in beam_methods
    return (
        scenario,
        str(size),
        str(repeat),
        method,
        args.support_policy,
        str(args.min_support_count),
        f"{args.min_support_ratio:.12g}",
        f"{args.min_area_normalized_support:.12g}",
        f"{args.adjacency_support_weight:.12g}",
        str(config.beam_width if has_beam_fields else None),
        str(config.candidate_pool_size if has_beam_fields else None),
        str(config.max_expansions_per_node if has_beam_fields else None),
        str(config.max_layer_expansions if has_beam_fields else None),
        str(config.diversity_bucket_limit if has_beam_fields else None),
        str(config.min_expansions_per_parent if has_beam_fields else None),
        str(config.unstable_min_expansions_per_parent if has_beam_fields else None),
        f"{config.unstable_layer_expansion_multiplier if has_beam_fields else None}",
        str(config.unstable_layer_expansion_bonus if has_beam_fields else None),
        str(config.completion_aware_prerank if has_beam_fields else None),
        str(config.unstable_completion_focus_count if has_beam_fields else None),
    )


def count_planned_method_tasks(
    sizes: tuple[int, ...],
    repeats: int,
    methods: tuple[str, ...],
) -> int:
    return max(repeats, 0) * len(sizes) * len(methods)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--scenario",
        choices=("grid", "clustered"),
        default="grid",
    )
    parser.add_argument("--process-aware-topology", action="store_true")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--beam-candidate-pool-size", type=int, default=None)
    parser.add_argument("--beam-max-expansions-per-node", type=int, default=None)
    parser.add_argument("--beam-max-layer-expansions", type=int, default=None)
    parser.add_argument("--beam-diversity-bucket-limit", type=int, default=None)
    parser.add_argument("--beam-min-expansions-per-parent", type=int, default=None)
    parser.add_argument("--beam-unstable-min-expansions-per-parent", type=int, default=None)
    parser.add_argument("--beam-unstable-layer-expansion-multiplier", type=float, default=None)
    parser.add_argument("--beam-unstable-layer-expansion-bonus", type=int, default=None)
    parser.add_argument("--beam-completion-aware-prerank", action="store_true", default=None)
    parser.add_argument("--beam-unstable-completion-focus-count", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "scalability_results.csv",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=0.0,
        help="Skip one scalability method run if it exceeds this many seconds; 0 disables timeout.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable terminal progress bar output.",
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
    methods = tuple(args.methods)
    sizes = tuple(args.sizes)
    progress_output = args.progress_output or default_progress_log_path(args.output)
    args.progress_output = progress_output
    progress_bar = TerminalProgressBar(
        count_planned_method_tasks(sizes, args.repeats, methods),
        enabled=not args.no_progress_bar,
    )
    progress_bar.start("scalability method tasks")
    existing_rows = load_existing_rows(args.output) if args.resume else ()
    completed_keys = {scalability_key(row) for row in existing_rows}
    prepare_stream_output(args.output, resume=args.resume)
    prepare_progress_log(progress_output, resume=args.resume)
    rows: list[ScalabilityRow] = list(existing_rows)
    if args.resume and existing_rows:
        print(f"resume: loaded {len(existing_rows)} existing rows from {args.output}")

    for repeat in range(args.repeats):
        for size in sizes:
            for row in iter_size_rows(
                size,
                repeat=repeat,
                seed=args.seed,
                scenario=args.scenario,
                methods=methods,
                stability_args=args,
                process_aware_topology=args.process_aware_topology,
                completed_keys=completed_keys,
                progress_output=progress_output,
                task_timeout_seconds=args.task_timeout_seconds,
                progress_bar=progress_bar,
            ):
                append_rows((row,), args.output)
                rows.append(row)
                completed_keys.add(scalability_key(row))

    if not rows:
        raise ValueError("no scalability rows produced")

    manifest_output = args.manifest_output or default_manifest_path(args.output)
    write_experiment_manifest(
        manifest_output,
        experiment_name="synthetic_scalability",
        args=args,
        outputs=(args.output, progress_output),
        root=ROOT,
        extra={
            "row_count": len(rows),
            "methods": methods,
            "sizes": sizes,
            "scenario": args.scenario,
        },
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {progress_output}")
    print(f"wrote: {manifest_output}")
    for row in rows:
        print(
            f"{row.scenario:<9} {row.size:>4} {row.method:<22} "
            f"{row.runtime_ms:>9.3f} ms "
            f"air={row.air_move_distance:>10.3f} "
            f"mode_cost={row.travel_mode_cost:>10.3f} "
            f"hard={row.hard_penalty:>6.3f} "
            f"stability={row.stability_penalty:>6.3f}"
        )


if __name__ == "__main__":
    main()
