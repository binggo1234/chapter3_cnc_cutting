from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

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
    plan_process_aware_beam_route,
    plan_topology_route,
)
from experiment_manifest import default_manifest_path, write_experiment_manifest
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (10, 50, 100, 200, 500)
DEFAULT_METHODS = (
    "greedy",
    "path_distance_local_search",
    "topology",
    "topology_process_aware",
    "process_aware_beam",
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
    return tuple(
        run_method(
            scenario,
            planners[method],
            layout=layout,
            candidate_unit_count=len(units),
            repeat=repeat,
            stability_args=stability_args,
        )
        for method in methods
    )


def write_rows(rows: tuple[ScalabilityRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


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
    add_experiment_preset_arg(parser)
    add_stability_model_args(parser)
    args = parser.parse_args()
    return apply_experiment_preset(args)


def main() -> None:
    args = parse_args()
    rows: list[ScalabilityRow] = []
    for repeat in range(args.repeats):
        for size in args.sizes:
            rows.extend(
                run_size(
                    size,
                    repeat=repeat,
                    seed=args.seed,
                    scenario=args.scenario,
                    methods=tuple(args.methods),
                    stability_args=args,
                    process_aware_topology=args.process_aware_topology,
                )
            )

    write_rows(tuple(rows), args.output)
    manifest_output = args.manifest_output or default_manifest_path(args.output)
    write_experiment_manifest(
        manifest_output,
        experiment_name="synthetic_scalability",
        args=args,
        outputs=(args.output,),
        root=ROOT,
        extra={
            "row_count": len(rows),
            "methods": tuple(args.methods),
            "sizes": tuple(args.sizes),
            "scenario": args.scenario,
        },
    )
    print(f"wrote: {args.output}")
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
