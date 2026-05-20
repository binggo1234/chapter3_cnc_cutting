from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

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
from process_options import add_stability_model_args, build_process_model_from_args


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path(
    "/Users/binggo/Desktop/codex_handoff_20260502/data/"
    "strong_baseline_overnight_20260407_001243.zip"
)


@dataclass(frozen=True)
class Chapter2RouteRow:
    archive: str
    placements_member: str
    board_id: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    method: str
    runtime_ms: float
    rectangle_count: int
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


def run_method(
    method: str,
    planner: Callable[[], RoutePlan],
    archive: Path,
    placements_member: str,
    board_id: str,
    support_policy: str,
    min_support_count: int,
    min_support_ratio: float,
    min_area_normalized_support: float,
    adjacency_support_weight: float,
    layout: Layout,
    candidate_unit_count: int,
) -> Chapter2RouteRow:
    start = perf_counter()
    plan = planner()
    runtime_ms = (perf_counter() - start) * 1000.0
    return Chapter2RouteRow(
        archive=archive.name,
        placements_member=placements_member,
        board_id=board_id,
        support_policy=support_policy,
        min_support_count=min_support_count,
        min_support_ratio=min_support_ratio,
        min_area_normalized_support=min_area_normalized_support,
        adjacency_support_weight=adjacency_support_weight,
        method=method,
        runtime_ms=runtime_ms,
        rectangle_count=len(layout.rectangles),
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


def write_rows(rows: tuple[Chapter2RouteRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def resolve_placements_member(
    zip_path: Path,
    placements_member: str | None,
    member_index: int,
) -> str:
    if placements_member is not None:
        return placements_member
    members = discover_chapter2_placement_members(zip_path)
    if not members:
        raise ValueError(f"no placements.csv found in {zip_path}")
    if member_index < 0 or member_index >= len(members):
        raise ValueError(f"member_index out of range: {member_index}; available={len(members)}")
    return members[member_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--placements-member", default=None)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--board-id", default="1")
    parser.add_argument("--skip-local-search", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "chapter2_route_smoke.csv",
    )
    add_stability_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    placements_member = resolve_placements_member(
        args.zip_path,
        args.placements_member,
        args.member_index,
    )
    cfg = load_chapter2_config_from_zip(args.zip_path, placements_member=placements_member)
    tool = tool_config_from_chapter2_config(cfg)
    layouts = load_chapter2_layouts_from_zip(
        args.zip_path,
        placements_member=placements_member,
        board_ids=(args.board_id,),
    )
    layout = layouts[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    unit_counts = Counter(unit.unit_type.value for unit in units)
    size = len(layout.rectangles)
    pool_size = topology_pool_size(size)

    planners: list[tuple[str, Callable[[], RoutePlan]]] = [
        (
            "greedy",
            lambda: plan_greedy_route(units, panel, tool, process_model=process_model),
        ),
        (
            "topology",
            lambda: plan_topology_route(
                units,
                panel,
                tool,
                process_model=process_model,
                candidate_pool_size=pool_size,
            ),
        ),
        (
            "topology_process_aware",
            lambda: plan_topology_route(
                units,
                panel,
                tool,
                process_model=process_model,
                candidate_pool_size=pool_size,
                process_aware=True,
            ),
        ),
        (
            "process_aware_beam",
            lambda: plan_process_aware_beam_route(
                units,
                panel,
                tool,
                config=compact_beam_search_config(size),
                process_model=process_model,
            ),
        ),
    ]

    if not args.skip_local_search:
        planners.extend(
            [
                (
                    "path_distance_local_search",
                    lambda: plan_path_distance_local_search_route(
                        units,
                        panel,
                        tool,
                        config=compact_local_search_config(size, False),
                        process_model=process_model,
                    ),
                ),
                (
                    "topology_local_search",
                    lambda: plan_local_search_route(
                        units,
                        panel,
                        tool,
                        config=compact_local_search_config(size, False),
                        process_model=process_model,
                    ),
                ),
                (
                    "topology_local_search_process_aware",
                    lambda: plan_local_search_route(
                        units,
                        panel,
                        tool,
                        config=compact_local_search_config(size, True),
                        process_model=process_model,
                    ),
                ),
            ]
        )

    rows = tuple(
        run_method(
            method,
            planner,
            archive=args.zip_path,
            placements_member=placements_member,
            board_id=str(args.board_id),
            support_policy=args.support_policy,
            min_support_count=args.min_support_count,
            min_support_ratio=args.min_support_ratio,
            min_area_normalized_support=args.min_area_normalized_support,
            adjacency_support_weight=args.adjacency_support_weight,
            layout=layout,
            candidate_unit_count=len(units),
        )
        for method, planner in planners
    )
    write_rows(rows, args.output)

    print(f"archive: {args.zip_path}")
    print(f"placements_member: {placements_member}")
    print(f"board_id: {args.board_id}")
    print(f"panel: {layout.panel_width:.1f} x {layout.panel_height:.1f}")
    print(f"rectangles: {len(layout.rectangles)}")
    print(f"candidate_units: {len(units)} {dict(sorted(unit_counts.items()))}")
    print(f"tool: trim={tool.trim_margin:.1f}, diameter={tool.tool_diameter:.1f}")
    print(
        "stability_model: "
        f"support_policy={args.support_policy}, "
        f"min_support_count={args.min_support_count}, "
        f"min_support_ratio={args.min_support_ratio:.3f}, "
        f"min_area_normalized_support={args.min_area_normalized_support:.3f}, "
        f"adjacency_support_weight={args.adjacency_support_weight:.3f}"
    )
    print(f"wrote: {args.output}")
    for row in rows:
        print(
            f"{row.method:<36} {row.runtime_ms:>9.3f} ms "
            f"air={row.air_move_distance:>10.3f} "
            f"collision={row.collision_penalty:>6.3f} "
            f"boundary={row.boundary_penalty:>6.3f} "
            f"stability={row.stability_penalty:>6.3f} "
            f"safe_lift={row.safe_lift_count:>3} "
            f"detour={row.detour_count:>3} "
            f"mode_cost={row.travel_mode_cost:>10.3f}"
        )


if __name__ == "__main__":
    main()
