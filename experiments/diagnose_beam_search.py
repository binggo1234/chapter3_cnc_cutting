from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import (
    discover_chapter2_placement_members,
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import BeamSearchConfig, process_aware_beam_search_order
from cnc_cutting.models import Panel
from cnc_cutting.optimizer import select_coverage_units
from process_options import add_stability_model_args, build_process_model_from_args
from run_chapter2_route import DEFAULT_ZIP


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BeamLayerDiagnosticsRow:
    archive: str
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
    unstable_layer_expansion_multiplier: float
    unstable_layer_expansion_bonus: int
    diversity_bucket_limit: int
    min_expansions_per_parent: int
    unstable_min_expansions_per_parent: int
    rectangle_count: int
    candidate_unit_count: int
    selected_unit_count: int
    runtime_ms: float
    final_air_move_distance: float
    final_travel_mode_cost: float
    final_stability_penalty: float
    final_hard_penalty: float
    final_expanded_nodes: int
    depth: int
    input_beam_count: int
    unstable_input_prefix_count: int
    raw_expansion_count: int
    effective_layer_expansion_limit: int
    layer_expansion_count: int
    layer_pruned_count: int
    parent_quota_added_count: int
    parent_quota_pruned_count: int
    evaluated_node_count: int
    duplicate_pruned_count: int
    diversity_pruned_count: int
    fallback_added_count: int
    output_beam_count: int
    best_hard_penalty: float
    best_stability_penalty: float
    best_travel_mode_cost: float
    best_air_move_distance: float
    worst_stability_penalty: float
    worst_travel_mode_cost: float
    unstable_prefix_count: int
    released_part_count_max: int


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


def write_rows(rows: tuple[BeamLayerDiagnosticsRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--placements-member", default=None)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--board-id", default="1")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--max-expansions-per-node", type=int, default=48)
    parser.add_argument("--max-layer-expansions", type=int, default=56)
    parser.add_argument("--unstable-layer-expansion-multiplier", type=float, default=1.0)
    parser.add_argument("--unstable-layer-expansion-bonus", type=int, default=0)
    parser.add_argument("--diversity-bucket-limit", type=int, default=1)
    parser.add_argument("--min-expansions-per-parent", type=int, default=0)
    parser.add_argument("--unstable-min-expansions-per-parent", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "beam_layer_diagnostics.csv",
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
    layout = load_chapter2_layouts_from_zip(
        args.zip_path,
        placements_member=placements_member,
        board_ids=(args.board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    selected_units = select_coverage_units(units)
    config = BeamSearchConfig(
        beam_width=args.beam_width,
        candidate_pool_size=args.candidate_pool_size,
        max_expansions_per_node=args.max_expansions_per_node,
        max_layer_expansions=args.max_layer_expansions,
        unstable_layer_expansion_multiplier=args.unstable_layer_expansion_multiplier,
        unstable_layer_expansion_bonus=args.unstable_layer_expansion_bonus,
        diversity_bucket_limit=(
            args.diversity_bucket_limit
            if args.diversity_bucket_limit > 0
            else None
        ),
        min_expansions_per_parent=args.min_expansions_per_parent,
        unstable_min_expansions_per_parent=args.unstable_min_expansions_per_parent,
    )

    start = perf_counter()
    result = process_aware_beam_search_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=config,
    )
    runtime_ms = (perf_counter() - start) * 1000.0

    rows = tuple(
        BeamLayerDiagnosticsRow(
            archive=args.zip_path.name,
            placements_member=placements_member,
            board_id=str(args.board_id),
            support_policy=args.support_policy,
            min_support_count=args.min_support_count,
            min_support_ratio=args.min_support_ratio,
            min_area_normalized_support=args.min_area_normalized_support,
            adjacency_support_weight=args.adjacency_support_weight,
            beam_width=args.beam_width,
            candidate_pool_size=args.candidate_pool_size,
            max_expansions_per_node=args.max_expansions_per_node,
            max_layer_expansions=args.max_layer_expansions,
            unstable_layer_expansion_multiplier=args.unstable_layer_expansion_multiplier,
            unstable_layer_expansion_bonus=args.unstable_layer_expansion_bonus,
            diversity_bucket_limit=args.diversity_bucket_limit,
            min_expansions_per_parent=args.min_expansions_per_parent,
            unstable_min_expansions_per_parent=args.unstable_min_expansions_per_parent,
            rectangle_count=len(layout.rectangles),
            candidate_unit_count=len(units),
            selected_unit_count=len(selected_units),
            runtime_ms=runtime_ms,
            final_air_move_distance=result.metrics.air_move_distance,
            final_travel_mode_cost=result.metrics.travel_mode_cost,
            final_stability_penalty=result.metrics.stability_penalty,
            final_hard_penalty=result.metrics.hard_penalty,
            final_expanded_nodes=result.expanded_nodes,
            depth=layer.depth,
            input_beam_count=layer.input_beam_count,
            unstable_input_prefix_count=layer.unstable_input_prefix_count,
            raw_expansion_count=layer.raw_expansion_count,
            effective_layer_expansion_limit=layer.effective_layer_expansion_limit,
            layer_expansion_count=layer.layer_expansion_count,
            layer_pruned_count=layer.layer_pruned_count,
            parent_quota_added_count=layer.parent_quota_added_count,
            parent_quota_pruned_count=layer.parent_quota_pruned_count,
            evaluated_node_count=layer.evaluated_node_count,
            duplicate_pruned_count=layer.duplicate_pruned_count,
            diversity_pruned_count=layer.diversity_pruned_count,
            fallback_added_count=layer.fallback_added_count,
            output_beam_count=layer.output_beam_count,
            best_hard_penalty=layer.best_hard_penalty,
            best_stability_penalty=layer.best_stability_penalty,
            best_travel_mode_cost=layer.best_travel_mode_cost,
            best_air_move_distance=layer.best_air_move_distance,
            worst_stability_penalty=layer.worst_stability_penalty,
            worst_travel_mode_cost=layer.worst_travel_mode_cost,
            unstable_prefix_count=layer.unstable_prefix_count,
            released_part_count_max=layer.released_part_count_max,
        )
        for layer in result.diagnostics
    )
    if not rows:
        raise ValueError("no beam diagnostics produced")

    write_rows(rows, args.output)
    print(f"archive: {args.zip_path}")
    print(f"placements_member: {placements_member}")
    print(f"board_id: {args.board_id}")
    print(f"rectangles: {len(layout.rectangles)}")
    print(f"candidate_units: {len(units)}")
    print(f"selected_units: {len(selected_units)}")
    print(
        f"beam: width={args.beam_width}, pool={args.candidate_pool_size}, "
        f"per_node={args.max_expansions_per_node}, "
        f"layer={args.max_layer_expansions}, "
        f"unstable_layer_multiplier={args.unstable_layer_expansion_multiplier}, "
        f"unstable_layer_bonus={args.unstable_layer_expansion_bonus}, "
        f"diversity={args.diversity_bucket_limit}, "
        f"parent_min={args.min_expansions_per_parent}, "
        f"unstable_parent_min={args.unstable_min_expansions_per_parent}"
    )
    print(
        f"final: runtime={runtime_ms:.3f} ms, "
        f"expanded={result.expanded_nodes}, "
        f"travel_mode_cost={result.metrics.travel_mode_cost:.3f}, "
        f"stability={result.metrics.stability_penalty:.3f}, "
        f"hard={result.metrics.hard_penalty:.3f}"
    )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
