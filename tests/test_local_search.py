from itertools import product, permutations

import pytest

from cnc_cutting.exact_dp import ExactDPConfig, exact_process_dp_order
from cnc_cutting.local_search import (
    BeamSearchConfig,
    LocalSearchConfig,
    build_prefix_states,
    evaluate_directed_units,
    evaluate_directed_units_from_prefix,
    evaluate_neighbor_move_metrics,
    improve_directed_unit_order_by_path_distance,
    improve_directed_unit_order,
    local_neighbor_moves,
    local_neighbors,
    multistart_process_initial_orders,
    nearest_neighbor_unit_order,
    path_distance_metric_key,
    process_metric_key,
    process_local_search_multistart_order,
    process_aware_beam_search_order,
    process_aware_beam_polished_search_order,
    process_aware_candidate_units,
    process_aware_unit_order,
    sweep_unit_order,
)
from cnc_cutting.models import (
    CuttingUnit,
    CuttingUnitType,
    EdgeRole,
    EdgeSegment,
    IncrementalMetricsState,
    Layout,
    Panel,
    PathMetrics,
    PlacedRectangle,
    Point,
    ToolConfig,
)
from cnc_cutting.process_model import build_process_model
from cnc_cutting.topology_operators import (
    DirectedUnitCandidate,
    directed_unit_candidates,
)


def _candidate(unit_id: str, start: Point, end: Point) -> DirectedUnitCandidate:
    segment = EdgeSegment(f"{unit_id}:edge", unit_id, EdgeRole.BOTTOM, start, end)
    unit = CuttingUnit(
        unit_id=unit_id,
        unit_type=CuttingUnitType.SINGLE_EDGE,
        segments=(segment,),
        start=start,
        end=end,
        covered_segment_ids=(segment.segment_id,),
    )
    return DirectedUnitCandidate(
        unit=unit,
        start=start,
        end=end,
        direction=(1.0, 0.0),
    )


def _edge_unit(segment_id: str, role: EdgeRole) -> CuttingUnit:
    segment = EdgeSegment(segment_id, "A", role, Point(0, 0), Point(10, 0))
    return CuttingUnit(
        unit_id=f"single:{segment_id}",
        unit_type=CuttingUnitType.SINGLE_EDGE,
        segments=(segment,),
        start=segment.start,
        end=segment.end,
        covered_segment_ids=(segment.segment_id,),
    )


def _edge_unit_for_part(
    part_id: str,
    segment_id: str,
    role: EdgeRole,
    start: Point,
    end: Point,
) -> CuttingUnit:
    segment = EdgeSegment(segment_id, part_id, role, start, end)
    return CuttingUnit(
        unit_id=f"single:{segment_id}",
        unit_type=CuttingUnitType.SINGLE_EDGE,
        segments=(segment,),
        start=segment.start,
        end=segment.end,
        covered_segment_ids=(segment.segment_id,),
    )


def test_local_search_relocates_unit_to_reduce_air_move() -> None:
    initial_order = (
        _candidate("near_start", Point(0, 0), Point(10, 0)),
        _candidate("far", Point(100, 0), Point(110, 0)),
        _candidate("middle", Point(20, 0), Point(30, 0)),
    )

    result = improve_directed_unit_order(
        initial_order,
        Panel("P", 200, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert result.improved
    assert [candidate.unit.unit_id for candidate in result.directed_units] == [
        "near_start",
        "middle",
        "far",
    ]
    assert result.metrics.air_move_distance == 80


def test_path_distance_local_search_uses_path_only_objective() -> None:
    initial_order = (
        _candidate("near_start", Point(0, 0), Point(10, 0)),
        _candidate("far", Point(100, 0), Point(110, 0)),
        _candidate("middle", Point(20, 0), Point(30, 0)),
    )

    result = improve_directed_unit_order_by_path_distance(
        initial_order,
        Panel("P", 200, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )
    _, initial_metrics = evaluate_directed_units(
        initial_order,
        Panel("P", 200, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert result.improved
    assert path_distance_metric_key(result.metrics) < path_distance_metric_key(initial_metrics)
    assert [candidate.unit.unit_id for candidate in result.directed_units] == [
        "near_start",
        "middle",
        "far",
    ]


def test_process_metric_key_accounts_for_cutting_length_before_tool_events() -> None:
    shared_edge_like = PathMetrics(
        cutting_length=100,
        travel_mode_cost=100,
        pierce_count=4,
        lift_count=4,
    )
    repeated_single_edge_like = PathMetrics(
        cutting_length=180,
        travel_mode_cost=80,
        pierce_count=2,
        lift_count=2,
    )

    assert process_metric_key(shared_edge_like) < process_metric_key(
        repeated_single_edge_like
    )


def test_process_metric_key_penalizes_repeated_cut_before_machining_cost() -> None:
    repeat_free = PathMetrics(
        cutting_length=200,
        travel_mode_cost=100,
        pierce_count=4,
        lift_count=4,
    )
    repeated_shorter = PathMetrics(
        cutting_length=120,
        travel_mode_cost=80,
        pierce_count=2,
        lift_count=2,
        repeated_cut_segment_count=1,
        repeated_cut_length=20,
    )

    assert process_metric_key(repeat_free) < process_metric_key(repeated_shorter)


def test_exact_process_dp_matches_bruteforce_process_metric() -> None:
    units = (
        _candidate("near_start", Point(0, 0), Point(10, 0)).unit,
        _candidate("far", Point(100, 0), Point(110, 0)).unit,
        _candidate("middle", Point(20, 0), Point(30, 0)).unit,
    )
    panel = Panel("P", 200, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0))

    result = exact_process_dp_order(units, panel, tool)
    brute_force_metrics = []
    for unit_order in permutations(units):
        for direction_choices in product((0, 1), repeat=len(unit_order)):
            directed_order = tuple(
                directed_unit_candidates(unit)[choice]
                for unit, choice in zip(unit_order, direction_choices)
            )
            _, metrics = evaluate_directed_units(directed_order, panel, tool)
            brute_force_metrics.append(metrics)

    assert process_metric_key(result.metrics) == min(
        process_metric_key(metrics)
        for metrics in brute_force_metrics
    )
    assert [candidate.unit.unit_id for candidate in result.directed_units] == [
        "near_start",
        "middle",
        "far",
    ]
    assert result.expanded_nodes > 0
    assert result.retained_states > 0


def test_exact_process_dp_rejects_large_instances_by_default() -> None:
    units = tuple(
        _candidate(f"u{index}", Point(index * 10, 0), Point(index * 10 + 5, 0)).unit
        for index in range(13)
    )

    with pytest.raises(ValueError, match="supports at most 12 units"):
        exact_process_dp_order(
            units,
            Panel("P", 200, 100),
            ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
        )


def test_nearest_neighbor_unit_order_chooses_closer_reversed_endpoint() -> None:
    unit = _edge_unit_for_part(
        "A",
        "A:bottom",
        EdgeRole.BOTTOM,
        Point(0, 0),
        Point(10, 0),
    )

    ordered = nearest_neighbor_unit_order((unit,), Point(9, 0))

    assert ordered[0].is_reversed
    assert ordered[0].start == Point(10, 0)


def test_sweep_unit_order_uses_requested_primary_axis() -> None:
    units = (
        _edge_unit_for_part("A", "A:e", EdgeRole.BOTTOM, Point(30, 0), Point(40, 0)),
        _edge_unit_for_part("B", "B:e", EdgeRole.BOTTOM, Point(0, 30), Point(10, 30)),
        _edge_unit_for_part("C", "C:e", EdgeRole.BOTTOM, Point(0, 0), Point(10, 0)),
    )

    x_order = sweep_unit_order(units, Point(0, 0), primary_axis="x")
    y_order = sweep_unit_order(units, Point(0, 0), primary_axis="y")

    assert [candidate.unit.unit_id for candidate in x_order] == [
        "single:C:e",
        "single:B:e",
        "single:A:e",
    ]
    assert [candidate.unit.unit_id for candidate in y_order] == [
        "single:C:e",
        "single:A:e",
        "single:B:e",
    ]


def test_local_neighbors_respects_configured_limit() -> None:
    initial_order = (
        _candidate("u1", Point(0, 0), Point(10, 0)),
        _candidate("u2", Point(20, 0), Point(30, 0)),
        _candidate("u3", Point(40, 0), Point(50, 0)),
        _candidate("u4", Point(60, 0), Point(70, 0)),
    )

    neighbors = local_neighbors(
        initial_order,
        LocalSearchConfig(
            max_swap_span=1,
            max_relocate_span=1,
            max_two_opt_span=1,
            max_neighbors_per_iteration=5,
        ),
    )

    assert len(neighbors) == 5


def test_prefix_neighbor_evaluation_matches_full_evaluation() -> None:
    initial_order = (
        _candidate("u1", Point(0, 0), Point(10, 0)),
        _candidate("u2", Point(40, 0), Point(50, 0)),
        _candidate("u3", Point(20, 0), Point(30, 0)),
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0))
    moves = local_neighbor_moves(
        initial_order,
        LocalSearchConfig(
            enable_swap=False,
            enable_relocate=True,
            enable_two_opt=False,
            max_neighbors_per_iteration=1,
        ),
    )
    prefix_states = build_prefix_states(initial_order, panel, tool)
    move = moves[0]
    _, current_metrics = evaluate_directed_units(initial_order, panel, tool)

    _, full_metrics = evaluate_directed_units(move.directed_units, panel, tool)
    prefix_metrics = evaluate_directed_units_from_prefix(
        move.directed_units,
        move.affected_start_index,
        prefix_states[move.affected_start_index],
        panel,
        tool,
    )
    bounded_metrics = evaluate_neighbor_move_metrics(
        move,
        prefix_states,
        current_metrics,
        panel,
        tool,
    )

    assert prefix_metrics == full_metrics
    assert bounded_metrics == full_metrics


def test_first_improvement_local_search_still_returns_valid_metrics() -> None:
    initial_order = (
        _candidate("near_start", Point(0, 0), Point(10, 0)),
        _candidate("far", Point(100, 0), Point(110, 0)),
        _candidate("middle", Point(20, 0), Point(30, 0)),
    )

    result = improve_directed_unit_order(
        initial_order,
        Panel("P", 200, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
        config=LocalSearchConfig(first_improvement=True),
    )

    assert result.improved
    assert result.metrics.air_move_distance < 160


def test_process_aware_order_delays_support_edge_when_tied() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(layout)
    units = (
        _edge_unit("A:top", EdgeRole.TOP),
        _edge_unit("A:bottom", EdgeRole.BOTTOM),
    )

    ordered = process_aware_unit_order(
        units,
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(0, 0)),
        process_model=process_model,
    )

    assert ordered[0].unit.covered_segment_ids == ("A:bottom",)


def test_process_aware_candidate_pool_prioritizes_unstable_part_completion() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=200,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 100, 10, 10, 10),
            PlacedRectangle("B", 112, 10, 10, 10),
        ),
    )
    process_model = build_process_model(
        layout,
        support_edge_roles=(
            EdgeRole.BOTTOM,
            EdgeRole.RIGHT,
            EdgeRole.TOP,
            EdgeRole.LEFT,
        ),
        min_remaining_support_length_ratio=0.5,
    )
    remaining = [
        _edge_unit_for_part(
            "B",
            "B:bottom",
            EdgeRole.BOTTOM,
            Point(112, 10),
            Point(122, 10),
        ),
        _edge_unit_for_part(
            "A",
            "A:left",
            EdgeRole.LEFT,
            Point(100, 20),
            Point(100, 10),
        ),
    ]
    from cnc_cutting.models import IncrementalMetricsState

    state = IncrementalMetricsState(
        current_point=Point(112, 10),
        unstable_part_ids={"A"},
    )

    candidates = process_aware_candidate_units(
        remaining,
        state,
        process_model,
        candidate_pool_size=1,
    )

    assert [unit.unit_id for unit in candidates] == ["single:A:left"]


def test_completion_candidate_pool_can_focus_nearest_release() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=200,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 10, 10, 10, 10),
            PlacedRectangle("B", 40, 10, 10, 10),
        ),
    )
    process_model = build_process_model(
        layout,
        support_edge_roles=(
            EdgeRole.BOTTOM,
            EdgeRole.RIGHT,
            EdgeRole.TOP,
            EdgeRole.LEFT,
        ),
        min_remaining_support_length_ratio=0.5,
    )
    remaining = [
        _edge_unit_for_part(
            "A",
            "A:left",
            EdgeRole.LEFT,
            Point(10, 20),
            Point(10, 10),
        ),
        _edge_unit_for_part(
            "B",
            "B:bottom",
            EdgeRole.BOTTOM,
            Point(40, 10),
            Point(50, 10),
        ),
        _edge_unit_for_part(
            "B",
            "B:left",
            EdgeRole.LEFT,
            Point(40, 20),
            Point(40, 10),
        ),
    ]
    state = IncrementalMetricsState(
        current_point=Point(12, 12),
        processed_segments={"A:bottom", "A:right", "A:top"},
        unstable_part_ids={"A", "B"},
    )

    candidates = process_aware_candidate_units(
        remaining,
        state,
        process_model,
        candidate_pool_size=None,
        unstable_completion_focus_count=1,
    )

    assert [unit.unit_id for unit in candidates] == ["single:A:left"]


def test_process_aware_beam_search_returns_complete_route() -> None:
    units = (
        _candidate("near_start", Point(0, 0), Point(10, 0)).unit,
        _candidate("far", Point(100, 0), Point(110, 0)).unit,
        _candidate("middle", Point(20, 0), Point(30, 0)).unit,
    )

    result = process_aware_beam_search_order(
        units,
        Panel("P", 200, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
        config=BeamSearchConfig(
            beam_width=2,
            candidate_pool_size=3,
            max_expansions_per_node=6,
            diversity_bucket_limit=1,
        ),
    )

    assert len(result.directed_units) == len(units)
    assert {candidate.unit.unit_id for candidate in result.directed_units} == {
        unit.unit_id for unit in units
    }
    assert result.metrics.cutting_length == 30
    assert result.expanded_nodes > len(units)
    assert len(result.diagnostics) == len(units)
    assert result.diagnostics[0].input_beam_count == 1
    assert result.diagnostics[-1].output_beam_count >= 1


def test_process_local_search_multistart_selects_best_initial_seed_without_iterations() -> None:
    units = (
        _candidate("near_start", Point(0, 0), Point(10, 0)).unit,
        _candidate("far", Point(100, 0), Point(110, 0)).unit,
        _candidate("middle", Point(20, 0), Point(30, 0)).unit,
    )
    panel = Panel("P", 200, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0))
    config = LocalSearchConfig(max_iterations=0)

    result = process_local_search_multistart_order(
        units,
        panel,
        tool,
        config=config,
    )
    initial_metrics = tuple(
        evaluate_directed_units(order, panel, tool)[1]
        for order in multistart_process_initial_orders(
            units,
            panel=panel,
            tool=tool,
            config=config,
        )
    )

    assert process_metric_key(result.metrics) == min(
        process_metric_key(metrics)
        for metrics in initial_metrics
    )
    assert {candidate.unit.unit_id for candidate in result.directed_units} == {
        unit.unit_id for unit in units
    }


def test_process_aware_beam_polish_does_not_worsen_process_metric() -> None:
    units = (
        _candidate("near_start", Point(0, 0), Point(10, 0)).unit,
        _candidate("far", Point(100, 0), Point(110, 0)).unit,
        _candidate("middle", Point(20, 0), Point(30, 0)).unit,
    )
    panel = Panel("P", 200, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0))

    beam = process_aware_beam_search_order(
        units,
        panel,
        tool,
        config=BeamSearchConfig(beam_width=1, candidate_pool_size=3),
    )
    polished = process_aware_beam_polished_search_order(
        units,
        panel,
        tool,
        beam_config=BeamSearchConfig(beam_width=1, candidate_pool_size=3),
        polish_config=LocalSearchConfig(max_iterations=2),
    )

    assert process_metric_key(polished.metrics) <= process_metric_key(beam.metrics)
    assert {candidate.unit.unit_id for candidate in polished.directed_units} == {
        unit.unit_id for unit in units
    }


def test_process_aware_beam_search_respects_layer_expansion_limit() -> None:
    units = (
        _candidate("u1", Point(0, 0), Point(10, 0)).unit,
        _candidate("u2", Point(20, 0), Point(30, 0)).unit,
        _candidate("u3", Point(40, 0), Point(50, 0)).unit,
    )

    result = process_aware_beam_search_order(
        units,
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
        config=BeamSearchConfig(
            beam_width=3,
            candidate_pool_size=3,
            max_expansions_per_node=6,
            max_layer_expansions=1,
        ),
    )

    assert len(result.directed_units) == len(units)
    assert result.expanded_nodes == len(units)
    assert all(layer.layer_expansion_count == 1 for layer in result.diagnostics)
    assert all(layer.layer_pruned_count >= 0 for layer in result.diagnostics)


def test_process_aware_beam_search_completes_low_support_part() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=200,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(
        layout,
        support_edge_roles=(
            EdgeRole.BOTTOM,
            EdgeRole.RIGHT,
            EdgeRole.TOP,
            EdgeRole.LEFT,
        ),
        min_remaining_support_length_ratio=0.75,
    )
    units = (
        _edge_unit_for_part(
            "A",
            "A:bottom",
            EdgeRole.BOTTOM,
            Point(10, 10),
            Point(30, 10),
        ),
        _edge_unit_for_part(
            "A",
            "A:right",
            EdgeRole.RIGHT,
            Point(30, 10),
            Point(30, 20),
        ),
        _edge_unit_for_part(
            "A",
            "A:top",
            EdgeRole.TOP,
            Point(30, 20),
            Point(10, 20),
        ),
        _edge_unit_for_part(
            "A",
            "A:left",
            EdgeRole.LEFT,
            Point(10, 20),
            Point(10, 10),
        ),
    )

    result = process_aware_beam_search_order(
        units,
        Panel("P", 200, 100),
        ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10)),
        process_model=process_model,
        config=BeamSearchConfig(
            beam_width=3,
            candidate_pool_size=4,
            max_expansions_per_node=8,
        ),
    )

    assert result.metrics.stability_penalty == 0
    assert len(result.directed_units) == 4
