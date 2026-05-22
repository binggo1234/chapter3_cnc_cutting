from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.models import (
    CuttingUnit,
    CuttingUnitType,
    EdgeRole,
    EdgeSegment,
    Layout,
    Panel,
    PathMetrics,
    PlacedRectangle,
    Point,
    ToolConfig,
)
from cnc_cutting.optimizer import (
    RoutePlan,
    ToolEventGateConfig,
    _best_process_route,
    greedy_unit_actions,
    plan_exact_process_dp_route,
    plan_path_distance_local_search_route,
    plan_greedy_route,
    plan_local_search_route,
    plan_process_local_search_multistart_route,
    plan_process_aware_beam_adaptive_route,
    plan_process_aware_beam_adaptive_polished_route,
    plan_process_aware_beam_polished_route,
    plan_process_aware_beam_route,
    plan_topology_route,
    select_coverage_units,
)


def _single_unit(unit_id: str, start: Point, end: Point) -> CuttingUnit:
    segment = EdgeSegment(f"{unit_id}:edge", unit_id, EdgeRole.BOTTOM, start, end)
    return CuttingUnit(
        unit_id=unit_id,
        unit_type=CuttingUnitType.SINGLE_EDGE,
        segments=(segment,),
        start=start,
        end=end,
        covered_segment_ids=(segment.segment_id,),
    )


def test_best_process_route_rejects_extra_tool_events_for_small_travel_gain() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            travel_mode_cost=100,
            pierce_count=1,
            lift_count=1,
        ),
    )
    lower_travel_more_events = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            travel_mode_cost=80,
            pierce_count=3,
            lift_count=3,
        ),
    )

    assert _best_process_route((incumbent, lower_travel_more_events)) is incumbent


def test_best_process_route_accepts_extra_tool_events_for_large_travel_gain() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            travel_mode_cost=400,
            pierce_count=1,
            lift_count=1,
        ),
    )
    lower_travel_more_events = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            travel_mode_cost=150,
            pierce_count=2,
            lift_count=2,
        ),
    )

    assert _best_process_route((incumbent, lower_travel_more_events)) is (
        lower_travel_more_events
    )


def test_best_process_route_rejects_extra_tool_events_below_relative_gain() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            travel_mode_cost=10000,
            pierce_count=1,
            lift_count=1,
        ),
    )
    lower_travel_more_events = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            travel_mode_cost=9500,
            pierce_count=2,
            lift_count=3,
        ),
    )

    assert _best_process_route((incumbent, lower_travel_more_events)) is incumbent


def test_best_process_route_can_disable_tool_event_gate() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=5000,
            travel_mode_cost=10000,
            pierce_count=1,
            lift_count=1,
        ),
    )
    lower_travel_more_events = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=5000,
            travel_mode_cost=9500,
            pierce_count=2,
            lift_count=3,
        ),
    )

    assert (
        _best_process_route(
            (incumbent, lower_travel_more_events),
            tool_event_gate=ToolEventGateConfig(enabled=False),
        )
        is lower_travel_more_events
    )


def test_best_process_route_uses_configured_tool_event_gate_threshold() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=5000,
            travel_mode_cost=400,
            pierce_count=1,
            lift_count=1,
        ),
    )
    lower_travel_more_events = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=5000,
            travel_mode_cost=150,
            pierce_count=2,
            lift_count=2,
        ),
    )

    assert (
        _best_process_route(
            (incumbent, lower_travel_more_events),
            tool_event_gate=ToolEventGateConfig(
                min_travel_saving_per_extra_event=200.0,
                min_travel_saving_ratio_per_extra_event=0.0,
            ),
        )
        is incumbent
    )


def test_best_process_route_keeps_protected_plan_even_below_relative_gain() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=5000,
            travel_mode_cost=10000,
            pierce_count=1,
            lift_count=1,
        ),
    )
    protected_lower_travel_more_events = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=5000,
            travel_mode_cost=9500,
            pierce_count=2,
            lift_count=3,
        ),
    )

    assert (
        _best_process_route(
            (incumbent, protected_lower_travel_more_events),
            protected_plans=(protected_lower_travel_more_events,),
        )
        is protected_lower_travel_more_events
    )


def test_best_process_route_uses_process_key_when_tool_events_do_not_increase() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))
    incumbent = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=100,
            travel_mode_cost=100,
            pierce_count=1,
            lift_count=1,
        ),
    )
    lower_cutting_length = RoutePlan(
        selected_units=(unit,),
        actions=(),
        metrics=PathMetrics(
            cutting_length=90,
            travel_mode_cost=100,
            pierce_count=1,
            lift_count=1,
        ),
    )

    assert _best_process_route((incumbent, lower_cutting_length)) is lower_cutting_length


def test_select_coverage_units_prefers_near_shared_unit_over_duplicate_edges() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 0, 0, 10, 10),
            PlacedRectangle("B", 16, 0, 10, 10),
        ),
    )
    units = build_candidate_cutting_units(layout, ToolConfig(tool_diameter=6))

    selected = select_coverage_units(units)
    selected_unit_ids = {unit.unit_id for unit in selected}

    assert any(unit.unit_type == CuttingUnitType.NEAR_SHARED_CHANNEL for unit in selected)
    assert "single:A:right" not in selected_unit_ids
    assert "single:B:left" not in selected_unit_ids


def test_greedy_unit_actions_chooses_reversed_direction_when_closer() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    actions = greedy_unit_actions((unit,), Point(9, 0))

    assert actions[0].start == Point(9, 0)
    assert actions[0].end == Point(10, 0)
    assert actions[1].start == Point(10, 0)
    assert actions[1].end == Point(0, 0)


def test_plan_greedy_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_greedy_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_topology_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_topology_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_local_search_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_local_search_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_path_distance_local_search_route_returns_process_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_path_distance_local_search_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_process_aware_beam_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_process_aware_beam_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_process_aware_beam_adaptive_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_process_aware_beam_adaptive_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_process_aware_beam_adaptive_polished_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_process_aware_beam_adaptive_polished_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_process_local_search_multistart_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_process_local_search_multistart_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_exact_process_dp_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_exact_process_dp_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1


def test_plan_process_aware_beam_polished_route_returns_incremental_metrics() -> None:
    unit = _single_unit("u1", Point(0, 0), Point(10, 0))

    plan = plan_process_aware_beam_polished_route(
        (unit,),
        Panel("P", 100, 100),
        ToolConfig(trim_margin=0, tool_diameter=6, start_point=Point(0, 0)),
    )

    assert len(plan.selected_units) == 1
    assert plan.metrics.cutting_length == 10
    assert plan.metrics.pierce_count == 1
    assert plan.metrics.lift_count == 1
