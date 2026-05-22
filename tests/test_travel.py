from cnc_cutting.models import IncrementalMetricsState, Panel, Point, ToolConfig, TravelMode
from cnc_cutting.travel import plan_travel_actions


def test_low_clearance_detour_routes_around_released_rectangle() -> None:
    state = IncrementalMetricsState(
        current_point=Point(20, 50),
        processed_polygon_bounds=((40, 40, 60, 60),),
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(
        trim_margin=0,
        tool_diameter=0,
        centerline_boundary_margin=0,
        allow_low_clearance_detour=True,
        allow_safe_lift_over_released_parts=True,
    )

    actions = plan_travel_actions(Point(20, 50), Point(80, 50), state, panel, tool)

    assert len(actions) > 1
    assert all(action.travel_mode == TravelMode.LOW_CLEARANCE_DETOUR for action in actions)
    assert actions[0].start == Point(20, 50)
    assert actions[-1].end == Point(80, 50)


def test_safe_lift_used_when_detour_is_disabled() -> None:
    state = IncrementalMetricsState(
        current_point=Point(20, 50),
        processed_polygon_bounds=((40, 40, 60, 60),),
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(
        trim_margin=0,
        tool_diameter=0,
        centerline_boundary_margin=0,
        allow_low_clearance_detour=False,
        allow_safe_lift_over_released_parts=True,
    )

    actions = plan_travel_actions(Point(20, 50), Point(80, 50), state, panel, tool)

    assert len(actions) == 1
    assert actions[0].travel_mode == TravelMode.SAFE_LIFT


def test_safe_lift_used_when_it_is_cheaper_than_detour() -> None:
    state = IncrementalMetricsState(
        current_point=Point(20, 50),
        processed_polygon_bounds=((40, 40, 60, 60),),
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(
        trim_margin=0,
        tool_diameter=0,
        centerline_boundary_margin=0,
        allow_low_clearance_detour=True,
        allow_safe_lift_over_released_parts=True,
        safe_lift_fixed_cost=0,
    )

    actions = plan_travel_actions(Point(20, 50), Point(80, 50), state, panel, tool)

    assert len(actions) == 1
    assert actions[0].travel_mode == TravelMode.SAFE_LIFT


def test_safe_lift_used_when_detour_visibility_graph_is_too_large() -> None:
    obstacle_bounds = tuple(
        (
            20.0 + index * 7.0,
            20.0 + index * 7.0,
            23.0 + index * 7.0,
            23.0 + index * 7.0,
        )
        for index in range(64)
    )
    state = IncrementalMetricsState(
        current_point=Point(5, 5),
        processed_polygon_bounds=obstacle_bounds,
    )
    panel = Panel("P", 520, 520)
    tool = ToolConfig(
        trim_margin=0,
        tool_diameter=0,
        centerline_boundary_margin=0,
        allow_low_clearance_detour=True,
        allow_safe_lift_over_released_parts=True,
        safe_lift_fixed_cost=10_000,
    )

    actions = plan_travel_actions(Point(5, 5), Point(500, 500), state, panel, tool)

    assert len(actions) == 1
    assert actions[0].travel_mode == TravelMode.SAFE_LIFT
