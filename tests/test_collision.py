from cnc_cutting.collision import action_boundary_penalty, action_collision_penalty
from cnc_cutting.models import (
    CuttingAction,
    CuttingActionType,
    Panel,
    Point,
    ToolConfig,
    TravelMode,
)


def test_travel_crossing_processed_polygon_gets_collision_penalty() -> None:
    processed_square = (
        Point(5, 5),
        Point(15, 5),
        Point(15, 15),
        Point(5, 15),
    )
    action = CuttingAction(CuttingActionType.TRAVEL, Point(0, 10), Point(20, 10))

    assert action_collision_penalty(action, (processed_square,)) == 1.0


def test_travel_collision_uses_cached_polygon_bounds() -> None:
    processed_square = (
        Point(5, 5),
        Point(15, 5),
        Point(15, 15),
        Point(5, 15),
    )
    action = CuttingAction(CuttingActionType.TRAVEL, Point(0, 10), Point(20, 10))

    assert action_collision_penalty(action, (processed_square,), ((5, 5, 15, 15),)) == 1.0


def test_safe_lift_travel_crossing_processed_polygon_has_no_hard_collision() -> None:
    processed_square = (
        Point(5, 5),
        Point(15, 5),
        Point(15, 15),
        Point(5, 15),
    )
    action = CuttingAction(
        CuttingActionType.TRAVEL,
        Point(0, 10),
        Point(20, 10),
        travel_mode=TravelMode.SAFE_LIFT,
    )

    assert action_collision_penalty(action, (processed_square,)) == 0.0


def test_cutting_action_does_not_use_travel_collision_penalty() -> None:
    processed_square = (
        Point(5, 5),
        Point(15, 5),
        Point(15, 15),
        Point(5, 15),
    )
    action = CuttingAction(CuttingActionType.CUT, Point(0, 10), Point(20, 10), "s1")

    assert action_collision_penalty(action, (processed_square,)) == 0.0


def test_boundary_penalty_uses_trim_margin_and_tool_radius() -> None:
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    unsafe_action = CuttingAction(CuttingActionType.CUT, Point(5, 20), Point(20, 20), "s1")
    safe_action = CuttingAction(CuttingActionType.CUT, Point(8, 20), Point(20, 20), "s2")

    assert action_boundary_penalty(unsafe_action, panel, tool) == 1.0
    assert action_boundary_penalty(safe_action, panel, tool) == 0.0


def test_boundary_penalty_allows_imported_trimmed_layout_override() -> None:
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=5, tool_diameter=6, centerline_boundary_margin=3)
    action = CuttingAction(CuttingActionType.CUT, Point(5, 20), Point(20, 20), "s1")

    assert action_boundary_penalty(action, panel, tool) == 0.0
