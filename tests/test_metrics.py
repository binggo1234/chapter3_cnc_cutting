from cnc_cutting.metrics import apply_action_incremental, evaluate_actions, finalize_metrics
from cnc_cutting.models import (
    CuttingAction,
    CuttingActionType,
    EdgeRole,
    IncrementalMetricsState,
    Panel,
    PathMetrics,
    Point,
    ToolConfig,
)
from cnc_cutting.process_model import build_process_model


def test_incremental_metrics_matches_global_evaluation() -> None:
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6)
    actions = (
        CuttingAction(CuttingActionType.TRAVEL, Point(0, 0), Point(3, 4)),
        CuttingAction(CuttingActionType.CUT, Point(3, 4), Point(13, 4), "s1"),
    )

    state = IncrementalMetricsState(current_point=Point(0, 0))
    for action in actions:
        state = apply_action_incremental(action, state, panel, tool)

    global_metrics = evaluate_actions(actions, panel, tool)

    assert state.metrics.air_move_distance == global_metrics.air_move_distance
    assert state.metrics.cutting_length == global_metrics.cutting_length
    assert state.metrics.pierce_count == 1


def test_path_metrics_machining_cost_combines_cutting_and_travel_modes() -> None:
    metrics = PathMetrics(cutting_length=12.5, travel_mode_cost=7.5)

    assert metrics.machining_cost == 20.0


def test_continuous_cut_keeps_tool_down_until_final_lift() -> None:
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6)
    actions = (
        CuttingAction(CuttingActionType.CUT, Point(0, 0), Point(10, 0), "s1"),
        CuttingAction(CuttingActionType.CUT, Point(10, 0), Point(20, 0), "s2"),
    )

    metrics = evaluate_actions(actions, panel, tool)

    assert metrics.cutting_length == 20
    assert metrics.pierce_count == 1
    assert metrics.lift_count == 1


def test_travel_after_cut_counts_one_lift() -> None:
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=6)
    actions = (
        CuttingAction(CuttingActionType.CUT, Point(0, 0), Point(10, 0), "s1"),
        CuttingAction(CuttingActionType.TRAVEL, Point(10, 0), Point(10, 10)),
    )

    metrics = evaluate_actions(actions, panel, tool)

    assert metrics.pierce_count == 1
    assert metrics.lift_count == 1
    assert metrics.air_move_distance == 10


def _rectangle_cut_actions(part_id: str = "A") -> tuple[CuttingAction, ...]:
    return (
        CuttingAction(
            CuttingActionType.CUT,
            Point(10, 10),
            Point(30, 10),
            f"{part_id}:bottom",
        ),
        CuttingAction(
            CuttingActionType.CUT,
            Point(30, 10),
            Point(30, 20),
            f"{part_id}:right",
        ),
        CuttingAction(
            CuttingActionType.CUT,
            Point(30, 20),
            Point(10, 20),
            f"{part_id}:top",
        ),
        CuttingAction(
            CuttingActionType.CUT,
            Point(10, 20),
            Point(10, 10),
            f"{part_id}:left",
        ),
    )


def test_incremental_state_releases_part_after_all_edges_are_cut() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(layout)
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    state = IncrementalMetricsState(current_point=tool.start_point)

    for action in _rectangle_cut_actions():
        state = apply_action_incremental(
            action,
            state,
            panel,
            tool,
            process_model=process_model,
        )

    assert state.released_part_ids == {"A"}
    assert state.processed_polygons == (process_model.part_polygons["A"],)
    assert state.processed_polygon_bounds == ((10, 10, 30, 20),)


def test_travel_after_part_release_gets_dynamic_collision_penalty() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(layout)
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    actions = _rectangle_cut_actions() + (
        CuttingAction(CuttingActionType.TRAVEL, Point(0, 15), Point(40, 15)),
    )

    metrics = evaluate_actions(actions, panel, tool, process_model=process_model)

    assert metrics.collision_penalty == 1.0


def test_safe_lift_option_converts_release_collision_to_soft_cost() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(layout)
    panel = Panel("P", 100, 100)
    tool = ToolConfig(
        trim_margin=0,
        tool_diameter=0,
        start_point=Point(10, 10),
        allow_safe_lift_over_released_parts=True,
    )
    actions = _rectangle_cut_actions() + (
        CuttingAction(CuttingActionType.TRAVEL, Point(0, 15), Point(40, 15)),
    )

    metrics = evaluate_actions(actions, panel, tool, process_model=process_model)

    assert metrics.collision_penalty == 0.0
    assert metrics.safe_lift_count == 1
    assert metrics.safe_lift_distance == 40
    assert metrics.travel_mode_cost == 290


def test_cutting_support_edge_too_early_gets_stability_penalty() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(layout)
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 20))
    actions = (
        CuttingAction(CuttingActionType.CUT, Point(30, 20), Point(10, 20), "A:top"),
    )

    metrics = evaluate_actions(actions, panel, tool, process_model=process_model)

    assert metrics.stability_penalty == 1.0


def test_cutting_support_edge_last_has_no_stability_penalty() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )
    process_model = build_process_model(layout)
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))

    actions = (
        CuttingAction(CuttingActionType.CUT, Point(10, 10), Point(30, 10), "A:bottom"),
        CuttingAction(CuttingActionType.CUT, Point(30, 10), Point(30, 20), "A:right"),
        CuttingAction(CuttingActionType.TRAVEL, Point(30, 20), Point(10, 20)),
        CuttingAction(CuttingActionType.CUT, Point(10, 20), Point(10, 10), "A:left"),
        CuttingAction(CuttingActionType.TRAVEL, Point(10, 10), Point(10, 20)),
        CuttingAction(CuttingActionType.CUT, Point(10, 20), Point(30, 20), "A:top"),
    )

    metrics = evaluate_actions(actions, panel, tool, process_model=process_model)

    assert metrics.stability_penalty == 0.0


def test_multi_edge_support_penalizes_low_remaining_support_ratio() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
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
        min_remaining_support_length_ratio=0.5,
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))

    stable_prefix = _rectangle_cut_actions()[:2]
    unstable_prefix = _rectangle_cut_actions()[:3]

    stable_metrics = evaluate_actions(
        stable_prefix,
        panel,
        tool,
        process_model=process_model,
    )
    unstable_metrics = evaluate_actions(
        unstable_prefix,
        panel,
        tool,
        process_model=process_model,
    )

    assert stable_metrics.stability_penalty == 0.0
    assert unstable_metrics.stability_penalty == 1.0


def test_multi_edge_support_allows_immediate_part_completion() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
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
        min_remaining_support_length_ratio=0.5,
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))

    metrics = evaluate_actions(
        _rectangle_cut_actions(),
        panel,
        tool,
        process_model=process_model,
    )

    assert metrics.stability_penalty == 0.0


def test_completion_travel_keeps_unstable_part_context() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
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
        min_remaining_support_length_ratio=0.5,
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    state = IncrementalMetricsState(current_point=tool.start_point)
    for action in _rectangle_cut_actions()[:3]:
        state = apply_action_incremental(
            action,
            state,
            panel,
            tool,
            process_model=process_model,
        )

    assert state.unstable_part_ids == {"A"}

    completion_travel = CuttingAction(
        action_type=CuttingActionType.TRAVEL,
        start=Point(10, 20),
        end=Point(10, 18),
        cutting_unit_id="single:A:left",
        covered_segment_ids=("A:left",),
    )
    next_state = apply_action_incremental(
        completion_travel,
        state,
        panel,
        tool,
        process_model=process_model,
    )

    assert next_state.metrics.stability_penalty == 0.0
    assert next_state.unstable_part_ids == {"A"}


def test_unannotated_travel_abandons_unstable_part() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
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
        min_remaining_support_length_ratio=0.5,
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    state = IncrementalMetricsState(current_point=tool.start_point)
    for action in _rectangle_cut_actions()[:3]:
        state = apply_action_incremental(
            action,
            state,
            panel,
            tool,
            process_model=process_model,
        )

    abandoned_state = apply_action_incremental(
        CuttingAction(
            action_type=CuttingActionType.TRAVEL,
            start=Point(10, 20),
            end=Point(50, 50),
        ),
        state,
        panel,
        tool,
        process_model=process_model,
    )

    assert abandoned_state.metrics.stability_penalty == 1.0
    assert abandoned_state.unstable_part_ids == set()


def test_continuing_one_unstable_part_retains_other_unstable_parts() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 10, 10, 10, 10),
            PlacedRectangle("B", 30, 10, 10, 10),
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
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    state = IncrementalMetricsState(
        current_point=Point(10, 20),
        unstable_part_ids={"A", "B"},
    )

    next_state = apply_action_incremental(
        CuttingAction(
            CuttingActionType.CUT,
            Point(10, 20),
            Point(10, 10),
            "A:left",
        ),
        state,
        panel,
        tool,
        process_model=process_model,
    )

    assert next_state.metrics.stability_penalty == 0.0
    assert "B" in next_state.unstable_part_ids


def test_adjacent_unreleased_part_contributes_temporary_support() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 10, 10, 10, 10),
            PlacedRectangle("B", 20, 10, 10, 10),
        ),
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    actions = (
        CuttingAction(CuttingActionType.CUT, Point(10, 10), Point(20, 10), "A:bottom"),
        CuttingAction(CuttingActionType.CUT, Point(20, 20), Point(10, 20), "A:top"),
        CuttingAction(CuttingActionType.CUT, Point(10, 20), Point(10, 10), "A:left"),
    )
    unsupported_model = build_process_model(
        layout,
        support_edge_roles=(
            EdgeRole.BOTTOM,
            EdgeRole.RIGHT,
            EdgeRole.TOP,
            EdgeRole.LEFT,
        ),
        min_remaining_support_length_ratio=0.5,
        adjacency_support_weight=0.0,
    )
    adjacency_supported_model = build_process_model(
        layout,
        support_edge_roles=(
            EdgeRole.BOTTOM,
            EdgeRole.RIGHT,
            EdgeRole.TOP,
            EdgeRole.LEFT,
        ),
        min_remaining_support_length_ratio=0.5,
        adjacency_support_weight=1.0,
    )

    unsupported_metrics = evaluate_actions(
        actions,
        panel,
        tool,
        process_model=unsupported_model,
    )
    adjacency_supported_metrics = evaluate_actions(
        actions,
        panel,
        tool,
        process_model=adjacency_supported_model,
    )

    assert unsupported_metrics.stability_penalty == 1.0
    assert adjacency_supported_metrics.stability_penalty == 0.0


def test_released_adjacent_part_no_longer_contributes_support() -> None:
    from cnc_cutting.models import Layout, PlacedRectangle

    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 10, 10, 10, 10),
            PlacedRectangle("B", 20, 10, 10, 10),
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
        adjacency_support_weight=1.0,
    )
    panel = Panel("P", 100, 100)
    tool = ToolConfig(trim_margin=0, tool_diameter=0, start_point=Point(10, 10))
    state = IncrementalMetricsState(current_point=tool.start_point)
    for action in _rectangle_cut_actions("B"):
        state = apply_action_incremental(
            action,
            state,
            panel,
            tool,
            process_model=process_model,
        )
    for action in (
        CuttingAction(CuttingActionType.CUT, Point(10, 10), Point(20, 10), "A:bottom"),
        CuttingAction(CuttingActionType.CUT, Point(20, 20), Point(10, 20), "A:top"),
        CuttingAction(CuttingActionType.CUT, Point(10, 20), Point(10, 10), "A:left"),
    ):
        state = apply_action_incremental(
            action,
            state,
            panel,
            tool,
            process_model=process_model,
        )

    assert "B" in state.released_part_ids
    assert state.unstable_part_ids == {"A"}
    assert finalize_metrics(state).stability_penalty == 1.0
