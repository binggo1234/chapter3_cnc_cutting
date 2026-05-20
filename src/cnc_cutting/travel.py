from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush

from .collision import effective_centerline_margin
from .geometry import (
    euclidean_distance,
    point_inside_panel_work_area,
    segment_crosses_axis_aligned_rectangle_interior,
)
from .models import (
    CuttingAction,
    CuttingActionType,
    IncrementalMetricsState,
    Panel,
    Point,
    ToolConfig,
    TravelMode,
)


MAX_DETOUR_OBSTACLES = 64
DETOUR_EPSILON = 1e-6
DETOUR_CACHE_MAX_SIZE = 20000

_DETOUR_POINTS_CACHE: dict[
    tuple[
        Point,
        Point,
        tuple[tuple[float, float, float, float], ...],
        float,
        float,
        float,
        float,
    ],
    tuple[Point, ...] | None,
] = {}


def clear_detour_cache() -> None:
    _DETOUR_POINTS_CACHE.clear()


def travel_action(
    start: Point,
    end: Point,
    travel_mode: TravelMode = TravelMode.LOW_CLEARANCE,
    cutting_unit_id: str | None = None,
    covered_segment_ids: tuple[str, ...] = (),
) -> CuttingAction:
    return CuttingAction(
        action_type=CuttingActionType.TRAVEL,
        start=start,
        end=end,
        cutting_unit_id=cutting_unit_id,
        covered_segment_ids=covered_segment_ids,
        travel_mode=travel_mode,
    )


def plan_travel_actions(
    start: Point,
    end: Point,
    state: IncrementalMetricsState,
    panel: Panel,
    tool: ToolConfig,
    cutting_unit_id: str | None = None,
    covered_segment_ids: tuple[str, ...] = (),
) -> tuple[CuttingAction, ...]:
    if euclidean_distance(start, end) <= 1e-9:
        return ()

    direct = travel_action(
        start,
        end,
        cutting_unit_id=cutting_unit_id,
        covered_segment_ids=covered_segment_ids,
    )
    if _low_clearance_segment_clear(
        start,
        end,
        state.processed_polygon_bounds,
        panel,
        tool,
    ):
        return (direct,)

    safe_lift = (
        travel_action(
            start,
            end,
            TravelMode.SAFE_LIFT,
            cutting_unit_id=cutting_unit_id,
            covered_segment_ids=covered_segment_ids,
        )
        if tool.allow_safe_lift_over_released_parts
        else None
    )
    safe_lift_cost = (
        _travel_action_mode_cost(safe_lift, tool)
        if safe_lift is not None
        else float("inf")
    )
    direct_distance = euclidean_distance(start, end)
    detour_lower_bound = direct_distance * tool.detour_travel_weight
    if tool.allow_low_clearance_detour and safe_lift_cost > detour_lower_bound:
        detour_actions = _low_clearance_detour_actions(
            start,
            end,
            state.processed_polygon_bounds,
            panel,
            tool,
            cutting_unit_id=cutting_unit_id,
            covered_segment_ids=covered_segment_ids,
        )
        if detour_actions is not None:
            detour_cost = sum(_travel_action_mode_cost(action, tool) for action in detour_actions)
            if detour_cost <= safe_lift_cost:
                return detour_actions

    if safe_lift is not None:
        return (safe_lift,)
    return (direct,)


def _low_clearance_detour_actions(
    start: Point,
    end: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    tool: ToolConfig,
    cutting_unit_id: str | None = None,
    covered_segment_ids: tuple[str, ...] = (),
) -> tuple[CuttingAction, ...] | None:
    if tool.allow_low_clearance_detour:
        detour_points = _low_clearance_detour_points(
            start,
            end,
            obstacle_bounds,
            panel,
            tool,
        )
        if detour_points is not None:
            return tuple(
                travel_action(
                    first,
                    second,
                    TravelMode.LOW_CLEARANCE_DETOUR,
                    cutting_unit_id=cutting_unit_id,
                    covered_segment_ids=covered_segment_ids,
                )
                for first, second in zip(detour_points, detour_points[1:])
                if euclidean_distance(first, second) > 1e-9
            )
    return None


def _travel_action_mode_cost(action: CuttingAction, tool: ToolConfig) -> float:
    length = euclidean_distance(action.start, action.end)
    if action.travel_mode == TravelMode.SAFE_LIFT:
        return tool.safe_lift_fixed_cost + length * tool.safe_lift_travel_weight
    if action.travel_mode == TravelMode.LOW_CLEARANCE_DETOUR:
        return length * tool.detour_travel_weight
    return length * tool.low_clearance_travel_weight


def _low_clearance_detour_points(
    start: Point,
    end: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    tool: ToolConfig,
) -> tuple[Point, ...] | None:
    if not obstacle_bounds or len(obstacle_bounds) > MAX_DETOUR_OBSTACLES:
        return None

    margin = effective_centerline_margin(tool)
    offset = max(tool.tool_radius + tool.safe_clearance, DETOUR_EPSILON)
    canonical_obstacle_bounds = tuple(sorted(obstacle_bounds))
    cache_key = (
        start,
        end,
        canonical_obstacle_bounds,
        panel.panel_width,
        panel.panel_height,
        margin,
        offset,
    )
    if cache_key in _DETOUR_POINTS_CACHE:
        return _DETOUR_POINTS_CACHE[cache_key]
    reverse_cache_key = (
        end,
        start,
        canonical_obstacle_bounds,
        panel.panel_width,
        panel.panel_height,
        margin,
        offset,
    )
    if reverse_cache_key in _DETOUR_POINTS_CACHE:
        reverse_result = _DETOUR_POINTS_CACHE[reverse_cache_key]
        if reverse_result is None:
            return None
        return tuple(reversed(reverse_result))

    result = _compute_low_clearance_detour_points(
        start,
        end,
        canonical_obstacle_bounds,
        panel,
        margin,
        offset,
    )
    if len(_DETOUR_POINTS_CACHE) >= DETOUR_CACHE_MAX_SIZE:
        _DETOUR_POINTS_CACHE.clear()
    _DETOUR_POINTS_CACHE[cache_key] = result
    return result


def _compute_low_clearance_detour_points(
    start: Point,
    end: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    margin: float,
    offset: float,
) -> tuple[Point, ...] | None:
    x_min = margin
    y_min = margin
    x_max = panel.panel_width - margin
    y_max = panel.panel_height - margin

    xs = {start.x, end.x, x_min, x_max}
    ys = {start.y, end.y, y_min, y_max}
    for min_x, min_y, max_x, max_y in obstacle_bounds:
        for x in (min_x - offset, max_x + offset):
            if x_min <= x <= x_max:
                xs.add(x)
        for y in (min_y - offset, max_y + offset):
            if y_min <= y <= y_max:
                ys.add(y)

    nodes: set[Point] = set()
    sorted_xs = sorted(xs)
    for y in sorted(ys):
        row_obstacle_bounds = _horizontal_relevant_obstacle_bounds(
            y,
            obstacle_bounds,
        )
        for x in sorted_xs:
            point = Point(x, y)
            if _point_valid_for_low_clearance_row(
                point,
                row_obstacle_bounds,
                panel,
                margin,
            ):
                nodes.add(point)
    nodes.add(start)
    nodes.add(end)

    row_nodes: dict[float, list[Point]] = defaultdict(list)
    column_nodes: dict[float, list[Point]] = defaultdict(list)
    for node in nodes:
        row_nodes[node.y].append(node)
        column_nodes[node.x].append(node)

    adjacency: dict[Point, list[tuple[float, Point]]] = defaultdict(list)
    for row in row_nodes.values():
        row.sort(key=lambda point: point.x)
        _connect_visible_neighbors(row, adjacency, obstacle_bounds, panel, margin)
    for column in column_nodes.values():
        column.sort(key=lambda point: point.y)
        _connect_visible_neighbors(column, adjacency, obstacle_bounds, panel, margin)

    return _shortest_point_path(start, end, adjacency)


def _connect_visible_neighbors(
    ordered_nodes: list[Point],
    adjacency: dict[Point, list[tuple[float, Point]]],
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    margin: float,
) -> None:
    relevant_obstacle_bounds = _axis_relevant_obstacle_bounds(
        ordered_nodes,
        obstacle_bounds,
    )
    for first, second in zip(ordered_nodes, ordered_nodes[1:]):
        if not _low_clearance_segment_clear_between_valid_points(
            first,
            second,
            relevant_obstacle_bounds,
            panel,
            margin,
        ):
            continue
        distance = euclidean_distance(first, second)
        adjacency[first].append((distance, second))
        adjacency[second].append((distance, first))


def _axis_relevant_obstacle_bounds(
    ordered_nodes: list[Point],
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float, float, float], ...]:
    if len(ordered_nodes) < 2 or not obstacle_bounds:
        return obstacle_bounds

    first = ordered_nodes[0]
    last = ordered_nodes[-1]
    if abs(first.y - last.y) <= tolerance:
        return _horizontal_relevant_obstacle_bounds(
            first.y,
            obstacle_bounds,
            tolerance=tolerance,
        )
    if abs(first.x - last.x) <= tolerance:
        return _vertical_relevant_obstacle_bounds(
            first.x,
            obstacle_bounds,
            tolerance=tolerance,
        )
    return obstacle_bounds


def _horizontal_relevant_obstacle_bounds(
    y: float,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        bounds
        for bounds in obstacle_bounds
        if bounds[1] + tolerance < y < bounds[3] - tolerance
    )


def _vertical_relevant_obstacle_bounds(
    x: float,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        bounds
        for bounds in obstacle_bounds
        if bounds[0] + tolerance < x < bounds[2] - tolerance
    )


def _low_clearance_segment_clear_between_valid_points(
    start: Point,
    end: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    margin: float,
) -> bool:
    if not (
        margin <= start.x <= panel.panel_width - margin
        and margin <= end.x <= panel.panel_width - margin
        and margin <= start.y <= panel.panel_height - margin
        and margin <= end.y <= panel.panel_height - margin
    ):
        return False
    return not any(
        _segment_crosses_axis_aligned_bounds_interior(start, end, bounds)
        for bounds in obstacle_bounds
    )


def _segment_crosses_axis_aligned_bounds_interior(
    start: Point,
    end: Point,
    bounds: tuple[float, float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    min_x, min_y, max_x, max_y = bounds
    if abs(start.y - end.y) <= tolerance:
        y = start.y
        if y <= min_y + tolerance or y >= max_y - tolerance:
            return False
        segment_min_x = min(start.x, end.x)
        segment_max_x = max(start.x, end.x)
        return max(segment_min_x, min_x) < min(segment_max_x, max_x) - tolerance
    if abs(start.x - end.x) <= tolerance:
        x = start.x
        if x <= min_x + tolerance or x >= max_x - tolerance:
            return False
        segment_min_y = min(start.y, end.y)
        segment_max_y = max(start.y, end.y)
        return max(segment_min_y, min_y) < min(segment_max_y, max_y) - tolerance
    return segment_crosses_axis_aligned_rectangle_interior(start, end, bounds)


def _shortest_point_path(
    start: Point,
    end: Point,
    adjacency: dict[Point, list[tuple[float, Point]]],
) -> tuple[Point, ...] | None:
    queue: list[tuple[float, int, Point]] = [(0.0, 0, start)]
    best_distance = {start: 0.0}
    previous: dict[Point, Point] = {}
    counter = 1

    while queue:
        distance, _, node = heappop(queue)
        if node == end:
            break
        if distance > best_distance.get(node, float("inf")) + 1e-9:
            continue
        for edge_distance, neighbor in adjacency.get(node, ()):
            candidate_distance = distance + edge_distance
            if candidate_distance + 1e-9 >= best_distance.get(neighbor, float("inf")):
                continue
            best_distance[neighbor] = candidate_distance
            previous[neighbor] = node
            heappush(queue, (candidate_distance, counter, neighbor))
            counter += 1

    if end not in best_distance:
        return None

    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return tuple(path)


def _point_valid_for_low_clearance(
    point: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    margin: float,
) -> bool:
    if not point_inside_panel_work_area(
        point,
        panel.panel_width,
        panel.panel_height,
        margin,
    ):
        return False
    return not any(_point_inside_bounds_interior(point, bounds) for bounds in obstacle_bounds)


def _point_valid_for_low_clearance_row(
    point: Point,
    row_obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    margin: float,
) -> bool:
    if not point_inside_panel_work_area(
        point,
        panel.panel_width,
        panel.panel_height,
        margin,
    ):
        return False
    return not any(
        bounds[0] + 1e-9 < point.x < bounds[2] - 1e-9
        for bounds in row_obstacle_bounds
    )


def _point_inside_bounds_interior(
    point: Point,
    bounds: tuple[float, float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return (
        min_x + tolerance < point.x < max_x - tolerance
        and min_y + tolerance < point.y < max_y - tolerance
    )


def _low_clearance_segment_clear(
    start: Point,
    end: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    tool: ToolConfig,
) -> bool:
    margin = effective_centerline_margin(tool)
    return _low_clearance_segment_clear_with_margin(
        start,
        end,
        obstacle_bounds,
        panel,
        margin,
    )


def _low_clearance_segment_clear_with_margin(
    start: Point,
    end: Point,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    panel: Panel,
    margin: float,
) -> bool:
    if not (
        _point_valid_for_low_clearance(start, obstacle_bounds, panel, margin)
        and _point_valid_for_low_clearance(end, obstacle_bounds, panel, margin)
    ):
        return False
    return not any(
        segment_crosses_axis_aligned_rectangle_interior(start, end, bounds)
        for bounds in obstacle_bounds
    )
