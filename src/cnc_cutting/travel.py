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
MAX_DETOUR_GRID_NODES = 16384
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

    sorted_xs = sorted(x for x in xs if x_min <= x <= x_max)
    sorted_ys = sorted(y for y in ys if y_min <= y <= y_max)
    if len(sorted_xs) * len(sorted_ys) > MAX_DETOUR_GRID_NODES:
        return None

    points: list[Point] = []
    row_nodes: dict[float, list[int]] = {}
    column_nodes: dict[float, list[int]] = defaultdict(list)
    horizontal_intervals_by_y: dict[float, tuple[tuple[float, float], ...]] = {}
    rows_needing_sort: set[float] = set()
    columns_needing_sort: set[float] = set()
    for y in sorted_ys:
        intervals = _horizontal_blocking_intervals(
            y,
            obstacle_bounds,
        )
        horizontal_intervals_by_y[y] = intervals
        row: list[Point] = []
        interval_count = len(intervals)
        interval_index = 0
        for x in sorted_xs:
            while (
                interval_index < interval_count
                and intervals[interval_index][1] <= x + 1e-9
            ):
                interval_index += 1
            if (
                interval_index < interval_count
                and intervals[interval_index][0] + 1e-9 < x < intervals[interval_index][1] - 1e-9
            ):
                continue
            point = Point(x, y)
            point_id = len(points)
            points.append(point)
            row.append(point_id)
            column_nodes[x].append(point_id)
        row_nodes[y] = row

    start_node_id: int | None = None
    end_node_id: int | None = None
    for endpoint in (start, end):
        if not point_inside_panel_work_area(
            endpoint,
            panel.panel_width,
            panel.panel_height,
            margin,
        ):
            continue
        row = row_nodes.setdefault(endpoint.y, [])
        existing_id = next(
            (node_id for node_id in row if points[node_id].x == endpoint.x),
            None,
        )
        if existing_id is None:
            existing_id = len(points)
            points.append(endpoint)
            row.append(existing_id)
            column_nodes[endpoint.x].append(existing_id)
            rows_needing_sort.add(endpoint.y)
            columns_needing_sort.add(endpoint.x)
            horizontal_intervals_by_y.setdefault(
                endpoint.y,
                _horizontal_blocking_intervals(endpoint.y, obstacle_bounds),
            )
        if endpoint == start:
            start_node_id = existing_id
        if endpoint == end:
            end_node_id = existing_id

    if start_node_id is None or end_node_id is None:
        return None

    vertical_intervals_by_x = {
        x: _vertical_blocking_intervals(x, obstacle_bounds)
        for x in column_nodes
    }

    adjacency: list[list[tuple[float, int]]] = [[] for _ in points]
    for y, row in row_nodes.items():
        if y in rows_needing_sort:
            row.sort(key=lambda node_id: points[node_id].x)
        _connect_horizontal_neighbors(
            row,
            points,
            adjacency,
            horizontal_intervals_by_y[y],
        )
    for x, column in column_nodes.items():
        if x in columns_needing_sort:
            column.sort(key=lambda node_id: points[node_id].y)
        _connect_vertical_neighbors(
            column,
            points,
            adjacency,
            vertical_intervals_by_x[x],
        )

    return _shortest_point_path(start_node_id, end_node_id, points, adjacency)


def _connect_horizontal_neighbors(
    ordered_nodes: list[int],
    points: list[Point],
    adjacency: list[list[tuple[float, int]]],
    intervals: tuple[tuple[float, float], ...],
) -> None:
    interval_count = len(intervals)
    interval_index = 0
    for first_id, second_id in zip(ordered_nodes, ordered_nodes[1:]):
        first = points[first_id]
        second = points[second_id]
        lower = first.x
        upper = second.x
        while (
            interval_index < interval_count
            and intervals[interval_index][1] <= lower + 1e-9
        ):
            interval_index += 1
        if interval_index < interval_count and intervals[interval_index][0] < upper - 1e-9:
            continue
        distance = upper - lower
        adjacency[first_id].append((distance, second_id))
        adjacency[second_id].append((distance, first_id))


def _connect_vertical_neighbors(
    ordered_nodes: list[int],
    points: list[Point],
    adjacency: list[list[tuple[float, int]]],
    intervals: tuple[tuple[float, float], ...],
) -> None:
    interval_count = len(intervals)
    interval_index = 0
    for first_id, second_id in zip(ordered_nodes, ordered_nodes[1:]):
        first = points[first_id]
        second = points[second_id]
        lower = first.y
        upper = second.y
        while (
            interval_index < interval_count
            and intervals[interval_index][1] <= lower + 1e-9
        ):
            interval_index += 1
        if interval_index < interval_count and intervals[interval_index][0] < upper - 1e-9:
            continue
        distance = upper - lower
        adjacency[first_id].append((distance, second_id))
        adjacency[second_id].append((distance, first_id))


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


def _horizontal_blocking_intervals(
    y: float,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        sorted(
            (min_x, max_x)
            for min_x, min_y, max_x, max_y in obstacle_bounds
            if min_y + tolerance < y < max_y - tolerance
        )
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


def _vertical_blocking_intervals(
    x: float,
    obstacle_bounds: tuple[tuple[float, float, float, float], ...],
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        sorted(
            (min_y, max_y)
            for min_x, min_y, max_x, max_y in obstacle_bounds
            if min_x + tolerance < x < max_x - tolerance
        )
    )


def _manhattan_distance(first: Point, second: Point) -> float:
    return abs(first.x - second.x) + abs(first.y - second.y)


def _shortest_point_path(
    start_id: int,
    end_id: int,
    points: list[Point],
    adjacency: list[list[tuple[float, int]]],
) -> tuple[Point, ...] | None:
    start = points[start_id]
    end = points[end_id]
    queue: list[tuple[float, float, int, int]] = [
        (_manhattan_distance(start, end), 0.0, 0, start_id)
    ]
    best_distance = [float("inf")] * len(points)
    best_distance[start_id] = 0.0
    previous = [-1] * len(points)
    counter = 1

    while queue:
        _, distance, _, node_id = heappop(queue)
        if node_id == end_id:
            break
        if distance > best_distance[node_id] + 1e-9:
            continue
        for edge_distance, neighbor_id in adjacency[node_id]:
            candidate_distance = distance + edge_distance
            if candidate_distance + 1e-9 >= best_distance[neighbor_id]:
                continue
            best_distance[neighbor_id] = candidate_distance
            previous[neighbor_id] = node_id
            heappush(
                queue,
                (
                    candidate_distance + _manhattan_distance(points[neighbor_id], end),
                    candidate_distance,
                    counter,
                    neighbor_id,
                ),
            )
            counter += 1

    if best_distance[end_id] == float("inf"):
        return None

    path_ids = [end_id]
    while path_ids[-1] != start_id:
        previous_id = previous[path_ids[-1]]
        if previous_id < 0:
            return None
        path_ids.append(previous_id)
    path_ids.reverse()
    return tuple(points[node_id] for node_id in path_ids)


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
