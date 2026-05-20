from __future__ import annotations

from math import hypot, isclose

from .models import EdgeRole, EdgeSegment, PlacedRectangle, Point


def euclidean_distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def manhattan_distance(a: Point, b: Point) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)


def rectangle_corners(rectangle: PlacedRectangle) -> tuple[Point, Point, Point, Point]:
    """Return rectangle corners in clockwise order from the lower-left corner."""

    x0 = rectangle.x
    y0 = rectangle.y
    x1 = rectangle.x + rectangle.width
    y1 = rectangle.y + rectangle.height
    return (
        Point(x0, y0),
        Point(x1, y0),
        Point(x1, y1),
        Point(x0, y1),
    )


def rectangle_perimeter(rectangle: PlacedRectangle) -> float:
    return 2.0 * (rectangle.width + rectangle.height)


def rectangle_edges(rectangle: PlacedRectangle) -> tuple[EdgeSegment, ...]:
    lower_left, lower_right, upper_right, upper_left = rectangle_corners(rectangle)
    part_id = rectangle.part_id
    return (
        EdgeSegment(f"{part_id}:bottom", part_id, EdgeRole.BOTTOM, lower_left, lower_right),
        EdgeSegment(f"{part_id}:right", part_id, EdgeRole.RIGHT, lower_right, upper_right),
        EdgeSegment(f"{part_id}:top", part_id, EdgeRole.TOP, upper_right, upper_left),
        EdgeSegment(f"{part_id}:left", part_id, EdgeRole.LEFT, upper_left, lower_left),
    )


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    a_lo, a_hi = min(a0, a1), max(a0, a1)
    b_lo, b_hi = min(b0, b1), max(b0, b1)
    if a_hi < b_lo:
        return b_lo - a_hi
    if b_hi < a_lo:
        return a_lo - b_hi
    return 0.0


def segment_gap(a: EdgeSegment, b: EdgeSegment) -> float:
    if a.orientation != b.orientation:
        return float("inf")
    if a.orientation.value == "horizontal":
        return abs(a.start.y - b.start.y)
    return abs(a.start.x - b.start.x)


def perpendicular_gap(a: EdgeSegment, b: EdgeSegment) -> float:
    return segment_gap(a, b)


def axial_gap(a: EdgeSegment, b: EdgeSegment) -> float:
    if a.orientation != b.orientation:
        return float("inf")
    if a.orientation.value == "horizontal":
        return interval_gap(a.start.x, a.end.x, b.start.x, b.end.x)
    return interval_gap(a.start.y, a.end.y, b.start.y, b.end.y)


def collinear(a: EdgeSegment, b: EdgeSegment, tolerance: float = 1e-6) -> bool:
    if a.orientation != b.orientation:
        return False
    if a.orientation.value == "horizontal":
        return isclose(a.start.y, b.start.y, abs_tol=tolerance)
    return isclose(a.start.x, b.start.x, abs_tol=tolerance)


def parallel_overlap_length(a: EdgeSegment, b: EdgeSegment) -> float:
    if a.orientation != b.orientation:
        return 0.0
    if a.orientation.value == "horizontal":
        return interval_overlap(a.start.x, a.end.x, b.start.x, b.end.x)
    return interval_overlap(a.start.y, a.end.y, b.start.y, b.end.y)


def direction_vector(start: Point, end: Point) -> tuple[float, float]:
    length = euclidean_distance(start, end)
    if length == 0:
        return (0.0, 0.0)
    return ((end.x - start.x) / length, (end.y - start.y) / length)


def cross_product(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def point_on_segment(point: Point, start: Point, end: Point, tolerance: float = 1e-9) -> bool:
    return (
        abs(cross_product(start, end, point)) <= tolerance
        and min(start.x, end.x) - tolerance <= point.x <= max(start.x, end.x) + tolerance
        and min(start.y, end.y) - tolerance <= point.y <= max(start.y, end.y) + tolerance
    )


def segments_intersect(
    a_start: Point,
    a_end: Point,
    b_start: Point,
    b_end: Point,
    tolerance: float = 1e-9,
) -> bool:
    d1 = cross_product(a_start, a_end, b_start)
    d2 = cross_product(a_start, a_end, b_end)
    d3 = cross_product(b_start, b_end, a_start)
    d4 = cross_product(b_start, b_end, a_end)

    if (
        (d1 > tolerance and d2 < -tolerance or d1 < -tolerance and d2 > tolerance)
        and (d3 > tolerance and d4 < -tolerance or d3 < -tolerance and d4 > tolerance)
    ):
        return True

    return (
        point_on_segment(b_start, a_start, a_end, tolerance)
        or point_on_segment(b_end, a_start, a_end, tolerance)
        or point_on_segment(a_start, b_start, b_end, tolerance)
        or point_on_segment(a_end, b_start, b_end, tolerance)
    )


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        crosses = (current.y > point.y) != (previous.y > point.y)
        if crosses:
            x_at_y = (previous.x - current.x) * (point.y - current.y) / (
                previous.y - current.y
            ) + current.x
            if point.x < x_at_y:
                inside = not inside
        j = i
    return inside


def bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    first_min_x, first_min_y, first_max_x, first_max_y = first
    second_min_x, second_min_y, second_max_x, second_max_y = second
    return not (
        first_max_x < second_min_x - tolerance
        or second_max_x < first_min_x - tolerance
        or first_max_y < second_min_y - tolerance
        or second_max_y < first_min_y - tolerance
    )


def segment_bounds(start: Point, end: Point) -> tuple[float, float, float, float]:
    return (
        min(start.x, end.x),
        min(start.y, end.y),
        max(start.x, end.x),
        max(start.y, end.y),
    )


def polygon_bounds(polygon: tuple[Point, ...]) -> tuple[float, float, float, float]:
    return (
        min(point.x for point in polygon),
        min(point.y for point in polygon),
        max(point.x for point in polygon),
        max(point.y for point in polygon),
    )


def axis_aligned_rectangle_bounds(
    polygon: tuple[Point, ...],
    tolerance: float = 1e-9,
) -> tuple[float, float, float, float] | None:
    if len(polygon) != 4:
        return None
    x_values = sorted({round(point.x / tolerance) for point in polygon})
    y_values = sorted({round(point.y / tolerance) for point in polygon})
    if len(x_values) != 2 or len(y_values) != 2:
        return None
    return polygon_bounds(polygon)


def point_in_bounds(
    point: Point,
    bounds: tuple[float, float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return (
        min_x - tolerance <= point.x <= max_x + tolerance
        and min_y - tolerance <= point.y <= max_y + tolerance
    )


def segment_intersects_axis_aligned_rectangle(
    start: Point,
    end: Point,
    rectangle_bounds: tuple[float, float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    if not bounds_overlap(segment_bounds(start, end), rectangle_bounds, tolerance):
        return False
    if point_in_bounds(start, rectangle_bounds, tolerance) or point_in_bounds(
        end,
        rectangle_bounds,
        tolerance,
    ):
        return True

    min_x, min_y, max_x, max_y = rectangle_bounds
    dx = end.x - start.x
    dy = end.y - start.y
    t_min = 0.0
    t_max = 1.0

    for start_value, delta, lower, upper in (
        (start.x, dx, min_x, max_x),
        (start.y, dy, min_y, max_y),
    ):
        if abs(delta) <= tolerance:
            if start_value < lower - tolerance or start_value > upper + tolerance:
                return False
            continue
        inv_delta = 1.0 / delta
        t1 = (lower - start_value) * inv_delta
        t2 = (upper - start_value) * inv_delta
        t_enter = min(t1, t2)
        t_exit = max(t1, t2)
        t_min = max(t_min, t_enter)
        t_max = min(t_max, t_exit)
        if t_min > t_max + tolerance:
            return False
    return True


def segment_crosses_axis_aligned_rectangle_interior(
    start: Point,
    end: Point,
    rectangle_bounds: tuple[float, float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    if not bounds_overlap(segment_bounds(start, end), rectangle_bounds, tolerance):
        return False

    min_x, min_y, max_x, max_y = rectangle_bounds
    dx = end.x - start.x
    dy = end.y - start.y
    t_min = 0.0
    t_max = 1.0

    for start_value, delta, lower, upper in (
        (start.x, dx, min_x, max_x),
        (start.y, dy, min_y, max_y),
    ):
        if abs(delta) <= tolerance:
            if start_value <= lower + tolerance or start_value >= upper - tolerance:
                return False
            continue
        inv_delta = 1.0 / delta
        t1 = (lower - start_value) * inv_delta
        t2 = (upper - start_value) * inv_delta
        t_enter = min(t1, t2)
        t_exit = max(t1, t2)
        t_min = max(t_min, t_enter)
        t_max = min(t_max, t_exit)
        if t_min > t_max + tolerance:
            return False

    if t_max <= tolerance or t_min >= 1.0 - tolerance:
        return False

    sample_t = min(max(t_min + tolerance * 10.0, tolerance), 1.0 - tolerance)
    if sample_t > t_max - tolerance:
        sample_t = (max(t_min, 0.0) + min(t_max, 1.0)) / 2.0
    sample = Point(start.x + dx * sample_t, start.y + dy * sample_t)
    return (
        min_x + tolerance < sample.x < max_x - tolerance
        and min_y + tolerance < sample.y < max_y - tolerance
    )


def segment_intersects_polygon(start: Point, end: Point, polygon: tuple[Point, ...]) -> bool:
    segment_box = segment_bounds(start, end)
    polygon_box = polygon_bounds(polygon)
    if not bounds_overlap(segment_box, polygon_box):
        return False

    rectangle_bounds = axis_aligned_rectangle_bounds(polygon)
    if rectangle_bounds is not None:
        return segment_intersects_axis_aligned_rectangle(start, end, rectangle_bounds)

    if point_in_polygon(start, polygon) or point_in_polygon(end, polygon):
        return True
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        if segments_intersect(start, end, point, next_point):
            return True
    return False


def point_inside_panel_work_area(
    point: Point,
    panel_width: float,
    panel_height: float,
    trim_margin: float,
) -> bool:
    return (
        trim_margin <= point.x <= panel_width - trim_margin
        and trim_margin <= point.y <= panel_height - trim_margin
    )
