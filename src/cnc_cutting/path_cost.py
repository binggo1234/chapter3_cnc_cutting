from __future__ import annotations

from .geometry import euclidean_distance, rectangle_corners
from .models import CuttingConfig, PlacedRectangle, Point


def entry_point(rectangle: PlacedRectangle) -> Point:
    """Use the lower-left corner as the default entry point."""

    return rectangle_corners(rectangle)[0]


def total_air_move_distance(
    rectangles: tuple[PlacedRectangle, ...],
    order: tuple[int, ...],
    config: CuttingConfig | None = None,
) -> float:
    """Estimate non-cutting travel distance for a given rectangle visit order."""

    if config is None:
        config = CuttingConfig()

    if not order:
        return 0.0

    total = 0.0
    current = config.start_point

    for index in order:
        target = entry_point(rectangles[index])
        total += euclidean_distance(current, target)
        current = target

    if config.return_to_start:
        total += euclidean_distance(current, config.start_point)

    return total


def nearest_neighbor_order(
    rectangles: tuple[PlacedRectangle, ...],
    start_point: Point = Point(0.0, 0.0),
) -> tuple[int, ...]:
    """Build a simple nearest-neighbor baseline order using entry points."""

    remaining = set(range(len(rectangles)))
    order: list[int] = []
    current = start_point

    while remaining:
        next_index = min(
            remaining,
            key=lambda index: euclidean_distance(current, entry_point(rectangles[index])),
        )
        order.append(next_index)
        current = entry_point(rectangles[next_index])
        remaining.remove(next_index)

    return tuple(order)

