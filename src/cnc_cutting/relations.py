from __future__ import annotations

from .geometry import axial_gap, collinear, parallel_overlap_length, perpendicular_gap
from .models import (
    BoundaryRelation,
    BoundaryRelationType,
    EdgeOrientation,
    EdgeRole,
    EdgeSegment,
)


def complementary_roles(first: EdgeSegment, second: EdgeSegment) -> bool:
    return frozenset((first.role, second.role)) in {
        frozenset((EdgeRole.LEFT, EdgeRole.RIGHT)),
        frozenset((EdgeRole.TOP, EdgeRole.BOTTOM)),
    }


def fixed_coordinate(segment: EdgeSegment) -> float:
    if segment.orientation == EdgeOrientation.HORIZONTAL:
        return segment.start.y
    return segment.start.x


def possibly_related(
    first: EdgeSegment,
    second: EdgeSegment,
    min_channel_width: float,
    max_collinear_gap: float,
    tolerance: float,
) -> bool:
    if first.part_id == second.part_id or first.orientation != second.orientation:
        return False

    fixed_gap = abs(fixed_coordinate(first) - fixed_coordinate(second))
    same_line = fixed_gap <= tolerance
    near_line = abs(fixed_gap - min_channel_width) <= tolerance
    if not same_line and not near_line:
        return False

    overlap = parallel_overlap_length(first, second)
    if near_line:
        return overlap > tolerance and complementary_roles(first, second)

    if overlap > tolerance and complementary_roles(first, second):
        return True
    return axial_gap(first, second) <= max_collinear_gap


def candidate_pairs(
    segments: tuple[EdgeSegment, ...],
    min_channel_width: float,
    max_collinear_gap: float,
    tolerance: float,
) -> tuple[tuple[EdgeSegment, EdgeSegment], ...]:
    pairs: list[tuple[EdgeSegment, EdgeSegment]] = []
    max_fixed_gap = min_channel_width + tolerance
    for orientation in (EdgeOrientation.HORIZONTAL, EdgeOrientation.VERTICAL):
        oriented = sorted(
            [segment for segment in segments if segment.orientation == orientation],
            key=lambda segment: (fixed_coordinate(segment), segment.segment_id),
        )
        for first_index, first in enumerate(oriented):
            first_fixed = fixed_coordinate(first)
            for second in oriented[first_index + 1 :]:
                fixed_gap = fixed_coordinate(second) - first_fixed
                if fixed_gap > max_fixed_gap:
                    break
                if possibly_related(
                    first,
                    second,
                    min_channel_width=min_channel_width,
                    max_collinear_gap=max_collinear_gap,
                    tolerance=tolerance,
                ):
                    pairs.append((first, second))
    return tuple(pairs)


def classify_boundary_relation(
    first: EdgeSegment,
    second: EdgeSegment,
    min_channel_width: float,
    max_collinear_gap: float = float("inf"),
    tolerance: float = 1e-6,
) -> BoundaryRelation | None:
    if not possibly_related(
        first,
        second,
        min_channel_width=min_channel_width,
        max_collinear_gap=max_collinear_gap,
        tolerance=tolerance,
    ):
        return None

    overlap = parallel_overlap_length(first, second)
    perp_gap = perpendicular_gap(first, second)
    axis_gap = axial_gap(first, second)

    if perp_gap <= tolerance and overlap > tolerance and complementary_roles(first, second):
        relation_type = BoundaryRelationType.SHARED_EDGE
    elif (
        abs(perp_gap - min_channel_width) <= tolerance
        and overlap > tolerance
        and complementary_roles(first, second)
    ):
        relation_type = BoundaryRelationType.NEAR_SHARED_EDGE
    elif collinear(first, second, tolerance) and axis_gap <= max_collinear_gap:
        relation_type = BoundaryRelationType.COLLINEAR_EDGE
    else:
        return None

    return BoundaryRelation(
        relation_type=relation_type,
        first=first,
        second=second,
        overlap_length=overlap,
        gap=perp_gap,
        axial_gap=axis_gap,
        perpendicular_gap=perp_gap,
    )


def detect_boundary_relations(
    segments: tuple[EdgeSegment, ...],
    min_channel_width: float,
    max_collinear_gap: float = float("inf"),
    tolerance: float = 1e-6,
) -> tuple[BoundaryRelation, ...]:
    relations: list[BoundaryRelation] = []
    for first, second in candidate_pairs(
        segments,
        min_channel_width=min_channel_width,
        max_collinear_gap=max_collinear_gap,
        tolerance=tolerance,
    ):
        relation = classify_boundary_relation(
            first,
            second,
            min_channel_width=min_channel_width,
            max_collinear_gap=max_collinear_gap,
            tolerance=tolerance,
        )
        if relation is not None:
            relations.append(relation)
    return tuple(relations)
