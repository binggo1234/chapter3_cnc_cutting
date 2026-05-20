from __future__ import annotations

from .geometry import rectangle_edges
from .models import (
    BoundaryRelation,
    BoundaryRelationType,
    CuttingUnit,
    CuttingUnitType,
    EdgeOrientation,
    EdgeSegment,
    Layout,
    Point,
    ToolConfig,
)
from .relations import detect_boundary_relations


def _axis_bounds(segment: EdgeSegment) -> tuple[float, float]:
    if segment.orientation == EdgeOrientation.HORIZONTAL:
        return (min(segment.start.x, segment.end.x), max(segment.start.x, segment.end.x))
    return (min(segment.start.y, segment.end.y), max(segment.start.y, segment.end.y))


def _fixed_coordinate(segment: EdgeSegment) -> float:
    if segment.orientation == EdgeOrientation.HORIZONTAL:
        return segment.start.y
    return segment.start.x


def _point_from_axis(
    orientation: EdgeOrientation,
    fixed_coordinate: float,
    axis_coordinate: float,
) -> Point:
    if orientation == EdgeOrientation.HORIZONTAL:
        return Point(axis_coordinate, fixed_coordinate)
    return Point(fixed_coordinate, axis_coordinate)


def _overlap_bounds(first: EdgeSegment, second: EdgeSegment) -> tuple[float, float]:
    first_lo, first_hi = _axis_bounds(first)
    second_lo, second_hi = _axis_bounds(second)
    return (max(first_lo, second_lo), min(first_hi, second_hi))


def _all_bounds(segments: tuple[EdgeSegment, ...]) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for segment in segments:
        lo, hi = _axis_bounds(segment)
        lows.append(lo)
        highs.append(hi)
    return (min(lows), max(highs))


def _covered_segment_ids(segments: tuple[EdgeSegment, ...]) -> tuple[str, ...]:
    return tuple(segment.segment_id for segment in segments)


def primitive_cutting_units(segments: tuple[EdgeSegment, ...]) -> tuple[CuttingUnit, ...]:
    return tuple(
        CuttingUnit(
            unit_id=f"single:{segment.segment_id}",
            unit_type=CuttingUnitType.SINGLE_EDGE,
            segments=(segment,),
            start=segment.start,
            end=segment.end,
            covered_segment_ids=(segment.segment_id,),
        )
        for segment in segments
    )


def relation_cutting_unit(
    relation: BoundaryRelation,
    tolerance: float = 1e-6,
) -> CuttingUnit | None:
    first = relation.first
    second = relation.second
    segments = tuple(
        sorted(
            (first, second),
            key=lambda segment: (_axis_bounds(segment)[0], segment.segment_id),
        )
    )
    relation_key = f"{relation.relation_type.value}:{first.segment_id}|{second.segment_id}"

    if relation.relation_type == BoundaryRelationType.SHARED_EDGE:
        lo, hi = _overlap_bounds(first, second)
        if hi - lo <= tolerance:
            return None
        fixed_coordinate = _fixed_coordinate(first)
        unit_type = CuttingUnitType.SHARED_EDGE
        requires_bridge_cut = False
    elif relation.relation_type == BoundaryRelationType.NEAR_SHARED_EDGE:
        lo, hi = _overlap_bounds(first, second)
        if hi - lo <= tolerance:
            return None
        fixed_coordinate = (_fixed_coordinate(first) + _fixed_coordinate(second)) / 2.0
        unit_type = CuttingUnitType.NEAR_SHARED_CHANNEL
        requires_bridge_cut = False
    elif relation.relation_type == BoundaryRelationType.COLLINEAR_EDGE:
        lo, hi = _all_bounds(segments)
        fixed_coordinate = _fixed_coordinate(first)
        unit_type = CuttingUnitType.COLLINEAR_CHAIN
        requires_bridge_cut = relation.axial_gap > tolerance
    else:
        return None

    return CuttingUnit(
        unit_id=relation_key,
        unit_type=unit_type,
        segments=segments,
        start=_point_from_axis(first.orientation, fixed_coordinate, lo),
        end=_point_from_axis(first.orientation, fixed_coordinate, hi),
        relation_types=(relation.relation_type,),
        covered_segment_ids=_covered_segment_ids(segments),
        requires_bridge_cut=requires_bridge_cut,
    )


def relation_cutting_units(
    relations: tuple[BoundaryRelation, ...],
    tolerance: float = 1e-6,
) -> tuple[CuttingUnit, ...]:
    units: list[CuttingUnit] = []
    seen_unit_ids: set[str] = set()
    for relation in relations:
        unit = relation_cutting_unit(relation, tolerance=tolerance)
        if unit is None or unit.unit_id in seen_unit_ids:
            continue
        units.append(unit)
        seen_unit_ids.add(unit.unit_id)
    return tuple(units)


def build_candidate_cutting_units(
    layout: Layout,
    tool: ToolConfig,
    max_collinear_gap: float = float("inf"),
    tolerance: float = 1e-6,
) -> tuple[CuttingUnit, ...]:
    segments = tuple(
        segment
        for rectangle in layout.rectangles
        for segment in rectangle_edges(rectangle)
    )
    relations = detect_boundary_relations(
        segments,
        min_channel_width=tool.min_channel_width,
        max_collinear_gap=max_collinear_gap,
        tolerance=tolerance,
    )
    return primitive_cutting_units(segments) + relation_cutting_units(
        relations,
        tolerance=tolerance,
    )
