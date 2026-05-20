from __future__ import annotations

from collections.abc import Iterable

from .geometry import rectangle_corners, rectangle_edges
from .models import (
    BoundaryRelationType,
    CuttingProcessModel,
    EdgeRole,
    Layout,
)
from .relations import detect_boundary_relations


def build_process_model(
    layout: Layout,
    support_edge_role: EdgeRole | None = EdgeRole.TOP,
    support_edge_roles: Iterable[EdgeRole] | None = None,
    min_remaining_support_count: int = 1,
    min_remaining_support_length_ratio: float = 0.0,
    min_area_normalized_support_length: float = 0.0,
    adjacency_support_weight: float = 0.0,
) -> CuttingProcessModel:
    if support_edge_roles is None:
        support_roles = () if support_edge_role is None else (support_edge_role,)
    else:
        support_roles = tuple(dict.fromkeys(support_edge_roles))

    part_segment_ids: dict[str, frozenset[str]] = {}
    segment_part_ids: dict[str, str] = {}
    part_polygons = {
        rectangle.part_id: rectangle_corners(rectangle)
        for rectangle in layout.rectangles
    }
    support_segment_ids: dict[str, frozenset[str]] = {}
    segment_lengths: dict[str, float] = {}
    part_areas: dict[str, float] = {}
    part_support_lengths: dict[str, float] = {}
    all_edges = []

    for rectangle in layout.rectangles:
        edges = rectangle_edges(rectangle)
        all_edges.extend(edges)
        part_segment_ids[rectangle.part_id] = frozenset(
            edge.segment_id for edge in edges
        )
        part_areas[rectangle.part_id] = rectangle.width * rectangle.height
        for edge in edges:
            segment_part_ids[edge.segment_id] = rectangle.part_id
            segment_lengths[edge.segment_id] = edge.length
        if support_roles:
            support_ids = frozenset(
                edge.segment_id
                for edge in edges
                if edge.role in support_roles
            )
            support_segment_ids[rectangle.part_id] = support_ids
            part_support_lengths[rectangle.part_id] = sum(
                segment_lengths[segment_id] for segment_id in support_ids
            )

    adjacency_support_lengths = build_adjacency_support_lengths(
        tuple(all_edges),
        min_channel_width=0.0,
    )

    return CuttingProcessModel(
        part_segment_ids=part_segment_ids,
        segment_part_ids=segment_part_ids,
        part_polygons=part_polygons,
        support_segment_ids=support_segment_ids,
        segment_lengths=segment_lengths,
        part_areas=part_areas,
        part_support_lengths=part_support_lengths,
        part_adjacency_support_lengths=adjacency_support_lengths,
        adjacency_support_weight=adjacency_support_weight,
        min_remaining_support_count=min_remaining_support_count,
        min_remaining_support_length_ratio=min_remaining_support_length_ratio,
        min_area_normalized_support_length=min_area_normalized_support_length,
    )


def build_adjacency_support_lengths(
    edges,
    min_channel_width: float = 0.0,
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = {}
    relations = detect_boundary_relations(
        edges,
        min_channel_width=min_channel_width,
        max_collinear_gap=0.0,
    )
    for relation in relations:
        if relation.relation_type != BoundaryRelationType.SHARED_EDGE:
            continue
        first_part = relation.first.part_id
        second_part = relation.second.part_id
        if first_part == second_part:
            continue
        adjacency.setdefault(first_part, {})[second_part] = (
            adjacency.setdefault(first_part, {}).get(second_part, 0.0)
            + relation.overlap_length
        )
        adjacency.setdefault(second_part, {})[first_part] = (
            adjacency.setdefault(second_part, {}).get(first_part, 0.0)
            + relation.overlap_length
        )
    return adjacency
