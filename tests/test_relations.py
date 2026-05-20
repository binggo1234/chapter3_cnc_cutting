from cnc_cutting.geometry import rectangle_edges
from cnc_cutting.models import BoundaryRelationType, PlacedRectangle, ToolConfig
from cnc_cutting.relations import candidate_pairs, detect_boundary_relations


def test_detect_near_shared_edge_when_gap_matches_tool_channel() -> None:
    tool = ToolConfig(tool_diameter=6.0)
    first = PlacedRectangle("A", 0, 0, 10, 10)
    second = PlacedRectangle("B", 16, 0, 10, 10)

    relations = detect_boundary_relations(
        rectangle_edges(first) + rectangle_edges(second),
        min_channel_width=tool.min_channel_width,
    )

    assert any(
        relation.relation_type == BoundaryRelationType.NEAR_SHARED_EDGE
        and relation.overlap_length == 10
        for relation in relations
    )


def test_near_shared_edge_requires_facing_roles() -> None:
    tool = ToolConfig(tool_diameter=6.0)
    first = PlacedRectangle("A", 0, 0, 10, 10)
    second = PlacedRectangle("B", 0, 6, 10, 10)

    relations = detect_boundary_relations(
        rectangle_edges(first) + rectangle_edges(second),
        min_channel_width=tool.min_channel_width,
    )

    assert not any(
        relation.relation_type == BoundaryRelationType.NEAR_SHARED_EDGE
        and relation.first.role == relation.second.role
        for relation in relations
    )


def test_detect_shared_edge_between_complementary_roles() -> None:
    first = PlacedRectangle("A", 0, 0, 10, 10)
    second = PlacedRectangle("B", 10, 0, 10, 10)

    relations = detect_boundary_relations(
        rectangle_edges(first) + rectangle_edges(second),
        min_channel_width=6.0,
    )

    assert any(
        relation.relation_type == BoundaryRelationType.SHARED_EDGE
        and relation.first.part_id != relation.second.part_id
        and relation.overlap_length == 10
        for relation in relations
    )


def test_detect_collinear_edge_with_axial_gap() -> None:
    first = PlacedRectangle("A", 0, 0, 10, 10)
    second = PlacedRectangle("B", 15, 0, 10, 10)

    relations = detect_boundary_relations(
        rectangle_edges(first) + rectangle_edges(second),
        min_channel_width=6.0,
        max_collinear_gap=10.0,
    )

    assert any(
        relation.relation_type == BoundaryRelationType.COLLINEAR_EDGE
        and relation.axial_gap == 5
        and relation.perpendicular_gap == 0
        for relation in relations
    )


def test_candidate_pairs_prunes_unrelated_far_edges() -> None:
    first = PlacedRectangle("A", 0, 0, 10, 10)
    second = PlacedRectangle("B", 100, 100, 10, 10)

    pairs = candidate_pairs(
        rectangle_edges(first) + rectangle_edges(second),
        min_channel_width=6.0,
        max_collinear_gap=6.0,
        tolerance=1e-6,
    )

    assert pairs == ()
