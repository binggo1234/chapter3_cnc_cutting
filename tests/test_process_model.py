from cnc_cutting.models import EdgeRole, Layout, PlacedRectangle, Point
from cnc_cutting.process_model import build_process_model


def test_build_process_model_maps_segments_to_parts_and_polygons() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )

    process_model = build_process_model(layout, support_edge_role=EdgeRole.TOP)

    assert process_model.part_segment_ids["A"] == frozenset(
        {"A:bottom", "A:right", "A:top", "A:left"}
    )
    assert process_model.segment_part_ids["A:right"] == "A"
    assert process_model.part_polygons["A"] == (
        Point(10, 10),
        Point(30, 10),
        Point(30, 20),
        Point(10, 20),
    )
    assert process_model.support_segment_ids["A"] == frozenset({"A:top"})
    assert process_model.segment_lengths["A:top"] == 20
    assert process_model.part_areas["A"] == 200
    assert process_model.part_support_lengths["A"] == 20


def test_build_process_model_supports_multiple_stability_edges() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 10, 10, 20, 10),),
    )

    process_model = build_process_model(
        layout,
        support_edge_roles=(EdgeRole.BOTTOM, EdgeRole.TOP),
        min_remaining_support_length_ratio=0.5,
    )

    assert process_model.support_segment_ids["A"] == frozenset(
        {"A:bottom", "A:top"}
    )
    assert process_model.part_support_lengths["A"] == 40
    assert process_model.min_remaining_support_length_ratio == 0.5


def test_build_process_model_records_shared_edge_adjacency_support() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 10, 10, 20, 10),
            PlacedRectangle("B", 30, 10, 20, 10),
        ),
    )

    process_model = build_process_model(layout, adjacency_support_weight=1.0)

    assert process_model.part_adjacency_support_lengths["A"]["B"] == 10
    assert process_model.part_adjacency_support_lengths["B"]["A"] == 10
    assert process_model.adjacency_support_weight == 1.0
