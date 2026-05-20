from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.models import (
    CuttingUnitType,
    Layout,
    PlacedRectangle,
    Point,
    ToolConfig,
)


def test_build_candidate_units_keeps_primitive_edges() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(PlacedRectangle("A", 0, 0, 10, 10),),
    )

    units = build_candidate_cutting_units(layout, ToolConfig(tool_diameter=6))

    assert len([unit for unit in units if unit.unit_type == CuttingUnitType.SINGLE_EDGE]) == 4


def test_build_shared_edge_candidate_unit() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 0, 0, 10, 10),
            PlacedRectangle("B", 10, 0, 10, 10),
        ),
    )

    units = build_candidate_cutting_units(layout, ToolConfig(tool_diameter=6))

    shared_units = [unit for unit in units if unit.unit_type == CuttingUnitType.SHARED_EDGE]
    assert any(
        unit.covered_segment_ids == ("A:right", "B:left")
        and unit.start == Point(10, 0)
        and unit.end == Point(10, 10)
        for unit in shared_units
    )


def test_build_near_shared_channel_candidate_unit() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 0, 0, 10, 10),
            PlacedRectangle("B", 16, 0, 10, 10),
        ),
    )

    units = build_candidate_cutting_units(layout, ToolConfig(tool_diameter=6))

    channel_units = [
        unit for unit in units if unit.unit_type == CuttingUnitType.NEAR_SHARED_CHANNEL
    ]
    assert any(
        unit.covered_segment_ids == ("A:right", "B:left")
        and unit.start == Point(13, 0)
        and unit.end == Point(13, 10)
        for unit in channel_units
    )


def test_build_collinear_chain_marks_internal_bridge_cut() -> None:
    layout = Layout(
        panel_id="P",
        panel_width=100,
        panel_height=100,
        rectangles=(
            PlacedRectangle("A", 0, 0, 10, 10),
            PlacedRectangle("B", 15, 0, 10, 10),
        ),
    )

    units = build_candidate_cutting_units(
        layout,
        ToolConfig(tool_diameter=6),
        max_collinear_gap=10,
    )

    assert any(
        unit.unit_type == CuttingUnitType.COLLINEAR_CHAIN
        and unit.covered_segment_ids == ("A:bottom", "B:bottom")
        and unit.start == Point(0, 0)
        and unit.end == Point(25, 0)
        and unit.requires_bridge_cut
        for unit in units
    )
