from cnc_cutting.collision import effective_centerline_margin
from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.layout_generator import (
    ClusteredLayoutConfig,
    SyntheticLayoutConfig,
    generate_clustered_channel_layout,
    generate_synthetic_layout,
    grid_dimensions,
)
from cnc_cutting.models import CuttingUnitType, ToolConfig


def test_grid_dimensions_for_positive_count() -> None:
    assert grid_dimensions(10, columns=4) == (4, 3)


def test_grid_dimensions_rejects_non_positive_count() -> None:
    try:
        grid_dimensions(0)
    except ValueError as exc:
        assert "rectangle_count" in str(exc)
    else:
        raise AssertionError("grid_dimensions should reject non-positive counts")


def test_generate_synthetic_layout_is_deterministic() -> None:
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    config = SyntheticLayoutConfig(seed=7)

    first = generate_synthetic_layout(12, tool, config)
    second = generate_synthetic_layout(12, tool, config)

    assert first == second


def test_generate_synthetic_layout_stays_inside_effective_work_area() -> None:
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    margin = effective_centerline_margin(tool)

    layout = generate_synthetic_layout(25, tool, SyntheticLayoutConfig(seed=11))

    assert len(layout.rectangles) == 25
    for rectangle in layout.rectangles:
        assert rectangle.min_x >= margin
        assert rectangle.min_y >= margin
        assert rectangle.max_x <= layout.panel_width - margin
        assert rectangle.max_y <= layout.panel_height - margin


def test_generate_clustered_channel_layout_is_deterministic() -> None:
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    config = ClusteredLayoutConfig(seed=19, columns=3)

    first = generate_clustered_channel_layout(9, tool, config)
    second = generate_clustered_channel_layout(9, tool, config)

    assert first == second


def test_generate_clustered_channel_layout_creates_near_shared_channels() -> None:
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    layout = generate_clustered_channel_layout(
        6,
        tool,
        ClusteredLayoutConfig(seed=23, columns=3),
    )

    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )

    assert len(layout.rectangles) == 6
    assert (
        len(
            [
                unit
                for unit in units
                if unit.unit_type == CuttingUnitType.NEAR_SHARED_CHANNEL
            ]
        )
        >= 4
    )


def test_generate_clustered_channel_layout_stays_inside_effective_work_area() -> None:
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    margin = effective_centerline_margin(tool)

    layout = generate_clustered_channel_layout(
        20,
        tool,
        ClusteredLayoutConfig(seed=29, columns=5),
    )

    for rectangle in layout.rectangles:
        assert rectangle.min_x >= margin
        assert rectangle.min_y >= margin
        assert rectangle.max_x <= layout.panel_width - margin
        assert rectangle.max_y <= layout.panel_height - margin
