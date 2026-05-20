from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from random import Random

from .collision import effective_centerline_margin
from .models import Layout, PlacedRectangle, ToolConfig


@dataclass(frozen=True)
class SyntheticLayoutConfig:
    seed: int = 42
    cell_width: float = 140.0
    cell_height: float = 100.0
    min_part_width: float = 55.0
    max_part_width: float = 105.0
    min_part_height: float = 35.0
    max_part_height: float = 75.0
    columns: int | None = None
    panel_id_prefix: str = "synthetic"


@dataclass(frozen=True)
class ClusteredLayoutConfig:
    seed: int = 42
    min_part_width: float = 55.0
    max_part_width: float = 105.0
    min_part_height: float = 35.0
    max_part_height: float = 75.0
    row_gap: float | None = None
    columns: int | None = None
    panel_id_prefix: str = "clustered"


def grid_dimensions(rectangle_count: int, columns: int | None = None) -> tuple[int, int]:
    if rectangle_count <= 0:
        raise ValueError("rectangle_count must be positive")
    if columns is None:
        columns = max(1, ceil(sqrt(rectangle_count * 2.0)))
    rows = ceil(rectangle_count / columns)
    return columns, rows


def generate_synthetic_layout(
    rectangle_count: int,
    tool: ToolConfig,
    config: SyntheticLayoutConfig | None = None,
) -> Layout:
    if config is None:
        config = SyntheticLayoutConfig()
    rng = Random(config.seed + rectangle_count)
    columns, rows = grid_dimensions(rectangle_count, config.columns)
    margin = effective_centerline_margin(tool)
    panel_width = 2.0 * margin + columns * config.cell_width
    panel_height = 2.0 * margin + rows * config.cell_height

    rectangles: list[PlacedRectangle] = []
    for index in range(rectangle_count):
        row = index // columns
        column = index % columns
        width = rng.uniform(config.min_part_width, config.max_part_width)
        height = rng.uniform(config.min_part_height, config.max_part_height)
        max_offset_x = max(0.0, config.cell_width - width)
        max_offset_y = max(0.0, config.cell_height - height)
        x = margin + column * config.cell_width + rng.uniform(0.0, max_offset_x)
        y = margin + row * config.cell_height + rng.uniform(0.0, max_offset_y)
        rectangles.append(
            PlacedRectangle(
                part_id=f"P{index + 1:04d}",
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )

    return Layout(
        panel_id=f"{config.panel_id_prefix}_{rectangle_count}",
        panel_width=panel_width,
        panel_height=panel_height,
        rectangles=tuple(rectangles),
    )


def generate_clustered_channel_layout(
    rectangle_count: int,
    tool: ToolConfig,
    config: ClusteredLayoutConfig | None = None,
) -> Layout:
    if config is None:
        config = ClusteredLayoutConfig()
    rng = Random(config.seed + rectangle_count)
    columns, rows = grid_dimensions(rectangle_count, config.columns)
    margin = effective_centerline_margin(tool)
    row_gap = tool.min_channel_width if config.row_gap is None else config.row_gap

    rectangles: list[PlacedRectangle] = []
    max_x = margin
    y = margin
    part_index = 0

    for row in range(rows):
        row_height = rng.uniform(config.min_part_height, config.max_part_height)
        x = margin
        for _column in range(columns):
            if part_index >= rectangle_count:
                break
            width = rng.uniform(config.min_part_width, config.max_part_width)
            rectangles.append(
                PlacedRectangle(
                    part_id=f"P{part_index + 1:04d}",
                    x=x,
                    y=y,
                    width=width,
                    height=row_height,
                )
            )
            x += width + tool.min_channel_width
            part_index += 1
        if rectangles:
            max_x = max(max_x, max(rectangle.max_x for rectangle in rectangles))
        y += row_height + row_gap

    panel_width = max_x + margin
    panel_height = max(rectangle.max_y for rectangle in rectangles) + margin

    return Layout(
        panel_id=f"{config.panel_id_prefix}_{rectangle_count}",
        panel_width=panel_width,
        panel_height=panel_height,
        rectangles=tuple(rectangles),
    )
