from __future__ import annotations

from .geometry import (
    bounds_overlap,
    point_inside_panel_work_area,
    segment_bounds,
    segment_crosses_axis_aligned_rectangle_interior,
    segment_intersects_polygon,
)
from .models import CuttingAction, CuttingActionType, Panel, Point, ToolConfig, TravelMode


def effective_centerline_margin(tool: ToolConfig) -> float:
    if tool.centerline_boundary_margin is not None:
        return tool.centerline_boundary_margin
    return tool.trim_margin + tool.tool_radius


def action_within_panel_work_area(action: CuttingAction, panel: Panel, tool: ToolConfig) -> bool:
    margin = effective_centerline_margin(tool)
    return point_inside_panel_work_area(
        action.start,
        panel.panel_width,
        panel.panel_height,
        margin,
    ) and point_inside_panel_work_area(
        action.end,
        panel.panel_width,
        panel.panel_height,
        margin,
    )


def action_boundary_penalty(action: CuttingAction, panel: Panel, tool: ToolConfig) -> float:
    if action_within_panel_work_area(action, panel, tool):
        return 0.0
    return 1.0


def action_low_clearance_collision_penalty(
    action: CuttingAction,
    processed_polygons: tuple[tuple[Point, ...], ...] = (),
    processed_polygon_bounds: tuple[tuple[float, float, float, float], ...] = (),
) -> float:
    if action.action_type != CuttingActionType.TRAVEL:
        return 0.0
    action_bounds = segment_bounds(action.start, action.end)
    if processed_polygon_bounds and len(processed_polygon_bounds) == len(processed_polygons):
        for polygon, polygon_bounds in zip(processed_polygons, processed_polygon_bounds):
            if not bounds_overlap(action_bounds, polygon_bounds):
                continue
            if len(polygon) == 4:
                if segment_crosses_axis_aligned_rectangle_interior(
                    action.start,
                    action.end,
                    polygon_bounds,
                ):
                    return 1.0
                continue
            if segment_intersects_polygon(action.start, action.end, polygon):
                return 1.0
        return 0.0

    return float(
        any(
            (
                segment_crosses_axis_aligned_rectangle_interior(
                    action.start,
                    action.end,
                    segment_bounds(polygon[0], polygon[2]),
                )
                if len(polygon) == 4
                else segment_intersects_polygon(action.start, action.end, polygon)
            )
            for polygon in processed_polygons
        )
    )


def action_collision_penalty(
    action: CuttingAction,
    processed_polygons: tuple[tuple[Point, ...], ...] = (),
    processed_polygon_bounds: tuple[tuple[float, float, float, float], ...] = (),
) -> float:
    if action.action_type == CuttingActionType.TRAVEL and action.travel_mode == TravelMode.SAFE_LIFT:
        return 0.0
    return action_low_clearance_collision_penalty(
        action,
        processed_polygons,
        processed_polygon_bounds,
    )
