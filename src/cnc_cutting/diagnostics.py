from __future__ import annotations

from dataclasses import dataclass

from .collision import (
    action_boundary_penalty,
    action_collision_penalty,
    action_low_clearance_collision_penalty,
)
from .geometry import euclidean_distance
from .metrics import apply_action_incremental, classify_action_clearance
from .models import (
    CuttingAction,
    CuttingProcessModel,
    IncrementalMetricsState,
    Panel,
    ToolConfig,
)


@dataclass(frozen=True)
class ActionDiagnostic:
    action_index: int
    action_type: str
    travel_mode: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    length: float
    cutting_unit_id: str | None
    segment_id: str | None
    low_clearance_collision_penalty: float
    collision_penalty: float
    boundary_penalty: float
    safe_lift_count: int
    safe_lift_distance: float
    detour_count: int
    detour_distance: float
    travel_mode_cost: float
    stability_penalty_delta: float
    released_parts_before: int
    released_parts_after: int
    unstable_parts_before: int
    unstable_parts_after: int
    unstable_part_ids_before: str
    unstable_part_ids_after: str
    processed_segments_before: int
    processed_segments_after: int

    @property
    def hard_penalty(self) -> float:
        return self.collision_penalty + self.boundary_penalty


def diagnose_actions(
    actions: tuple[CuttingAction, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> tuple[ActionDiagnostic, ...]:
    state = IncrementalMetricsState(current_point=tool.start_point)
    diagnostics: list[ActionDiagnostic] = []

    for index, action in enumerate(actions):
        effective_action = classify_action_clearance(action, state, tool)
        low_clearance_collision_penalty = action_low_clearance_collision_penalty(
            effective_action,
            state.processed_polygons,
            state.processed_polygon_bounds,
        )
        collision_penalty = action_collision_penalty(
            effective_action,
            state.processed_polygons,
            state.processed_polygon_bounds,
        )
        boundary_penalty = action_boundary_penalty(effective_action, panel, tool)
        released_before = len(state.released_part_ids)
        unstable_before = set(state.unstable_part_ids)
        processed_before = len(state.processed_segments)
        next_state = apply_action_incremental(
            effective_action,
            state,
            panel,
            tool,
            process_model=process_model,
        )

        diagnostics.append(
            ActionDiagnostic(
                action_index=index,
                action_type=effective_action.action_type.value,
                travel_mode=effective_action.travel_mode.value,
                start_x=effective_action.start.x,
                start_y=effective_action.start.y,
                end_x=effective_action.end.x,
                end_y=effective_action.end.y,
                length=euclidean_distance(effective_action.start, effective_action.end),
                cutting_unit_id=effective_action.cutting_unit_id,
                segment_id=effective_action.segment_id,
                low_clearance_collision_penalty=low_clearance_collision_penalty,
                collision_penalty=collision_penalty,
                boundary_penalty=boundary_penalty,
                safe_lift_count=next_state.metrics.safe_lift_count
                - state.metrics.safe_lift_count,
                safe_lift_distance=next_state.metrics.safe_lift_distance
                - state.metrics.safe_lift_distance,
                detour_count=next_state.metrics.detour_count - state.metrics.detour_count,
                detour_distance=next_state.metrics.detour_distance
                - state.metrics.detour_distance,
                travel_mode_cost=next_state.metrics.travel_mode_cost
                - state.metrics.travel_mode_cost,
                stability_penalty_delta=next_state.metrics.stability_penalty
                - state.metrics.stability_penalty,
                released_parts_before=released_before,
                released_parts_after=len(next_state.released_part_ids),
                unstable_parts_before=len(unstable_before),
                unstable_parts_after=len(next_state.unstable_part_ids),
                unstable_part_ids_before=";".join(sorted(unstable_before)),
                unstable_part_ids_after=";".join(sorted(next_state.unstable_part_ids)),
                processed_segments_before=processed_before,
                processed_segments_after=len(next_state.processed_segments),
            )
        )
        state = next_state

    return tuple(diagnostics)
