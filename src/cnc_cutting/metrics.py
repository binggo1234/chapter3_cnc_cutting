from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

from .collision import (
    action_boundary_penalty,
    action_collision_penalty,
    action_low_clearance_collision_penalty,
)
from .geometry import direction_vector, euclidean_distance, polygon_bounds
from .models import (
    CuttingAction,
    CuttingActionType,
    CuttingProcessModel,
    IncrementalMetricsState,
    Panel,
    PathMetrics,
    ToolConfig,
    TravelMode,
)
from .travel import plan_travel_actions


@dataclass(frozen=True)
class MetricsComparatorConfig:
    hard_penalty_epsilon: float = 1e-9
    tool_event_epsilon: int = 0
    distance_epsilon: float = 1e-6
    penalty_epsilon: float = 1e-6


def add_metrics(base: PathMetrics, delta: PathMetrics) -> PathMetrics:
    return PathMetrics(
        air_move_distance=base.air_move_distance + delta.air_move_distance,
        cutting_length=base.cutting_length + delta.cutting_length,
        pierce_count=base.pierce_count + delta.pierce_count,
        lift_count=base.lift_count + delta.lift_count,
        turn_penalty=base.turn_penalty + delta.turn_penalty,
        collision_penalty=base.collision_penalty + delta.collision_penalty,
        boundary_penalty=base.boundary_penalty + delta.boundary_penalty,
        stability_penalty=base.stability_penalty + delta.stability_penalty,
        continuity_reward=base.continuity_reward + delta.continuity_reward,
        safe_lift_count=base.safe_lift_count + delta.safe_lift_count,
        safe_lift_distance=base.safe_lift_distance + delta.safe_lift_distance,
        detour_count=base.detour_count + delta.detour_count,
        detour_distance=base.detour_distance + delta.detour_distance,
        travel_mode_cost=base.travel_mode_cost + delta.travel_mode_cost,
    )


def subtract_metrics(total: PathMetrics, base: PathMetrics) -> PathMetrics:
    return PathMetrics(
        air_move_distance=total.air_move_distance - base.air_move_distance,
        cutting_length=total.cutting_length - base.cutting_length,
        pierce_count=total.pierce_count - base.pierce_count,
        lift_count=total.lift_count - base.lift_count,
        turn_penalty=total.turn_penalty - base.turn_penalty,
        collision_penalty=total.collision_penalty - base.collision_penalty,
        boundary_penalty=total.boundary_penalty - base.boundary_penalty,
        stability_penalty=total.stability_penalty - base.stability_penalty,
        continuity_reward=total.continuity_reward - base.continuity_reward,
        safe_lift_count=total.safe_lift_count - base.safe_lift_count,
        safe_lift_distance=total.safe_lift_distance - base.safe_lift_distance,
        detour_count=total.detour_count - base.detour_count,
        detour_distance=total.detour_distance - base.detour_distance,
        travel_mode_cost=total.travel_mode_cost - base.travel_mode_cost,
    )


def turn_penalty(
    previous_direction: tuple[float, float] | None,
    next_direction: tuple[float, float],
) -> float:
    if previous_direction is None:
        return 0.0
    return 1.0 - (
        previous_direction[0] * next_direction[0]
        + previous_direction[1] * next_direction[1]
    )


def action_covered_segment_ids(action: CuttingAction) -> tuple[str, ...]:
    if action.covered_segment_ids:
        return action.covered_segment_ids
    if action.segment_id is not None:
        return (action.segment_id,)
    return ()


def action_part_ids(
    action: CuttingAction,
    process_model: CuttingProcessModel | None,
) -> tuple[str, ...]:
    if process_model is None:
        return ()
    return tuple(
        sorted(
            {
                process_model.segment_part_ids[segment_id]
                for segment_id in action_covered_segment_ids(action)
                if segment_id in process_model.segment_part_ids
            }
        )
    )


def unstable_support_abandonment_penalty(
    action: CuttingAction,
    state: IncrementalMetricsState,
    process_model: CuttingProcessModel | None,
) -> float:
    if process_model is None or not state.unstable_part_ids:
        return 0.0

    continued_part_ids = set(action_part_ids(action, process_model))
    if not continued_part_ids:
        return float(len(state.unstable_part_ids))
    if state.unstable_part_ids & continued_part_ids:
        return 0.0
    return float(len(state.unstable_part_ids))


def retained_unstable_part_ids_after_action_start(
    action: CuttingAction,
    state: IncrementalMetricsState,
    process_model: CuttingProcessModel | None,
) -> set[str]:
    if process_model is None or not state.unstable_part_ids:
        return set(state.unstable_part_ids)
    continued_part_ids = set(action_part_ids(action, process_model))
    if not continued_part_ids:
        return set()
    if state.unstable_part_ids & continued_part_ids:
        return set(state.unstable_part_ids)
    return set()


def support_stability_shortfall(
    part_id: str,
    remaining_support_ids: set[str] | frozenset[str],
    process_model: CuttingProcessModel,
    released_part_ids: set[str] | frozenset[str] | None = None,
) -> float:
    shortfall = 0.0

    if len(remaining_support_ids) < process_model.min_remaining_support_count:
        missing_count = (
            process_model.min_remaining_support_count - len(remaining_support_ids)
        )
        shortfall += missing_count / max(process_model.min_remaining_support_count, 1)

    remaining_support_length = sum(
        process_model.segment_lengths.get(segment_id, 0.0)
        for segment_id in remaining_support_ids
    )
    remaining_support_length += adjacency_support_length(
        part_id,
        process_model,
        released_part_ids=released_part_ids,
    )
    total_support_length = process_model.part_support_lengths.get(part_id, 0.0)

    if (
        process_model.min_remaining_support_length_ratio > 0.0
        and total_support_length > 0.0
    ):
        required_length = (
            total_support_length
            * process_model.min_remaining_support_length_ratio
        )
        if remaining_support_length < required_length:
            shortfall += (required_length - remaining_support_length) / total_support_length

    part_area = process_model.part_areas.get(part_id, 0.0)
    if (
        process_model.min_area_normalized_support_length > 0.0
        and part_area > 0.0
    ):
        normalized_length = remaining_support_length / sqrt(part_area)
        if normalized_length < process_model.min_area_normalized_support_length:
            shortfall += (
                process_model.min_area_normalized_support_length
                - normalized_length
            )

    return shortfall


def adjacency_support_length(
    part_id: str,
    process_model: CuttingProcessModel,
    released_part_ids: set[str] | frozenset[str] | None = None,
) -> float:
    if process_model.adjacency_support_weight <= 0.0:
        return 0.0
    released = released_part_ids or frozenset()
    adjacent_lengths = process_model.part_adjacency_support_lengths.get(part_id, {})
    return process_model.adjacency_support_weight * sum(
        length
        for neighbor_part_id, length in adjacent_lengths.items()
        if neighbor_part_id not in released
    )


def update_released_part_state(
    action: CuttingAction,
    processed_segments: set[str],
    state: IncrementalMetricsState,
    process_model: CuttingProcessModel | None,
) -> tuple[
    set[str],
    tuple[tuple, ...],
    tuple[tuple[float, float, float, float], ...],
]:
    if process_model is None:
        return (
            set(state.released_part_ids),
            state.processed_polygons,
            state.processed_polygon_bounds,
        )

    released_part_ids = set(state.released_part_ids)
    newly_released: list[str] = []
    for part_id in action_part_ids(action, process_model):
        if part_id in released_part_ids:
            continue
        if process_model.part_segment_ids[part_id] <= processed_segments:
            released_part_ids.add(part_id)
            newly_released.append(part_id)

    if not newly_released:
        return (
            released_part_ids,
            state.processed_polygons,
            state.processed_polygon_bounds,
        )

    new_polygons = tuple(
        process_model.part_polygons[part_id]
        for part_id in sorted(newly_released)
        if part_id in process_model.part_polygons
    )
    new_polygon_bounds = tuple(polygon_bounds(polygon) for polygon in new_polygons)
    return (
        released_part_ids,
        state.processed_polygons + new_polygons,
        state.processed_polygon_bounds + new_polygon_bounds,
    )


def update_unstable_part_state(
    action: CuttingAction,
    processed_segments: set[str],
    released_part_ids: set[str],
    retained_unstable_part_ids: set[str],
    process_model: CuttingProcessModel | None,
) -> set[str]:
    if process_model is None:
        return retained_unstable_part_ids

    unstable_part_ids = set(retained_unstable_part_ids)
    unstable_part_ids.difference_update(released_part_ids)

    if action.action_type != CuttingActionType.CUT:
        return unstable_part_ids

    for part_id in action_part_ids(action, process_model):
        if part_id in released_part_ids:
            unstable_part_ids.discard(part_id)
            continue
        support_ids = process_model.support_segment_ids.get(part_id, frozenset())
        if not support_ids:
            continue
        remaining_support_ids = support_ids - processed_segments
        if support_stability_shortfall(
            part_id,
            remaining_support_ids,
            process_model,
            released_part_ids=released_part_ids,
        ) > 0.0:
            unstable_part_ids.add(part_id)
        else:
            unstable_part_ids.discard(part_id)

    return unstable_part_ids


def classify_action_clearance(
    action: CuttingAction,
    state: IncrementalMetricsState,
    tool: ToolConfig,
) -> CuttingAction:
    if (
        action.action_type != CuttingActionType.TRAVEL
        or action.travel_mode == TravelMode.SAFE_LIFT
        or not tool.allow_safe_lift_over_released_parts
    ):
        return action

    if (
        action_low_clearance_collision_penalty(
            action,
            state.processed_polygons,
            state.processed_polygon_bounds,
        )
        == 0.0
    ):
        return action

    return replace(action, travel_mode=TravelMode.SAFE_LIFT)


def evaluate_action_delta(
    action: CuttingAction,
    state: IncrementalMetricsState,
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> PathMetrics:
    length = euclidean_distance(action.start, action.end)
    next_direction = direction_vector(action.start, action.end)
    boundary_penalty = action_boundary_penalty(action, panel, tool)
    collision_penalty = action_collision_penalty(
        action,
        state.processed_polygons,
        state.processed_polygon_bounds,
    )
    stability_penalty = unstable_support_abandonment_penalty(
        action,
        state,
        process_model,
    )
    turn = turn_penalty(state.current_direction, next_direction)

    if action.action_type == CuttingActionType.CUT:
        covered_segment_ids = action_covered_segment_ids(action)
        already_cut = bool(covered_segment_ids) and all(
            segment_id in state.processed_segments
            for segment_id in covered_segment_ids
        )
        pierce = 0 if state.is_tool_down or already_cut else 1
        return PathMetrics(
            cutting_length=length,
            pierce_count=pierce,
            turn_penalty=turn,
            collision_penalty=collision_penalty,
            boundary_penalty=boundary_penalty,
            stability_penalty=stability_penalty,
        )

    lift = 1 if state.is_tool_down else 0
    safe_lift_count = 1 if action.travel_mode == TravelMode.SAFE_LIFT else 0
    safe_lift_distance = length if action.travel_mode == TravelMode.SAFE_LIFT else 0.0
    detour_count = 1 if action.travel_mode == TravelMode.LOW_CLEARANCE_DETOUR else 0
    detour_distance = length if action.travel_mode == TravelMode.LOW_CLEARANCE_DETOUR else 0.0
    travel_mode_cost = travel_action_mode_cost(action, tool)
    return PathMetrics(
        air_move_distance=length,
        lift_count=lift,
        turn_penalty=turn,
        collision_penalty=collision_penalty,
        boundary_penalty=boundary_penalty,
        stability_penalty=stability_penalty,
        safe_lift_count=safe_lift_count,
        safe_lift_distance=safe_lift_distance,
        detour_count=detour_count,
        detour_distance=detour_distance,
        travel_mode_cost=travel_mode_cost,
    )


def apply_action_incremental(
    action: CuttingAction,
    state: IncrementalMetricsState,
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> IncrementalMetricsState:
    effective_action = classify_action_clearance(action, state, tool)
    delta = evaluate_action_delta(
        effective_action,
        state,
        panel,
        tool,
        process_model=process_model,
    )
    retained_unstable_part_ids = retained_unstable_part_ids_after_action_start(
        effective_action,
        state,
        process_model,
    )
    processed_segments = set(state.processed_segments)
    if effective_action.action_type == CuttingActionType.CUT:
        processed_segments.update(action_covered_segment_ids(effective_action))
    released_part_ids, processed_polygons, processed_polygon_bounds = update_released_part_state(
        effective_action,
        processed_segments,
        state,
        process_model,
    )
    unstable_part_ids = update_unstable_part_state(
        effective_action,
        processed_segments,
        released_part_ids,
        retained_unstable_part_ids,
        process_model,
    )

    return IncrementalMetricsState(
        current_point=effective_action.end,
        current_direction=direction_vector(effective_action.start, effective_action.end),
        is_tool_down=effective_action.action_type == CuttingActionType.CUT,
        current_cutting_unit_id=(
            effective_action.cutting_unit_id
            if effective_action.action_type == CuttingActionType.CUT
            else None
        ),
        processed_segments=processed_segments,
        released_part_ids=released_part_ids,
        unstable_part_ids=unstable_part_ids,
        processed_polygons=processed_polygons,
        processed_polygon_bounds=processed_polygon_bounds,
        metrics=add_metrics(state.metrics, delta),
    )


def materialize_action_clearance(
    actions: tuple[CuttingAction, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> tuple[CuttingAction, ...]:
    state = IncrementalMetricsState(current_point=tool.start_point)
    materialized: list[CuttingAction] = []
    for action in actions:
        if action.action_type == CuttingActionType.TRAVEL and action.travel_mode != TravelMode.SAFE_LIFT:
            effective_actions = plan_travel_actions(
                action.start,
                action.end,
                state,
                panel,
                tool,
                cutting_unit_id=action.cutting_unit_id,
                covered_segment_ids=action.covered_segment_ids,
            )
        else:
            effective_actions = (classify_action_clearance(action, state, tool),)

        for effective_action in effective_actions:
            materialized.append(effective_action)
            state = apply_action_incremental(
                effective_action,
                state,
                panel,
                tool,
                process_model=process_model,
            )
    return tuple(materialized)


def finalize_metrics(state: IncrementalMetricsState) -> PathMetrics:
    metrics = state.metrics
    if state.unstable_part_ids:
        metrics = add_metrics(
            metrics,
            PathMetrics(stability_penalty=float(len(state.unstable_part_ids))),
        )
    if not state.is_tool_down:
        return metrics
    return add_metrics(metrics, PathMetrics(lift_count=1))


def evaluate_actions(
    actions: tuple[CuttingAction, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> PathMetrics:
    state = IncrementalMetricsState(current_point=tool.start_point)
    for action in actions:
        state = apply_action_incremental(
            action,
            state,
            panel,
            tool,
            process_model=process_model,
        )
    return finalize_metrics(state)


def compare_metrics(
    candidate: PathMetrics,
    incumbent: PathMetrics,
    config: MetricsComparatorConfig | None = None,
) -> int:
    """Return -1 if candidate is better, 1 if worse, and 0 if equivalent."""

    if config is None:
        config = MetricsComparatorConfig()

    comparisons = (
        (candidate.hard_penalty, incumbent.hard_penalty, config.hard_penalty_epsilon, False),
        (candidate.stability_penalty, incumbent.stability_penalty, config.penalty_epsilon, False),
        (
            candidate.pierce_count + candidate.lift_count + candidate.safe_lift_count,
            incumbent.pierce_count + incumbent.lift_count + incumbent.safe_lift_count,
            config.tool_event_epsilon,
            False,
        ),
        (
            candidate.travel_mode_cost,
            incumbent.travel_mode_cost,
            config.distance_epsilon,
            False,
        ),
        (
            candidate.detour_distance,
            incumbent.detour_distance,
            config.distance_epsilon,
            False,
        ),
        (candidate.air_move_distance, incumbent.air_move_distance, config.distance_epsilon, False),
        (candidate.turn_penalty, incumbent.turn_penalty, config.penalty_epsilon, False),
        (candidate.continuity_reward, incumbent.continuity_reward, config.penalty_epsilon, True),
    )

    for left, right, epsilon, larger_is_better in comparisons:
        if abs(left - right) <= epsilon:
            continue
        if larger_is_better:
            return -1 if left > right else 1
        return -1 if left < right else 1
    return 0


def metrics_better(
    candidate: PathMetrics,
    incumbent: PathMetrics,
    config: MetricsComparatorConfig | None = None,
) -> bool:
    return compare_metrics(candidate, incumbent, config) < 0


def travel_action_mode_cost(action: CuttingAction, tool: ToolConfig) -> float:
    if action.action_type != CuttingActionType.TRAVEL:
        return 0.0

    length = euclidean_distance(action.start, action.end)
    if action.travel_mode == TravelMode.SAFE_LIFT:
        return tool.safe_lift_fixed_cost + length * tool.safe_lift_travel_weight
    if action.travel_mode == TravelMode.LOW_CLEARANCE_DETOUR:
        return length * tool.detour_travel_weight
    return length * tool.low_clearance_travel_weight
