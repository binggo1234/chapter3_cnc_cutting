from __future__ import annotations

from dataclasses import dataclass

from .geometry import euclidean_distance
from .local_search import (
    BeamSearchConfig,
    LocalSearchConfig,
    path_distance_local_search_order,
    process_aware_beam_search_order,
    process_aware_unit_order,
    topology_local_search_order,
)
from .metrics import evaluate_actions, materialize_action_clearance
from .models import (
    CuttingAction,
    CuttingActionType,
    CuttingProcessModel,
    CuttingUnit,
    CuttingUnitType,
    Panel,
    PathMetrics,
    Point,
    ToolConfig,
)
from .topology_operators import TopologyWeights, topology_aware_unit_actions
from .topology_operators import directed_units_to_actions


@dataclass(frozen=True)
class RoutePlan:
    selected_units: tuple[CuttingUnit, ...]
    actions: tuple[CuttingAction, ...]
    metrics: PathMetrics


def _unit_priority(unit: CuttingUnit) -> tuple[int, int, float, str]:
    type_priority = {
        CuttingUnitType.SHARED_EDGE: 0,
        CuttingUnitType.NEAR_SHARED_CHANNEL: 1,
        CuttingUnitType.COLLINEAR_CHAIN: 2,
        CuttingUnitType.SINGLE_EDGE: 3,
    }[unit.unit_type]
    return (
        type_priority,
        -len(unit.covered_segment_ids),
        euclidean_distance(unit.start, unit.end),
        unit.unit_id,
    )


def select_coverage_units(
    units: tuple[CuttingUnit, ...],
    allow_bridge_cut: bool = False,
) -> tuple[CuttingUnit, ...]:
    selected: list[CuttingUnit] = []
    covered_segment_ids: set[str] = set()

    for unit in sorted(units, key=_unit_priority):
        if unit.requires_bridge_cut and not allow_bridge_cut:
            continue
        unit_segment_ids = set(unit.covered_segment_ids)
        if not unit_segment_ids or unit_segment_ids <= covered_segment_ids:
            continue
        selected.append(unit)
        covered_segment_ids.update(unit_segment_ids)

    return tuple(selected)


def _directed_unit_points(
    unit: CuttingUnit,
    current_point: Point,
) -> tuple[Point, Point]:
    forward_distance = euclidean_distance(current_point, unit.start)
    reverse_distance = euclidean_distance(current_point, unit.end)
    if unit.is_reversible and reverse_distance < forward_distance:
        return (unit.end, unit.start)
    return (unit.start, unit.end)


def _unit_segment_id(unit: CuttingUnit) -> str | None:
    if len(unit.covered_segment_ids) == 1:
        return unit.covered_segment_ids[0]
    return None


def greedy_unit_actions(
    units: tuple[CuttingUnit, ...],
    start_point: Point,
) -> tuple[CuttingAction, ...]:
    remaining = list(units)
    current_point = start_point
    actions: list[CuttingAction] = []

    while remaining:
        unit = min(
            remaining,
            key=lambda candidate: min(
                euclidean_distance(current_point, candidate.start),
                euclidean_distance(current_point, candidate.end)
                if candidate.is_reversible
                else float("inf"),
            ),
        )
        entry, exit_point = _directed_unit_points(unit, current_point)
        if euclidean_distance(current_point, entry) > 1e-9:
            actions.append(
                CuttingAction(
                    action_type=CuttingActionType.TRAVEL,
                    start=current_point,
                    end=entry,
                    cutting_unit_id=unit.unit_id,
                    covered_segment_ids=unit.covered_segment_ids,
                )
            )
        actions.append(
            CuttingAction(
                action_type=CuttingActionType.CUT,
                start=entry,
                end=exit_point,
                segment_id=_unit_segment_id(unit),
                cutting_unit_id=unit.unit_id,
                covered_segment_ids=unit.covered_segment_ids,
            )
        )
        current_point = exit_point
        remaining.remove(unit)

    return tuple(actions)


def plan_greedy_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    actions = materialize_action_clearance(
        greedy_unit_actions(selected_units, tool.start_point),
        panel,
        tool,
        process_model=process_model,
    )
    return RoutePlan(
        selected_units=selected_units,
        actions=actions,
        metrics=evaluate_actions(actions, panel, tool, process_model=process_model),
    )


def plan_topology_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    weights: TopologyWeights | None = None,
    process_model: CuttingProcessModel | None = None,
    candidate_pool_size: int | None = None,
    process_aware: bool = False,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    if process_model is None or not process_aware:
        actions = topology_aware_unit_actions(
            selected_units,
            start_point=tool.start_point,
            weights=weights,
            candidate_pool_size=candidate_pool_size,
        )
    else:
        actions = directed_units_to_actions(
            process_aware_unit_order(
                selected_units,
                panel=panel,
                tool=tool,
                process_model=process_model,
                candidate_pool_size=candidate_pool_size,
            ),
            start_point=tool.start_point,
        )
    actions = materialize_action_clearance(
        actions,
        panel,
        tool,
        process_model=process_model,
    )
    return RoutePlan(
        selected_units=selected_units,
        actions=actions,
        metrics=evaluate_actions(actions, panel, tool, process_model=process_model),
    )


def plan_local_search_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    result = topology_local_search_order(
        selected_units,
        panel=panel,
        tool=tool,
        config=config,
        process_model=process_model,
    )
    return RoutePlan(
        selected_units=selected_units,
        actions=result.actions,
        metrics=result.metrics,
    )


def plan_path_distance_local_search_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    result = path_distance_local_search_order(
        selected_units,
        panel=panel,
        tool=tool,
        config=config,
        process_model=process_model,
    )
    actions = materialize_action_clearance(
        directed_units_to_actions(result.directed_units, start_point=tool.start_point),
        panel,
        tool,
        process_model=process_model,
    )
    return RoutePlan(
        selected_units=selected_units,
        actions=actions,
        metrics=evaluate_actions(actions, panel, tool, process_model=process_model),
    )


def plan_process_aware_beam_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    config: BeamSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    result = process_aware_beam_search_order(
        selected_units,
        panel=panel,
        tool=tool,
        process_model=process_model,
        config=config,
    )
    return RoutePlan(
        selected_units=selected_units,
        actions=result.actions,
        metrics=result.metrics,
    )
