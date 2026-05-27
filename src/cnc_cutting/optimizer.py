from __future__ import annotations

from dataclasses import dataclass, replace

from .exact_dp import ExactDPConfig, exact_process_dp_order
from .geometry import euclidean_distance
from .local_search import (
    BeamSearchConfig,
    LocalSearchConfig,
    improve_directed_unit_order,
    path_distance_local_search_order,
    process_local_search_multistart_order,
    process_aware_beam_search_order,
    process_aware_beam_polished_search_order,
    process_metric_key,
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


MIN_TRAVEL_COST_SAVING_PER_EXTRA_TOOL_EVENT = 100.0
MIN_TRAVEL_COST_SAVING_RATIO_PER_EXTRA_TOOL_EVENT = 0.02
MIN_MACHINING_COST_SAVING_FOR_EXTRA_TOOL_EVENT = 1e-9


@dataclass(frozen=True)
class ToolEventGateConfig:
    enabled: bool = True
    min_travel_saving_per_extra_event: float = (
        MIN_TRAVEL_COST_SAVING_PER_EXTRA_TOOL_EVENT
    )
    min_travel_saving_ratio_per_extra_event: float = (
        MIN_TRAVEL_COST_SAVING_RATIO_PER_EXTRA_TOOL_EVENT
    )
    min_machining_saving: float = MIN_MACHINING_COST_SAVING_FOR_EXTRA_TOOL_EVENT

    def __post_init__(self) -> None:
        if self.min_travel_saving_per_extra_event < 0.0:
            raise ValueError("min_travel_saving_per_extra_event must be non-negative")
        if self.min_travel_saving_ratio_per_extra_event < 0.0:
            raise ValueError("min_travel_saving_ratio_per_extra_event must be non-negative")
        if self.min_machining_saving < 0.0:
            raise ValueError("min_machining_saving must be non-negative")


DEFAULT_TOOL_EVENT_GATE_CONFIG = ToolEventGateConfig()


def _tool_event_count(metrics: PathMetrics) -> int:
    return metrics.pierce_count + metrics.lift_count + metrics.safe_lift_count


def _tool_event_increase_justified(
    candidate: PathMetrics,
    alternative: PathMetrics,
    tool_event_gate: ToolEventGateConfig = DEFAULT_TOOL_EVENT_GATE_CONFIG,
) -> bool:
    if not tool_event_gate.enabled:
        return True

    extra_events = _tool_event_count(candidate) - _tool_event_count(alternative)
    if extra_events <= 0:
        return True

    travel_saving = alternative.travel_mode_cost - candidate.travel_mode_cost
    machining_saving = alternative.machining_cost - candidate.machining_cost
    required_saving_per_event = max(
        tool_event_gate.min_travel_saving_per_extra_event,
        alternative.travel_mode_cost
        * tool_event_gate.min_travel_saving_ratio_per_extra_event,
    )
    return (
        machining_saving > tool_event_gate.min_machining_saving
        and travel_saving >= required_saving_per_event * extra_events
    )


def _route_passes_tool_event_gate(
    candidate: RoutePlan,
    alternatives: tuple[RoutePlan, ...],
    tool_event_gate: ToolEventGateConfig = DEFAULT_TOOL_EVENT_GATE_CONFIG,
) -> bool:
    return all(
        _tool_event_increase_justified(
            candidate.metrics,
            alternative.metrics,
            tool_event_gate=tool_event_gate,
        )
        for alternative in alternatives
        if alternative is not candidate
    )


def wider_beam_search_config(config: BeamSearchConfig | None = None) -> BeamSearchConfig:
    """Return the wider beam setting used for failure-risk fallback cases."""

    if config is None:
        config = BeamSearchConfig()

    wide_width = config.beam_width + 2
    candidate_pool_size = (
        None if config.candidate_pool_size is None else config.candidate_pool_size + 6
    )
    max_expansions_per_node = (
        None
        if config.max_expansions_per_node is None
        else config.max_expansions_per_node + 12
    )
    max_layer_expansions = (
        None if config.max_layer_expansions is None else wide_width * 7
    )
    return replace(
        config,
        beam_width=wide_width,
        candidate_pool_size=candidate_pool_size,
        max_expansions_per_node=max_expansions_per_node,
        max_layer_expansions=max_layer_expansions,
    )


def _best_process_route(
    plans: tuple[RoutePlan, ...],
    protected_plans: tuple[RoutePlan, ...] = (),
    tool_event_gate: ToolEventGateConfig = DEFAULT_TOOL_EVENT_GATE_CONFIG,
) -> RoutePlan:
    if not plans:
        raise ValueError("at least one route plan is required")

    protected_plan_ids = {id(plan) for plan in protected_plans}
    gated_plans = tuple(
        plan
        for plan in plans
        if id(plan) in protected_plan_ids
        or _route_passes_tool_event_gate(plan, plans, tool_event_gate=tool_event_gate)
    )
    if not gated_plans:
        gated_plans = plans
    return min(gated_plans, key=lambda plan: process_metric_key(plan.metrics))


def _beam_fallback_needed(
    beam_plan: RoutePlan,
    reference_plan: RoutePlan,
    margin: float,
) -> bool:
    if process_metric_key(beam_plan.metrics) >= process_metric_key(reference_plan.metrics):
        return True
    return (beam_plan.metrics.travel_mode_cost - reference_plan.metrics.travel_mode_cost) > -margin


def _route_from_search_result(
    selected_units: tuple[CuttingUnit, ...],
    result,
) -> RoutePlan:
    return RoutePlan(
        selected_units=selected_units,
        actions=result.actions,
        metrics=result.metrics,
    )


def _polish_beam_route(
    selected_units: tuple[CuttingUnit, ...],
    beam_result,
    panel: Panel,
    tool: ToolConfig,
    polish_config: LocalSearchConfig | None,
    process_model: CuttingProcessModel | None,
) -> RoutePlan:
    if not beam_result.directed_units:
        return _route_from_search_result(selected_units, beam_result)
    if polish_config is None:
        polish_config = LocalSearchConfig(
            max_iterations=2,
            first_improvement=True,
            max_swap_span=6,
            max_relocate_span=6,
            max_two_opt_span=6,
            max_neighbors_per_iteration=160,
            process_aware_initial_order=True,
        )
    polished = improve_directed_unit_order(
        beam_result.directed_units,
        panel,
        tool,
        config=polish_config,
        process_model=process_model,
    )
    return _route_from_search_result(selected_units, polished)


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


def _dynamic_unit_priority(
    unit: CuttingUnit,
    covered_segment_ids: set[str],
) -> tuple[int, float, int, int, float, str]:
    unit_segment_ids = set(unit.covered_segment_ids)
    new_segment_ids = unit_segment_ids - covered_segment_ids
    repeated_segment_ids = unit_segment_ids & covered_segment_ids
    static_priority = _unit_priority(unit)
    return (
        len(repeated_segment_ids),
        sum(
            segment.length
            for segment in unit.segments
            if segment.segment_id in repeated_segment_ids
        ),
        -len(new_segment_ids),
        static_priority[0],
        static_priority[2],
        static_priority[3],
    )


def select_coverage_units(
    units: tuple[CuttingUnit, ...],
    allow_bridge_cut: bool = False,
) -> tuple[CuttingUnit, ...]:
    selected: list[CuttingUnit] = []
    covered_segment_ids: set[str] = set()
    remaining = [
        unit
        for unit in units
        if allow_bridge_cut or not unit.requires_bridge_cut
    ]

    while True:
        candidates = [
            unit
            for unit in remaining
            if set(unit.covered_segment_ids) - covered_segment_ids
        ]
        if not candidates:
            break
        unit = min(
            candidates,
            key=lambda candidate: _dynamic_unit_priority(candidate, covered_segment_ids),
        )
        selected.append(unit)
        unit_segment_ids = set(unit.covered_segment_ids)
        covered_segment_ids.update(unit_segment_ids)
        remaining.remove(unit)

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


def plan_process_local_search_multistart_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    result = process_local_search_multistart_order(
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


def plan_exact_process_dp_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    config: ExactDPConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    result = exact_process_dp_order(
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


def plan_process_aware_beam_adaptive_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    beam_config: BeamSearchConfig | None = None,
    fallback_beam_config: BeamSearchConfig | None = None,
    topology_candidate_pool_size: int | None = None,
    fallback_margin: float = 1000.0,
    process_model: CuttingProcessModel | None = None,
    tool_event_gate: ToolEventGateConfig = DEFAULT_TOOL_EVENT_GATE_CONFIG,
) -> RoutePlan:
    """Bounded portfolio planner that expands beam search only on risky cases.

    The topology route is a cheap process-aware reference. The default beam result is
    accepted when it clearly dominates that reference; otherwise a wider beam is
    evaluated and the best process-metric route is returned.
    """

    topology_plan = plan_topology_route(
        units,
        panel,
        tool,
        allow_bridge_cut=allow_bridge_cut,
        candidate_pool_size=topology_candidate_pool_size,
        process_aware=True,
        process_model=process_model,
    )
    beam_plan = plan_process_aware_beam_route(
        units,
        panel,
        tool,
        allow_bridge_cut=allow_bridge_cut,
        config=beam_config,
        process_model=process_model,
    )
    candidates = [topology_plan, beam_plan]
    if _beam_fallback_needed(beam_plan, topology_plan, fallback_margin):
        candidates.append(
            plan_process_aware_beam_route(
                units,
                panel,
                tool,
                allow_bridge_cut=allow_bridge_cut,
                config=(
                    fallback_beam_config
                    if fallback_beam_config is not None
                    else wider_beam_search_config(beam_config)
                ),
                process_model=process_model,
            )
        )
    return _best_process_route(
        tuple(candidates),
        protected_plans=(beam_plan,),
        tool_event_gate=tool_event_gate,
    )


def plan_process_aware_beam_adaptive_polished_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    beam_config: BeamSearchConfig | None = None,
    fallback_beam_config: BeamSearchConfig | None = None,
    polish_config: LocalSearchConfig | None = None,
    topology_candidate_pool_size: int | None = None,
    fallback_margin: float = 1000.0,
    process_model: CuttingProcessModel | None = None,
    tool_event_gate: ToolEventGateConfig = DEFAULT_TOOL_EVENT_GATE_CONFIG,
) -> RoutePlan:
    """Adaptive portfolio with process-aware local polishing as a candidate."""

    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    topology_plan = plan_topology_route(
        units,
        panel,
        tool,
        allow_bridge_cut=allow_bridge_cut,
        candidate_pool_size=topology_candidate_pool_size,
        process_aware=True,
        process_model=process_model,
    )
    beam_result = process_aware_beam_search_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=beam_config,
    )
    beam_plan = _route_from_search_result(selected_units, beam_result)
    polished_plan = _polish_beam_route(
        selected_units,
        beam_result,
        panel,
        tool,
        polish_config=polish_config,
        process_model=process_model,
    )
    candidates = [topology_plan, beam_plan, polished_plan]
    current_best = _best_process_route(
        tuple(candidates),
        protected_plans=(beam_plan,),
        tool_event_gate=tool_event_gate,
    )
    if _beam_fallback_needed(current_best, topology_plan, fallback_margin):
        fallback_result = process_aware_beam_search_order(
            selected_units,
            panel,
            tool,
            process_model=process_model,
            config=(
                fallback_beam_config
                if fallback_beam_config is not None
                else wider_beam_search_config(beam_config)
            ),
        )
        candidates.append(
            _polish_beam_route(
                selected_units,
                fallback_result,
                panel,
                tool,
                polish_config=polish_config,
                process_model=process_model,
            )
        )
    return _best_process_route(
        tuple(candidates),
        protected_plans=(beam_plan,),
        tool_event_gate=tool_event_gate,
    )


def plan_process_aware_beam_polished_route(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    allow_bridge_cut: bool = False,
    beam_config: BeamSearchConfig | None = None,
    polish_config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> RoutePlan:
    selected_units = select_coverage_units(units, allow_bridge_cut=allow_bridge_cut)
    result = process_aware_beam_polished_search_order(
        selected_units,
        panel=panel,
        tool=tool,
        process_model=process_model,
        beam_config=beam_config,
        polish_config=polish_config,
    )
    return RoutePlan(
        selected_units=selected_units,
        actions=result.actions,
        metrics=result.metrics,
    )
