from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .geometry import euclidean_distance
from .metrics import (
    add_metrics,
    apply_action_incremental,
    evaluate_actions,
    finalize_metrics,
    materialize_action_clearance,
    subtract_metrics,
)
from .models import (
    CuttingAction,
    CuttingActionType,
    CuttingProcessModel,
    IncrementalMetricsState,
    CuttingUnit,
    Panel,
    PathMetrics,
    Point,
    ToolConfig,
)
from .travel import plan_travel_actions
from .topology_operators import (
    DirectedUnitCandidate,
    directed_unit_candidates,
    directed_units_to_actions,
    nearest_candidate_units,
    topology_aware_unit_order,
    transition_score,
    unit_part_ids,
    unit_segment_id,
)


@dataclass(frozen=True)
class LocalSearchConfig:
    max_iterations: int = 50
    enable_swap: bool = True
    enable_relocate: bool = True
    enable_two_opt: bool = True
    first_improvement: bool = False
    max_swap_span: int | None = None
    max_relocate_span: int | None = None
    max_two_opt_span: int | None = None
    max_neighbors_per_iteration: int | None = None
    topology_candidate_pool_size: int | None = None
    process_aware_initial_order: bool = False


def _reached_neighbor_limit(
    neighbors: list,
    max_neighbors: int | None,
) -> bool:
    return max_neighbors is not None and len(neighbors) >= max_neighbors


@dataclass(frozen=True)
class LocalSearchResult:
    directed_units: tuple[DirectedUnitCandidate, ...]
    actions: tuple[CuttingAction, ...]
    metrics: PathMetrics
    iterations: int
    improved: bool


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_width: int = 8
    candidate_pool_size: int | None = 32
    max_expansions_per_node: int | None = None
    max_layer_expansions: int | None = None
    diversity_bucket_limit: int | None = None
    min_expansions_per_parent: int = 0
    unstable_min_expansions_per_parent: int = 0
    unstable_layer_expansion_multiplier: float = 1.0
    unstable_layer_expansion_bonus: int = 0
    completion_aware_prerank: bool = False
    unstable_completion_focus_count: int | None = None


@dataclass(frozen=True)
class BeamSearchResult:
    directed_units: tuple[DirectedUnitCandidate, ...]
    actions: tuple[CuttingAction, ...]
    metrics: PathMetrics
    expanded_nodes: int
    diagnostics: tuple["BeamLayerDiagnostics", ...] = ()


def process_aware_exact_candidate_limit(candidate_pool_size: int | None) -> int | None:
    if candidate_pool_size is None:
        return None
    return max(16, candidate_pool_size // 3)


@dataclass(frozen=True)
class BeamLayerDiagnostics:
    depth: int
    input_beam_count: int
    unstable_input_prefix_count: int
    raw_expansion_count: int
    effective_layer_expansion_limit: int
    layer_expansion_count: int
    layer_pruned_count: int
    parent_quota_added_count: int
    parent_quota_pruned_count: int
    evaluated_node_count: int
    duplicate_pruned_count: int
    diversity_pruned_count: int
    fallback_added_count: int
    output_beam_count: int
    best_hard_penalty: float
    best_stability_penalty: float
    best_travel_mode_cost: float
    best_air_move_distance: float
    worst_stability_penalty: float
    worst_travel_mode_cost: float
    unstable_prefix_count: int
    released_part_count_max: int


@dataclass(frozen=True)
class NeighborMove:
    directed_units: tuple[DirectedUnitCandidate, ...]
    affected_start_index: int
    affected_stop_index: int


@dataclass(frozen=True)
class BeamSearchNode:
    directed_units: tuple[DirectedUnitCandidate, ...]
    remaining_units: tuple[CuttingUnit, ...]
    state: IncrementalMetricsState
    previous_unit: CuttingUnit | None = None


@dataclass(frozen=True)
class BeamSearchExpansion:
    parent: BeamSearchNode
    candidate: DirectedUnitCandidate
    transition_cost: float
    rank: tuple


def reverse_candidate(candidate: DirectedUnitCandidate) -> DirectedUnitCandidate | None:
    if not candidate.unit.is_reversible:
        return None
    return DirectedUnitCandidate(
        unit=candidate.unit,
        start=candidate.end,
        end=candidate.start,
        direction=(-candidate.direction[0], -candidate.direction[1]),
        is_reversed=not candidate.is_reversed,
    )


def evaluate_directed_units(
    directed_units: tuple[DirectedUnitCandidate, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> tuple[tuple[CuttingAction, ...], PathMetrics]:
    actions = materialize_action_clearance(
        directed_units_to_actions(directed_units, start_point=tool.start_point),
        panel,
        tool,
        process_model=process_model,
    )
    return actions, evaluate_actions(actions, panel, tool, process_model=process_model)


def candidate_actions_from_state(
    candidate: DirectedUnitCandidate,
    state: IncrementalMetricsState,
) -> tuple[CuttingAction, ...]:
    actions: list[CuttingAction] = []
    if euclidean_distance(state.current_point, candidate.start) > 1e-9:
        actions.append(
            CuttingAction(
                action_type=CuttingActionType.TRAVEL,
                start=state.current_point,
                end=candidate.start,
                cutting_unit_id=candidate.unit.unit_id,
                covered_segment_ids=candidate.unit.covered_segment_ids,
            )
        )
    actions.append(
        CuttingAction(
            action_type=CuttingActionType.CUT,
            start=candidate.start,
            end=candidate.end,
            segment_id=unit_segment_id(candidate.unit),
            cutting_unit_id=candidate.unit.unit_id,
            covered_segment_ids=candidate.unit.covered_segment_ids,
        )
    )
    return tuple(actions)


def apply_candidate_incremental(
    candidate: DirectedUnitCandidate,
    state: IncrementalMetricsState,
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> IncrementalMetricsState:
    next_state = state
    for action in candidate_actions_from_state(candidate, state):
        if action.action_type == CuttingActionType.TRAVEL:
            planned_actions = plan_travel_actions(
                action.start,
                action.end,
                next_state,
                panel,
                tool,
                cutting_unit_id=action.cutting_unit_id,
                covered_segment_ids=action.covered_segment_ids,
            )
        else:
            planned_actions = (action,)
        for planned_action in planned_actions:
            next_state = apply_action_incremental(
                planned_action,
                next_state,
                panel,
                tool,
                process_model=process_model,
            )
    return next_state


def build_prefix_states(
    directed_units: tuple[DirectedUnitCandidate, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> tuple[IncrementalMetricsState, ...]:
    state = IncrementalMetricsState(current_point=tool.start_point)
    states = [state]
    for candidate in directed_units:
        state = apply_candidate_incremental(
            candidate,
            state,
            panel,
            tool,
            process_model=process_model,
        )
        states.append(state)
    return tuple(states)


def evaluate_directed_units_from_prefix(
    directed_units: tuple[DirectedUnitCandidate, ...],
    start_index: int,
    prefix_state: IncrementalMetricsState,
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> PathMetrics:
    state = prefix_state
    for candidate in directed_units[start_index:]:
        state = apply_candidate_incremental(
            candidate,
            state,
            panel,
            tool,
            process_model=process_model,
        )
    return finalize_metrics(state)


def _state_rejoins_suffix(
    state: IncrementalMetricsState,
    reference_state: IncrementalMetricsState,
) -> bool:
    return (
        state.current_point == reference_state.current_point
        and state.current_direction == reference_state.current_direction
        and state.is_tool_down == reference_state.is_tool_down
        and state.current_cutting_unit_id == reference_state.current_cutting_unit_id
        and state.processed_segments == reference_state.processed_segments
        and state.released_part_ids == reference_state.released_part_ids
        and state.unstable_part_ids == reference_state.unstable_part_ids
        and tuple(sorted(state.processed_polygon_bounds))
        == tuple(sorted(reference_state.processed_polygon_bounds))
    )


def evaluate_neighbor_move_metrics(
    move: NeighborMove,
    prefix_states: tuple[IncrementalMetricsState, ...],
    current_metrics: PathMetrics,
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
) -> PathMetrics:
    stop_index = min(move.affected_stop_index, len(move.directed_units))
    state = prefix_states[move.affected_start_index]
    for candidate in move.directed_units[move.affected_start_index : stop_index]:
        state = apply_candidate_incremental(
            candidate,
            state,
            panel,
            tool,
            process_model=process_model,
        )

    if stop_index >= len(move.directed_units):
        return finalize_metrics(state)

    reference_state = prefix_states[stop_index]
    if _state_rejoins_suffix(state, reference_state):
        suffix_delta = subtract_metrics(current_metrics, reference_state.metrics)
        return add_metrics(state.metrics, suffix_delta)

    return evaluate_directed_units_from_prefix(
        move.directed_units,
        stop_index,
        state,
        panel,
        tool,
        process_model=process_model,
    )


def process_metric_key(
    metrics: PathMetrics,
) -> tuple[float, float, float, int, float, float, float, float]:
    return (
        metrics.hard_penalty,
        metrics.stability_penalty,
        metrics.machining_cost,
        metrics.pierce_count + metrics.lift_count + metrics.safe_lift_count,
        metrics.travel_mode_cost,
        metrics.air_move_distance,
        metrics.turn_penalty,
        -metrics.continuity_reward,
    )


def path_distance_metric_key(
    metrics: PathMetrics,
) -> tuple[float, float, float, float, float, int]:
    return (
        metrics.hard_penalty,
        metrics.machining_cost,
        metrics.travel_mode_cost,
        metrics.air_move_distance,
        metrics.turn_penalty,
        metrics.pierce_count + metrics.lift_count + metrics.safe_lift_count,
    )


def process_state_metric_key(
    state: IncrementalMetricsState,
) -> tuple[float, float, int, float, float, float, float]:
    metrics = state.metrics
    return (
        metrics.hard_penalty,
        metrics.stability_penalty + float(len(state.unstable_part_ids)),
        metrics.pierce_count + metrics.lift_count + metrics.safe_lift_count,
        metrics.travel_mode_cost,
        metrics.air_move_distance,
        metrics.turn_penalty,
        -metrics.continuity_reward,
    )


def unit_process_part_ids(
    unit: CuttingUnit,
    process_model: CuttingProcessModel | None,
) -> frozenset[str]:
    if process_model is None:
        return frozenset()
    return frozenset(
        process_model.segment_part_ids[segment_id]
        for segment_id in unit.covered_segment_ids
        if segment_id in process_model.segment_part_ids
    )


def completion_candidate_units(
    remaining: list[CuttingUnit],
    state: IncrementalMetricsState,
    process_model: CuttingProcessModel | None,
    focus_count: int | None = None,
) -> tuple[CuttingUnit, ...]:
    if process_model is None or not state.unstable_part_ids:
        return ()
    unstable_part_ids = state.unstable_part_ids
    if focus_count is not None and focus_count > 0:
        unstable_part_ids = set(
            sorted(
                unstable_part_ids,
                key=lambda part_id: _unstable_part_completion_rank(
                    part_id,
                    remaining,
                    state,
                    process_model,
                ),
            )[:focus_count]
        )
    return tuple(
        unit
        for unit in remaining
        if unit_process_part_ids(unit, process_model) & unstable_part_ids
    )


def _unstable_part_completion_rank(
    part_id: str,
    remaining: list[CuttingUnit],
    state: IncrementalMetricsState,
    process_model: CuttingProcessModel,
) -> tuple[int, float, str]:
    remaining_segment_ids = (
        process_model.part_segment_ids.get(part_id, frozenset())
        - state.processed_segments
    )
    nearest_distance = min(
        (
            min(
                euclidean_distance(state.current_point, unit.start),
                euclidean_distance(state.current_point, unit.end)
                if unit.is_reversible
                else float("inf"),
            )
            for unit in remaining
            if part_id in unit_process_part_ids(unit, process_model)
        ),
        default=float("inf"),
    )
    return (len(remaining_segment_ids), nearest_distance, part_id)


def process_aware_candidate_units(
    remaining: list[CuttingUnit],
    state: IncrementalMetricsState,
    process_model: CuttingProcessModel | None,
    candidate_pool_size: int | None,
    unstable_completion_focus_count: int | None = None,
) -> tuple[CuttingUnit, ...]:
    completion_units = completion_candidate_units(
        remaining,
        state,
        process_model,
        focus_count=unstable_completion_focus_count,
    )
    if completion_units:
        return completion_units
    return nearest_candidate_units(
        remaining,
        current_point=state.current_point,
        candidate_pool_size=candidate_pool_size,
    )


def process_aware_unit_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
    candidate_pool_size: int | None = None,
) -> tuple[DirectedUnitCandidate, ...]:
    remaining = list(units)
    directed_candidate_cache = {
        unit.unit_id: directed_unit_candidates(unit)
        for unit in units
    }
    state = IncrementalMetricsState(current_point=tool.start_point)
    previous_unit: CuttingUnit | None = None
    ordered: list[DirectedUnitCandidate] = []

    while remaining:
        candidate_units = process_aware_candidate_units(
            remaining,
            state=state,
            process_model=process_model,
            candidate_pool_size=candidate_pool_size,
        )
        best_candidate: DirectedUnitCandidate | None = None
        best_state: IncrementalMetricsState | None = None
        best_key: tuple | None = None

        directed_candidates = tuple(
            directed_candidate
            for unit in candidate_units
            for directed_candidate in directed_candidate_cache[unit.unit_id]
        )
        exact_candidate_limit = process_aware_exact_candidate_limit(candidate_pool_size)
        if (
            exact_candidate_limit is not None
            and not state.unstable_part_ids
            and len(directed_candidates) > exact_candidate_limit
        ):
            directed_candidates = tuple(
                sorted(
                    directed_candidates,
                    key=lambda candidate: (
                        transition_score(
                            candidate,
                            current_point=state.current_point,
                            previous_direction=state.current_direction,
                            previous_unit=previous_unit,
                        ),
                        euclidean_distance(state.current_point, candidate.start),
                        candidate.unit.unit_id,
                        candidate.is_reversed,
                    ),
                )[:exact_candidate_limit]
            )

        for candidate in directed_candidates:
            next_state = apply_candidate_incremental(
                candidate,
                state,
                panel,
                tool,
                process_model=process_model,
            )
            key = (
                process_state_metric_key(next_state),
                transition_score(
                    candidate,
                    current_point=state.current_point,
                    previous_direction=state.current_direction,
                    previous_unit=previous_unit,
                ),
                candidate.unit.unit_id,
                candidate.is_reversed,
            )
            if best_key is None or key < best_key:
                best_candidate = candidate
                best_state = next_state
                best_key = key

        if best_candidate is None or best_state is None:
            break

        ordered.append(best_candidate)
        remaining.remove(best_candidate.unit)
        previous_unit = best_candidate.unit
        state = best_state

    return tuple(ordered)


def _beam_node_signature(node: BeamSearchNode) -> tuple:
    return (
        tuple(sorted(unit.unit_id for unit in node.remaining_units)),
        node.state.current_point,
        node.state.current_direction,
        tuple(sorted(node.state.unstable_part_ids)),
        tuple(sorted(node.state.released_part_ids)),
    )


def _beam_node_part_ids(
    node: BeamSearchNode,
    process_model: CuttingProcessModel | None,
) -> tuple[str, ...]:
    if node.previous_unit is None:
        return ()
    process_part_ids = unit_process_part_ids(node.previous_unit, process_model)
    if process_part_ids:
        return tuple(sorted(process_part_ids))
    return unit_part_ids(node.previous_unit)


def _beam_diversity_signature(
    node: BeamSearchNode,
    process_model: CuttingProcessModel | None,
) -> tuple:
    return (
        _beam_node_part_ids(node, process_model),
        tuple(sorted(node.state.unstable_part_ids)),
        len(node.state.released_part_ids),
        node.state.current_direction,
    )


def _select_beam_nodes(
    ranked_nodes: list[tuple[tuple, BeamSearchNode]],
    beam_width: int,
    diversity_bucket_limit: int | None,
    process_model: CuttingProcessModel | None,
) -> tuple[tuple[BeamSearchNode, ...], int, int]:
    ordered = sorted(ranked_nodes, key=lambda item: item[0])
    if diversity_bucket_limit is None or diversity_bucket_limit <= 0:
        return tuple(node for _, node in ordered[:beam_width]), 0, 0

    selected: list[BeamSearchNode] = []
    selected_ids: set[int] = set()
    bucket_counts: dict[tuple, int] = {}
    diversity_pruned_count = 0
    for _, node in ordered:
        signature = _beam_diversity_signature(node, process_model)
        if bucket_counts.get(signature, 0) >= diversity_bucket_limit:
            diversity_pruned_count += 1
            continue
        selected.append(node)
        selected_ids.add(id(node))
        bucket_counts[signature] = bucket_counts.get(signature, 0) + 1
        if len(selected) >= beam_width:
            return tuple(selected), diversity_pruned_count, 0

    fallback_added_count = 0
    for _, node in ordered:
        if id(node) in selected_ids:
            continue
        selected.append(node)
        fallback_added_count += 1
        if len(selected) >= beam_width:
            break
    return tuple(selected), diversity_pruned_count, fallback_added_count


def _beam_node_rank(
    node: BeamSearchNode,
    transition_cost: float = 0.0,
    is_reversed: bool = False,
) -> tuple:
    return (
        process_state_metric_key(node.state),
        transition_cost,
        len(node.remaining_units),
        node.previous_unit.unit_id if node.previous_unit is not None else "",
        is_reversed,
    )


def _beam_expansion_rank(
    node: BeamSearchNode,
    candidate: DirectedUnitCandidate,
    transition_cost: float,
    process_model: CuttingProcessModel | None = None,
    completion_aware_prerank: bool = False,
) -> tuple:
    unstable_overlap_count = 0
    covered_segment_count = 0
    if completion_aware_prerank:
        candidate_part_ids = unit_process_part_ids(candidate.unit, process_model)
        unstable_overlap_count = len(candidate_part_ids & node.state.unstable_part_ids)
        covered_segment_count = len(candidate.unit.covered_segment_ids)
    return (
        process_state_metric_key(node.state),
        -unstable_overlap_count,
        -covered_segment_count,
        transition_cost,
        euclidean_distance(node.state.current_point, candidate.start),
        len(node.remaining_units),
        candidate.unit.unit_id,
        candidate.is_reversed,
    )


def _limited_directed_candidates(
    candidate_units: tuple[CuttingUnit, ...],
    directed_candidate_cache: dict[str, tuple[DirectedUnitCandidate, ...]],
    state: IncrementalMetricsState,
    previous_unit: CuttingUnit | None,
    max_expansions: int | None,
    process_model: CuttingProcessModel | None,
    completion_aware_prerank: bool = False,
) -> tuple[tuple[DirectedUnitCandidate, float], ...]:
    candidates = tuple(
        (
            directed_candidate,
            transition_score(
                directed_candidate,
                current_point=state.current_point,
                previous_direction=state.current_direction,
                previous_unit=previous_unit,
            ),
        )
        for unit in candidate_units
        for directed_candidate in directed_candidate_cache[unit.unit_id]
    )
    unstable_part_ids = state.unstable_part_ids
    ordered = sorted(
        candidates,
        key=lambda item: (
            (
                -len(unit_process_part_ids(item[0].unit, process_model) & unstable_part_ids)
                if completion_aware_prerank
                else 0
            ),
            -len(item[0].unit.covered_segment_ids) if completion_aware_prerank else 0,
            item[1],
            euclidean_distance(state.current_point, item[0].start),
            item[0].unit.unit_id,
            item[0].is_reversed,
        ),
    )
    if max_expansions is None:
        return tuple(ordered)
    return tuple(ordered[:max_expansions])


def _effective_layer_expansion_limit(
    config: BeamSearchConfig,
    unstable_parent_count: int,
) -> int | None:
    if config.max_layer_expansions is None:
        return None

    base_limit = max(0, config.max_layer_expansions)
    if unstable_parent_count <= 0:
        return base_limit

    multiplier = max(1.0, config.unstable_layer_expansion_multiplier)
    bonus = max(0, config.unstable_layer_expansion_bonus)
    return max(base_limit, ceil(base_limit * multiplier) + bonus)


def _limited_beam_expansions(
    beam: tuple[BeamSearchNode, ...],
    directed_candidate_cache: dict[str, tuple[DirectedUnitCandidate, ...]],
    process_model: CuttingProcessModel | None,
    config: BeamSearchConfig,
) -> tuple[tuple[BeamSearchExpansion, ...], int, int, int, int, int]:
    parent_expansions: list[tuple[BeamSearchNode, tuple[BeamSearchExpansion, ...]]] = []
    raw_expansion_count = 0
    unstable_parent_count = sum(1 for node in beam if node.state.unstable_part_ids)
    for node in beam:
        candidate_units = process_aware_candidate_units(
            list(node.remaining_units),
            state=node.state,
            process_model=process_model,
            candidate_pool_size=config.candidate_pool_size,
            unstable_completion_focus_count=config.unstable_completion_focus_count,
        )
        node_expansions: list[BeamSearchExpansion] = []
        for candidate, transition_cost in _limited_directed_candidates(
            candidate_units,
            directed_candidate_cache,
            node.state,
            node.previous_unit,
            config.max_expansions_per_node,
            process_model,
            config.completion_aware_prerank,
        ):
            node_expansions.append(
                BeamSearchExpansion(
                    parent=node,
                    candidate=candidate,
                    transition_cost=transition_cost,
                    rank=_beam_expansion_rank(
                        node,
                        candidate,
                        transition_cost,
                        process_model=process_model,
                        completion_aware_prerank=config.completion_aware_prerank,
                    ),
                )
            )
        raw_expansion_count += len(node_expansions)
        parent_expansions.append((node, tuple(node_expansions)))

    expansions = [
        expansion
        for _, node_expansions in parent_expansions
        for expansion in node_expansions
    ]
    expansions.sort(key=lambda expansion: expansion.rank)
    layer_expansion_limit = _effective_layer_expansion_limit(
        config,
        unstable_parent_count=unstable_parent_count,
    )
    if layer_expansion_limit is None:
        return tuple(expansions), 0, 0, 0, raw_expansion_count, unstable_parent_count

    selected: list[BeamSearchExpansion] = []
    selected_ids: set[int] = set()
    quota_added_count = 0
    parent_quota = max(0, config.min_expansions_per_parent)
    unstable_parent_quota = max(parent_quota, config.unstable_min_expansions_per_parent)

    if parent_quota > 0 or unstable_parent_quota > 0:
        for node, node_expansions in parent_expansions:
            quota = (
                unstable_parent_quota
                if node.state.unstable_part_ids
                else parent_quota
            )
            if quota <= 0:
                continue
            for expansion in node_expansions[:quota]:
                if len(selected) >= layer_expansion_limit:
                    break
                selected.append(expansion)
                selected_ids.add(id(expansion))
                quota_added_count += 1
            if len(selected) >= layer_expansion_limit:
                break

    for expansion in expansions:
        if len(selected) >= layer_expansion_limit:
            break
        if id(expansion) in selected_ids:
            continue
        selected.append(expansion)
        selected_ids.add(id(expansion))

    layer_pruned_count = max(0, raw_expansion_count - len(selected))
    quota_pruned_count = max(
        0,
        sum(
            min(
                (
                    unstable_parent_quota
                    if node.state.unstable_part_ids
                    else parent_quota
                ),
                len(node_expansions),
            )
            for node, node_expansions in parent_expansions
        )
        - quota_added_count,
    )
    return (
        tuple(selected),
        layer_pruned_count,
        quota_added_count,
        quota_pruned_count,
        layer_expansion_limit,
        unstable_parent_count,
    )


def _beam_layer_diagnostics(
    depth: int,
    input_beam_count: int,
    unstable_input_prefix_count: int,
    raw_expansion_count: int,
    effective_layer_expansion_limit: int,
    layer_expansion_count: int,
    layer_pruned_count: int,
    parent_quota_added_count: int,
    parent_quota_pruned_count: int,
    evaluated_node_count: int,
    duplicate_pruned_count: int,
    diversity_pruned_count: int,
    fallback_added_count: int,
    output_beam: tuple[BeamSearchNode, ...],
) -> BeamLayerDiagnostics:
    if output_beam:
        metrics = tuple(finalize_metrics(node.state) for node in output_beam)
        best_by_process = min(metrics, key=process_metric_key)
        worst_stability = max(metric.stability_penalty for metric in metrics)
        worst_travel = max(metric.travel_mode_cost for metric in metrics)
        released_max = max(len(node.state.released_part_ids) for node in output_beam)
        unstable_count = sum(1 for node in output_beam if node.state.unstable_part_ids)
    else:
        best_by_process = PathMetrics()
        worst_stability = 0.0
        worst_travel = 0.0
        released_max = 0
        unstable_count = 0

    return BeamLayerDiagnostics(
        depth=depth,
        input_beam_count=input_beam_count,
        unstable_input_prefix_count=unstable_input_prefix_count,
        raw_expansion_count=raw_expansion_count,
        effective_layer_expansion_limit=effective_layer_expansion_limit,
        layer_expansion_count=layer_expansion_count,
        layer_pruned_count=layer_pruned_count,
        parent_quota_added_count=parent_quota_added_count,
        parent_quota_pruned_count=parent_quota_pruned_count,
        evaluated_node_count=evaluated_node_count,
        duplicate_pruned_count=duplicate_pruned_count,
        diversity_pruned_count=diversity_pruned_count,
        fallback_added_count=fallback_added_count,
        output_beam_count=len(output_beam),
        best_hard_penalty=best_by_process.hard_penalty,
        best_stability_penalty=best_by_process.stability_penalty,
        best_travel_mode_cost=best_by_process.travel_mode_cost,
        best_air_move_distance=best_by_process.air_move_distance,
        worst_stability_penalty=worst_stability,
        worst_travel_mode_cost=worst_travel,
        unstable_prefix_count=unstable_count,
        released_part_count_max=released_max,
    )


def process_aware_beam_search_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
    config: BeamSearchConfig | None = None,
) -> BeamSearchResult:
    if config is None:
        config = BeamSearchConfig()
    if config.beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    directed_candidate_cache = {
        unit.unit_id: directed_unit_candidates(unit)
        for unit in units
    }
    beam: tuple[BeamSearchNode, ...] = (
        BeamSearchNode(
            directed_units=(),
            remaining_units=units,
            state=IncrementalMetricsState(current_point=tool.start_point),
        ),
    )
    expanded_nodes = 0
    diagnostics: list[BeamLayerDiagnostics] = []

    for depth in range(len(units)):
        next_nodes: list[tuple[tuple, BeamSearchNode]] = []
        input_beam_count = len(beam)
        (
            expansions,
            layer_pruned_count,
            parent_quota_added_count,
            parent_quota_pruned_count,
            effective_layer_expansion_limit,
            unstable_input_prefix_count,
        ) = _limited_beam_expansions(
            beam,
            directed_candidate_cache,
            process_model,
            config,
        )
        raw_expansion_count = len(expansions) + layer_pruned_count
        for expansion in expansions:
            candidate = expansion.candidate
            node = expansion.parent
            next_state = apply_candidate_incremental(
                candidate,
                node.state,
                panel,
                tool,
                process_model=process_model,
            )
            remaining = tuple(
                unit
                for unit in node.remaining_units
                if unit.unit_id != candidate.unit.unit_id
            )
            next_node = BeamSearchNode(
                directed_units=node.directed_units + (candidate,),
                remaining_units=remaining,
                state=next_state,
                previous_unit=candidate.unit,
            )
            next_nodes.append(
                (
                    _beam_node_rank(
                        next_node,
                        transition_cost=expansion.transition_cost,
                        is_reversed=candidate.is_reversed,
                    ),
                    next_node,
                )
            )
            expanded_nodes += 1

        if not next_nodes:
            diagnostics.append(
                _beam_layer_diagnostics(
                    depth=depth,
                    input_beam_count=input_beam_count,
                    unstable_input_prefix_count=unstable_input_prefix_count,
                    raw_expansion_count=raw_expansion_count,
                    effective_layer_expansion_limit=effective_layer_expansion_limit,
                    layer_expansion_count=len(expansions),
                    layer_pruned_count=layer_pruned_count,
                    parent_quota_added_count=parent_quota_added_count,
                    parent_quota_pruned_count=parent_quota_pruned_count,
                    evaluated_node_count=0,
                    duplicate_pruned_count=0,
                    diversity_pruned_count=0,
                    fallback_added_count=0,
                    output_beam=(),
                )
            )
            break

        deduplicated: dict[tuple, tuple[tuple, BeamSearchNode]] = {}
        for rank, node in next_nodes:
            signature = _beam_node_signature(node)
            incumbent = deduplicated.get(signature)
            if incumbent is None or rank < incumbent[0]:
                deduplicated[signature] = (rank, node)

        duplicate_pruned_count = len(next_nodes) - len(deduplicated)
        beam, diversity_pruned_count, fallback_added_count = _select_beam_nodes(
            list(deduplicated.values()),
            beam_width=config.beam_width,
            diversity_bucket_limit=config.diversity_bucket_limit,
            process_model=process_model,
        )
        diagnostics.append(
            _beam_layer_diagnostics(
                depth=depth,
                input_beam_count=input_beam_count,
                unstable_input_prefix_count=unstable_input_prefix_count,
                raw_expansion_count=raw_expansion_count,
                effective_layer_expansion_limit=effective_layer_expansion_limit,
                layer_expansion_count=len(expansions),
                layer_pruned_count=layer_pruned_count,
                parent_quota_added_count=parent_quota_added_count,
                parent_quota_pruned_count=parent_quota_pruned_count,
                evaluated_node_count=len(next_nodes),
                duplicate_pruned_count=duplicate_pruned_count,
                diversity_pruned_count=diversity_pruned_count,
                fallback_added_count=fallback_added_count,
                output_beam=beam,
            )
        )

    if not beam:
        actions: tuple[CuttingAction, ...] = ()
        return BeamSearchResult(
            directed_units=(),
            actions=actions,
            metrics=evaluate_actions(actions, panel, tool, process_model=process_model),
            expanded_nodes=expanded_nodes,
            diagnostics=tuple(diagnostics),
        )

    final_nodes = sorted(
        beam,
        key=lambda node: (
            len(node.remaining_units),
            process_metric_key(finalize_metrics(node.state)),
            tuple(candidate.unit.unit_id for candidate in node.directed_units),
        ),
    )
    best = final_nodes[0]
    actions = materialize_action_clearance(
        directed_units_to_actions(best.directed_units, start_point=tool.start_point),
        panel,
        tool,
        process_model=process_model,
    )
    return BeamSearchResult(
        directed_units=best.directed_units,
        actions=actions,
        metrics=finalize_metrics(best.state),
        expanded_nodes=expanded_nodes,
        diagnostics=tuple(diagnostics),
    )


def nearest_neighbor_unit_order(
    units: tuple[CuttingUnit, ...],
    start_point: Point,
) -> tuple[DirectedUnitCandidate, ...]:
    remaining = list(units)
    directed_candidate_cache = {
        unit.unit_id: directed_unit_candidates(unit)
        for unit in units
    }
    current_point = start_point
    ordered: list[DirectedUnitCandidate] = []

    while remaining:
        best_unit = min(
            remaining,
            key=lambda unit: min(
                euclidean_distance(current_point, unit.start),
                euclidean_distance(current_point, unit.end)
                if unit.is_reversible
                else float("inf"),
            ),
        )
        best = min(
            directed_candidate_cache[best_unit.unit_id],
            key=lambda candidate: euclidean_distance(current_point, candidate.start),
        )
        ordered.append(best)
        remaining.remove(best_unit)
        current_point = best.end

    return tuple(ordered)


def swap_neighbor_moves(
    directed_units: tuple[DirectedUnitCandidate, ...],
    max_span: int | None = None,
    max_neighbors: int | None = None,
) -> tuple[NeighborMove, ...]:
    neighbors: list[NeighborMove] = []
    for first in range(len(directed_units)):
        for second in range(first + 1, len(directed_units)):
            if max_span is not None and second - first > max_span:
                continue
            candidate = list(directed_units)
            candidate[first], candidate[second] = candidate[second], candidate[first]
            neighbors.append(
                NeighborMove(
                    tuple(candidate),
                    first,
                    min(len(directed_units), second + 2),
                )
            )
            if _reached_neighbor_limit(neighbors, max_neighbors):
                return tuple(neighbors)
    return tuple(neighbors)


def swap_neighbors(
    directed_units: tuple[DirectedUnitCandidate, ...],
    max_span: int | None = None,
    max_neighbors: int | None = None,
) -> tuple[tuple[DirectedUnitCandidate, ...], ...]:
    return tuple(
        move.directed_units
        for move in swap_neighbor_moves(directed_units, max_span, max_neighbors)
    )


def relocate_neighbor_moves(
    directed_units: tuple[DirectedUnitCandidate, ...],
    max_span: int | None = None,
    max_neighbors: int | None = None,
) -> tuple[NeighborMove, ...]:
    neighbors: list[NeighborMove] = []
    for source in range(len(directed_units)):
        for target in range(len(directed_units)):
            if source == target:
                continue
            if max_span is not None and abs(target - source) > max_span:
                continue
            candidate = list(directed_units)
            item = candidate.pop(source)
            candidate.insert(target, item)
            affected_start = min(source, target)
            affected_end = max(source, target)
            neighbors.append(
                NeighborMove(
                    tuple(candidate),
                    affected_start,
                    min(len(directed_units), affected_end + 2),
                )
            )
            if _reached_neighbor_limit(neighbors, max_neighbors):
                return tuple(neighbors)
    return tuple(neighbors)


def relocate_neighbors(
    directed_units: tuple[DirectedUnitCandidate, ...],
    max_span: int | None = None,
    max_neighbors: int | None = None,
) -> tuple[tuple[DirectedUnitCandidate, ...], ...]:
    return tuple(
        move.directed_units
        for move in relocate_neighbor_moves(directed_units, max_span, max_neighbors)
    )


def two_opt_neighbor_moves(
    directed_units: tuple[DirectedUnitCandidate, ...],
    max_span: int | None = None,
    max_neighbors: int | None = None,
) -> tuple[NeighborMove, ...]:
    neighbors: list[NeighborMove] = []
    for start in range(len(directed_units)):
        for end in range(start + 1, len(directed_units)):
            if max_span is not None and end - start > max_span:
                continue
            block = directed_units[start : end + 1]
            reversed_block: list[DirectedUnitCandidate] = []
            feasible = True
            for candidate in reversed(block):
                reversed_candidate = reverse_candidate(candidate)
                if reversed_candidate is None:
                    feasible = False
                    break
                reversed_block.append(reversed_candidate)
            if not feasible:
                continue
            neighbors.append(
                NeighborMove(
                    directed_units[:start]
                    + tuple(reversed_block)
                    + directed_units[end + 1 :],
                    start,
                    min(len(directed_units), end + 2),
                )
            )
            if _reached_neighbor_limit(neighbors, max_neighbors):
                return tuple(neighbors)
    return tuple(neighbors)


def two_opt_neighbors(
    directed_units: tuple[DirectedUnitCandidate, ...],
    max_span: int | None = None,
    max_neighbors: int | None = None,
) -> tuple[tuple[DirectedUnitCandidate, ...], ...]:
    return tuple(
        move.directed_units
        for move in two_opt_neighbor_moves(directed_units, max_span, max_neighbors)
    )


def local_neighbor_moves(
    directed_units: tuple[DirectedUnitCandidate, ...],
    config: LocalSearchConfig,
) -> tuple[NeighborMove, ...]:
    neighbors: list[NeighborMove] = []
    max_neighbors = config.max_neighbors_per_iteration
    if config.enable_swap:
        remaining = None if max_neighbors is None else max_neighbors - len(neighbors)
        if remaining is None or remaining > 0:
            neighbors.extend(
                swap_neighbor_moves(
                    directed_units,
                    max_span=config.max_swap_span,
                    max_neighbors=remaining,
                )
            )
    if config.enable_relocate:
        remaining = None if max_neighbors is None else max_neighbors - len(neighbors)
        if remaining is None or remaining > 0:
            neighbors.extend(
                relocate_neighbor_moves(
                    directed_units,
                    max_span=config.max_relocate_span,
                    max_neighbors=remaining,
                )
            )
    if config.enable_two_opt:
        remaining = None if max_neighbors is None else max_neighbors - len(neighbors)
        if remaining is None or remaining > 0:
            neighbors.extend(
                two_opt_neighbor_moves(
                    directed_units,
                    max_span=config.max_two_opt_span,
                    max_neighbors=remaining,
                )
            )
    return tuple(neighbors)


def local_neighbors(
    directed_units: tuple[DirectedUnitCandidate, ...],
    config: LocalSearchConfig,
) -> tuple[tuple[DirectedUnitCandidate, ...], ...]:
    return tuple(move.directed_units for move in local_neighbor_moves(directed_units, config))


def improve_directed_unit_order(
    directed_units: tuple[DirectedUnitCandidate, ...],
    panel: Panel,
    tool: ToolConfig,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> LocalSearchResult:
    if config is None:
        config = LocalSearchConfig()

    current = directed_units
    current_actions, current_metrics = evaluate_directed_units(
        current,
        panel,
        tool,
        process_model=process_model,
    )
    improved = False

    for iteration in range(config.max_iterations):
        prefix_states = build_prefix_states(
            current,
            panel,
            tool,
            process_model=process_model,
        )
        best_neighbor = current
        best_metrics = current_metrics
        best_key = process_metric_key(current_metrics)

        for move in local_neighbor_moves(current, config):
            neighbor_metrics = evaluate_neighbor_move_metrics(
                move,
                prefix_states,
                current_metrics,
                panel,
                tool,
                process_model=process_model,
            )
            neighbor_key = process_metric_key(neighbor_metrics)
            if neighbor_key < best_key:
                best_neighbor = move.directed_units
                best_metrics = neighbor_metrics
                best_key = neighbor_key
                if config.first_improvement:
                    break

        if best_neighbor == current:
            return LocalSearchResult(
                directed_units=current,
                actions=current_actions,
                metrics=current_metrics,
                iterations=iteration,
                improved=improved,
            )

        current = best_neighbor
        current_actions = materialize_action_clearance(
            directed_units_to_actions(current, start_point=tool.start_point),
            panel,
            tool,
            process_model=process_model,
        )
        current_metrics = best_metrics
        improved = True

    return LocalSearchResult(
        directed_units=current,
        actions=current_actions,
        metrics=current_metrics,
        iterations=config.max_iterations,
        improved=improved,
    )


def improve_directed_unit_order_by_path_distance(
    directed_units: tuple[DirectedUnitCandidate, ...],
    panel: Panel,
    tool: ToolConfig,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> LocalSearchResult:
    if config is None:
        config = LocalSearchConfig()

    current = directed_units
    current_actions, current_metrics = evaluate_directed_units(
        current,
        panel,
        tool,
        process_model=process_model,
    )
    improved = False

    for iteration in range(config.max_iterations):
        prefix_states = build_prefix_states(
            current,
            panel,
            tool,
            process_model=process_model,
        )
        best_neighbor = current
        best_metrics = current_metrics
        best_key = path_distance_metric_key(current_metrics)

        for move in local_neighbor_moves(current, config):
            neighbor_metrics = evaluate_neighbor_move_metrics(
                move,
                prefix_states,
                current_metrics,
                panel,
                tool,
                process_model=process_model,
            )
            neighbor_key = path_distance_metric_key(neighbor_metrics)
            if neighbor_key < best_key:
                best_neighbor = move.directed_units
                best_metrics = neighbor_metrics
                best_key = neighbor_key
                if config.first_improvement:
                    break

        if best_neighbor == current:
            return LocalSearchResult(
                directed_units=current,
                actions=current_actions,
                metrics=current_metrics,
                iterations=iteration,
                improved=improved,
            )

        current = best_neighbor
        current_actions = materialize_action_clearance(
            directed_units_to_actions(current, start_point=tool.start_point),
            panel,
            tool,
            process_model=process_model,
        )
        current_metrics = best_metrics
        improved = True

    return LocalSearchResult(
        directed_units=current,
        actions=current_actions,
        metrics=current_metrics,
        iterations=config.max_iterations,
        improved=improved,
    )


def path_distance_local_search_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> LocalSearchResult:
    if config is None:
        config = LocalSearchConfig()
    initial_order = nearest_neighbor_unit_order(units, tool.start_point)
    return improve_directed_unit_order_by_path_distance(
        initial_order,
        panel,
        tool,
        config=config,
        process_model=process_model,
    )


def topology_local_search_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> LocalSearchResult:
    if config is None:
        config = LocalSearchConfig()
    if process_model is not None and config.process_aware_initial_order:
        initial_order = process_aware_unit_order(
            units,
            panel=panel,
            tool=tool,
            process_model=process_model,
            candidate_pool_size=config.topology_candidate_pool_size,
        )
    else:
        initial_order = topology_aware_unit_order(
            units,
            start_point=tool.start_point,
            candidate_pool_size=config.topology_candidate_pool_size,
        )
    return improve_directed_unit_order(
        initial_order,
        panel,
        tool,
        config=config,
        process_model=process_model,
    )


def _unit_midpoint(unit: CuttingUnit) -> tuple[float, float]:
    return ((unit.start.x + unit.end.x) / 2.0, (unit.start.y + unit.end.y) / 2.0)


def _direct_units_in_sequence(
    units: tuple[CuttingUnit, ...],
    start_point: Point,
) -> tuple[DirectedUnitCandidate, ...]:
    directed_candidate_cache = {
        unit.unit_id: directed_unit_candidates(unit)
        for unit in units
    }
    current_point = start_point
    ordered: list[DirectedUnitCandidate] = []
    for unit in units:
        best = min(
            directed_candidate_cache[unit.unit_id],
            key=lambda candidate: (
                euclidean_distance(current_point, candidate.start),
                candidate.unit.unit_id,
                candidate.is_reversed,
            ),
        )
        ordered.append(best)
        current_point = best.end
    return tuple(ordered)


def sweep_unit_order(
    units: tuple[CuttingUnit, ...],
    start_point: Point,
    primary_axis: str = "x",
    reverse: bool = False,
) -> tuple[DirectedUnitCandidate, ...]:
    """Build a deterministic row/column sweep order used as a strong baseline seed."""

    if primary_axis not in {"x", "y"}:
        raise ValueError("primary_axis must be 'x' or 'y'")

    def key(unit: CuttingUnit) -> tuple[float, float, int, str]:
        mid_x, mid_y = _unit_midpoint(unit)
        if primary_axis == "x":
            return (mid_x, mid_y, -len(unit.covered_segment_ids), unit.unit_id)
        return (mid_y, mid_x, -len(unit.covered_segment_ids), unit.unit_id)

    ordered_units = tuple(sorted(units, key=key, reverse=reverse))
    return _direct_units_in_sequence(ordered_units, start_point)


def _order_signature(
    directed_units: tuple[DirectedUnitCandidate, ...],
) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (candidate.unit.unit_id, candidate.is_reversed)
        for candidate in directed_units
    )


def multistart_process_initial_orders(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    config: LocalSearchConfig,
    process_model: CuttingProcessModel | None = None,
) -> tuple[tuple[DirectedUnitCandidate, ...], ...]:
    """Construct literature-style deterministic seeds for process local search."""

    initial_orders: list[tuple[DirectedUnitCandidate, ...]] = [
        nearest_neighbor_unit_order(units, tool.start_point),
        topology_aware_unit_order(
            units,
            start_point=tool.start_point,
            candidate_pool_size=config.topology_candidate_pool_size,
        ),
        sweep_unit_order(units, tool.start_point, primary_axis="x"),
        sweep_unit_order(units, tool.start_point, primary_axis="y"),
        sweep_unit_order(units, tool.start_point, primary_axis="x", reverse=True),
        sweep_unit_order(units, tool.start_point, primary_axis="y", reverse=True),
    ]
    if process_model is not None:
        initial_orders.append(
            process_aware_unit_order(
                units,
                panel=panel,
                tool=tool,
                process_model=process_model,
                candidate_pool_size=config.topology_candidate_pool_size,
            )
        )

    unique_orders: list[tuple[DirectedUnitCandidate, ...]] = []
    seen: set[tuple[tuple[str, bool], ...]] = set()
    for order in initial_orders:
        signature = _order_signature(order)
        if signature in seen:
            continue
        seen.add(signature)
        unique_orders.append(order)
    return tuple(unique_orders)


def process_local_search_multistart_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    config: LocalSearchConfig | None = None,
    process_model: CuttingProcessModel | None = None,
) -> LocalSearchResult:
    """Strong process-aware local-search baseline from multiple deterministic seeds."""

    if config is None:
        config = LocalSearchConfig()

    best_result: LocalSearchResult | None = None
    total_iterations = 0
    any_improved = False
    for initial_order in multistart_process_initial_orders(
        units,
        panel=panel,
        tool=tool,
        config=config,
        process_model=process_model,
    ):
        result = improve_directed_unit_order(
            initial_order,
            panel,
            tool,
            config=config,
            process_model=process_model,
        )
        total_iterations += result.iterations
        any_improved = any_improved or result.improved
        if best_result is None or (
            process_metric_key(result.metrics),
            _order_signature(result.directed_units),
        ) < (
            process_metric_key(best_result.metrics),
            _order_signature(best_result.directed_units),
        ):
            best_result = result

    if best_result is None:
        actions: tuple[CuttingAction, ...] = ()
        return LocalSearchResult(
            directed_units=(),
            actions=actions,
            metrics=evaluate_actions(actions, panel, tool, process_model=process_model),
            iterations=0,
            improved=False,
        )

    return LocalSearchResult(
        directed_units=best_result.directed_units,
        actions=best_result.actions,
        metrics=best_result.metrics,
        iterations=total_iterations,
        improved=any_improved,
    )


def process_aware_beam_polished_search_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
    beam_config: BeamSearchConfig | None = None,
    polish_config: LocalSearchConfig | None = None,
) -> BeamSearchResult:
    """Run process-aware beam search and polish the result with process local search."""

    beam_result = process_aware_beam_search_order(
        units,
        panel,
        tool,
        process_model=process_model,
        config=beam_config,
    )
    if not beam_result.directed_units:
        return beam_result
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
    return BeamSearchResult(
        directed_units=polished.directed_units,
        actions=polished.actions,
        metrics=polished.metrics,
        expanded_nodes=beam_result.expanded_nodes,
        diagnostics=beam_result.diagnostics,
    )
