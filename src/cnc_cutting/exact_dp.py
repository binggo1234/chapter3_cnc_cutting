from __future__ import annotations

from dataclasses import dataclass

from .local_search import (
    apply_candidate_incremental,
    process_metric_key,
    process_state_metric_key,
)
from .metrics import evaluate_actions, finalize_metrics, materialize_action_clearance
from .models import (
    CuttingAction,
    CuttingProcessModel,
    CuttingUnit,
    IncrementalMetricsState,
    Panel,
    PathMetrics,
    ToolConfig,
)
from .topology_operators import (
    DirectedUnitCandidate,
    directed_unit_candidates,
    directed_units_to_actions,
)


@dataclass(frozen=True)
class ExactDPConfig:
    max_units: int = 12


@dataclass(frozen=True)
class ExactDPNode:
    unit_mask: int
    directed_units: tuple[DirectedUnitCandidate, ...]
    state: IncrementalMetricsState


@dataclass(frozen=True)
class ExactDPResult:
    directed_units: tuple[DirectedUnitCandidate, ...]
    actions: tuple[CuttingAction, ...]
    metrics: PathMetrics
    expanded_nodes: int
    retained_states: int


def _candidate_signature(candidate: DirectedUnitCandidate) -> tuple[str, bool]:
    return (candidate.unit.unit_id, candidate.is_reversed)


def _node_state_signature(
    mask: int,
    candidate: DirectedUnitCandidate,
    state: IncrementalMetricsState,
) -> tuple:
    return (
        mask,
        _candidate_signature(candidate),
        state.current_point,
        state.current_direction,
        tuple(sorted(state.processed_segments)),
        tuple(sorted(state.released_part_ids)),
        tuple(sorted(state.unstable_part_ids)),
    )


def _sequence_signature(
    directed_units: tuple[DirectedUnitCandidate, ...],
) -> tuple[tuple[str, bool], ...]:
    return tuple(_candidate_signature(candidate) for candidate in directed_units)


def _node_rank(node: ExactDPNode) -> tuple:
    return (
        process_state_metric_key(node.state),
        _sequence_signature(node.directed_units),
    )


def exact_process_dp_order(
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool: ToolConfig,
    process_model: CuttingProcessModel | None = None,
    config: ExactDPConfig | None = None,
) -> ExactDPResult:
    """Solve a small cutting-unit ordering instance by dynamic programming.

    This routine is intentionally capped by ``ExactDPConfig.max_units`` and is
    meant for small optimality-gap experiments, not for production-scale boards.
    """

    if config is None:
        config = ExactDPConfig()
    if len(units) > config.max_units:
        raise ValueError(
            f"exact DP supports at most {config.max_units} units; got {len(units)}"
        )

    if not units:
        actions: tuple[CuttingAction, ...] = ()
        metrics = evaluate_actions(actions, panel, tool, process_model=process_model)
        return ExactDPResult(
            directed_units=(),
            actions=actions,
            metrics=metrics,
            expanded_nodes=0,
            retained_states=1,
        )

    directed_by_unit = tuple(directed_unit_candidates(unit) for unit in units)
    full_mask = (1 << len(units)) - 1
    layers: dict[tuple, ExactDPNode] = {
        (0,): ExactDPNode(
            unit_mask=0,
            directed_units=(),
            state=IncrementalMetricsState(current_point=tool.start_point),
        )
    }
    expanded_nodes = 0
    retained_state_count = 1

    for _ in range(len(units)):
        next_layers: dict[tuple, ExactDPNode] = {}
        for node in layers.values():
            for unit_index, directed_candidates in enumerate(directed_by_unit):
                unit_bit = 1 << unit_index
                if node.unit_mask & unit_bit:
                    continue
                next_mask = node.unit_mask | unit_bit
                for candidate in directed_candidates:
                    next_state = apply_candidate_incremental(
                        candidate,
                        node.state,
                        panel,
                        tool,
                        process_model=process_model,
                    )
                    next_node = ExactDPNode(
                        unit_mask=next_mask,
                        directed_units=node.directed_units + (candidate,),
                        state=next_state,
                    )
                    signature = _node_state_signature(
                        next_mask,
                        candidate,
                        next_state,
                    )
                    incumbent = next_layers.get(signature)
                    if incumbent is None or _node_rank(next_node) < _node_rank(incumbent):
                        next_layers[signature] = next_node
                    expanded_nodes += 1
        retained_state_count += len(next_layers)
        layers = next_layers

    complete_nodes = [
        node
        for node in layers.values()
        if node.unit_mask == full_mask and len(node.directed_units) == len(units)
    ]
    if not complete_nodes:
        actions = ()
        return ExactDPResult(
            directed_units=(),
            actions=actions,
            metrics=evaluate_actions(actions, panel, tool, process_model=process_model),
            expanded_nodes=expanded_nodes,
            retained_states=retained_state_count,
        )

    best = min(
        complete_nodes,
        key=lambda node: (
            process_metric_key(finalize_metrics(node.state)),
            _sequence_signature(node.directed_units),
        ),
    )
    actions = materialize_action_clearance(
        directed_units_to_actions(best.directed_units, start_point=tool.start_point),
        panel,
        tool,
        process_model=process_model,
    )
    return ExactDPResult(
        directed_units=best.directed_units,
        actions=actions,
        metrics=finalize_metrics(best.state),
        expanded_nodes=expanded_nodes,
        retained_states=retained_state_count,
    )
