from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest

from .geometry import direction_vector, euclidean_distance
from .metrics import turn_penalty
from .models import (
    CuttingAction,
    CuttingActionType,
    CuttingUnit,
    CuttingUnitType,
    Point,
)


@dataclass(frozen=True)
class TopologyWeights:
    travel_distance: float = 1.0
    turn_penalty: float = 30.0
    same_part_bonus: float = 25.0
    relation_unit_bonus: float = 20.0
    multi_segment_bonus: float = 15.0
    bridge_cut_penalty: float = 1000.0


@dataclass(frozen=True)
class DirectedUnitCandidate:
    unit: CuttingUnit
    start: Point
    end: Point
    direction: tuple[float, float]
    is_reversed: bool = False


def unit_part_ids(unit: CuttingUnit) -> tuple[str, ...]:
    return tuple(sorted({segment.part_id for segment in unit.segments}))


def unit_relation_strength(unit: CuttingUnit) -> int:
    strengths = {
        CuttingUnitType.SHARED_EDGE: 3,
        CuttingUnitType.NEAR_SHARED_CHANNEL: 2,
        CuttingUnitType.COLLINEAR_CHAIN: 1,
        CuttingUnitType.SINGLE_EDGE: 0,
    }
    return strengths[unit.unit_type]


def directed_unit_candidates(unit: CuttingUnit) -> tuple[DirectedUnitCandidate, ...]:
    forward = DirectedUnitCandidate(
        unit=unit,
        start=unit.start,
        end=unit.end,
        direction=direction_vector(unit.start, unit.end),
        is_reversed=False,
    )
    if not unit.is_reversible:
        return (forward,)
    return (
        forward,
        DirectedUnitCandidate(
            unit=unit,
            start=unit.end,
            end=unit.start,
            direction=direction_vector(unit.end, unit.start),
            is_reversed=True,
        ),
    )


def unit_distance_key(unit: CuttingUnit, current_point: Point) -> tuple[float, int, str]:
    return (
        min(
            euclidean_distance(current_point, unit.start),
            euclidean_distance(current_point, unit.end)
            if unit.is_reversible
            else float("inf"),
        ),
        -unit_relation_strength(unit),
        unit.unit_id,
    )


def nearest_candidate_units(
    units: list[CuttingUnit],
    current_point: Point,
    candidate_pool_size: int | None = None,
) -> tuple[CuttingUnit, ...]:
    if candidate_pool_size is None or candidate_pool_size >= len(units):
        return tuple(units)
    return tuple(
        nsmallest(
            candidate_pool_size,
            units,
            key=lambda unit: unit_distance_key(unit, current_point),
        )
    )


def transition_score(
    candidate: DirectedUnitCandidate,
    current_point: Point,
    previous_direction: tuple[float, float] | None,
    previous_unit: CuttingUnit | None = None,
    weights: TopologyWeights | None = None,
) -> float:
    if weights is None:
        weights = TopologyWeights()

    score = weights.travel_distance * euclidean_distance(current_point, candidate.start)
    score += weights.turn_penalty * turn_penalty(previous_direction, candidate.direction)
    score -= weights.relation_unit_bonus * unit_relation_strength(candidate.unit)
    score -= weights.multi_segment_bonus * max(0, len(candidate.unit.covered_segment_ids) - 1)

    if previous_unit is not None:
        previous_parts = set(unit_part_ids(previous_unit))
        current_parts = set(unit_part_ids(candidate.unit))
        if previous_parts & current_parts:
            score -= weights.same_part_bonus

    if candidate.unit.requires_bridge_cut:
        score += weights.bridge_cut_penalty

    return score


def topology_aware_unit_order(
    units: tuple[CuttingUnit, ...],
    start_point: Point,
    weights: TopologyWeights | None = None,
    candidate_pool_size: int | None = None,
) -> tuple[DirectedUnitCandidate, ...]:
    remaining = list(units)
    directed_candidate_cache = {
        unit.unit_id: directed_unit_candidates(unit)
        for unit in units
    }
    current_point = start_point
    previous_direction: tuple[float, float] | None = None
    previous_unit: CuttingUnit | None = None
    ordered: list[DirectedUnitCandidate] = []

    while remaining:
        candidate_units = nearest_candidate_units(
            remaining,
            current_point=current_point,
            candidate_pool_size=candidate_pool_size,
        )
        candidates = [
            candidate
            for unit in candidate_units
            for candidate in directed_candidate_cache[unit.unit_id]
        ]
        best = min(
            candidates,
            key=lambda candidate: (
                transition_score(
                    candidate,
                    current_point=current_point,
                    previous_direction=previous_direction,
                    previous_unit=previous_unit,
                    weights=weights,
                ),
                euclidean_distance(current_point, candidate.start),
                candidate.unit.unit_id,
                candidate.is_reversed,
            ),
        )
        ordered.append(best)
        remaining.remove(best.unit)
        current_point = best.end
        previous_direction = best.direction
        previous_unit = best.unit

    return tuple(ordered)


def unit_segment_id(unit: CuttingUnit) -> str | None:
    if len(unit.covered_segment_ids) == 1:
        return unit.covered_segment_ids[0]
    return None


def directed_units_to_actions(
    directed_units: tuple[DirectedUnitCandidate, ...],
    start_point: Point,
) -> tuple[CuttingAction, ...]:
    current_point = start_point
    actions: list[CuttingAction] = []

    for candidate in directed_units:
        if euclidean_distance(current_point, candidate.start) > 1e-9:
            actions.append(
                CuttingAction(
                    action_type=CuttingActionType.TRAVEL,
                    start=current_point,
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
        current_point = candidate.end

    return tuple(actions)


def topology_aware_unit_actions(
    units: tuple[CuttingUnit, ...],
    start_point: Point,
    weights: TopologyWeights | None = None,
    candidate_pool_size: int | None = None,
) -> tuple[CuttingAction, ...]:
    return directed_units_to_actions(
        topology_aware_unit_order(
            units,
            start_point=start_point,
            weights=weights,
            candidate_pool_size=candidate_pool_size,
        ),
        start_point=start_point,
    )
