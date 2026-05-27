from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import (
    CuttingAction,
    CuttingActionType,
    CuttingProcessModel,
    CuttingUnit,
    CuttingUnitType,
)


@dataclass(frozen=True)
class CoverageDiagnostics:
    original_segment_count: int
    coverage_covered_segment_count: int
    coverage_missing_segment_count: int
    coverage_repeated_segment_count: int
    coverage_duplicate_segment_count: int
    coverage_duplicate_segment_length: float
    cut_covered_segment_count: int
    cut_missing_segment_count: int
    cut_repeated_segment_count: int
    cut_duplicate_segment_count: int
    cut_duplicate_segment_length: float
    selected_single_edge_count: int
    selected_shared_edge_count: int
    selected_near_shared_channel_count: int
    selected_collinear_chain_count: int
    selected_composite_unit_count: int
    selected_composite_covered_segment_count: int
    selected_composite_coverage_ratio: float
    partially_overlapping_selected_unit_count: int
    redundant_selected_unit_count: int
    partial_repeated_cut_action_count: int
    redundant_cut_action_count: int


def diagnose_route_coverage(
    candidate_units: tuple[CuttingUnit, ...],
    selected_units: tuple[CuttingUnit, ...],
    actions: tuple[CuttingAction, ...],
    process_model: CuttingProcessModel | None = None,
) -> CoverageDiagnostics:
    original_segment_ids = _original_segment_ids(candidate_units, process_model)
    segment_lengths = _segment_lengths(candidate_units, process_model)
    selected_counts = _unit_coverage_counts(selected_units, original_segment_ids)
    cut_counts = _cut_coverage_counts(actions, original_segment_ids)
    selected_type_counts = Counter(unit.unit_type for unit in selected_units)
    composite_segment_ids = {
        segment_id
        for unit in selected_units
        if unit.unit_type != CuttingUnitType.SINGLE_EDGE
        for segment_id in _filter_segment_ids(unit.covered_segment_ids, original_segment_ids)
    }
    selected_overlap = _unit_overlap_counts(selected_units, original_segment_ids)
    cut_overlap = _cut_overlap_counts(actions, original_segment_ids)
    original_count = len(original_segment_ids)

    return CoverageDiagnostics(
        original_segment_count=original_count,
        coverage_covered_segment_count=len(set(selected_counts)),
        coverage_missing_segment_count=len(original_segment_ids - set(selected_counts)),
        coverage_repeated_segment_count=_repeated_segment_count(selected_counts),
        coverage_duplicate_segment_count=_duplicate_segment_count(selected_counts),
        coverage_duplicate_segment_length=_duplicate_segment_length(
            selected_counts,
            segment_lengths,
        ),
        cut_covered_segment_count=len(set(cut_counts)),
        cut_missing_segment_count=len(original_segment_ids - set(cut_counts)),
        cut_repeated_segment_count=_repeated_segment_count(cut_counts),
        cut_duplicate_segment_count=_duplicate_segment_count(cut_counts),
        cut_duplicate_segment_length=_duplicate_segment_length(cut_counts, segment_lengths),
        selected_single_edge_count=selected_type_counts[CuttingUnitType.SINGLE_EDGE],
        selected_shared_edge_count=selected_type_counts[CuttingUnitType.SHARED_EDGE],
        selected_near_shared_channel_count=selected_type_counts[
            CuttingUnitType.NEAR_SHARED_CHANNEL
        ],
        selected_collinear_chain_count=selected_type_counts[
            CuttingUnitType.COLLINEAR_CHAIN
        ],
        selected_composite_unit_count=sum(
            count
            for unit_type, count in selected_type_counts.items()
            if unit_type != CuttingUnitType.SINGLE_EDGE
        ),
        selected_composite_covered_segment_count=len(composite_segment_ids),
        selected_composite_coverage_ratio=(
            len(composite_segment_ids) / original_count if original_count else 0.0
        ),
        partially_overlapping_selected_unit_count=selected_overlap[0],
        redundant_selected_unit_count=selected_overlap[1],
        partial_repeated_cut_action_count=cut_overlap[0],
        redundant_cut_action_count=cut_overlap[1],
    )


def _original_segment_ids(
    candidate_units: tuple[CuttingUnit, ...],
    process_model: CuttingProcessModel | None,
) -> frozenset[str]:
    if process_model is not None and process_model.segment_part_ids:
        return frozenset(process_model.segment_part_ids)

    single_edge_ids = {
        segment_id
        for unit in candidate_units
        if unit.unit_type == CuttingUnitType.SINGLE_EDGE
        for segment_id in unit.covered_segment_ids
    }
    if single_edge_ids:
        return frozenset(single_edge_ids)

    return frozenset(
        segment_id
        for unit in candidate_units
        for segment_id in unit.covered_segment_ids
    )


def _segment_lengths(
    candidate_units: tuple[CuttingUnit, ...],
    process_model: CuttingProcessModel | None,
) -> dict[str, float]:
    lengths: dict[str, float] = {}
    if process_model is not None:
        lengths.update(process_model.segment_lengths)
    for unit in candidate_units:
        for segment in unit.segments:
            lengths.setdefault(segment.segment_id, segment.length)
    return lengths


def _filter_segment_ids(
    segment_ids: tuple[str, ...],
    original_segment_ids: frozenset[str],
) -> tuple[str, ...]:
    if not original_segment_ids:
        return segment_ids
    return tuple(segment_id for segment_id in segment_ids if segment_id in original_segment_ids)


def _unit_coverage_counts(
    units: tuple[CuttingUnit, ...],
    original_segment_ids: frozenset[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for unit in units:
        counts.update(_filter_segment_ids(unit.covered_segment_ids, original_segment_ids))
    return counts


def _cut_coverage_counts(
    actions: tuple[CuttingAction, ...],
    original_segment_ids: frozenset[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for action in actions:
        if action.action_type != CuttingActionType.CUT:
            continue
        counts.update(_filter_segment_ids(_action_segment_ids(action), original_segment_ids))
    return counts


def _unit_overlap_counts(
    units: tuple[CuttingUnit, ...],
    original_segment_ids: frozenset[str],
) -> tuple[int, int]:
    seen: set[str] = set()
    partial_overlap_count = 0
    redundant_count = 0
    for unit in units:
        segment_ids = set(_filter_segment_ids(unit.covered_segment_ids, original_segment_ids))
        if not segment_ids:
            continue
        overlap = segment_ids & seen
        new_segments = segment_ids - seen
        if overlap and new_segments:
            partial_overlap_count += 1
        elif overlap and not new_segments:
            redundant_count += 1
        seen.update(segment_ids)
    return partial_overlap_count, redundant_count


def _cut_overlap_counts(
    actions: tuple[CuttingAction, ...],
    original_segment_ids: frozenset[str],
) -> tuple[int, int]:
    seen: set[str] = set()
    partial_repeat_count = 0
    redundant_count = 0
    for action in actions:
        if action.action_type != CuttingActionType.CUT:
            continue
        segment_ids = set(_filter_segment_ids(_action_segment_ids(action), original_segment_ids))
        if not segment_ids:
            continue
        overlap = segment_ids & seen
        new_segments = segment_ids - seen
        if overlap and new_segments:
            partial_repeat_count += 1
        elif overlap and not new_segments:
            redundant_count += 1
        seen.update(segment_ids)
    return partial_repeat_count, redundant_count


def _action_segment_ids(action: CuttingAction) -> tuple[str, ...]:
    if action.covered_segment_ids:
        return action.covered_segment_ids
    if action.segment_id is not None:
        return (action.segment_id,)
    return ()


def _repeated_segment_count(counts: Counter[str]) -> int:
    return sum(1 for count in counts.values() if count > 1)


def _duplicate_segment_count(counts: Counter[str]) -> int:
    return sum(count - 1 for count in counts.values() if count > 1)


def _duplicate_segment_length(
    counts: Counter[str],
    segment_lengths: dict[str, float],
) -> float:
    return sum(
        (count - 1) * segment_lengths.get(segment_id, 0.0)
        for segment_id, count in counts.items()
        if count > 1
    )
