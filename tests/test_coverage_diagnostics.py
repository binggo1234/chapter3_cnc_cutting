from cnc_cutting.coverage_diagnostics import diagnose_route_coverage
from cnc_cutting.models import (
    CuttingAction,
    CuttingActionType,
    CuttingUnit,
    CuttingUnitType,
    EdgeRole,
    EdgeSegment,
    Point,
)


def _segment(segment_id: str, x: float) -> EdgeSegment:
    part_id, role = segment_id.split(":")
    return EdgeSegment(
        segment_id=segment_id,
        part_id=part_id,
        role=EdgeRole(role),
        start=Point(x, 0.0),
        end=Point(x + 10.0, 0.0),
    )


def _unit(
    unit_id: str,
    segment_ids: tuple[str, ...],
    unit_type: CuttingUnitType = CuttingUnitType.SINGLE_EDGE,
) -> CuttingUnit:
    segments = tuple(
        _segment(segment_id, index * 20.0)
        for index, segment_id in enumerate(segment_ids)
    )
    return CuttingUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        segments=segments,
        start=segments[0].start,
        end=segments[-1].end,
        covered_segment_ids=segment_ids,
    )


def _cut(unit: CuttingUnit) -> CuttingAction:
    return CuttingAction(
        action_type=CuttingActionType.CUT,
        start=unit.start,
        end=unit.end,
        cutting_unit_id=unit.unit_id,
        covered_segment_ids=unit.covered_segment_ids,
    )


def test_coverage_diagnostics_reports_complete_single_edge_route() -> None:
    units = (
        _unit("A:bottom", ("A:bottom",)),
        _unit("B:bottom", ("B:bottom",)),
    )

    diagnostics = diagnose_route_coverage(
        candidate_units=units,
        selected_units=units,
        actions=tuple(_cut(unit) for unit in units),
    )

    assert diagnostics.original_segment_count == 2
    assert diagnostics.coverage_missing_segment_count == 0
    assert diagnostics.coverage_repeated_segment_count == 0
    assert diagnostics.cut_missing_segment_count == 0
    assert diagnostics.cut_repeated_segment_count == 0
    assert diagnostics.selected_single_edge_count == 2
    assert diagnostics.selected_composite_coverage_ratio == 0.0


def test_coverage_diagnostics_separates_partial_overlap_from_missing_coverage() -> None:
    singles = (
        _unit("A:bottom", ("A:bottom",)),
        _unit("B:bottom", ("B:bottom",)),
        _unit("C:bottom", ("C:bottom",)),
    )
    shared = _unit(
        "shared:A:B",
        ("A:bottom", "B:bottom"),
        CuttingUnitType.SHARED_EDGE,
    )
    chain = _unit(
        "chain:B:C",
        ("B:bottom", "C:bottom"),
        CuttingUnitType.COLLINEAR_CHAIN,
    )

    diagnostics = diagnose_route_coverage(
        candidate_units=singles + (shared, chain),
        selected_units=(shared, chain),
        actions=(_cut(shared), _cut(chain)),
    )

    assert diagnostics.coverage_covered_segment_count == 3
    assert diagnostics.coverage_missing_segment_count == 0
    assert diagnostics.coverage_repeated_segment_count == 1
    assert diagnostics.coverage_duplicate_segment_count == 1
    assert diagnostics.coverage_duplicate_segment_length == 10.0
    assert diagnostics.cut_repeated_segment_count == 1
    assert diagnostics.cut_duplicate_segment_length == 10.0
    assert diagnostics.partially_overlapping_selected_unit_count == 1
    assert diagnostics.partial_repeated_cut_action_count == 1
    assert diagnostics.redundant_cut_action_count == 0
    assert diagnostics.selected_shared_edge_count == 1
    assert diagnostics.selected_collinear_chain_count == 1
    assert diagnostics.selected_composite_unit_count == 2
    assert diagnostics.selected_composite_covered_segment_count == 3
    assert diagnostics.selected_composite_coverage_ratio == 1.0


def test_coverage_diagnostics_flags_redundant_selected_and_cut_actions() -> None:
    single_a = _unit("A:bottom", ("A:bottom",))
    single_b = _unit("B:bottom", ("B:bottom",))
    shared = _unit(
        "shared:A:B",
        ("A:bottom", "B:bottom"),
        CuttingUnitType.SHARED_EDGE,
    )

    diagnostics = diagnose_route_coverage(
        candidate_units=(single_a, single_b, shared),
        selected_units=(shared, single_a),
        actions=(_cut(shared), _cut(single_a)),
    )

    assert diagnostics.coverage_missing_segment_count == 0
    assert diagnostics.coverage_repeated_segment_count == 1
    assert diagnostics.redundant_selected_unit_count == 1
    assert diagnostics.cut_repeated_segment_count == 1
    assert diagnostics.redundant_cut_action_count == 1
