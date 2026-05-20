from cnc_cutting.models import (
    CuttingUnit,
    CuttingUnitType,
    EdgeRole,
    EdgeSegment,
    Point,
)
from cnc_cutting.topology_operators import (
    DirectedUnitCandidate,
    nearest_candidate_units,
    topology_aware_unit_actions,
    topology_aware_unit_order,
    transition_score,
    unit_distance_key,
)


def _unit(
    unit_id: str,
    part_id: str,
    start: Point,
    end: Point,
    unit_type: CuttingUnitType = CuttingUnitType.SINGLE_EDGE,
) -> CuttingUnit:
    segment = EdgeSegment(f"{part_id}:{unit_id}", part_id, EdgeRole.BOTTOM, start, end)
    return CuttingUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        segments=(segment,),
        start=start,
        end=end,
        covered_segment_ids=(segment.segment_id,),
    )


def test_transition_score_prefers_same_part_when_distance_ties() -> None:
    previous = _unit("u0", "A", Point(0, 0), Point(5, 0))
    same_part = _unit("u1", "A", Point(10, 0), Point(20, 0))
    other_part = _unit("u2", "B", Point(10, 0), Point(20, 0))
    same_candidate = DirectedUnitCandidate(
        same_part,
        same_part.start,
        same_part.end,
        (1.0, 0.0),
    )
    other_candidate = DirectedUnitCandidate(
        other_part,
        other_part.start,
        other_part.end,
        (1.0, 0.0),
    )

    assert transition_score(
        same_candidate,
        current_point=Point(0, 0),
        previous_direction=(1.0, 0.0),
        previous_unit=previous,
    ) < transition_score(
        other_candidate,
        current_point=Point(0, 0),
        previous_direction=(1.0, 0.0),
        previous_unit=previous,
    )


def test_topology_order_prefers_relation_unit_when_distance_ties() -> None:
    single = _unit("single", "A", Point(10, 0), Point(20, 0))
    shared = _unit(
        "shared",
        "B",
        Point(10, 0),
        Point(20, 0),
        unit_type=CuttingUnitType.SHARED_EDGE,
    )

    order = topology_aware_unit_order((single, shared), Point(0, 0))

    assert order[0].unit.unit_id == "shared"


def test_topology_actions_preserve_unit_coverage_information() -> None:
    unit = _unit("u1", "A", Point(10, 0), Point(20, 0))

    actions = topology_aware_unit_actions((unit,), Point(0, 0))

    assert actions[0].start == Point(0, 0)
    assert actions[0].end == Point(10, 0)
    assert actions[1].cutting_unit_id == "u1"
    assert actions[1].covered_segment_ids == ("A:u1",)


def test_nearest_candidate_units_limits_pool_by_distance() -> None:
    near = _unit("near", "A", Point(10, 0), Point(20, 0))
    far = _unit("far", "B", Point(100, 0), Point(110, 0))
    middle = _unit("middle", "C", Point(30, 0), Point(40, 0))

    pool = nearest_candidate_units([far, middle, near], Point(0, 0), candidate_pool_size=2)

    assert [unit.unit_id for unit in pool] == ["near", "middle"]


def test_unit_distance_key_prefers_relation_strength_when_distance_ties() -> None:
    single = _unit("single", "A", Point(10, 0), Point(20, 0))
    shared = _unit(
        "shared",
        "B",
        Point(10, 0),
        Point(20, 0),
        unit_type=CuttingUnitType.SHARED_EDGE,
    )

    assert unit_distance_key(shared, Point(0, 0)) < unit_distance_key(single, Point(0, 0))
