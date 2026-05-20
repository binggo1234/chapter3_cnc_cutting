from cnc_cutting.cutting_graph import build_unit_cutting_graph
from cnc_cutting.models import (
    CuttingGraphEdgeKind,
    CuttingUnit,
    CuttingUnitType,
    EdgeRole,
    EdgeSegment,
    Point,
    ToolConfig,
)


def _unit(unit_id: str, start: Point, end: Point) -> CuttingUnit:
    segment = EdgeSegment(f"{unit_id}:edge", unit_id, EdgeRole.BOTTOM, start, end)
    return CuttingUnit(
        unit_id=unit_id,
        unit_type=CuttingUnitType.SINGLE_EDGE,
        segments=(segment,),
        start=start,
        end=end,
        covered_segment_ids=(segment.segment_id,),
    )


def test_build_graph_adds_forward_and_reverse_cut_edges() -> None:
    unit = _unit("u1", Point(0, 0), Point(10, 0))

    graph = build_unit_cutting_graph((unit,), ToolConfig(start_point=Point(0, 0)))

    cut_edges = [edge for edge in graph.edges if edge.edge_kind == CuttingGraphEdgeKind.CUT_EDGE]
    assert len(cut_edges) == 2
    assert {edge.action.cutting_unit_id for edge in cut_edges} == {"u1"}
    assert {edge.base_cost for edge in cut_edges} == {10.0}


def test_build_graph_adds_start_and_inter_unit_travel_edges() -> None:
    first = _unit("u1", Point(0, 0), Point(10, 0))
    second = _unit("u2", Point(20, 0), Point(30, 0))

    graph = build_unit_cutting_graph(
        (first, second),
        ToolConfig(start_point=Point(0, 0)),
    )

    start_edges = [
        edge for edge in graph.edges if edge.source == "tool:start"
    ]
    inter_unit_edges = [
        edge
        for edge in graph.edges
        if edge.edge_kind == CuttingGraphEdgeKind.TRAVEL_EDGE
        and edge.source != "tool:start"
    ]

    assert len(start_edges) == 4
    assert len(inter_unit_edges) == 8
    assert all(edge.source.split(":")[0] != edge.target.split(":")[0] for edge in inter_unit_edges)
