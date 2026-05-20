from __future__ import annotations

from dataclasses import dataclass

from .geometry import direction_vector, euclidean_distance
from .models import (
    CuttingAction,
    CuttingGraphEdgeKind,
    CuttingActionType,
    CuttingUnit,
    CuttingVertexKind,
    Point,
    ToolConfig,
)


@dataclass(frozen=True)
class CuttingVertex:
    vertex_id: str
    vertex_kind: CuttingVertexKind
    point: Point
    segment_id: str | None = None
    cutting_unit_id: str | None = None
    direction: tuple[float, float] | None = None
    is_reversible: bool = True
    release_allowed: bool = True


@dataclass(frozen=True)
class CuttingGraphEdge:
    edge_id: str
    edge_kind: CuttingGraphEdgeKind
    source: str
    target: str
    action: CuttingAction
    base_cost: float = 0.0


@dataclass(frozen=True)
class CuttingGraph:
    """Directed heterogeneous cutting graph.

    Vertices represent candidate process states. Edges represent either actual
    cutting actions or non-cutting travel actions.
    """

    vertices: tuple[CuttingVertex, ...]
    edges: tuple[CuttingGraphEdge, ...]


def _directed_unit_specs(unit: CuttingUnit) -> tuple[tuple[str, Point, Point], ...]:
    forward = (f"{unit.unit_id}:forward", unit.start, unit.end)
    if not unit.is_reversible:
        return (forward,)
    return (
        forward,
        (f"{unit.unit_id}:reverse", unit.end, unit.start),
    )


def _unit_segment_id(unit: CuttingUnit) -> str | None:
    if len(unit.covered_segment_ids) == 1:
        return unit.covered_segment_ids[0]
    return None


def build_unit_cutting_graph(
    units: tuple[CuttingUnit, ...],
    tool: ToolConfig,
    include_start_edges: bool = True,
    include_travel_edges: bool = True,
) -> CuttingGraph:
    vertices: list[CuttingVertex] = []
    edges: list[CuttingGraphEdge] = []
    entry_vertices: list[CuttingVertex] = []
    exit_vertices: list[CuttingVertex] = []

    if include_start_edges:
        vertices.append(
            CuttingVertex(
                vertex_id="tool:start",
                vertex_kind=CuttingVertexKind.ENTRY_POINT,
                point=tool.start_point,
                is_reversible=False,
            )
        )

    for unit in units:
        for direction_id, start, end in _directed_unit_specs(unit):
            direction = direction_vector(start, end)
            entry = CuttingVertex(
                vertex_id=f"{direction_id}:entry",
                vertex_kind=CuttingVertexKind.ENTRY_POINT,
                point=start,
                cutting_unit_id=unit.unit_id,
                direction=direction,
                is_reversible=unit.is_reversible,
                release_allowed=False,
            )
            exit_vertex = CuttingVertex(
                vertex_id=f"{direction_id}:exit",
                vertex_kind=CuttingVertexKind.EXIT_POINT,
                point=end,
                cutting_unit_id=unit.unit_id,
                direction=direction,
                is_reversible=unit.is_reversible,
                release_allowed=True,
            )
            vertices.extend((entry, exit_vertex))
            entry_vertices.append(entry)
            exit_vertices.append(exit_vertex)
            edges.append(
                CuttingGraphEdge(
                    edge_id=f"cut:{direction_id}",
                    edge_kind=CuttingGraphEdgeKind.CUT_EDGE,
                    source=entry.vertex_id,
                    target=exit_vertex.vertex_id,
                    action=CuttingAction(
                        action_type=CuttingActionType.CUT,
                        start=start,
                        end=end,
                        segment_id=_unit_segment_id(unit),
                        cutting_unit_id=unit.unit_id,
                        covered_segment_ids=unit.covered_segment_ids,
                    ),
                    base_cost=euclidean_distance(start, end),
                )
            )

    if include_start_edges:
        for entry in entry_vertices:
            edges.append(
                CuttingGraphEdge(
                    edge_id=f"travel:tool:start->{entry.vertex_id}",
                    edge_kind=CuttingGraphEdgeKind.TRAVEL_EDGE,
                    source="tool:start",
                    target=entry.vertex_id,
                    action=CuttingAction(
                        action_type=CuttingActionType.TRAVEL,
                        start=tool.start_point,
                        end=entry.point,
                    ),
                    base_cost=euclidean_distance(tool.start_point, entry.point),
                )
            )

    if include_travel_edges:
        for source in exit_vertices:
            for target in entry_vertices:
                if source.cutting_unit_id == target.cutting_unit_id:
                    continue
                edges.append(
                    CuttingGraphEdge(
                        edge_id=f"travel:{source.vertex_id}->{target.vertex_id}",
                        edge_kind=CuttingGraphEdgeKind.TRAVEL_EDGE,
                        source=source.vertex_id,
                        target=target.vertex_id,
                        action=CuttingAction(
                            action_type=CuttingActionType.TRAVEL,
                            start=source.point,
                            end=target.point,
                        ),
                        base_cost=euclidean_distance(source.point, target.point),
                    )
                )

    return CuttingGraph(vertices=tuple(vertices), edges=tuple(edges))
