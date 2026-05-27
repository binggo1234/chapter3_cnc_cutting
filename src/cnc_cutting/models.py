from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class PlacedRectangle:
    """A rectangular part placed on a stock panel."""

    part_id: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_y(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class Panel:
    panel_id: str
    panel_width: float
    panel_height: float


@dataclass(frozen=True)
class Layout:
    """A complete nesting result for one stock panel."""

    panel_id: str
    panel_width: float
    panel_height: float
    rectangles: tuple[PlacedRectangle, ...]


@dataclass(frozen=True)
class CuttingConfig:
    """Basic CNC cutting assumptions used by path-cost estimation."""

    start_point: Point = Point(0.0, 0.0)
    return_to_start: bool = False


@dataclass(frozen=True)
class ToolConfig:
    """CNC process parameters used by feasibility checks and path metrics."""

    trim_margin: float = 5.0
    tool_diameter: float = 6.0
    safe_clearance: float = 0.0
    centerline_boundary_margin: float | None = None
    allow_safe_lift_over_released_parts: bool = False
    allow_low_clearance_detour: bool = False
    low_clearance_travel_weight: float = 1.0
    detour_travel_weight: float = 1.0
    safe_lift_travel_weight: float = 1.0
    safe_lift_fixed_cost: float = 250.0
    start_point: Point = Point(0.0, 0.0)
    return_to_start: bool = False

    @property
    def tool_radius(self) -> float:
        return self.tool_diameter / 2.0

    @property
    def min_channel_width(self) -> float:
        return self.tool_diameter + self.safe_clearance


class EdgeOrientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class EdgeRole(str, Enum):
    BOTTOM = "bottom"
    RIGHT = "right"
    TOP = "top"
    LEFT = "left"


class BoundaryRelationType(str, Enum):
    SHARED_EDGE = "shared_edge"
    NEAR_SHARED_EDGE = "near_shared_edge"
    COLLINEAR_EDGE = "collinear_edge"


class CuttingActionType(str, Enum):
    CUT = "cut"
    TRAVEL = "travel"


class TravelMode(str, Enum):
    LOW_CLEARANCE = "low_clearance"
    LOW_CLEARANCE_DETOUR = "low_clearance_detour"
    SAFE_LIFT = "safe_lift"


class CuttingVertexKind(str, Enum):
    ENTRY_POINT = "entry_point"
    EXIT_POINT = "exit_point"
    DIRECTED_SEGMENT_STATE = "directed_segment_state"
    CUTTING_UNIT_STATE = "cutting_unit_state"


class CuttingGraphEdgeKind(str, Enum):
    CUT_EDGE = "cut_edge"
    TRAVEL_EDGE = "travel_edge"
    TRANSITION_EDGE = "transition_edge"


class CuttingUnitType(str, Enum):
    SINGLE_EDGE = "single_edge"
    SHARED_EDGE = "shared_edge"
    NEAR_SHARED_CHANNEL = "near_shared_channel"
    COLLINEAR_CHAIN = "collinear_chain"


@dataclass(frozen=True)
class EdgeSegment:
    """An oriented-agnostic rectangle boundary segment."""

    segment_id: str
    part_id: str
    role: EdgeRole
    start: Point
    end: Point

    @property
    def orientation(self) -> EdgeOrientation:
        if abs(self.start.y - self.end.y) < 1e-9:
            return EdgeOrientation.HORIZONTAL
        return EdgeOrientation.VERTICAL

    @property
    def length(self) -> float:
        if self.orientation == EdgeOrientation.HORIZONTAL:
            return abs(self.end.x - self.start.x)
        return abs(self.end.y - self.start.y)


@dataclass(frozen=True)
class DirectedCutSegment:
    """A directed executable cut on an edge segment."""

    segment: EdgeSegment
    start: Point
    end: Point


@dataclass(frozen=True)
class BoundaryRelation:
    relation_type: BoundaryRelationType
    first: EdgeSegment
    second: EdgeSegment
    overlap_length: float
    gap: float
    axial_gap: float = 0.0
    perpendicular_gap: float = 0.0


@dataclass(frozen=True)
class CuttingUnit:
    """A candidate executable unit derived from one or more boundary segments."""

    unit_id: str
    unit_type: CuttingUnitType
    segments: tuple[EdgeSegment, ...]
    start: Point
    end: Point
    relation_types: tuple[BoundaryRelationType, ...] = ()
    covered_segment_ids: tuple[str, ...] = ()
    is_reversible: bool = True
    requires_bridge_cut: bool = False


@dataclass(frozen=True)
class CuttingAction:
    action_type: CuttingActionType
    start: Point
    end: Point
    segment_id: str | None = None
    cutting_unit_id: str | None = None
    covered_segment_ids: tuple[str, ...] = ()
    travel_mode: TravelMode = TravelMode.LOW_CLEARANCE


@dataclass(frozen=True)
class CuttingProcessModel:
    """Static process context used by dynamic release and stability checks."""

    part_segment_ids: dict[str, frozenset[str]]
    segment_part_ids: dict[str, str]
    part_polygons: dict[str, tuple[Point, ...]]
    support_segment_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    segment_lengths: dict[str, float] = field(default_factory=dict)
    part_areas: dict[str, float] = field(default_factory=dict)
    part_support_lengths: dict[str, float] = field(default_factory=dict)
    part_adjacency_support_lengths: dict[str, dict[str, float]] = field(default_factory=dict)
    adjacency_support_weight: float = 0.0
    min_remaining_support_count: int = 1
    min_remaining_support_length_ratio: float = 0.0
    min_area_normalized_support_length: float = 0.0


@dataclass(frozen=True)
class PathMetrics:
    air_move_distance: float = 0.0
    cutting_length: float = 0.0
    pierce_count: int = 0
    lift_count: int = 0
    turn_penalty: float = 0.0
    collision_penalty: float = 0.0
    boundary_penalty: float = 0.0
    stability_penalty: float = 0.0
    repeated_cut_segment_count: int = 0
    repeated_cut_length: float = 0.0
    redundant_cut_action_count: int = 0
    continuity_reward: float = 0.0
    safe_lift_count: int = 0
    safe_lift_distance: float = 0.0
    detour_count: int = 0
    detour_distance: float = 0.0
    travel_mode_cost: float = 0.0

    @property
    def hard_penalty(self) -> float:
        return self.collision_penalty + self.boundary_penalty

    @property
    def machining_cost(self) -> float:
        return self.cutting_length + self.travel_mode_cost


@dataclass
class IncrementalMetricsState:
    """Mutable state for local path-cost updates during prefix search."""

    current_point: Point
    current_direction: tuple[float, float] | None = None
    is_tool_down: bool = False
    current_cutting_unit_id: str | None = None
    processed_segments: set[str] = field(default_factory=set)
    released_part_ids: set[str] = field(default_factory=set)
    unstable_part_ids: set[str] = field(default_factory=set)
    processed_polygons: tuple[tuple[Point, ...], ...] = ()
    processed_polygon_bounds: tuple[tuple[float, float, float, float], ...] = ()
    metrics: PathMetrics = field(default_factory=PathMetrics)


@dataclass
class CuttingState:
    """Dynamic feasibility state separated from accumulated metric values."""

    current_point: Point
    current_direction: tuple[float, float] | None = None
    is_tool_down: bool = False
    processed_segments: set[str] = field(default_factory=set)
    released_parts: set[str] = field(default_factory=set)
    unstable_parts: set[str] = field(default_factory=set)
    processed_polygons: tuple[tuple[Point, ...], ...] = ()
    processed_polygon_bounds: tuple[tuple[float, float, float, float], ...] = ()
