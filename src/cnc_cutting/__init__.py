"""Utilities for CNC rectangular cutting path optimization."""

from .cutting_units import build_candidate_cutting_units
from .cutting_graph import build_unit_cutting_graph
from .layout_generator import (
    ClusteredLayoutConfig,
    SyntheticLayoutConfig,
    generate_clustered_channel_layout,
    generate_synthetic_layout,
)
from .local_search import LocalSearchConfig
from .models import (
    CuttingConfig,
    CuttingProcessModel,
    CuttingUnit,
    CuttingUnitType,
    Layout,
    Panel,
    PlacedRectangle,
    Point,
    ToolConfig,
)
from .optimizer import (
    plan_greedy_route,
    plan_local_search_route,
    plan_topology_route,
    select_coverage_units,
)
from .path_cost import nearest_neighbor_order, total_air_move_distance
from .process_model import build_process_model
from .topology_operators import TopologyWeights

__all__ = [
    "CuttingConfig",
    "CuttingProcessModel",
    "CuttingUnit",
    "CuttingUnitType",
    "ClusteredLayoutConfig",
    "Layout",
    "LocalSearchConfig",
    "Panel",
    "PlacedRectangle",
    "Point",
    "SyntheticLayoutConfig",
    "ToolConfig",
    "build_candidate_cutting_units",
    "build_unit_cutting_graph",
    "build_process_model",
    "generate_synthetic_layout",
    "generate_clustered_channel_layout",
    "nearest_neighbor_order",
    "plan_greedy_route",
    "plan_local_search_route",
    "plan_topology_route",
    "select_coverage_units",
    "TopologyWeights",
    "total_air_move_distance",
]
