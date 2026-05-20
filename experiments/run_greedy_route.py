from __future__ import annotations

from pathlib import Path

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import load_layout
from cnc_cutting.models import Panel, ToolConfig
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_greedy_route,
    plan_local_search_route,
    plan_topology_route,
)
from cnc_cutting.process_model import build_process_model


ROOT = Path(__file__).resolve().parents[1]


def print_plan(label: str, plan: RoutePlan) -> None:
    print(f"[{label}]")
    print(f"selected_unit_count: {len(plan.selected_units)}")
    print(f"action_count: {len(plan.actions)}")
    print(f"air_move_distance: {plan.metrics.air_move_distance:.3f}")
    print(f"cutting_length: {plan.metrics.cutting_length:.3f}")
    print(f"pierce_count: {plan.metrics.pierce_count}")
    print(f"lift_count: {plan.metrics.lift_count}")
    print(f"safe_lift_count: {plan.metrics.safe_lift_count}")
    print(f"safe_lift_distance: {plan.metrics.safe_lift_distance:.3f}")
    print(f"detour_count: {plan.metrics.detour_count}")
    print(f"detour_distance: {plan.metrics.detour_distance:.3f}")
    print(f"travel_mode_cost: {plan.metrics.travel_mode_cost:.3f}")
    print(f"stability_penalty: {plan.metrics.stability_penalty:.3f}")
    print(f"hard_penalty: {plan.metrics.hard_penalty:.3f}")
    print(
        "feasibility_status: "
        + ("ok" if plan.metrics.hard_penalty == 0 else "violates boundary/collision constraints")
    )


def main() -> None:
    layout = load_layout(ROOT / "data" / "sample_layouts" / "demo_layout.json")
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    tool = ToolConfig(trim_margin=5, tool_diameter=6)
    process_model = build_process_model(layout)
    units = build_candidate_cutting_units(layout, tool, max_collinear_gap=tool.min_channel_width)
    greedy_plan = plan_greedy_route(units, panel, tool, process_model=process_model)
    topology_plan = plan_topology_route(units, panel, tool, process_model=process_model)
    local_search_plan = plan_local_search_route(
        units,
        panel,
        tool,
        process_model=process_model,
    )

    print(f"panel_id: {layout.panel_id}")
    print(f"rectangle_count: {len(layout.rectangles)}")
    print(f"candidate_unit_count: {len(units)}")
    print_plan("greedy", greedy_plan)
    print_plan("topology", topology_plan)
    print_plan("topology_local_search", local_search_plan)


if __name__ == "__main__":
    main()
