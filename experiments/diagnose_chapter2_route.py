from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.diagnostics import ActionDiagnostic, diagnose_actions
from cnc_cutting.io import (
    discover_chapter2_placement_members,
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import LocalSearchConfig
from cnc_cutting.models import CuttingActionType, Layout, Panel
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_greedy_route,
    plan_local_search_route,
    plan_topology_route,
)
from process_options import add_stability_model_args, build_process_model_from_args


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path(
    "/Users/binggo/Desktop/codex_handoff_20260502/data/"
    "strong_baseline_overnight_20260407_001243.zip"
)


def compact_local_search_config(size: int, process_aware_initial_order: bool) -> LocalSearchConfig:
    return LocalSearchConfig(
        max_iterations=1 if size > 50 else 2,
        max_swap_span=5 if size > 50 else 6,
        max_relocate_span=5 if size > 50 else 6,
        max_two_opt_span=5 if size > 50 else 6,
        max_neighbors_per_iteration=120 if size > 50 else 160,
        first_improvement=True,
        topology_candidate_pool_size=64 if size > 50 else 96,
        process_aware_initial_order=process_aware_initial_order,
    )


def topology_pool_size(size: int) -> int:
    if size <= 50:
        return 96
    if size <= 200:
        return 64
    return 48


def resolve_placements_member(
    zip_path: Path,
    placements_member: str | None,
    member_index: int,
) -> str:
    if placements_member is not None:
        return placements_member
    members = discover_chapter2_placement_members(zip_path)
    if not members:
        raise ValueError(f"no placements.csv found in {zip_path}")
    if member_index < 0 or member_index >= len(members):
        raise ValueError(f"member_index out of range: {member_index}; available={len(members)}")
    return members[member_index]


def build_plan(
    method: str,
    units,
    panel: Panel,
    tool,
    process_model,
    size: int,
) -> RoutePlan:
    if method == "greedy":
        return plan_greedy_route(units, panel, tool, process_model=process_model)
    if method == "topology":
        return plan_topology_route(
            units,
            panel,
            tool,
            process_model=process_model,
            candidate_pool_size=topology_pool_size(size),
        )
    if method == "topology_process_aware":
        return plan_topology_route(
            units,
            panel,
            tool,
            process_model=process_model,
            candidate_pool_size=topology_pool_size(size),
            process_aware=True,
        )
    if method == "topology_local_search":
        return plan_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, False),
            process_model=process_model,
        )
    if method == "topology_local_search_process_aware":
        return plan_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    raise ValueError(f"unsupported method: {method}")


def write_diagnostics(rows: tuple[ActionDiagnostic, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_route(
    layout: Layout,
    plan: RoutePlan,
    diagnostics: tuple[ActionDiagnostic, ...],
    output_path: Path,
    label_parts: bool = False,
    max_action_labels: int = 80,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    ax.add_patch(
        Rectangle(
            (0, 0),
            layout.panel_width,
            layout.panel_height,
            fill=False,
            edgecolor="#111827",
            linewidth=1.4,
        )
    )

    for rectangle in layout.rectangles:
        ax.add_patch(
            Rectangle(
                (rectangle.x, rectangle.y),
                rectangle.width,
                rectangle.height,
                facecolor="#f3f4f6",
                edgecolor="#6b7280",
                linewidth=0.6,
            )
        )
        if label_parts:
            ax.text(
                rectangle.center.x,
                rectangle.center.y,
                rectangle.part_id,
                ha="center",
                va="center",
                fontsize=6,
                color="#374151",
            )

    collision_indices = {
        row.action_index
        for row in diagnostics
        if row.collision_penalty > 0 or row.boundary_penalty > 0
    }
    safe_lift_indices = {
        row.action_index
        for row in diagnostics
        if row.safe_lift_count > 0 or row.travel_mode == "safe_lift"
    }
    detour_indices = {
        row.action_index
        for row in diagnostics
        if row.detour_count > 0 or row.travel_mode == "low_clearance_detour"
    }
    for index, action in enumerate(plan.actions):
        if action.action_type == CuttingActionType.CUT:
            color = "#2563eb"
            linewidth = 1.1
            linestyle = "-"
            alpha = 0.75
        elif index in collision_indices:
            color = "#dc2626"
            linewidth = 2.2
            linestyle = "--"
            alpha = 0.95
        elif index in safe_lift_indices:
            color = "#7c3aed"
            linewidth = 2.0
            linestyle = "--"
            alpha = 0.9
        elif index in detour_indices:
            color = "#0891b2"
            linewidth = 1.6
            linestyle = "--"
            alpha = 0.85
        else:
            color = "#f59e0b"
            linewidth = 0.8
            linestyle = "--"
            alpha = 0.45

        ax.plot(
            [action.start.x, action.end.x],
            [action.start.y, action.end.y],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        if (
            index in collision_indices
            or index in safe_lift_indices
            or index in detour_indices
            or index < max_action_labels
        ):
            mid_x = (action.start.x + action.end.x) / 2.0
            mid_y = (action.start.y + action.end.y) / 2.0
            ax.text(
                mid_x,
                mid_y,
                str(index),
                fontsize=5,
                color=(
                    "#991b1b"
                    if index in collision_indices
                    else "#5b21b6"
                    if index in safe_lift_indices
                    else "#155e75"
                    if index in detour_indices
                    else "#1f2937"
                ),
            )

    ax.scatter(
        [plan.actions[0].start.x if plan.actions else 0],
        [plan.actions[0].start.y if plan.actions else 0],
        marker="o",
        s=30,
        color="#16a34a",
        zorder=5,
        label="start",
    )

    summary = (
        f"rectangles={len(layout.rectangles)}, actions={len(plan.actions)}, "
        f"air={plan.metrics.air_move_distance:.1f}, "
        f"collision={plan.metrics.collision_penalty:.0f}, "
        f"boundary={plan.metrics.boundary_penalty:.0f}, "
        f"stability={plan.metrics.stability_penalty:.0f}, "
        f"safe_lift={plan.metrics.safe_lift_count}, "
        f"detour={plan.metrics.detour_count}, "
        f"mode_cost={plan.metrics.travel_mode_cost:.1f}"
    )
    ax.set_title(summary, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-20, layout.panel_width + 20)
    ax.set_ylim(-20, layout.panel_height + 20)
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.grid(True, linewidth=0.3, color="#e5e7eb")

    cut_proxy = plt.Line2D([0], [0], color="#2563eb", linewidth=1.4, label="cut")
    travel_proxy = plt.Line2D([0], [0], color="#f59e0b", linestyle="--", label="travel")
    collision_proxy = plt.Line2D(
        [0],
        [0],
        color="#dc2626",
        linestyle="--",
        linewidth=2.2,
        label="penalized travel",
    )
    safe_lift_proxy = plt.Line2D(
        [0],
        [0],
        color="#7c3aed",
        linestyle="--",
        linewidth=2.0,
        label="safe-lift travel",
    )
    detour_proxy = plt.Line2D(
        [0],
        [0],
        color="#0891b2",
        linestyle="--",
        linewidth=1.6,
        label="low-clearance detour",
    )
    ax.legend(
        handles=[cut_proxy, travel_proxy, detour_proxy, safe_lift_proxy, collision_proxy],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--placements-member", default=None)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--board-id", default="1")
    parser.add_argument(
        "--method",
        choices=(
            "greedy",
            "topology",
            "topology_process_aware",
            "topology_local_search",
            "topology_local_search_process_aware",
        ),
        default="topology_local_search_process_aware",
    )
    parser.add_argument("--label-parts", action="store_true")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "results" / "chapter2_route_diagnostics.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "figures" / "chapter2_route_diagnostics.png",
    )
    add_stability_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    placements_member = resolve_placements_member(
        args.zip_path,
        args.placements_member,
        args.member_index,
    )
    cfg = load_chapter2_config_from_zip(args.zip_path, placements_member=placements_member)
    tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        args.zip_path,
        placements_member=placements_member,
        board_ids=(args.board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    plan = build_plan(
        args.method,
        units,
        panel,
        tool,
        process_model,
        len(layout.rectangles),
    )
    diagnostics = diagnose_actions(plan.actions, panel, tool, process_model=process_model)
    write_diagnostics(diagnostics, args.csv_output)
    plot_route(
        layout,
        plan,
        diagnostics,
        args.figure_output,
        label_parts=args.label_parts,
    )

    unit_counts = Counter(unit.unit_type.value for unit in units)
    penalized = [row for row in diagnostics if row.hard_penalty > 0]
    safe_lift = [row for row in diagnostics if row.safe_lift_count > 0]
    detour = [row for row in diagnostics if row.detour_count > 0]
    stability_events = [row for row in diagnostics if row.stability_penalty_delta > 0]
    unstable_events = [row for row in diagnostics if row.unstable_parts_after > 0]
    low_clearance_conflicts = [
        row for row in diagnostics if row.low_clearance_collision_penalty > 0
    ]
    print(f"archive: {args.zip_path}")
    print(f"placements_member: {placements_member}")
    print(f"board_id: {args.board_id}")
    print(f"method: {args.method}")
    print(
        "stability_model: "
        f"support_policy={args.support_policy}, "
        f"min_support_count={args.min_support_count}, "
        f"min_support_ratio={args.min_support_ratio:.3f}, "
        f"min_area_normalized_support={args.min_area_normalized_support:.3f}, "
        f"adjacency_support_weight={args.adjacency_support_weight:.3f}"
    )
    print(f"rectangles: {len(layout.rectangles)}")
    print(f"candidate_units: {len(units)} {dict(sorted(unit_counts.items()))}")
    print(f"actions: {len(plan.actions)}")
    print(f"penalized_actions: {len(penalized)}")
    print(f"low_clearance_conflicts: {len(low_clearance_conflicts)}")
    print(f"safe_lift_actions: {len(safe_lift)}")
    print(f"detour_actions: {len(detour)}")
    print(f"stability_penalty_actions: {len(stability_events)}")
    print(f"unstable_state_actions: {len(unstable_events)}")
    print(f"collision_penalty: {plan.metrics.collision_penalty:.3f}")
    print(f"boundary_penalty: {plan.metrics.boundary_penalty:.3f}")
    print(f"stability_penalty: {plan.metrics.stability_penalty:.3f}")
    print(f"safe_lift_count: {plan.metrics.safe_lift_count}")
    print(f"safe_lift_distance: {plan.metrics.safe_lift_distance:.3f}")
    print(f"detour_count: {plan.metrics.detour_count}")
    print(f"detour_distance: {plan.metrics.detour_distance:.3f}")
    print(f"travel_mode_cost: {plan.metrics.travel_mode_cost:.3f}")
    print(f"csv: {args.csv_output}")
    print(f"figure: {args.figure_output}")


if __name__ == "__main__":
    main()
