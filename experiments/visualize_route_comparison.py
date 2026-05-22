from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

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
from cnc_cutting.local_search import BeamSearchConfig, LocalSearchConfig
from cnc_cutting.models import (
    CuttingAction,
    CuttingActionType,
    CuttingUnit,
    CuttingUnitType,
    Layout,
    Panel,
)
from cnc_cutting.optimizer import (
    RoutePlan,
    plan_greedy_route,
    plan_local_search_route,
    plan_path_distance_local_search_route,
    plan_process_aware_beam_adaptive_polished_route,
    plan_process_aware_beam_adaptive_route,
    plan_process_aware_beam_polished_route,
    plan_process_local_search_multistart_route,
    plan_process_aware_beam_route,
    plan_topology_route,
)
from process_options import add_stability_model_args, build_process_model_from_args


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path(
    "/Users/binggo/Desktop/codex_handoff_20260502/data/"
    "review_overnight_20260422.zip"
)
DEFAULT_METHODS = (
    "greedy",
    "path_distance_local_search",
    "topology_process_aware",
    "process_local_search_multistart",
    "process_aware_beam",
    "process_aware_beam_adaptive",
    "process_aware_beam_polished",
    "process_aware_beam_adaptive_polished",
)
METHOD_LABELS = {
    "greedy": "Greedy",
    "path_distance_local_search": "Path-LS",
    "topology": "Topology",
    "topology_process_aware": "Process-aware",
    "process_local_search_multistart": "Multi-start process LS",
    "process_aware_beam": "Process-aware beam",
    "process_aware_beam_adaptive": "Adaptive beam",
    "process_aware_beam_polished": "Beam+process LS",
    "process_aware_beam_adaptive_polished": "Adaptive beam+LS",
    "topology_local_search": "Topology+LS",
    "topology_local_search_process_aware": "Process-aware+LS",
    "full_process_aware_beam": "Full",
    "single_edges_only": "Single edges",
    "no_stability_guidance": "No stability",
    "no_adjacency_support_guidance": "No adjacency",
    "topology_no_beam": "No beam",
    "path_distance_baseline": "Path-LS",
    "no_detour_operator": "No detour",
    "no_safe_travel_modes": "No safe travel",
}
UNIT_COLORS = {
    CuttingUnitType.SINGLE_EDGE.value: "#6B7280",
    CuttingUnitType.SHARED_EDGE.value: "#0072B2",
    CuttingUnitType.NEAR_SHARED_CHANNEL.value: "#009E73",
    CuttingUnitType.COLLINEAR_CHAIN.value: "#E69F00",
}


@dataclass(frozen=True)
class RouteRun:
    method: str
    runtime_ms: float
    plan: RoutePlan
    diagnostics: tuple[ActionDiagnostic, ...]
    unit_type_by_id: dict[str, str]


@dataclass(frozen=True)
class RouteMetricsRow:
    archive: str
    placements_member: str
    board_id: str
    method: str
    runtime_ms: float
    rectangle_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    cutting_length: float
    pierce_count: int
    lift_count: int
    collision_penalty: float
    boundary_penalty: float
    hard_penalty: float
    stability_penalty: float
    safe_lift_count: int
    safe_lift_distance: float
    detour_count: int
    detour_distance: float
    travel_mode_cost: float
    machining_cost: float


def compact_local_search_config(size: int, process_aware_initial_order: bool) -> LocalSearchConfig:
    if size <= 50:
        return LocalSearchConfig(
            max_iterations=2,
            max_swap_span=6,
            max_relocate_span=6,
            max_two_opt_span=6,
            max_neighbors_per_iteration=160,
            first_improvement=True,
            topology_candidate_pool_size=96,
            process_aware_initial_order=process_aware_initial_order,
        )
    return LocalSearchConfig(
        max_iterations=1,
        max_swap_span=5,
        max_relocate_span=5,
        max_two_opt_span=5,
        max_neighbors_per_iteration=120,
        first_improvement=True,
        topology_candidate_pool_size=64,
        process_aware_initial_order=process_aware_initial_order,
    )


def compact_beam_search_config(size: int) -> BeamSearchConfig:
    if size <= 20:
        return BeamSearchConfig(
            beam_width=8,
            candidate_pool_size=24,
            max_expansions_per_node=48,
            max_layer_expansions=56,
            diversity_bucket_limit=1,
            min_expansions_per_parent=0,
            unstable_min_expansions_per_parent=2,
            unstable_layer_expansion_multiplier=1.0,
        )
    if size <= 75:
        return BeamSearchConfig(
            beam_width=6,
            candidate_pool_size=18,
            max_expansions_per_node=36,
            max_layer_expansions=42,
            diversity_bucket_limit=1,
            min_expansions_per_parent=0,
            unstable_min_expansions_per_parent=2,
            unstable_layer_expansion_multiplier=1.0,
        )
    return BeamSearchConfig(
        beam_width=4,
        candidate_pool_size=12,
        max_expansions_per_node=24,
        max_layer_expansions=28,
        diversity_bucket_limit=1,
        min_expansions_per_parent=0,
        unstable_min_expansions_per_parent=2,
        unstable_layer_expansion_multiplier=1.0,
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
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool,
    process_model,
    size: int,
) -> RoutePlan:
    if method == "greedy":
        return plan_greedy_route(units, panel, tool, process_model=process_model)
    if method == "path_distance_local_search":
        return plan_path_distance_local_search_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, False),
            process_model=process_model,
        )
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
    if method == "process_local_search_multistart":
        return plan_process_local_search_multistart_route(
            units,
            panel,
            tool,
            config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    if method == "process_aware_beam":
        return plan_process_aware_beam_route(
            units,
            panel,
            tool,
            config=compact_beam_search_config(size),
            process_model=process_model,
        )
    if method == "process_aware_beam_adaptive":
        return plan_process_aware_beam_adaptive_route(
            units,
            panel,
            tool,
            beam_config=compact_beam_search_config(size),
            topology_candidate_pool_size=topology_pool_size(size),
            process_model=process_model,
        )
    if method == "process_aware_beam_polished":
        return plan_process_aware_beam_polished_route(
            units,
            panel,
            tool,
            beam_config=compact_beam_search_config(size),
            polish_config=compact_local_search_config(size, True),
            process_model=process_model,
        )
    if method == "process_aware_beam_adaptive_polished":
        return plan_process_aware_beam_adaptive_polished_route(
            units,
            panel,
            tool,
            beam_config=compact_beam_search_config(size),
            polish_config=compact_local_search_config(size, True),
            topology_candidate_pool_size=topology_pool_size(size),
            process_model=process_model,
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


def run_method(
    method: str,
    units: tuple[CuttingUnit, ...],
    panel: Panel,
    tool,
    process_model,
    size: int,
) -> RouteRun:
    start = perf_counter()
    plan = build_plan(method, units, panel, tool, process_model, size)
    runtime_ms = (perf_counter() - start) * 1000.0
    diagnostics = diagnose_actions(plan.actions, panel, tool, process_model=process_model)
    return RouteRun(
        method=method,
        runtime_ms=runtime_ms,
        plan=plan,
        diagnostics=diagnostics,
        unit_type_by_id={
            unit.unit_id: unit.unit_type.value for unit in plan.selected_units
        },
    )


def route_metrics_row(
    archive: Path,
    placements_member: str,
    board_id: str,
    layout: Layout,
    run: RouteRun,
) -> RouteMetricsRow:
    metrics = run.plan.metrics
    return RouteMetricsRow(
        archive=archive.name,
        placements_member=placements_member,
        board_id=board_id,
        method=run.method,
        runtime_ms=run.runtime_ms,
        rectangle_count=len(layout.rectangles),
        selected_unit_count=len(run.plan.selected_units),
        action_count=len(run.plan.actions),
        air_move_distance=metrics.air_move_distance,
        cutting_length=metrics.cutting_length,
        pierce_count=metrics.pierce_count,
        lift_count=metrics.lift_count,
        collision_penalty=metrics.collision_penalty,
        boundary_penalty=metrics.boundary_penalty,
        hard_penalty=metrics.hard_penalty,
        stability_penalty=metrics.stability_penalty,
        safe_lift_count=metrics.safe_lift_count,
        safe_lift_distance=metrics.safe_lift_distance,
        detour_count=metrics.detour_count,
        detour_distance=metrics.detour_distance,
        travel_mode_cost=metrics.travel_mode_cost,
        machining_cost=metrics.machining_cost,
    )


def write_route_metrics(rows: tuple[RouteMetricsRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_diagnostics(rows: tuple[ActionDiagnostic, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def method_slug(method: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in method).strip("_")


def draw_panel_base(
    ax: plt.Axes,
    layout: Layout,
    label_parts: bool,
) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            layout.panel_width,
            layout.panel_height,
            fill=False,
            edgecolor="#111827",
            linewidth=1.2,
        )
    )
    for rectangle in layout.rectangles:
        ax.add_patch(
            Rectangle(
                (rectangle.x, rectangle.y),
                rectangle.width,
                rectangle.height,
                facecolor="#F3F4F6",
                edgecolor="#9CA3AF",
                linewidth=0.55,
            )
        )
        if label_parts:
            ax.text(
                rectangle.center.x,
                rectangle.center.y,
                rectangle.part_id,
                ha="center",
                va="center",
                fontsize=5.5,
                color="#374151",
            )


def action_midpoint(action: CuttingAction) -> tuple[float, float]:
    return ((action.start.x + action.end.x) / 2.0, (action.start.y + action.end.y) / 2.0)


def route_line_style(
    action: CuttingAction,
    diagnostic: ActionDiagnostic,
    unit_type_by_id: dict[str, str],
) -> tuple[str, str, float, float]:
    if action.action_type == CuttingActionType.CUT:
        unit_type = unit_type_by_id.get(action.cutting_unit_id or "", "")
        return (UNIT_COLORS.get(unit_type, "#6B7280"), "-", 1.15, 0.82)
    if diagnostic.collision_penalty > 0 or diagnostic.boundary_penalty > 0:
        return ("#DC2626", "--", 2.0, 0.95)
    if diagnostic.safe_lift_count > 0 or diagnostic.travel_mode == "safe_lift":
        return ("#7C3AED", "--", 1.7, 0.9)
    if diagnostic.detour_count > 0 or diagnostic.travel_mode == "low_clearance_detour":
        return ("#0891B2", "--", 1.5, 0.88)
    return ("#F59E0B", "--", 0.75, 0.42)


def plot_single_route(
    ax: plt.Axes,
    layout: Layout,
    run: RouteRun,
    label_parts: bool,
    label_actions: bool,
    action_label_stride: int,
) -> None:
    draw_panel_base(ax, layout, label_parts=label_parts)
    diagnostic_by_index = {row.action_index: row for row in run.diagnostics}

    for index, action in enumerate(run.plan.actions):
        diagnostic = diagnostic_by_index[index]
        color, linestyle, linewidth, alpha = route_line_style(
            action,
            diagnostic,
            run.unit_type_by_id,
        )
        ax.plot(
            [action.start.x, action.end.x],
            [action.start.y, action.end.y],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        should_label = (
            label_actions
            and (index % action_label_stride == 0)
            or diagnostic.collision_penalty > 0
            or diagnostic.boundary_penalty > 0
            or diagnostic.safe_lift_count > 0
            or diagnostic.detour_count > 0
            or diagnostic.stability_penalty_delta > 0
        )
        if should_label:
            mid_x, mid_y = action_midpoint(action)
            ax.text(mid_x, mid_y, str(index), fontsize=5, color="#111827")
        if diagnostic.stability_penalty_delta > 0:
            mid_x, mid_y = action_midpoint(action)
            ax.scatter(
                [mid_x],
                [mid_y],
                marker="x",
                s=22,
                color="#BE123C",
                linewidths=1.2,
                zorder=5,
            )

    if run.plan.actions:
        first = run.plan.actions[0]
        last = run.plan.actions[-1]
        ax.scatter(
            [first.start.x],
            [first.start.y],
            marker="o",
            s=24,
            color="#16A34A",
            zorder=6,
        )
        ax.scatter(
            [last.end.x],
            [last.end.y],
            marker="s",
            s=20,
            color="#111827",
            zorder=6,
        )

    metrics = run.plan.metrics
    label = METHOD_LABELS.get(run.method, run.method)
    ax.set_title(
        (
            f"{label}: runtime={run.runtime_ms:.1f} ms, "
            f"cost={metrics.travel_mode_cost:.1f}, "
            f"machining={metrics.machining_cost:.1f}, "
            f"stability={metrics.stability_penalty:.0f}, "
            f"safe_lift={metrics.safe_lift_count}, detour={metrics.detour_count}"
        ),
        fontsize=9.5,
    )
    ax.set_aspect("equal", adjustable="box")
    padding = max(2.0, min(20.0, 0.04 * max(layout.panel_width, layout.panel_height)))
    ax.set_xlim(-padding, layout.panel_width + padding)
    ax.set_ylim(-padding, layout.panel_height + padding)
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.grid(True, linewidth=0.3, color="#E5E7EB")


def add_route_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([0], [0], color="#0072B2", linewidth=1.4, label="shared-edge cut"),
        plt.Line2D([0], [0], color="#009E73", linewidth=1.4, label="near-shared cut"),
        plt.Line2D([0], [0], color="#E69F00", linewidth=1.4, label="collinear chain"),
        plt.Line2D([0], [0], color="#6B7280", linewidth=1.4, label="single edge"),
        plt.Line2D([0], [0], color="#F59E0B", linestyle="--", label="travel"),
        plt.Line2D([0], [0], color="#0891B2", linestyle="--", label="detour"),
        plt.Line2D([0], [0], color="#7C3AED", linestyle="--", label="safe lift"),
        plt.Line2D([0], [0], color="#DC2626", linestyle="--", label="penalized travel"),
        plt.Line2D([0], [0], marker="x", color="#BE123C", linestyle="", label="stability event"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=5,
        frameon=False,
        fontsize=8,
    )


def plot_route_comparison(
    layout: Layout,
    runs: tuple[RouteRun, ...],
    output_path: Path,
    label_parts: bool,
    label_actions: bool,
    action_label_stride: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(3.2, 3.3 * len(runs))
    fig, axes = plt.subplots(len(runs), 1, figsize=(13.5, fig_height), dpi=220)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        plot_single_route(
            ax,
            layout,
            run,
            label_parts=label_parts,
            label_actions=label_actions,
            action_label_stride=action_label_stride,
        )
    add_route_legend(fig)
    fig.suptitle(
        f"Route comparison: rectangles={len(layout.rectangles)}, panel={layout.panel_width:.0f} x {layout.panel_height:.0f} mm",
        y=0.996,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    fig.savefig(output_path)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--placements-member", default=None)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--board-id", default="346")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=(
            "greedy",
            "path_distance_local_search",
            "topology",
            "topology_process_aware",
	            "process_local_search_multistart",
	            "process_aware_beam",
	            "process_aware_beam_adaptive",
	            "process_aware_beam_polished",
	            "process_aware_beam_adaptive_polished",
	            "topology_local_search",
	            "topology_local_search_process_aware",
	        ),
    )
    parser.add_argument("--label-parts", action="store_true")
    parser.add_argument("--label-actions", action="store_true")
    parser.add_argument("--action-label-stride", type=int, default=20)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "results" / "route_comparison_metrics.csv",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=ROOT / "results" / "route_comparison_diagnostics",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "figures" / "route_comparison.png",
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

    runs = tuple(
        run_method(method, units, panel, tool, process_model, len(layout.rectangles))
        for method in args.methods
    )
    metrics_rows = tuple(
        route_metrics_row(args.zip_path, placements_member, str(args.board_id), layout, run)
        for run in runs
    )
    write_route_metrics(metrics_rows, args.metrics_output)
    for run in runs:
        write_diagnostics(
            run.diagnostics,
            args.diagnostics_dir / f"{method_slug(run.method)}_diagnostics.csv",
        )
    plot_route_comparison(
        layout,
        runs,
        args.figure_output,
        label_parts=args.label_parts,
        label_actions=args.label_actions,
        action_label_stride=max(1, args.action_label_stride),
    )

    unit_counts = Counter(unit.unit_type.value for unit in units)
    print(f"archive: {args.zip_path}")
    print(f"placements_member: {placements_member}")
    print(f"board_id: {args.board_id}")
    print(f"rectangles: {len(layout.rectangles)}")
    print(f"candidate_units: {len(units)} {dict(sorted(unit_counts.items()))}")
    print(f"metrics: {args.metrics_output}")
    print(f"diagnostics_dir: {args.diagnostics_dir}")
    print(f"figure: {args.figure_output}")
    for row in metrics_rows:
        print(
            f"{row.method:<36} runtime={row.runtime_ms:>8.2f} ms "
            f"cost={row.travel_mode_cost:>10.2f} "
            f"stability={row.stability_penalty:>5.1f} "
            f"safe_lift={row.safe_lift_count:>3} "
            f"detour={row.detour_count:>3}"
        )


if __name__ == "__main__":
    main()
