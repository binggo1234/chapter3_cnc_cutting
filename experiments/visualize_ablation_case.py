from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.diagnostics import diagnose_actions
from cnc_cutting.io import (
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.models import Panel
from cnc_cutting.optimizer import RoutePlan
from process_options import add_stability_model_args, build_process_model_from_args
from run_ablation import (
    DEFAULT_ARCHIVES,
    ablation_spec,
    build_plan,
    evaluated_metrics_and_actions,
    planning_process_args,
    planning_tool,
    single_edge_units,
)
from visualize_route_comparison import (
    RouteRun,
    method_slug,
    plot_route_comparison,
    route_metrics_row,
    write_diagnostics,
    write_route_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS = ("full_process_aware_beam", "single_edges_only")


def load_selected_case(path: Path, selection_priority: str | None) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no selected cases in {path}")
    if selection_priority is None:
        return rows[0]
    for row in rows:
        if row["selection_priority"] == selection_priority:
            return row
    raise ValueError(f"selection priority not found: {selection_priority}")


def resolve_archive(archive_value: str | Path) -> Path:
    path = Path(archive_value)
    if path.exists():
        return path
    for archive in DEFAULT_ARCHIVES:
        if archive.name == archive_value:
            return archive
    raise ValueError(f"cannot resolve archive: {archive_value}")


def run_variant(
    variant: str,
    layout,
    units,
    panel: Panel,
    base_tool,
    evaluation_process_model,
    args: argparse.Namespace,
) -> RouteRun:
    spec = ablation_spec(variant)
    variant_units = single_edge_units(layout) if spec.use_single_edge_units_only else units
    plan_args = planning_process_args(args, spec)
    planning_process_model = build_process_model_from_args(layout, plan_args)
    plan_tool = planning_tool(base_tool, spec)
    eval_tool = plan_tool if spec.evaluate_with_planning_tool else base_tool
    start = perf_counter()
    plan = build_plan(
        spec,
        variant_units,
        panel,
        plan_tool,
        planning_process_model,
        len(layout.rectangles),
    )
    runtime_ms = (perf_counter() - start) * 1000.0
    eval_metrics, eval_actions = evaluated_metrics_and_actions(
        plan,
        panel,
        eval_tool,
        evaluation_process_model,
    )
    evaluated_plan = RoutePlan(
        selected_units=plan.selected_units,
        actions=eval_actions,
        metrics=eval_metrics,
    )
    diagnostics = diagnose_actions(
        eval_actions,
        panel,
        eval_tool,
        process_model=evaluation_process_model,
    )
    return RouteRun(
        method=variant,
        runtime_ms=runtime_ms,
        plan=evaluated_plan,
        diagnostics=diagnostics,
        unit_type_by_id={
            unit.unit_id: unit.unit_type.value for unit in evaluated_plan.selected_units
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-cases",
        type=Path,
        default=ROOT / "results" / "ablation_representative_cases.csv",
    )
    parser.add_argument("--selection-priority", default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--placements-member", default=None)
    parser.add_argument("--board-id", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=(
            "full_process_aware_beam",
            "single_edges_only",
            "no_stability_guidance",
            "no_adjacency_support_guidance",
            "topology_no_beam",
            "path_distance_baseline",
            "no_detour_operator",
            "no_safe_travel_modes",
        ),
    )
    parser.add_argument("--label-parts", action="store_true")
    parser.add_argument("--label-actions", action="store_true")
    parser.add_argument("--action-label-stride", type=int, default=20)
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "figures" / "ablation_case_comparison.png",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "results" / "ablation_case_metrics.csv",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=ROOT / "results" / "ablation_case_diagnostics",
    )
    add_stability_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_case = (
        {}
        if args.zip_path is not None and args.placements_member is not None and args.board_id is not None
        else load_selected_case(args.selected_cases, args.selection_priority)
    )
    archive = resolve_archive(args.zip_path or selected_case["archive"])
    placements_member = args.placements_member or selected_case["placements_member"]
    board_id = str(args.board_id or selected_case["board_id"])

    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    base_tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=placements_member,
        board_ids=(board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    evaluation_process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        base_tool,
        max_collinear_gap=base_tool.min_channel_width,
    )

    runs = tuple(
        run_variant(
            variant,
            layout,
            units,
            panel,
            base_tool,
            evaluation_process_model,
            args,
        )
        for variant in args.variants
    )
    metrics_rows = tuple(
        route_metrics_row(archive, placements_member, board_id, layout, run)
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

    print(f"archive: {archive}")
    print(f"placements_member: {placements_member}")
    print(f"board_id: {board_id}")
    print(f"variants: {', '.join(args.variants)}")
    print(f"figure: {args.figure_output}")
    print(f"metrics: {args.metrics_output}")
    print(f"diagnostics_dir: {args.diagnostics_dir}")
    for row in metrics_rows:
        print(
            f"{row.method:<34} machining={row.machining_cost:>10.3f} "
            f"travel={row.travel_mode_cost:>10.3f} "
            f"stability={row.stability_penalty:>5.1f} "
            f"hard={row.hard_penalty:>5.1f}"
        )


if __name__ == "__main__":
    main()
