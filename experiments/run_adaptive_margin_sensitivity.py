from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cnc_cutting_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import (
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import improve_directed_unit_order, process_aware_beam_search_order
from cnc_cutting.local_search import process_metric_key
from cnc_cutting.models import Layout, Panel
from cnc_cutting.optimizer import RoutePlan, plan_topology_route
from cnc_cutting.optimizer import select_coverage_units, wider_beam_search_config
from cnc_cutting.travel import clear_detour_cache
from progress_log import (
    append_progress_event,
    default_progress_log_path,
    new_progress_event,
    prepare_progress_log,
)
from progress_bar import TerminalProgressBar
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)
from run_chapter2_batch import (
    DEFAULT_ARCHIVES,
    board_counts_from_zip,
    compact_beam_search_config,
    compact_local_search_config,
    parse_member_metadata,
    rectangle_count_bin,
    sample_placement_members,
    select_board_ids,
    topology_pool_size,
)
from task_timeout import TaskTimeoutError, task_timeout


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("adaptive", "adaptive_polished")


@dataclass(frozen=True)
class CandidatePlan:
    label: str
    plan: RoutePlan
    runtime_ms: float


@dataclass(frozen=True)
class MarginRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    variant: str
    fallback_margin: float
    selected_source: str
    fallback_triggered: bool
    estimated_runtime_ms: float
    topology_runtime_ms: float
    default_beam_runtime_ms: float
    default_polish_runtime_ms: float
    fallback_beam_runtime_ms: float
    fallback_polish_runtime_ms: float
    rectangle_count: int
    rectangle_count_bin: str
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    cutting_length: float
    pierce_count: int
    lift_count: int
    safe_lift_count: int
    detour_count: int
    travel_mode_cost: float
    hard_penalty: float
    stability_penalty: float


@dataclass(frozen=True)
class MarginSummaryRow:
    variant: str
    fallback_margin: float
    n: int
    travel_mode_cost_mean: float
    travel_mode_cost_std: float
    machining_cost_mean: float
    estimated_runtime_ms_mean: float
    hard_penalty_mean: float
    stability_penalty_mean: float
    fallback_trigger_rate_pct: float
    source_topology_count: int
    source_default_beam_count: int
    source_default_polished_count: int
    source_fallback_beam_count: int
    source_fallback_polished_count: int


def plan_from_result(selected_units, result) -> RoutePlan:
    return RoutePlan(
        selected_units=selected_units,
        actions=result.actions,
        metrics=result.metrics,
    )


def polish_plan(selected_units, beam_result, panel, tool, polish_config, process_model) -> RoutePlan:
    if not beam_result.directed_units:
        return plan_from_result(selected_units, beam_result)
    polished = improve_directed_unit_order(
        beam_result.directed_units,
        panel,
        tool,
        config=polish_config,
        process_model=process_model,
    )
    return plan_from_result(selected_units, polished)


def best_candidate(candidates: tuple[CandidatePlan, ...]) -> CandidatePlan:
    return min(candidates, key=lambda candidate: process_metric_key(candidate.plan.metrics))


def fallback_needed(candidate: CandidatePlan, topology: CandidatePlan, margin: float) -> bool:
    if process_metric_key(candidate.plan.metrics) >= process_metric_key(topology.plan.metrics):
        return True
    return (
        candidate.plan.metrics.travel_mode_cost - topology.plan.metrics.travel_mode_cost
    ) > -margin


def same_metrics(left: CandidatePlan, right: CandidatePlan, tolerance: float = 1e-6) -> bool:
    fields_to_match = (
        "travel_mode_cost",
        "machining_cost",
        "hard_penalty",
        "stability_penalty",
        "pierce_count",
        "lift_count",
        "safe_lift_count",
    )
    return all(
        abs(getattr(left.plan.metrics, field) - getattr(right.plan.metrics, field))
        <= tolerance
        for field in fields_to_match
    )


def preferred_label(selected: CandidatePlan, candidates: tuple[CandidatePlan, ...]) -> str:
    preference = (
        "fallback_polished",
        "fallback_beam",
        "default_polished",
        "default_beam",
        "topology",
    )
    by_label = {candidate.label: candidate for candidate in candidates}
    for label in preference:
        candidate = by_label.get(label)
        if candidate is not None and same_metrics(selected, candidate):
            return label
    return selected.label


def estimated_runtime(
    variant: str,
    fallback_triggered: bool,
    candidates: dict[str, CandidatePlan],
) -> float:
    total = candidates["topology"].runtime_ms + candidates["default_beam"].runtime_ms
    if variant == "adaptive_polished":
        total += candidates["default_polished"].runtime_ms
        if fallback_triggered:
            total += (
                candidates["fallback_beam"].runtime_ms
                + candidates["fallback_polished"].runtime_ms
            )
    elif fallback_triggered:
        total += candidates["fallback_beam"].runtime_ms
    return total


def candidate_for_variant(
    variant: str,
    margin: float,
    candidates: dict[str, CandidatePlan],
) -> tuple[CandidatePlan, bool, tuple[CandidatePlan, ...]]:
    if variant == "adaptive":
        base = (candidates["topology"], candidates["default_beam"])
        current_best = best_candidate(base)
        triggered = fallback_needed(current_best, candidates["topology"], margin)
        expanded = base + ((candidates["fallback_beam"],) if triggered else ())
        return best_candidate(expanded), triggered, expanded

    if variant == "adaptive_polished":
        base = (
            candidates["topology"],
            candidates["default_beam"],
            candidates["default_polished"],
        )
        current_best = best_candidate(base)
        triggered = fallback_needed(current_best, candidates["topology"], margin)
        expanded = base + ((candidates["fallback_polished"],) if triggered else ())
        return best_candidate(expanded), triggered, expanded

    raise ValueError(f"unsupported variant: {variant}")


def build_case_candidates(
    archive: Path,
    placements_member: str,
    board_id: str,
    args: argparse.Namespace,
) -> tuple[Layout, dict[str, CandidatePlan], int]:
    cfg = load_chapter2_config_from_zip(archive, placements_member=placements_member)
    tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=placements_member,
        board_ids=(board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    selected_units = select_coverage_units(units)
    size = len(layout.rectangles)
    beam_config = compact_beam_search_config(size)
    fallback_beam_config = wider_beam_search_config(beam_config)
    polish_config = compact_local_search_config(size, True)

    clear_detour_cache()
    start = perf_counter()
    topology_plan = plan_topology_route(
        units,
        panel,
        tool,
        candidate_pool_size=topology_pool_size(size),
        process_aware=True,
        process_model=process_model,
    )
    topology = CandidatePlan("topology", topology_plan, (perf_counter() - start) * 1000.0)

    start = perf_counter()
    default_beam_result = process_aware_beam_search_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=beam_config,
    )
    default_beam = CandidatePlan(
        "default_beam",
        plan_from_result(selected_units, default_beam_result),
        (perf_counter() - start) * 1000.0,
    )

    start = perf_counter()
    default_polished = CandidatePlan(
        "default_polished",
        polish_plan(selected_units, default_beam_result, panel, tool, polish_config, process_model),
        (perf_counter() - start) * 1000.0,
    )

    max_margin = max(args.margins) if args.margins else 0.0
    needs_fallback = any(
        fallback_needed(
            best_candidate(
                (
                    topology,
                    default_beam,
                    default_polished,
                )
                if variant == "adaptive_polished"
                else (topology, default_beam)
            ),
            topology,
            margin,
        )
        for variant in args.variants
        for margin in (max_margin,)
    )

    fallback_beam = CandidatePlan("fallback_beam", default_beam.plan, 0.0)
    fallback_polished = CandidatePlan("fallback_polished", default_polished.plan, 0.0)
    if needs_fallback:
        start = perf_counter()
        fallback_beam_result = process_aware_beam_search_order(
            selected_units,
            panel,
            tool,
            process_model=process_model,
            config=fallback_beam_config,
        )
        fallback_beam = CandidatePlan(
            "fallback_beam",
            plan_from_result(selected_units, fallback_beam_result),
            (perf_counter() - start) * 1000.0,
        )
        start = perf_counter()
        fallback_polished = CandidatePlan(
            "fallback_polished",
            polish_plan(
                selected_units,
                fallback_beam_result,
                panel,
                tool,
                polish_config,
                process_model,
            ),
            (perf_counter() - start) * 1000.0,
        )

    return (
        layout,
        {
            "topology": topology,
            "default_beam": default_beam,
            "default_polished": default_polished,
            "fallback_beam": fallback_beam,
            "fallback_polished": fallback_polished,
        },
        len(units),
    )


def row_from_selection(
    archive: Path,
    placements_member: str,
    board_id: str,
    variant: str,
    margin: float,
    layout: Layout,
    candidate_count: int,
    candidates: dict[str, CandidatePlan],
) -> MarginRow:
    selected, triggered, expanded = candidate_for_variant(variant, margin, candidates)
    plan = selected.plan
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    return MarginRow(
        archive=archive.name,
        case_name=case_name,
        placement_method=placement_method,
        seed=seed,
        placements_member=placements_member,
        board_id=board_id,
        variant=variant,
        fallback_margin=margin,
        selected_source=preferred_label(selected, expanded),
        fallback_triggered=triggered,
        estimated_runtime_ms=estimated_runtime(variant, triggered, candidates),
        topology_runtime_ms=candidates["topology"].runtime_ms,
        default_beam_runtime_ms=candidates["default_beam"].runtime_ms,
        default_polish_runtime_ms=candidates["default_polished"].runtime_ms,
        fallback_beam_runtime_ms=candidates["fallback_beam"].runtime_ms if triggered else 0.0,
        fallback_polish_runtime_ms=(
            candidates["fallback_polished"].runtime_ms
            if triggered and variant == "adaptive_polished"
            else 0.0
        ),
        rectangle_count=len(layout.rectangles),
        rectangle_count_bin=rectangle_count_bin(len(layout.rectangles)),
        candidate_unit_count=candidate_count,
        selected_unit_count=len(plan.selected_units),
        action_count=len(plan.actions),
        air_move_distance=plan.metrics.air_move_distance,
        cutting_length=plan.metrics.cutting_length,
        pierce_count=plan.metrics.pierce_count,
        lift_count=plan.metrics.lift_count,
        safe_lift_count=plan.metrics.safe_lift_count,
        detour_count=plan.metrics.detour_count,
        travel_mode_cost=plan.metrics.travel_mode_cost,
        hard_penalty=plan.metrics.hard_penalty,
        stability_penalty=plan.metrics.stability_penalty,
    )


def summarize(rows: list[MarginRow]) -> list[MarginSummaryRow]:
    grouped: dict[tuple[str, float], list[MarginRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.variant, row.fallback_margin)].append(row)

    output: list[MarginSummaryRow] = []
    for (variant, margin), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        source_counts = Counter(row.selected_source for row in group)
        output.append(
            MarginSummaryRow(
                variant=variant,
                fallback_margin=margin,
                n=len(group),
                travel_mode_cost_mean=mean(row.travel_mode_cost for row in group),
                travel_mode_cost_std=pstdev(row.travel_mode_cost for row in group)
                if len(group) > 1
                else 0.0,
                machining_cost_mean=mean(
                    row.travel_mode_cost + row.cutting_length for row in group
                ),
                estimated_runtime_ms_mean=mean(row.estimated_runtime_ms for row in group),
                hard_penalty_mean=mean(row.hard_penalty for row in group),
                stability_penalty_mean=mean(row.stability_penalty for row in group),
                fallback_trigger_rate_pct=(
                    100.0
                    * sum(1 for row in group if row.fallback_triggered)
                    / len(group)
                ),
                source_topology_count=source_counts["topology"],
                source_default_beam_count=source_counts["default_beam"],
                source_default_polished_count=source_counts["default_polished"],
                source_fallback_beam_count=source_counts["fallback_beam"],
                source_fallback_polished_count=source_counts["fallback_polished"],
            )
        )
    return output


def write_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [field.name for field in fields(rows[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def append_rows(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(rows[0])]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def prepare_stream_output(path: Path, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.write_text("", encoding="utf-8")


def load_existing_rows(path: Path) -> list[MarginRow]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [margin_row_from_dict(row) for row in csv.DictReader(handle)]


def margin_row_from_dict(raw: dict[str, str]) -> MarginRow:
    return MarginRow(
        **{
            field.name: coerce_margin_value(field.name, raw.get(field.name, ""))
            for field in fields(MarginRow)
        }
    )


def coerce_margin_value(name: str, value: str):
    if value == "":
        return None
    if name in MARGIN_BOOL_FIELDS:
        return value in {"True", "true", "1"}
    if name in MARGIN_INT_FIELDS:
        return int(value)
    if name in MARGIN_FLOAT_FIELDS:
        return float(value)
    return value


MARGIN_BOOL_FIELDS = {"fallback_triggered"}
MARGIN_INT_FIELDS = {
    "rectangle_count",
    "candidate_unit_count",
    "selected_unit_count",
    "action_count",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
}
MARGIN_FLOAT_FIELDS = {
    "fallback_margin",
    "estimated_runtime_ms",
    "topology_runtime_ms",
    "default_beam_runtime_ms",
    "default_polish_runtime_ms",
    "fallback_beam_runtime_ms",
    "fallback_polish_runtime_ms",
    "air_move_distance",
    "cutting_length",
    "travel_mode_cost",
    "hard_penalty",
    "stability_penalty",
}


def margin_key(row: MarginRow) -> tuple[str, ...]:
    return margin_key_from_parts(
        row.archive,
        row.placements_member,
        row.board_id,
        row.variant,
        row.fallback_margin,
    )


def margin_key_from_parts(
    archive_name: str,
    placements_member: str,
    board_id: str,
    variant: str,
    margin: float,
) -> tuple[str, ...]:
    return (
        archive_name,
        placements_member,
        board_id,
        variant,
        f"{margin:.12g}",
    )


def case_completed(
    completed_keys: set[tuple[str, ...]],
    archive: Path,
    placements_member: str,
    board_id: str,
    variants: tuple[str, ...],
    margins: tuple[float, ...],
) -> bool:
    return all(
        margin_key_from_parts(
            archive.name,
            placements_member,
            board_id,
            variant,
            margin,
        )
        in completed_keys
        for variant in variants
        for margin in margins
    )


def plot_summary(rows: list[MarginSummaryRow], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = tuple(dict.fromkeys(row.variant for row in rows))
    colors = {
        "adaptive": "#0072B2",
        "adaptive_polished": "#009E73",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0))
    for variant in variants:
        subset = [row for row in rows if row.variant == variant]
        x = [row.fallback_margin for row in subset]
        axes[0].plot(
            x,
            [row.travel_mode_cost_mean for row in subset],
            marker="o",
            color=colors.get(variant, "#555555"),
            label=variant,
        )
        axes[1].plot(
            x,
            [row.estimated_runtime_ms_mean for row in subset],
            marker="o",
            color=colors.get(variant, "#555555"),
            label=variant,
        )
        axes[2].plot(
            x,
            [row.fallback_trigger_rate_pct for row in subset],
            marker="o",
            color=colors.get(variant, "#555555"),
            label=variant,
        )
    axes[0].set_title("Travel-mode cost")
    axes[0].set_ylabel("Mean cost")
    axes[1].set_title("Estimated runtime")
    axes[1].set_ylabel("Mean runtime (ms)")
    axes[2].set_title("Fallback trigger rate")
    axes[2].set_ylabel("Triggered cases (%)")
    for ax in axes:
        ax.set_xlabel("Fallback margin")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_adaptive_margin_sensitivity.pdf")
    fig.savefig(output_dir / "fig_adaptive_margin_sensitivity.png", dpi=300)
    plt.close(fig)


def iter_cases(args: argparse.Namespace):
    for archive in args.zip_paths:
        members = sample_placement_members(archive, args.max_members_per_archive)
        print(f"archive: {archive.name}, sampled_members={len(members)}")
        for member in members:
            board_ids = select_board_ids(
                board_counts_from_zip(archive, member),
                boards_per_member=args.boards_per_member,
                min_rectangles=args.min_rectangles,
                max_rectangles=args.max_rectangles,
            )
            print(f"  member: {member} boards={','.join(board_ids) if board_ids else 'none'}")
            for board_id in board_ids:
                yield archive, member, board_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", dest="zip_paths", action="append", type=Path)
    parser.add_argument("--max-members-per-archive", type=int, default=20)
    parser.add_argument("--boards-per-member", type=int, default=3)
    parser.add_argument("--min-rectangles", type=int, default=20)
    parser.add_argument("--max-rectangles", type=int, default=50)
    parser.add_argument(
        "--margins",
        nargs="+",
        type=float,
        default=(0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0),
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--task-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "adaptive_margin_sensitivity_real_20_50.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "adaptive_margin_sensitivity_real_20_50_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures" / "adaptive_margin_sensitivity_real_20_50",
    )
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable terminal progress bar output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output CSV and skip rows already present in it.",
    )
    add_experiment_preset_arg(parser)
    add_stability_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = apply_experiment_preset(parse_args())
    args.zip_paths = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    args.margins = tuple(sorted(set(args.margins)))
    args.variants = tuple(args.variants)
    progress_output = args.progress_output or default_progress_log_path(args.output)
    args.progress_output = progress_output

    cases = tuple(iter_cases(args))
    progress_bar = TerminalProgressBar(
        len(cases),
        enabled=not args.no_progress_bar,
    )
    progress_bar.start("adaptive margin cases")
    rows: list[MarginRow] = load_existing_rows(args.output) if args.resume else []
    completed_keys = {margin_key(row) for row in rows}
    prepare_stream_output(args.output, resume=args.resume)
    prepare_progress_log(progress_output, resume=args.resume)
    if args.resume and rows:
        print(f"resume: loaded {len(rows)} existing rows from {args.output}")

    case_count = 0
    timeout_count = 0
    skipped_case_count = 0
    for archive, member, board_id in cases:
        case_count += 1
        if case_completed(
            completed_keys,
            archive,
            member,
            board_id,
            args.variants,
            args.margins,
        ):
            skipped_case_count += 1
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="skipped",
                    archive=archive.name,
                    placements_member=member,
                    board_id=board_id,
                    method="adaptive_margin_candidates",
                    message="all margin rows already present in output CSV",
                ),
            )
            print(f"    skip completed candidates: board={board_id}")
            progress_bar.advance("skipped", f"board={board_id}")
            continue
        print(f"    run candidates: board={board_id}")
        append_progress_event(
            progress_output,
            new_progress_event(
                event="started",
                archive=archive.name,
                placements_member=member,
                board_id=board_id,
                method="adaptive_margin_candidates",
            ),
        )
        try:
            with task_timeout(args.task_timeout_seconds):
                layout, candidates, candidate_count = build_case_candidates(
                    archive,
                    member,
                    board_id,
                    args,
                )
        except TaskTimeoutError:
            timeout_count += 1
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="timed_out",
                    archive=archive.name,
                    placements_member=member,
                    board_id=board_id,
                    method="adaptive_margin_candidates",
                    message=f"timeout after {args.task_timeout_seconds:g} seconds",
                ),
            )
            print(
                f"    timed out: board={board_id} "
                f"after {args.task_timeout_seconds:.1f}s"
            )
            progress_bar.advance("timed_out", f"board={board_id}")
            continue
        new_rows: list[MarginRow] = []
        for variant in args.variants:
            for margin in args.margins:
                key = margin_key_from_parts(
                    archive.name,
                    member,
                    board_id,
                    variant,
                    margin,
                )
                if key in completed_keys:
                    continue
                row = row_from_selection(
                    archive,
                    member,
                    board_id,
                    variant,
                    margin,
                    layout,
                    candidate_count,
                    candidates,
                )
                rows.append(row)
                new_rows.append(row)
                completed_keys.add(key)
        append_rows(args.output, new_rows)
        append_progress_event(
            progress_output,
            new_progress_event(
                event="completed",
                archive=archive.name,
                placements_member=member,
                board_id=board_id,
                method="adaptive_margin_candidates",
                rectangle_count=len(layout.rectangles),
                candidate_unit_count=candidate_count,
                message=f"appended {len(new_rows)} margin rows",
            ),
        )
        print(
            f"    done candidates: board={board_id} rows={len(new_rows)}"
        )
        progress_bar.advance("completed", f"board={board_id}")

    summary = summarize(rows)
    write_rows(args.summary_output, summary)
    plot_summary(summary, args.figure_dir)
    print(
        f"cases={case_count} skipped={skipped_case_count} "
        f"timed_out={timeout_count} rows={len(rows)}"
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {progress_output}")
    print(f"figures: {args.figure_dir}")
    for row in summary:
        print(
            f"{row.variant:18s} margin={row.fallback_margin:7.1f} "
            f"cost={row.travel_mode_cost_mean:10.3f} "
            f"runtime={row.estimated_runtime_ms_mean:8.3f} ms "
            f"fallback={row.fallback_trigger_rate_pct:6.2f}%"
        )


if __name__ == "__main__":
    main()
