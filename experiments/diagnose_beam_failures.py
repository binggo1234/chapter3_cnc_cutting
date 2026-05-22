from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnc_cutting.cutting_units import build_candidate_cutting_units
from cnc_cutting.io import (
    load_chapter2_config_from_zip,
    load_chapter2_layouts_from_zip,
    tool_config_from_chapter2_config,
)
from cnc_cutting.local_search import BeamSearchConfig, process_aware_beam_search_order
from cnc_cutting.models import Panel
from cnc_cutting.optimizer import select_coverage_units
from cnc_cutting.travel import clear_detour_cache
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)
from run_chapter2_batch import DEFAULT_ARCHIVES, parse_member_metadata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = (
    "topology_no_beam",
    "process_local_search_multistart",
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)


@dataclass(frozen=True)
class FailureCase:
    case_id: str
    archive: str
    zip_path: str
    placements_member: str
    board_id: str
    baseline_method: str
    target_method: str
    rectangle_count: int
    baseline_travel_mode_cost: float
    target_travel_mode_cost: float
    travel_mode_cost_delta: float
    travel_mode_cost_reduction_pct: float
    baseline_machining_cost: float
    target_machining_cost: float
    machining_cost_reduction_pct: float
    baseline_hard_penalty: float
    target_hard_penalty: float
    baseline_stability_penalty: float
    target_stability_penalty: float
    baseline_tool_event_count: float
    target_tool_event_count: float
    process_key_winner: str
    failure_score: float


@dataclass(frozen=True)
class SweepConfigSpec:
    config_name: str
    beam_width: int
    candidate_pool_size: int
    max_expansions_per_node: int
    max_layer_expansions: int
    diversity_bucket_limit: int
    min_expansions_per_parent: int
    unstable_min_expansions_per_parent: int
    unstable_layer_expansion_multiplier: float
    unstable_layer_expansion_bonus: int


@dataclass(frozen=True)
class SweepRow:
    case_id: str
    archive: str
    placements_member: str
    board_id: str
    baseline_method: str
    target_method: str
    config_name: str
    support_policy: str
    min_support_count: int
    min_support_ratio: float
    min_area_normalized_support: float
    adjacency_support_weight: float
    beam_width: int
    candidate_pool_size: int
    max_expansions_per_node: int
    max_layer_expansions: int
    diversity_bucket_limit: int
    min_expansions_per_parent: int
    unstable_min_expansions_per_parent: int
    unstable_layer_expansion_multiplier: float
    unstable_layer_expansion_bonus: int
    runtime_ms: float
    expanded_nodes: int
    rectangle_count: int
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    air_move_distance: float
    travel_mode_cost: float
    machining_cost: float
    hard_penalty: float
    stability_penalty: float
    safe_lift_count: int
    detour_count: int
    baseline_travel_mode_cost: float
    default_target_travel_mode_cost: float
    travel_delta_vs_baseline: float
    travel_delta_vs_default: float
    travel_reduction_vs_baseline_pct: float
    travel_reduction_vs_default_pct: float
    process_feasible: bool
    layer_pruned_total: int
    diversity_pruned_total: int
    duplicate_pruned_total: int
    parent_quota_added_total: int
    fallback_added_total: int
    max_layer_expansion_count: int
    max_effective_layer_expansion_limit: int


@dataclass(frozen=True)
class SweepSummaryRow:
    config_name: str
    n: int
    feasible_rate: float
    win_rate_vs_baseline: float
    mean_travel_mode_cost: float
    mean_travel_delta_vs_baseline: float
    mean_travel_delta_vs_default: float
    mean_travel_reduction_vs_baseline_pct: float
    mean_travel_reduction_vs_default_pct: float
    mean_runtime_ms: float
    mean_expanded_nodes: float
    mean_layer_pruned_total: float
    mean_diversity_pruned_total: float
    mean_parent_quota_added_total: float


def parse_float(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    return float(value)


def archive_paths() -> dict[str, Path]:
    return {archive.name: archive for archive in DEFAULT_ARCHIVES}


def split_case_id(case_id: str) -> tuple[str, str, str]:
    parts = case_id.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected case_id format: {case_id}")
    return parts[0], parts[1], parts[2]


def load_paired_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def select_failure_cases(
    paired_rows: tuple[dict[str, str], ...],
    baseline_methods: tuple[str, ...],
    archive_by_name: dict[str, Path],
    max_cases: int,
    min_travel_loss: float,
    unique_cases: bool,
) -> tuple[FailureCase, ...]:
    failures: list[FailureCase] = []
    for row in paired_rows:
        baseline_method = row["baseline_method"]
        if baseline_method not in baseline_methods:
            continue
        travel_delta = parse_float(row["travel_mode_cost_delta"])
        process_winner = row["process_key_winner"]
        if process_winner != "baseline" and travel_delta <= min_travel_loss:
            continue
        archive_name, placements_member, board_id = split_case_id(row["case_id"])
        if archive_name not in archive_by_name:
            raise ValueError(f"archive not configured: {archive_name}")
        baseline_stability = parse_float(row["baseline_stability_penalty"])
        target_stability = parse_float(row["target_stability_penalty"])
        failure_score = travel_delta
        if process_winner == "baseline":
            failure_score += 100000.0
        failure_score += max(0.0, target_stability - baseline_stability) * 10000.0
        failures.append(
            FailureCase(
                case_id=row["case_id"],
                archive=archive_name,
                zip_path=str(archive_by_name[archive_name]),
                placements_member=placements_member,
                board_id=board_id,
                baseline_method=baseline_method,
                target_method=row["target_method"],
                rectangle_count=int(float(row["rectangle_count"])),
                baseline_travel_mode_cost=parse_float(
                    row["baseline_travel_mode_cost"]
                ),
                target_travel_mode_cost=parse_float(row["target_travel_mode_cost"]),
                travel_mode_cost_delta=travel_delta,
                travel_mode_cost_reduction_pct=parse_float(
                    row["travel_mode_cost_reduction_pct"]
                ),
                baseline_machining_cost=parse_float(row["baseline_machining_cost"]),
                target_machining_cost=parse_float(row["target_machining_cost"]),
                machining_cost_reduction_pct=parse_float(
                    row["machining_cost_reduction_pct"]
                ),
                baseline_hard_penalty=parse_float(row["baseline_hard_penalty"]),
                target_hard_penalty=parse_float(row["target_hard_penalty"]),
                baseline_stability_penalty=baseline_stability,
                target_stability_penalty=target_stability,
                baseline_tool_event_count=parse_float(
                    row["baseline_tool_event_count"]
                ),
                target_tool_event_count=parse_float(row["target_tool_event_count"]),
                process_key_winner=process_winner,
                failure_score=failure_score,
            )
        )

    failures.sort(
        key=lambda item: (
            item.failure_score,
            item.travel_mode_cost_delta,
            item.rectangle_count,
            item.case_id,
        ),
        reverse=True,
    )
    if not unique_cases:
        return tuple(failures[:max_cases])

    selected: list[FailureCase] = []
    seen_case_ids: set[str] = set()
    for failure in failures:
        if failure.case_id in seen_case_ids:
            continue
        selected.append(failure)
        seen_case_ids.add(failure.case_id)
        if len(selected) >= max_cases:
            break
    return tuple(selected)


def default_config_for_size(size: int) -> SweepConfigSpec:
    if size <= 20:
        return SweepConfigSpec(
            config_name="default",
            beam_width=8,
            candidate_pool_size=24,
            max_expansions_per_node=48,
            max_layer_expansions=56,
            diversity_bucket_limit=1,
            min_expansions_per_parent=0,
            unstable_min_expansions_per_parent=2,
            unstable_layer_expansion_multiplier=1.0,
            unstable_layer_expansion_bonus=0,
        )
    if size <= 75:
        return SweepConfigSpec(
            config_name="default",
            beam_width=6,
            candidate_pool_size=18,
            max_expansions_per_node=36,
            max_layer_expansions=42,
            diversity_bucket_limit=1,
            min_expansions_per_parent=0,
            unstable_min_expansions_per_parent=2,
            unstable_layer_expansion_multiplier=1.0,
            unstable_layer_expansion_bonus=0,
        )
    return SweepConfigSpec(
        config_name="default",
        beam_width=4,
        candidate_pool_size=12,
        max_expansions_per_node=24,
        max_layer_expansions=28,
        diversity_bucket_limit=1,
        min_expansions_per_parent=0,
        unstable_min_expansions_per_parent=2,
        unstable_layer_expansion_multiplier=1.0,
        unstable_layer_expansion_bonus=0,
    )


def sweep_configs_for_size(size: int) -> tuple[SweepConfigSpec, ...]:
    base = default_config_for_size(size)
    wide_width = base.beam_width + 2
    wide_pool = base.candidate_pool_size + 6
    wide_per_node = base.max_expansions_per_node + 12
    return (
        base,
        replace_config(base, "no_diversity", diversity_bucket_limit=0),
        replace_config(base, "diversity_2", diversity_bucket_limit=2),
        replace_config(
            base,
            "wider_beam",
            beam_width=wide_width,
            candidate_pool_size=wide_pool,
            max_expansions_per_node=wide_per_node,
            max_layer_expansions=wide_width * 7,
        ),
        replace_config(
            base,
            "wider_layer",
            max_layer_expansions=base.beam_width * 10,
        ),
        replace_config(
            base,
            "larger_pool",
            candidate_pool_size=base.candidate_pool_size + 12,
            max_expansions_per_node=base.max_expansions_per_node + 24,
            max_layer_expansions=base.beam_width * 10,
        ),
        replace_config(
            base,
            "parent_quota_1",
            min_expansions_per_parent=1,
        ),
        replace_config(
            base,
            "unstable_parent_4",
            unstable_min_expansions_per_parent=4,
        ),
        replace_config(
            base,
            "wide_no_diversity",
            beam_width=wide_width,
            candidate_pool_size=wide_pool,
            max_expansions_per_node=wide_per_node,
            max_layer_expansions=wide_width * 10,
            diversity_bucket_limit=0,
        ),
    )


def replace_config(base: SweepConfigSpec, config_name: str, **updates) -> SweepConfigSpec:
    values = asdict(base)
    values.update(updates)
    values["config_name"] = config_name
    return SweepConfigSpec(**values)


def run_sweep_case(
    failure: FailureCase,
    config_spec: SweepConfigSpec,
    args: argparse.Namespace,
) -> SweepRow:
    archive = Path(failure.zip_path)
    cfg = load_chapter2_config_from_zip(
        archive,
        placements_member=failure.placements_member,
    )
    tool = tool_config_from_chapter2_config(cfg)
    layout = load_chapter2_layouts_from_zip(
        archive,
        placements_member=failure.placements_member,
        board_ids=(failure.board_id,),
    )[0]
    panel = Panel(layout.panel_id, layout.panel_width, layout.panel_height)
    process_model = build_process_model_from_args(layout, args)
    units = build_candidate_cutting_units(
        layout,
        tool,
        max_collinear_gap=tool.min_channel_width,
    )
    selected_units = select_coverage_units(units)
    config = BeamSearchConfig(
        beam_width=config_spec.beam_width,
        candidate_pool_size=config_spec.candidate_pool_size,
        max_expansions_per_node=config_spec.max_expansions_per_node,
        max_layer_expansions=config_spec.max_layer_expansions,
        diversity_bucket_limit=(
            config_spec.diversity_bucket_limit
            if config_spec.diversity_bucket_limit > 0
            else None
        ),
        min_expansions_per_parent=config_spec.min_expansions_per_parent,
        unstable_min_expansions_per_parent=(
            config_spec.unstable_min_expansions_per_parent
        ),
        unstable_layer_expansion_multiplier=(
            config_spec.unstable_layer_expansion_multiplier
        ),
        unstable_layer_expansion_bonus=config_spec.unstable_layer_expansion_bonus,
    )

    clear_detour_cache()
    start = perf_counter()
    result = process_aware_beam_search_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=config,
    )
    runtime_ms = (perf_counter() - start) * 1000.0
    diagnostics = result.diagnostics
    travel_delta_vs_baseline = (
        result.metrics.travel_mode_cost - failure.baseline_travel_mode_cost
    )
    travel_delta_vs_default = (
        result.metrics.travel_mode_cost - failure.target_travel_mode_cost
    )
    return SweepRow(
        case_id=failure.case_id,
        archive=failure.archive,
        placements_member=failure.placements_member,
        board_id=failure.board_id,
        baseline_method=failure.baseline_method,
        target_method=failure.target_method,
        config_name=config_spec.config_name,
        support_policy=args.support_policy,
        min_support_count=args.min_support_count,
        min_support_ratio=args.min_support_ratio,
        min_area_normalized_support=args.min_area_normalized_support,
        adjacency_support_weight=args.adjacency_support_weight,
        beam_width=config_spec.beam_width,
        candidate_pool_size=config_spec.candidate_pool_size,
        max_expansions_per_node=config_spec.max_expansions_per_node,
        max_layer_expansions=config_spec.max_layer_expansions,
        diversity_bucket_limit=config_spec.diversity_bucket_limit,
        min_expansions_per_parent=config_spec.min_expansions_per_parent,
        unstable_min_expansions_per_parent=(
            config_spec.unstable_min_expansions_per_parent
        ),
        unstable_layer_expansion_multiplier=(
            config_spec.unstable_layer_expansion_multiplier
        ),
        unstable_layer_expansion_bonus=config_spec.unstable_layer_expansion_bonus,
        runtime_ms=runtime_ms,
        expanded_nodes=result.expanded_nodes,
        rectangle_count=len(layout.rectangles),
        candidate_unit_count=len(units),
        selected_unit_count=len(selected_units),
        action_count=len(result.actions),
        air_move_distance=result.metrics.air_move_distance,
        travel_mode_cost=result.metrics.travel_mode_cost,
        machining_cost=result.metrics.machining_cost,
        hard_penalty=result.metrics.hard_penalty,
        stability_penalty=result.metrics.stability_penalty,
        safe_lift_count=result.metrics.safe_lift_count,
        detour_count=result.metrics.detour_count,
        baseline_travel_mode_cost=failure.baseline_travel_mode_cost,
        default_target_travel_mode_cost=failure.target_travel_mode_cost,
        travel_delta_vs_baseline=travel_delta_vs_baseline,
        travel_delta_vs_default=travel_delta_vs_default,
        travel_reduction_vs_baseline_pct=relative_reduction_pct(
            failure.baseline_travel_mode_cost,
            result.metrics.travel_mode_cost,
        ),
        travel_reduction_vs_default_pct=relative_reduction_pct(
            failure.target_travel_mode_cost,
            result.metrics.travel_mode_cost,
        ),
        process_feasible=(
            result.metrics.hard_penalty <= 1e-9
            and result.metrics.stability_penalty <= 1e-9
        ),
        layer_pruned_total=sum(row.layer_pruned_count for row in diagnostics),
        diversity_pruned_total=sum(row.diversity_pruned_count for row in diagnostics),
        duplicate_pruned_total=sum(row.duplicate_pruned_count for row in diagnostics),
        parent_quota_added_total=sum(
            row.parent_quota_added_count for row in diagnostics
        ),
        fallback_added_total=sum(row.fallback_added_count for row in diagnostics),
        max_layer_expansion_count=max(
            (row.layer_expansion_count for row in diagnostics),
            default=0,
        ),
        max_effective_layer_expansion_limit=max(
            (row.effective_layer_expansion_limit for row in diagnostics),
            default=0,
        ),
    )


def relative_reduction_pct(baseline: float, target: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return (baseline - target) / abs(baseline) * 100.0


def summarize_sweep(rows: tuple[SweepRow, ...]) -> tuple[SweepSummaryRow, ...]:
    grouped: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        grouped[row.config_name].append(row)
    summary: list[SweepSummaryRow] = []
    for config_name, items in sorted(grouped.items()):
        summary.append(
            SweepSummaryRow(
                config_name=config_name,
                n=len(items),
                feasible_rate=mean(1.0 if row.process_feasible else 0.0 for row in items),
                win_rate_vs_baseline=mean(
                    1.0 if row.travel_delta_vs_baseline < -1e-9 else 0.0
                    for row in items
                ),
                mean_travel_mode_cost=mean(row.travel_mode_cost for row in items),
                mean_travel_delta_vs_baseline=mean(
                    row.travel_delta_vs_baseline for row in items
                ),
                mean_travel_delta_vs_default=mean(
                    row.travel_delta_vs_default for row in items
                ),
                mean_travel_reduction_vs_baseline_pct=mean(
                    row.travel_reduction_vs_baseline_pct for row in items
                ),
                mean_travel_reduction_vs_default_pct=mean(
                    row.travel_reduction_vs_default_pct for row in items
                ),
                mean_runtime_ms=mean(row.runtime_ms for row in items),
                mean_expanded_nodes=mean(row.expanded_nodes for row in items),
                mean_layer_pruned_total=mean(row.layer_pruned_total for row in items),
                mean_diversity_pruned_total=mean(
                    row.diversity_pruned_total for row in items
                ),
                mean_parent_quota_added_total=mean(
                    row.parent_quota_added_total for row in items
                ),
            )
        )
    return tuple(
        sorted(
            summary,
            key=lambda row: (
                -row.win_rate_vs_baseline,
                row.mean_travel_delta_vs_baseline,
                row.mean_runtime_ms,
            ),
        )
    )


def write_dataclass_rows(rows: tuple, output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_sweep_summary(summary: tuple[SweepSummaryRow, ...], output_dir: Path) -> None:
    if not summary:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(summary, key=lambda row: row.mean_travel_delta_vs_baseline)
    labels = [row.config_name for row in rows]
    y = range(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, max(3.0, 0.36 * len(rows) + 1.0)))
    axes[0].barh(y, [row.mean_travel_delta_vs_baseline for row in rows], color="#E69F00")
    axes[0].axvline(0.0, color="#111827", linewidth=0.7)
    axes[0].set_title("Gap to baseline")
    axes[0].set_xlabel("Mean travel delta")

    axes[1].barh(y, [100.0 * row.win_rate_vs_baseline for row in rows], color="#56B4E9")
    axes[1].axvline(50.0, color="#111827", linewidth=0.7)
    axes[1].set_title("Recovered cases")
    axes[1].set_xlabel("Win rate vs baseline (%)")

    axes[2].barh(y, [row.mean_runtime_ms for row in rows], color="#009E73")
    axes[2].set_title("Runtime")
    axes[2].set_xlabel("Mean runtime (ms)")

    for ax in axes:
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_beam_failure_sweep_summary.pdf")
    fig.savefig(output_dir / "fig_beam_failure_sweep_summary.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired-comparison",
        type=Path,
        default=ROOT
        / "results"
        / "analysis_ablation_full_real_20_50"
        / "paired_comparison.csv",
    )
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=list(DEFAULT_BASELINES),
    )
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--min-travel-loss", type=float, default=1e-9)
    parser.add_argument("--allow-duplicate-cases", action="store_true")
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument(
        "--config-names",
        nargs="+",
        default=None,
        help=(
            "Optional subset of sweep configs. Available: default, no_diversity, "
            "diversity_2, wider_beam, wider_layer, larger_pool, parent_quota_1, "
            "unstable_parent_4, wide_no_diversity."
        ),
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=ROOT / "results" / "beam_failure_cases.csv",
    )
    parser.add_argument(
        "--sweep-output",
        type=Path,
        default=ROOT / "results" / "beam_failure_sweep.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "beam_failure_sweep_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures" / "beam_failure_diagnostics",
    )
    add_experiment_preset_arg(parser)
    add_stability_model_args(parser)
    return apply_experiment_preset(parser.parse_args())


def main() -> None:
    args = parse_args()
    archive_by_name = archive_paths()
    paired_rows = load_paired_rows(args.paired_comparison)
    failures = select_failure_cases(
        paired_rows,
        baseline_methods=tuple(args.baseline_methods),
        archive_by_name=archive_by_name,
        max_cases=args.max_cases,
        min_travel_loss=args.min_travel_loss,
        unique_cases=not args.allow_duplicate_cases,
    )
    if not failures:
        raise ValueError("no beam failure cases found")
    write_dataclass_rows(failures, args.failure_output)
    print(f"paired_comparison: {args.paired_comparison}")
    print(f"failure_cases: {len(failures)}")
    print(f"wrote: {args.failure_output}")
    for failure in failures:
        case_name, placement_method, seed = parse_member_metadata(
            failure.placements_member
        )
        print(
            f"case board={failure.board_id:<6} baseline={failure.baseline_method:<32} "
            f"delta={failure.travel_mode_cost_delta:>9.3f} "
            f"rect={failure.rectangle_count:<3} "
            f"placement={placement_method or case_name or seed}"
        )

    if not args.run_sweep:
        return

    sweep_rows: list[SweepRow] = []
    for failure in failures:
        print(
            f"sweep case: board={failure.board_id} "
            f"baseline={failure.baseline_method} default_delta={failure.travel_mode_cost_delta:.3f}"
        )
        configs = sweep_configs_for_size(failure.rectangle_count)
        if args.config_names is not None:
            allowed = set(args.config_names)
            configs = tuple(config for config in configs if config.config_name in allowed)
            missing = sorted(allowed - {config.config_name for config in configs})
            if missing:
                raise ValueError(f"unknown config names for size {failure.rectangle_count}: {missing}")
        for config in configs:
            row = run_sweep_case(failure, config, args)
            sweep_rows.append(row)
            print(
                f"  {config.config_name:<18} cost={row.travel_mode_cost:>10.3f} "
                f"delta_base={row.travel_delta_vs_baseline:>9.3f} "
                f"delta_default={row.travel_delta_vs_default:>9.3f} "
                f"runtime={row.runtime_ms:>8.1f} ms "
                f"pruned={row.layer_pruned_total}"
            )

    result_rows = tuple(sweep_rows)
    summary = summarize_sweep(result_rows)
    write_dataclass_rows(result_rows, args.sweep_output)
    write_dataclass_rows(summary, args.summary_output)
    plot_sweep_summary(summary, args.figure_dir)
    print(f"wrote: {args.sweep_output}")
    print(f"wrote: {args.summary_output}")
    print(f"figures: {args.figure_dir}")
    for row in summary:
        print(
            f"{row.config_name:<18} n={row.n:<3} "
            f"win={row.win_rate_vs_baseline * 100:>5.1f}% "
            f"delta_base={row.mean_travel_delta_vs_baseline:>9.3f} "
            f"delta_default={row.mean_travel_delta_vs_default:>9.3f} "
            f"runtime={row.mean_runtime_ms:>8.1f} ms"
        )


if __name__ == "__main__":
    main()
