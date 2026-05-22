from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import mean
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
from cnc_cutting.models import Layout, Panel, PathMetrics
from cnc_cutting.optimizer import (
    DEFAULT_TOOL_EVENT_GATE_CONFIG,
    RoutePlan,
    ToolEventGateConfig,
    _beam_fallback_needed,
    _best_process_route,
    plan_topology_route,
    select_coverage_units,
    wider_beam_search_config,
)
from cnc_cutting.travel import clear_detour_cache
from process_options import (
    add_experiment_preset_arg,
    add_stability_model_args,
    apply_experiment_preset,
    build_process_model_from_args,
)
from progress_bar import TerminalProgressBar
from progress_log import (
    append_progress_event,
    default_progress_log_path,
    new_progress_event,
    prepare_progress_log,
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
DEFAULT_STRATEGIES = ("no_gate", "fixed_100", "current", "strict")


@dataclass(frozen=True)
class GateStrategy:
    name: str
    config: ToolEventGateConfig


@dataclass(frozen=True)
class CandidatePlan:
    label: str
    plan: RoutePlan
    runtime_ms: float


@dataclass(frozen=True)
class GateSensitivityRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    strategy: str
    gate_enabled: bool
    min_travel_saving_per_extra_event: float
    min_travel_saving_ratio_per_extra_event: float
    min_machining_saving: float
    fallback_triggered: bool
    selected_source: str
    estimated_runtime_ms: float
    topology_runtime_ms: float
    beam_runtime_ms: float
    polish_runtime_ms: float
    fallback_beam_runtime_ms: float
    fallback_polish_runtime_ms: float
    rectangle_count: int
    rectangle_count_bin: str
    candidate_unit_count: int
    selected_unit_count: int
    action_count: int
    baseline_tool_event_count: int
    target_tool_event_count: int
    tool_event_delta: int
    pierce_delta: int
    lift_delta: int
    safe_lift_delta: int
    baseline_travel_mode_cost: float
    target_travel_mode_cost: float
    travel_mode_cost_delta: float
    travel_mode_cost_reduction_pct: float
    baseline_machining_cost: float
    target_machining_cost: float
    machining_cost_delta: float
    machining_cost_reduction_pct: float
    baseline_detour_count: int
    target_detour_count: int
    detour_delta: int
    hard_penalty: float
    stability_penalty: float
    air_move_distance: float
    cutting_length: float


@dataclass(frozen=True)
class GateSensitivitySummaryRow:
    strategy: str
    gate_enabled: bool
    min_travel_saving_per_extra_event: float
    min_travel_saving_ratio_per_extra_event: float
    min_machining_saving: float
    n: int
    target_travel_mode_cost_mean: float
    target_machining_cost_mean: float
    estimated_runtime_ms_mean: float
    tool_event_decrease_count: int
    tool_event_tie_count: int
    tool_event_increase_count: int
    tool_event_delta_mean: float
    travel_mode_cost_reduction_pct_mean: float
    machining_cost_reduction_pct_mean: float
    increase_cases_travel_reduction_pct_mean: float
    increase_cases_machining_reduction_pct_mean: float
    fallback_trigger_rate_pct: float
    source_topology_count: int
    source_beam_count: int
    source_polished_count: int
    source_fallback_polished_count: int


@dataclass(frozen=True)
class GateDecisionRow:
    archive: str
    case_name: str
    placement_method: str
    seed: str
    placements_member: str
    board_id: str
    strategy: str
    candidate_source: str
    candidate_evaluated: bool
    selected: bool
    protected_candidate: bool
    gate_passed: bool
    gate_reason: str
    required_travel_saving: float
    travel_saving: float
    machining_saving: float
    extra_tool_events: int
    baseline_tool_event_count: int
    candidate_tool_event_count: int
    baseline_travel_mode_cost: float
    candidate_travel_mode_cost: float
    baseline_machining_cost: float
    candidate_machining_cost: float
    baseline_detour_count: int
    candidate_detour_count: int
    hard_penalty: float
    stability_penalty: float


def strategy_by_name(name: str) -> GateStrategy:
    if name == "no_gate":
        return GateStrategy(name, ToolEventGateConfig(enabled=False))
    if name == "fixed_100":
        return GateStrategy(
            name,
            ToolEventGateConfig(
                enabled=True,
                min_travel_saving_per_extra_event=100.0,
                min_travel_saving_ratio_per_extra_event=0.0,
                min_machining_saving=(
                    DEFAULT_TOOL_EVENT_GATE_CONFIG.min_machining_saving
                ),
            ),
        )
    if name == "current":
        return GateStrategy(name, DEFAULT_TOOL_EVENT_GATE_CONFIG)
    if name == "strict":
        return GateStrategy(
            name,
            ToolEventGateConfig(
                enabled=True,
                min_travel_saving_per_extra_event=200.0,
                min_travel_saving_ratio_per_extra_event=0.05,
                min_machining_saving=(
                    DEFAULT_TOOL_EVENT_GATE_CONFIG.min_machining_saving
                ),
            ),
        )
    raise ValueError(f"unsupported gate strategy: {name}")


def tool_event_count(metrics: PathMetrics) -> int:
    return metrics.pierce_count + metrics.lift_count + metrics.safe_lift_count


def reduction_pct(baseline: float, target: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 100.0 * (baseline - target) / baseline


def gate_requirement(
    baseline: PathMetrics,
    candidate: PathMetrics,
    config: ToolEventGateConfig,
) -> tuple[bool, str, float, float, float, int]:
    baseline_events = tool_event_count(baseline)
    candidate_events = tool_event_count(candidate)
    extra_events = candidate_events - baseline_events
    travel_saving = baseline.travel_mode_cost - candidate.travel_mode_cost
    machining_saving = baseline.machining_cost - candidate.machining_cost
    if extra_events <= 0:
        return True, "no_extra_tool_events", 0.0, travel_saving, machining_saving, extra_events
    if not config.enabled:
        return True, "gate_disabled", 0.0, travel_saving, machining_saving, extra_events
    required_per_event = max(
        config.min_travel_saving_per_extra_event,
        baseline.travel_mode_cost * config.min_travel_saving_ratio_per_extra_event,
    )
    required = required_per_event * extra_events
    if machining_saving <= config.min_machining_saving:
        return False, "insufficient_machining_saving", required, travel_saving, machining_saving, extra_events
    if travel_saving < required:
        return False, "insufficient_travel_saving", required, travel_saving, machining_saving, extra_events
    return True, "extra_events_justified", required, travel_saving, machining_saving, extra_events


def plan_from_result(selected_units, result) -> RoutePlan:
    return RoutePlan(
        selected_units=selected_units,
        actions=result.actions,
        metrics=result.metrics,
    )


def polish_plan(selected_units, beam_result, panel, tool, polish_config, process_model):
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


def choose_candidate(
    candidates: tuple[CandidatePlan, ...],
    protected: tuple[CandidatePlan, ...],
    strategy: GateStrategy,
) -> CandidatePlan:
    chosen = _best_process_route(
        tuple(candidate.plan for candidate in candidates),
        protected_plans=tuple(candidate.plan for candidate in protected),
        tool_event_gate=strategy.config,
    )
    for candidate in candidates:
        if candidate.plan is chosen:
            return candidate
    raise AssertionError("selected plan not found among candidates")


def select_for_strategy(
    candidates: dict[str, CandidatePlan],
    strategy: GateStrategy,
    fallback_margin: float,
) -> tuple[CandidatePlan, bool]:
    base = (
        candidates["topology"],
        candidates["beam"],
        candidates["polished"],
    )
    protected = (candidates["beam"],)
    current_best = choose_candidate(base, protected, strategy)
    fallback_triggered = _beam_fallback_needed(
        current_best.plan,
        candidates["topology"].plan,
        fallback_margin,
    )
    if not fallback_triggered:
        return current_best, False
    expanded = base + (candidates["fallback_polished"],)
    return choose_candidate(expanded, protected, strategy), True


def fallback_required_by_any_strategy(
    candidates: dict[str, CandidatePlan],
    strategies: tuple[GateStrategy, ...],
    fallback_margin: float,
) -> bool:
    return any(
        _beam_fallback_needed(
            choose_candidate(
                (
                    candidates["topology"],
                    candidates["beam"],
                    candidates["polished"],
                ),
                (candidates["beam"],),
                strategy,
            ).plan,
            candidates["topology"].plan,
            fallback_margin,
        )
        for strategy in strategies
    )


def estimated_runtime(candidates: dict[str, CandidatePlan], fallback_triggered: bool) -> float:
    total = (
        candidates["topology"].runtime_ms
        + candidates["beam"].runtime_ms
        + candidates["polished"].runtime_ms
    )
    if fallback_triggered:
        total += (
            candidates["fallback_beam"].runtime_ms
            + candidates["fallback_polished"].runtime_ms
        )
    return total


def build_case_candidates(
    archive: Path,
    placements_member: str,
    board_id: str,
    args: argparse.Namespace,
    strategies: tuple[GateStrategy, ...],
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
    beam_result = process_aware_beam_search_order(
        selected_units,
        panel,
        tool,
        process_model=process_model,
        config=beam_config,
    )
    beam = CandidatePlan(
        "beam",
        plan_from_result(selected_units, beam_result),
        (perf_counter() - start) * 1000.0,
    )

    start = perf_counter()
    polished = CandidatePlan(
        "polished",
        polish_plan(selected_units, beam_result, panel, tool, polish_config, process_model),
        (perf_counter() - start) * 1000.0,
    )

    candidates = {
        "topology": topology,
        "beam": beam,
        "polished": polished,
        "fallback_beam": CandidatePlan("fallback_beam", beam.plan, 0.0),
        "fallback_polished": CandidatePlan("fallback_polished", polished.plan, 0.0),
    }

    if fallback_required_by_any_strategy(candidates, strategies, args.fallback_margin):
        fallback_beam_config = wider_beam_search_config(beam_config)
        start = perf_counter()
        fallback_beam_result = process_aware_beam_search_order(
            selected_units,
            panel,
            tool,
            process_model=process_model,
            config=fallback_beam_config,
        )
        candidates["fallback_beam"] = CandidatePlan(
            "fallback_beam",
            plan_from_result(selected_units, fallback_beam_result),
            (perf_counter() - start) * 1000.0,
        )
        start = perf_counter()
        candidates["fallback_polished"] = CandidatePlan(
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

    return layout, candidates, len(units)


def row_from_selection(
    archive: Path,
    placements_member: str,
    board_id: str,
    strategy: GateStrategy,
    fallback_margin: float,
    layout: Layout,
    candidate_unit_count: int,
    candidates: dict[str, CandidatePlan],
) -> GateSensitivityRow:
    selected, fallback_triggered = select_for_strategy(
        candidates,
        strategy,
        fallback_margin=fallback_margin,
    )
    baseline = candidates["beam"].plan
    target = selected.plan
    baseline_metrics = baseline.metrics
    target_metrics = target.metrics
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    baseline_events = tool_event_count(baseline_metrics)
    target_events = tool_event_count(target_metrics)
    config = strategy.config
    return GateSensitivityRow(
        archive=archive.name,
        case_name=case_name,
        placement_method=placement_method,
        seed=seed,
        placements_member=placements_member,
        board_id=board_id,
        strategy=strategy.name,
        gate_enabled=config.enabled,
        min_travel_saving_per_extra_event=config.min_travel_saving_per_extra_event,
        min_travel_saving_ratio_per_extra_event=(
            config.min_travel_saving_ratio_per_extra_event
        ),
        min_machining_saving=config.min_machining_saving,
        fallback_triggered=fallback_triggered,
        selected_source=selected.label,
        estimated_runtime_ms=estimated_runtime(candidates, fallback_triggered),
        topology_runtime_ms=candidates["topology"].runtime_ms,
        beam_runtime_ms=candidates["beam"].runtime_ms,
        polish_runtime_ms=candidates["polished"].runtime_ms,
        fallback_beam_runtime_ms=(
            candidates["fallback_beam"].runtime_ms if fallback_triggered else 0.0
        ),
        fallback_polish_runtime_ms=(
            candidates["fallback_polished"].runtime_ms if fallback_triggered else 0.0
        ),
        rectangle_count=len(layout.rectangles),
        rectangle_count_bin=rectangle_count_bin(len(layout.rectangles)),
        candidate_unit_count=candidate_unit_count,
        selected_unit_count=len(target.selected_units),
        action_count=len(target.actions),
        baseline_tool_event_count=baseline_events,
        target_tool_event_count=target_events,
        tool_event_delta=target_events - baseline_events,
        pierce_delta=target_metrics.pierce_count - baseline_metrics.pierce_count,
        lift_delta=target_metrics.lift_count - baseline_metrics.lift_count,
        safe_lift_delta=(
            target_metrics.safe_lift_count - baseline_metrics.safe_lift_count
        ),
        baseline_travel_mode_cost=baseline_metrics.travel_mode_cost,
        target_travel_mode_cost=target_metrics.travel_mode_cost,
        travel_mode_cost_delta=(
            target_metrics.travel_mode_cost - baseline_metrics.travel_mode_cost
        ),
        travel_mode_cost_reduction_pct=reduction_pct(
            baseline_metrics.travel_mode_cost,
            target_metrics.travel_mode_cost,
        ),
        baseline_machining_cost=baseline_metrics.machining_cost,
        target_machining_cost=target_metrics.machining_cost,
        machining_cost_delta=(
            target_metrics.machining_cost - baseline_metrics.machining_cost
        ),
        machining_cost_reduction_pct=reduction_pct(
            baseline_metrics.machining_cost,
            target_metrics.machining_cost,
        ),
        baseline_detour_count=baseline_metrics.detour_count,
        target_detour_count=target_metrics.detour_count,
        detour_delta=target_metrics.detour_count - baseline_metrics.detour_count,
        hard_penalty=target_metrics.hard_penalty,
        stability_penalty=target_metrics.stability_penalty,
        air_move_distance=target_metrics.air_move_distance,
        cutting_length=target_metrics.cutting_length,
    )


def decision_rows_for_strategy(
    archive: Path,
    placements_member: str,
    board_id: str,
    strategy: GateStrategy,
    fallback_margin: float,
    candidates: dict[str, CandidatePlan],
) -> list[GateDecisionRow]:
    selected, fallback_triggered = select_for_strategy(
        candidates,
        strategy,
        fallback_margin=fallback_margin,
    )
    baseline = candidates["beam"].plan
    baseline_metrics = baseline.metrics
    case_name, placement_method, seed = parse_member_metadata(placements_member)
    candidate_labels = ("topology", "beam", "polished")
    if fallback_triggered:
        candidate_labels += ("fallback_polished",)

    rows: list[GateDecisionRow] = []
    for label in candidate_labels:
        candidate = candidates[label]
        metrics = candidate.plan.metrics
        protected = label == "beam"
        passed, reason, required, travel_saving, machining_saving, extra_events = (
            gate_requirement(baseline_metrics, metrics, strategy.config)
        )
        if protected:
            passed = True
            reason = "protected_beam"
        rows.append(
            GateDecisionRow(
                archive=archive.name,
                case_name=case_name,
                placement_method=placement_method,
                seed=seed,
                placements_member=placements_member,
                board_id=board_id,
                strategy=strategy.name,
                candidate_source=label,
                candidate_evaluated=True,
                selected=candidate.plan is selected.plan,
                protected_candidate=protected,
                gate_passed=passed,
                gate_reason=reason,
                required_travel_saving=required,
                travel_saving=travel_saving,
                machining_saving=machining_saving,
                extra_tool_events=extra_events,
                baseline_tool_event_count=tool_event_count(baseline_metrics),
                candidate_tool_event_count=tool_event_count(metrics),
                baseline_travel_mode_cost=baseline_metrics.travel_mode_cost,
                candidate_travel_mode_cost=metrics.travel_mode_cost,
                baseline_machining_cost=baseline_metrics.machining_cost,
                candidate_machining_cost=metrics.machining_cost,
                baseline_detour_count=baseline_metrics.detour_count,
                candidate_detour_count=metrics.detour_count,
                hard_penalty=metrics.hard_penalty,
                stability_penalty=metrics.stability_penalty,
            )
        )
    return rows


def summarize_rows(rows: list[GateSensitivityRow]) -> list[GateSensitivitySummaryRow]:
    grouped: dict[str, list[GateSensitivityRow]] = defaultdict(list)
    for row in rows:
        grouped[row.strategy].append(row)

    output: list[GateSensitivitySummaryRow] = []
    for strategy in DEFAULT_STRATEGIES:
        group = grouped.get(strategy, [])
        if not group:
            continue
        deltas = [row.tool_event_delta for row in group]
        increase_rows = [row for row in group if row.tool_event_delta > 0]
        source_counts = Counter(row.selected_source for row in group)
        first = group[0]
        output.append(
            GateSensitivitySummaryRow(
                strategy=strategy,
                gate_enabled=first.gate_enabled,
                min_travel_saving_per_extra_event=(
                    first.min_travel_saving_per_extra_event
                ),
                min_travel_saving_ratio_per_extra_event=(
                    first.min_travel_saving_ratio_per_extra_event
                ),
                min_machining_saving=first.min_machining_saving,
                n=len(group),
                target_travel_mode_cost_mean=mean(
                    row.target_travel_mode_cost for row in group
                ),
                target_machining_cost_mean=mean(
                    row.target_machining_cost for row in group
                ),
                estimated_runtime_ms_mean=mean(row.estimated_runtime_ms for row in group),
                tool_event_decrease_count=sum(1 for delta in deltas if delta < 0),
                tool_event_tie_count=sum(1 for delta in deltas if delta == 0),
                tool_event_increase_count=sum(1 for delta in deltas if delta > 0),
                tool_event_delta_mean=mean(deltas),
                travel_mode_cost_reduction_pct_mean=mean(
                    row.travel_mode_cost_reduction_pct for row in group
                ),
                machining_cost_reduction_pct_mean=mean(
                    row.machining_cost_reduction_pct for row in group
                ),
                increase_cases_travel_reduction_pct_mean=(
                    mean(row.travel_mode_cost_reduction_pct for row in increase_rows)
                    if increase_rows
                    else 0.0
                ),
                increase_cases_machining_reduction_pct_mean=(
                    mean(row.machining_cost_reduction_pct for row in increase_rows)
                    if increase_rows
                    else 0.0
                ),
                fallback_trigger_rate_pct=(
                    100.0
                    * sum(1 for row in group if row.fallback_triggered)
                    / len(group)
                ),
                source_topology_count=source_counts["topology"],
                source_beam_count=source_counts["beam"],
                source_polished_count=source_counts["polished"],
                source_fallback_polished_count=source_counts["fallback_polished"],
            )
        )
    return output


def write_dataclass_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [field.name for field in fields(rows[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def append_dataclass_rows(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(rows[0])]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def coerce_value(name: str, value: str):
    if value == "":
        return None
    if name in ROW_BOOL_FIELDS:
        return value in {"True", "true", "1"}
    if name in ROW_INT_FIELDS:
        return int(float(value))
    if name in ROW_FLOAT_FIELDS:
        return float(value)
    return value


ROW_BOOL_FIELDS = {
    "gate_enabled",
    "fallback_triggered",
    "candidate_evaluated",
    "selected",
    "protected_candidate",
    "gate_passed",
}
ROW_INT_FIELDS = {
    "rectangle_count",
    "candidate_unit_count",
    "selected_unit_count",
    "action_count",
    "baseline_tool_event_count",
    "target_tool_event_count",
    "tool_event_delta",
    "pierce_delta",
    "lift_delta",
    "safe_lift_delta",
    "baseline_detour_count",
    "target_detour_count",
    "detour_delta",
    "extra_tool_events",
    "candidate_tool_event_count",
    "candidate_detour_count",
}
ROW_FLOAT_FIELDS = {
    "min_travel_saving_per_extra_event",
    "min_travel_saving_ratio_per_extra_event",
    "min_machining_saving",
    "estimated_runtime_ms",
    "topology_runtime_ms",
    "beam_runtime_ms",
    "polish_runtime_ms",
    "fallback_beam_runtime_ms",
    "fallback_polish_runtime_ms",
    "baseline_travel_mode_cost",
    "target_travel_mode_cost",
    "candidate_travel_mode_cost",
    "travel_mode_cost_delta",
    "travel_mode_cost_reduction_pct",
    "baseline_machining_cost",
    "target_machining_cost",
    "candidate_machining_cost",
    "machining_cost_delta",
    "machining_cost_reduction_pct",
    "required_travel_saving",
    "travel_saving",
    "machining_saving",
    "hard_penalty",
    "stability_penalty",
    "air_move_distance",
    "cutting_length",
}


def row_from_dict(raw: dict[str, str]) -> GateSensitivityRow:
    return GateSensitivityRow(
        **{
            field.name: coerce_value(field.name, raw.get(field.name, ""))
            for field in fields(GateSensitivityRow)
        }
    )


def decision_row_from_dict(raw: dict[str, str]) -> GateDecisionRow:
    return GateDecisionRow(
        **{
            field.name: coerce_value(field.name, raw.get(field.name, ""))
            for field in fields(GateDecisionRow)
        }
    )


def load_existing_rows(path: Path) -> list[GateSensitivityRow]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row_from_dict(row) for row in csv.DictReader(handle)]


def load_existing_decision_rows(path: Path) -> list[GateDecisionRow]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [decision_row_from_dict(row) for row in csv.DictReader(handle)]


def row_key(row: GateSensitivityRow) -> tuple[str, ...]:
    return row_key_from_parts(
        row.archive,
        row.placements_member,
        row.board_id,
        row.strategy,
    )


def row_key_from_parts(
    archive_name: str,
    placements_member: str,
    board_id: str,
    strategy: str,
) -> tuple[str, ...]:
    return archive_name, placements_member, board_id, strategy


def decision_row_key(row: GateDecisionRow) -> tuple[str, ...]:
    return row_key_from_parts(
        row.archive,
        row.placements_member,
        row.board_id,
        row.strategy,
    )


def case_completed(
    completed_keys: set[tuple[str, ...]],
    archive: Path,
    placements_member: str,
    board_id: str,
    strategies: tuple[GateStrategy, ...],
) -> bool:
    return all(
        row_key_from_parts(archive.name, placements_member, board_id, strategy.name)
        in completed_keys
        for strategy in strategies
    )


def plot_summary(rows: list[GateSensitivitySummaryRow], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row.strategy for row in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))
    axes[0][0].bar(
        x,
        [row.travel_mode_cost_reduction_pct_mean for row in rows],
        color="#009E73",
    )
    axes[0][0].set_title("Travel reduction")
    axes[0][0].set_ylabel("Mean reduction (%)")
    axes[0][1].bar(
        x,
        [row.tool_event_increase_count for row in rows],
        color="#D55E00",
    )
    axes[0][1].set_title("Extra tool-event cases")
    axes[0][1].set_ylabel("Case count")
    axes[1][0].bar(
        x,
        [row.tool_event_delta_mean for row in rows],
        color="#0072B2",
    )
    axes[1][0].set_title("Mean tool-event delta")
    axes[1][0].set_ylabel("Events")
    axes[1][1].bar(
        x,
        [row.estimated_runtime_ms_mean for row in rows],
        color="#CC79A7",
    )
    axes[1][1].set_title("Estimated runtime")
    axes[1][1].set_ylabel("Mean ms")
    for ax in axes.ravel():
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_tool_event_gate_sensitivity.pdf")
    fig.savefig(output_dir / "fig_tool_event_gate_sensitivity.png", dpi=300)
    plt.close(fig)


def prepare_stream_output(path: Path, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.write_text("", encoding="utf-8")


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
    parser.add_argument("--fallback-margin", type=float, default=1000.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--task-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "tool_event_gate_sensitivity_real_20_50.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT
        / "results"
        / "tool_event_gate_sensitivity_real_20_50_summary.csv",
    )
    parser.add_argument(
        "--increase-output",
        type=Path,
        default=ROOT
        / "results"
        / "tool_event_gate_sensitivity_real_20_50_increase_cases.csv",
    )
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=ROOT
        / "results"
        / "tool_event_gate_current_decisions_real_20_50.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures" / "tool_event_gate_sensitivity_real_20_50",
    )
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--resume", action="store_true")
    add_experiment_preset_arg(parser)
    add_stability_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = apply_experiment_preset(parse_args())
    args.zip_paths = tuple(args.zip_paths) if args.zip_paths else DEFAULT_ARCHIVES
    strategies = tuple(strategy_by_name(name) for name in args.strategies)
    progress_output = args.progress_output or default_progress_log_path(args.output)
    args.progress_output = progress_output

    cases = tuple(iter_cases(args))
    progress_bar = TerminalProgressBar(len(cases), enabled=not args.no_progress_bar)
    progress_bar.start("tool-event gate cases")
    rows: list[GateSensitivityRow] = load_existing_rows(args.output) if args.resume else []
    decision_rows: list[GateDecisionRow] = (
        load_existing_decision_rows(args.decision_output) if args.resume else []
    )
    completed_keys = {row_key(row) for row in rows}
    completed_decision_keys = {decision_row_key(row) for row in decision_rows}
    prepare_stream_output(args.output, resume=args.resume)
    prepare_progress_log(progress_output, resume=args.resume)
    if args.resume and rows:
        print(f"resume: loaded {len(rows)} existing rows from {args.output}")
    if args.resume and decision_rows:
        print(
            f"resume: loaded {len(decision_rows)} existing decision rows "
            f"from {args.decision_output}"
        )

    case_count = 0
    skipped_case_count = 0
    timeout_count = 0
    for archive, member, board_id in cases:
        case_count += 1
        sensitivity_complete = case_completed(
            completed_keys,
            archive,
            member,
            board_id,
            strategies,
        )
        current_decision_key = row_key_from_parts(
            archive.name,
            member,
            board_id,
            "current",
        )
        needs_current_decisions = (
            any(strategy.name == "current" for strategy in strategies)
            and current_decision_key not in completed_decision_keys
        )
        if sensitivity_complete and not needs_current_decisions:
            skipped_case_count += 1
            append_progress_event(
                progress_output,
                new_progress_event(
                    event="skipped",
                    archive=archive.name,
                    placements_member=member,
                    board_id=board_id,
                    method="tool_event_gate_sensitivity",
                    message="all strategy rows already present in output CSV",
                ),
            )
            print(f"    skip completed gate case: board={board_id}")
            progress_bar.advance("skipped", f"board={board_id}")
            continue

        print(f"    run gate candidates: board={board_id}")
        append_progress_event(
            progress_output,
            new_progress_event(
                event="started",
                archive=archive.name,
                placements_member=member,
                board_id=board_id,
                method="tool_event_gate_sensitivity",
            ),
        )
        try:
            with task_timeout(args.task_timeout_seconds):
                layout, candidates, candidate_count = build_case_candidates(
                    archive,
                    member,
                    board_id,
                    args,
                    strategies,
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
                    method="tool_event_gate_sensitivity",
                    message=f"timeout after {args.task_timeout_seconds:g} seconds",
                ),
            )
            print(f"    timed out gate case: board={board_id}")
            progress_bar.advance("timed_out", f"board={board_id}")
            continue

        new_rows = []
        new_decision_rows = []
        for strategy in strategies:
            key = row_key_from_parts(archive.name, member, board_id, strategy.name)
            if key not in completed_keys:
                row = row_from_selection(
                    archive,
                    member,
                    board_id,
                    strategy,
                    args.fallback_margin,
                    layout,
                    candidate_count,
                    candidates,
                )
                rows.append(row)
                new_rows.append(row)
                completed_keys.add(key)
            if strategy.name == "current" and key not in completed_decision_keys:
                new_decision_rows.extend(
                    decision_rows_for_strategy(
                        archive,
                        member,
                        board_id,
                        strategy,
                        args.fallback_margin,
                        candidates,
                    )
                )
                completed_decision_keys.add(key)
        append_dataclass_rows(args.output, new_rows)
        decision_rows.extend(new_decision_rows)
        append_progress_event(
            progress_output,
            new_progress_event(
                event="completed",
                archive=archive.name,
                placements_member=member,
                board_id=board_id,
                method="tool_event_gate_sensitivity",
                rectangle_count=len(layout.rectangles),
                candidate_unit_count=candidate_count,
                message=(
                    f"appended {len(new_rows)} gate-sensitivity rows and "
                    f"{len(new_decision_rows)} gate-decision rows"
                ),
            ),
        )
        print(
            f"    done gate candidates: board={board_id} "
            f"rows={len(new_rows)} decisions={len(new_decision_rows)}"
        )
        progress_bar.advance("completed", f"board={board_id}")

    summary = summarize_rows(rows)
    increase_rows = [row for row in rows if row.tool_event_delta > 0]
    write_dataclass_rows(args.summary_output, summary)
    write_dataclass_rows(args.increase_output, increase_rows)
    write_dataclass_rows(args.decision_output, decision_rows)
    plot_summary(summary, args.figure_dir)
    print(
        f"cases={case_count} skipped={skipped_case_count} "
        f"timed_out={timeout_count} rows={len(rows)}"
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.increase_output}")
    print(f"wrote: {args.decision_output}")
    print(f"wrote: {progress_output}")
    print(f"figures: {args.figure_dir}")
    for row in summary:
        print(
            f"{row.strategy:<10s} n={row.n:<3} "
            f"events={row.tool_event_decrease_count}/"
            f"{row.tool_event_tie_count}/{row.tool_event_increase_count} "
            f"travel={row.travel_mode_cost_reduction_pct_mean:6.2f}% "
            f"machining={row.machining_cost_reduction_pct_mean:6.2f}%"
        )


if __name__ == "__main__":
    main()
