from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass, fields
from math import comb, sqrt
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CASE_KEY_CANDIDATES = (
    ("archive", "placements_member", "board_id"),
    ("scenario", "size", "repeat"),
    ("case_name", "placement_method", "seed", "board_id"),
)
DEFAULT_METRICS = (
    "process_key",
    "machining_cost",
    "travel_mode_cost",
    "stability_penalty",
    "hard_penalty",
    "tool_event_count",
    "detour_count",
    "runtime_ms",
)
PROCESS_KEY_FIELDS = (
    "hard_penalty",
    "stability_penalty",
    "machining_cost",
    "tool_event_count",
    "travel_mode_cost",
    "air_move_distance",
    "turn_penalty",
    "negative_continuity_reward",
)


@dataclass(frozen=True)
class ResultRow:
    values: dict[str, str]
    case_id: str
    method: str

    def number(self, field: str, default: float = 0.0) -> float:
        if field == "machining_cost" and field not in self.values:
            return self.number("cutting_length", default) + self.number(
                "travel_mode_cost",
                default,
            )
        if field == "tool_event_count":
            return (
                self.number("pierce_count", default)
                + self.number("lift_count", default)
                + self.number("safe_lift_count", default)
            )
        if field == "negative_continuity_reward":
            return -self.number("continuity_reward", default)
        value = self.values.get(field, "")
        if value == "":
            return default
        return float(value)


@dataclass(frozen=True)
class RobustnessRow:
    baseline_method: str
    target_method: str
    metric: str
    paired_cases: int
    baseline_mean: float
    baseline_std: float
    baseline_median: float
    target_mean: float
    target_std: float
    target_median: float
    reduction_mean: float
    reduction_std: float
    reduction_median: float
    reduction_ci95_low: float
    reduction_ci95_high: float
    absolute_delta_mean: float
    absolute_delta_median: float
    wins: int
    ties: int
    losses: int
    win_rate: float
    sign_test_p: float | None
    effect_size: float


def infer_case_key_fields(fieldnames: Iterable[str]) -> tuple[str, ...]:
    available = set(fieldnames)
    for candidate in CASE_KEY_CANDIDATES:
        if all(field in available for field in candidate):
            return candidate
    fallback = tuple(
        field
        for field in ("case_id", "board_id", "size", "repeat")
        if field in available
    )
    if fallback:
        return fallback
    raise ValueError(
        "cannot infer case identity fields; expected archive/placements_member/board_id "
        "or scenario/size/repeat"
    )


def case_id(row: dict[str, str], key_fields: tuple[str, ...]) -> str:
    return "|".join(row.get(field, "") for field in key_fields)


def read_rows(path: Path) -> tuple[ResultRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        key_fields = infer_case_key_fields(reader.fieldnames)
        rows = [
            ResultRow(row, case_id(row, key_fields), row["method"])
            for row in reader
            if row.get("method", "")
        ]
    if not rows:
        raise ValueError(f"no method rows found in {path}")
    return tuple(rows)


def group_by_case(rows: tuple[ResultRow, ...]) -> dict[str, dict[str, ResultRow]]:
    grouped: dict[str, dict[str, ResultRow]] = defaultdict(dict)
    for row in rows:
        grouped[row.case_id][row.method] = row
    return grouped


def process_key(row: ResultRow) -> tuple[float, ...]:
    return tuple(row.number(field) for field in PROCESS_KEY_FIELDS)


def value_for_metric(row: ResultRow, metric: str) -> float | tuple[float, ...]:
    if metric == "process_key":
        return process_key(row)
    return row.number(metric)


def numeric_value_for_summary(row: ResultRow, metric: str) -> float:
    if metric == "process_key":
        # Process-key is lexicographic. Machining cost is the most interpretable
        # scalar companion for mean/std columns while wins use the full key.
        return row.number("machining_cost")
    return row.number(metric)


def metric_winner(target: ResultRow, baseline: ResultRow, metric: str) -> str:
    target_value = value_for_metric(target, metric)
    baseline_value = value_for_metric(baseline, metric)
    if target_value < baseline_value:
        return "target"
    if target_value > baseline_value:
        return "baseline"
    return "tie"


def relative_reduction_pct(baseline: float, target: float) -> float:
    if abs(baseline) < 1e-12:
        if abs(target) < 1e-12:
            return 0.0
        return -100.0 if target > baseline else 100.0
    return 100.0 * (baseline - target) / abs(baseline)


def exact_sign_test_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    lower_tail = sum(comb(trials, k) for k in range(observed + 1)) / (2**trials)
    return min(1.0, 2.0 * lower_tail)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    means = [
        mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def cohen_d(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    std = pstdev(values)
    if std < 1e-12:
        return 0.0
    return mean(values) / std


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def robustness_rows(
    rows: tuple[ResultRow, ...],
    *,
    target_method: str,
    baseline_methods: tuple[str, ...],
    metrics: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[RobustnessRow]:
    grouped = group_by_case(rows)
    output: list[RobustnessRow] = []
    for baseline_method in baseline_methods:
        pairs = [
            (case_rows[baseline_method], case_rows[target_method])
            for case_rows in grouped.values()
            if baseline_method in case_rows and target_method in case_rows
        ]
        if not pairs:
            continue
        for metric_index, metric in enumerate(metrics):
            baseline_values = [
                numeric_value_for_summary(baseline, metric)
                for baseline, _target in pairs
            ]
            target_values = [
                numeric_value_for_summary(target, metric)
                for _baseline, target in pairs
            ]
            reductions = [
                relative_reduction_pct(baseline_value, target_value)
                for baseline_value, target_value in zip(baseline_values, target_values)
            ]
            absolute_deltas = [
                target_value - baseline_value
                for baseline_value, target_value in zip(baseline_values, target_values)
            ]
            wins = sum(
                1
                for baseline, target in pairs
                if metric_winner(target, baseline, metric) == "target"
            )
            ties = sum(
                1
                for baseline, target in pairs
                if metric_winner(target, baseline, metric) == "tie"
            )
            losses = sum(
                1
                for baseline, target in pairs
                if metric_winner(target, baseline, metric) == "baseline"
            )
            ci_low, ci_high = bootstrap_mean_ci(
                reductions,
                samples=bootstrap_samples,
                seed=bootstrap_seed + metric_index + 1009 * len(output),
            )
            output.append(
                RobustnessRow(
                    baseline_method=baseline_method,
                    target_method=target_method,
                    metric=metric,
                    paired_cases=len(pairs),
                    baseline_mean=mean(baseline_values) if baseline_values else 0.0,
                    baseline_std=(
                        pstdev(baseline_values) if len(baseline_values) > 1 else 0.0
                    ),
                    baseline_median=median(baseline_values) if baseline_values else 0.0,
                    target_mean=mean(target_values) if target_values else 0.0,
                    target_std=(
                        pstdev(target_values) if len(target_values) > 1 else 0.0
                    ),
                    target_median=median(target_values) if target_values else 0.0,
                    reduction_mean=mean(reductions) if reductions else 0.0,
                    reduction_std=pstdev(reductions) if len(reductions) > 1 else 0.0,
                    reduction_median=median(reductions) if reductions else 0.0,
                    reduction_ci95_low=ci_low,
                    reduction_ci95_high=ci_high,
                    absolute_delta_mean=mean(absolute_deltas) if absolute_deltas else 0.0,
                    absolute_delta_median=(
                        median(absolute_deltas) if absolute_deltas else 0.0
                    ),
                    wins=wins,
                    ties=ties,
                    losses=losses,
                    win_rate=wins / len(pairs) if pairs else 0.0,
                    sign_test_p=exact_sign_test_p_value(wins, losses),
                    effect_size=cohen_d(reductions),
                )
            )
    return output


def write_rows(path: Path, rows: list[RobustnessRow]) -> None:
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


def write_markdown(path: Path, rows: list[RobustnessRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical robustness summary",
        "",
        "Positive reduction values mean the target method is lower than the baseline.",
        "Win/tie/loss counts use lower-is-better comparison; `process_key` uses the full lexicographic process objective.",
        "",
        "| baseline | metric | n | reduction mean | 95% CI | win/tie/loss | sign-test p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        p_value = "<1e-4" if row.sign_test_p is not None and row.sign_test_p < 1e-4 else fmt(row.sign_test_p)
        lines.append(
            "| "
            + " | ".join(
                (
                    row.baseline_method,
                    row.metric,
                    str(row.paired_cases),
                    f"{row.reduction_mean:.2f}%",
                    f"[{row.reduction_ci95_low:.2f}, {row.reduction_ci95_high:.2f}]",
                    f"{row.wins}/{row.ties}/{row.losses}",
                    p_value,
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "adaptive_event_gate_protected_real_20x3_20_50.csv",
    )
    parser.add_argument("--target-method", default="process_aware_beam_adaptive_polished")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=(
            "topology_process_aware",
            "process_local_search_multistart",
            "process_aware_beam",
            "path_distance_local_search",
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260522)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "statistical_robustness_real_20_50.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "statistical_robustness_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = robustness_rows(
        read_rows(args.input),
        target_method=args.target_method,
        baseline_methods=tuple(args.baseline_methods),
        metrics=tuple(args.metrics),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_rows(args.output, rows)
    write_markdown(args.markdown_output, rows)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.markdown_output}")
    for row in rows:
        if row.metric in {"process_key", "machining_cost", "travel_mode_cost"}:
            p_value = (
                ""
                if row.sign_test_p is None
                else "<1e-4"
                if row.sign_test_p < 1e-4
                else f"{row.sign_test_p:.4f}"
            )
            print(
                f"{row.baseline_method:<32s} {row.metric:<18s} "
                f"n={row.paired_cases:<3} reduction={row.reduction_mean:6.2f}% "
                f"ci=[{row.reduction_ci95_low:6.2f}, {row.reduction_ci95_high:6.2f}] "
                f"w/t/l={row.wins}/{row.ties}/{row.losses} p={p_value}"
            )


if __name__ == "__main__":
    main()
