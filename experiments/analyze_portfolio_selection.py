from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CASE_KEY = ("archive", "placements_member", "board_id")
MATCH_FIELDS = (
    "travel_mode_cost",
    "machining_cost",
    "hard_penalty",
    "stability_penalty",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
)


@dataclass(frozen=True)
class AttributionRow:
    archive: str
    placements_member: str
    board_id: str
    target_method: str
    source_label: str
    rectangle_count: int
    target_travel_mode_cost: float
    target_machining_cost: float
    target_runtime_ms: float


@dataclass(frozen=True)
class AttributionSummaryRow:
    target_method: str
    source_label: str
    n: int
    share_pct: float
    travel_mode_cost_mean: float
    machining_cost_mean: float
    runtime_ms_mean: float
    rectangle_count_mean: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in CASE_KEY)


def numeric(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return 0.0 if value == "" else float(value)


def same_solution(
    left: dict[str, str],
    right: dict[str, str],
    tolerance: float,
) -> bool:
    for field in MATCH_FIELDS:
        if abs(numeric(left, field) - numeric(right, field)) > tolerance:
            return False
    return True


def load_by_case_method(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[case_key(row)][row["method"]] = row
    return grouped


def attribute_target(
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]],
    target_method: str,
    component_methods: tuple[str, ...],
    unmatched_label: str,
    tolerance: float,
) -> list[AttributionRow]:
    output: list[AttributionRow] = []
    for key, by_method in sorted(grouped.items()):
        target = by_method.get(target_method)
        if target is None:
            continue
        source_label = unmatched_label
        for method in component_methods:
            component = by_method.get(method)
            if component is not None and same_solution(target, component, tolerance):
                source_label = method
                break
        output.append(
            AttributionRow(
                archive=key[0],
                placements_member=key[1],
                board_id=key[2],
                target_method=target_method,
                source_label=source_label,
                rectangle_count=int(float(target["rectangle_count"])),
                target_travel_mode_cost=numeric(target, "travel_mode_cost"),
                target_machining_cost=numeric(target, "machining_cost"),
                target_runtime_ms=numeric(target, "runtime_ms"),
            )
        )
    return output


def summarize(rows: list[AttributionRow]) -> list[AttributionSummaryRow]:
    by_source: dict[str, list[AttributionRow]] = defaultdict(list)
    for row in rows:
        by_source[row.source_label].append(row)
    total = len(rows)
    summaries: list[AttributionSummaryRow] = []
    for source_label, source_rows in sorted(
        by_source.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        summaries.append(
            AttributionSummaryRow(
                target_method=source_rows[0].target_method,
                source_label=source_label,
                n=len(source_rows),
                share_pct=(100.0 * len(source_rows) / total) if total else 0.0,
                travel_mode_cost_mean=mean(
                    row.target_travel_mode_cost for row in source_rows
                ),
                machining_cost_mean=mean(row.target_machining_cost for row in source_rows),
                runtime_ms_mean=mean(row.target_runtime_ms for row in source_rows),
                rectangle_count_mean=mean(row.rectangle_count for row in source_rows),
            )
        )
    return summaries


def write_dataclass_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "paper_main_with_adaptive_polished_reuse_real_20_50.csv",
    )
    parser.add_argument("--target-method", default="process_aware_beam_adaptive_polished")
    parser.add_argument(
        "--component-methods",
        nargs="+",
        default=(
            "process_aware_beam_polished",
            "process_aware_beam_adaptive",
            "process_aware_beam",
            "topology_process_aware",
        ),
    )
    parser.add_argument("--unmatched-label", default="portfolio_fallback_candidate")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "portfolio_selection_attribution.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "portfolio_selection_attribution_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = load_by_case_method(read_rows(args.input))
    rows = attribute_target(
        grouped,
        target_method=args.target_method,
        component_methods=tuple(args.component_methods),
        unmatched_label=args.unmatched_label,
        tolerance=args.tolerance,
    )
    summary = summarize(rows)
    write_dataclass_rows(args.output, rows)
    write_dataclass_rows(args.summary_output, summary)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")
    for row in summary:
        print(
            f"{row.source_label:32s} n={row.n:<4d} "
            f"share={row.share_pct:6.2f}% "
            f"cost={row.travel_mode_cost_mean:10.3f} "
            f"runtime={row.runtime_ms_mean:8.3f} ms"
        )


if __name__ == "__main__":
    main()
