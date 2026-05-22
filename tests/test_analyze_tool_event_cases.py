from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_tool_event_cases import (  # noqa: E402
    compare_tool_events,
    read_rows,
    summarize,
)


FIELDS = (
    "archive",
    "case_name",
    "placement_method",
    "seed",
    "placements_member",
    "board_id",
    "method",
    "rectangle_count",
    "air_move_distance",
    "cutting_length",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
    "travel_mode_cost",
    "runtime_ms",
)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(case: str, method: str, pierce: int, lift: int, safe: int, travel: float) -> dict[str, str]:
    return {
        "archive": "a.zip",
        "case_name": "case",
        "placement_method": "placement",
        "seed": "seed",
        "placements_member": f"{case}/placements.csv",
        "board_id": case,
        "method": method,
        "rectangle_count": "10",
        "air_move_distance": f"{travel}",
        "cutting_length": "1000",
        "pierce_count": str(pierce),
        "lift_count": str(lift),
        "safe_lift_count": str(safe),
        "detour_count": "4",
        "travel_mode_cost": f"{travel}",
        "runtime_ms": "12.5",
    }


def test_tool_event_analysis_counts_decrease_tie_and_increase(tmp_path: Path) -> None:
    path = tmp_path / "routes.csv"
    write_rows(
        path,
        [
            row("decrease", "process_aware_beam", 5, 5, 1, 100.0),
            row("decrease", "process_aware_beam_adaptive_polished", 4, 5, 1, 90.0),
            row("tie", "process_aware_beam", 5, 5, 0, 100.0),
            row("tie", "process_aware_beam_adaptive_polished", 5, 5, 0, 95.0),
            row("increase", "process_aware_beam", 5, 5, 0, 100.0),
            row("increase", "process_aware_beam_adaptive_polished", 6, 5, 1, 80.0),
        ],
    )

    comparisons = compare_tool_events(
        read_rows(path),
        baseline_method="process_aware_beam",
        target_method="process_aware_beam_adaptive_polished",
    )
    summary = summarize(comparisons)
    increased = [row for row in comparisons if row.tool_event_delta > 0.0]

    assert summary.paired_cases == 3
    assert summary.tool_event_decrease_count == 1
    assert summary.tool_event_tie_count == 1
    assert summary.tool_event_increase_count == 1
    assert summary.tool_event_delta_mean == 1 / 3
    assert increased[0].board_id == "increase"
    assert increased[0].tool_event_delta == 2
    assert increased[0].pierce_delta == 1
    assert increased[0].safe_lift_delta == 1
    assert increased[0].travel_mode_cost_reduction_pct == 20.0
