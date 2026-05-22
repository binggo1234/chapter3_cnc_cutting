from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_statistical_robustness import (  # noqa: E402
    bootstrap_mean_ci,
    exact_sign_test_p_value,
    read_rows,
    robustness_rows,
)


FIELDS = (
    "archive",
    "case_name",
    "placement_method",
    "seed",
    "placements_member",
    "board_id",
    "method",
    "runtime_ms",
    "rectangle_count",
    "air_move_distance",
    "cutting_length",
    "pierce_count",
    "lift_count",
    "safe_lift_count",
    "detour_count",
    "detour_distance",
    "travel_mode_cost",
    "hard_penalty",
    "stability_penalty",
)


def row(
    case: str,
    method: str,
    travel: float,
    runtime: float,
    stability: float = 0.0,
) -> dict[str, str]:
    return {
        "archive": "a.zip",
        "case_name": "case",
        "placement_method": "placement",
        "seed": "seed",
        "placements_member": f"{case}/placements.csv",
        "board_id": case,
        "method": method,
        "runtime_ms": f"{runtime}",
        "rectangle_count": "10",
        "air_move_distance": f"{travel}",
        "cutting_length": "1000",
        "pierce_count": "5",
        "lift_count": "5",
        "safe_lift_count": "0",
        "detour_count": "2",
        "detour_distance": "20",
        "travel_mode_cost": f"{travel}",
        "hard_penalty": "0",
        "stability_penalty": f"{stability}",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_sign_test_p_value_is_two_sided() -> None:
    assert exact_sign_test_p_value(3, 0) == 0.25
    assert exact_sign_test_p_value(2, 1) == 1.0
    assert exact_sign_test_p_value(0, 0) is None


def test_bootstrap_mean_ci_is_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0], samples=100, seed=7)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0], samples=100, seed=7)

    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_robustness_rows_report_reduction_and_win_loss(tmp_path: Path) -> None:
    path = tmp_path / "routes.csv"
    write_rows(
        path,
        [
            row("1", "baseline", 100.0, 10.0),
            row("1", "target", 90.0, 12.0),
            row("2", "baseline", 100.0, 10.0),
            row("2", "target", 95.0, 11.0),
            row("3", "baseline", 100.0, 10.0),
            row("3", "target", 100.0, 9.0),
        ],
    )

    rows = robustness_rows(
        read_rows(path),
        target_method="target",
        baseline_methods=("baseline",),
        metrics=("travel_mode_cost", "runtime_ms", "process_key"),
        bootstrap_samples=100,
        bootstrap_seed=3,
    )
    by_metric = {row.metric: row for row in rows}

    travel = by_metric["travel_mode_cost"]
    runtime = by_metric["runtime_ms"]
    process = by_metric["process_key"]

    assert travel.paired_cases == 3
    assert travel.reduction_mean == 5.0
    assert travel.wins == 2
    assert travel.ties == 1
    assert travel.losses == 0
    assert runtime.losses == 2
    assert runtime.wins == 1
    assert process.wins == 2
    assert process.ties == 1
