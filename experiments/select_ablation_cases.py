from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIORITIES = (
    "single_edges_only:machining_cost_reduction_pct",
    "topology_no_beam:machining_cost_reduction_pct",
    "no_stability_guidance:stability_penalty_reduction",
    "path_distance_baseline:stability_penalty_reduction",
    "no_adjacency_support_guidance:machining_cost_reduction_pct",
    "no_safe_travel_modes:hard_penalty_delta_abs",
)


def parse_float(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    return float(value)


def load_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def split_case_id(case_id: str) -> tuple[str, str, str]:
    parts = case_id.split("|", 2)
    if len(parts) != 3:
        return (case_id, "", "")
    return (parts[0], parts[1], parts[2])


def score_row(row: dict[str, str], metric: str) -> float:
    if metric == "hard_penalty_delta_abs":
        return abs(parse_float(row["hard_penalty_delta"]))
    return parse_float(row[metric])


def select_cases(
    rows: tuple[dict[str, str], ...],
    priorities: tuple[str, ...],
    top_k: int,
) -> tuple[dict[str, str], ...]:
    by_baseline: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_baseline[row["baseline_method"]].append(row)

    selected: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for priority in priorities:
        baseline, metric = priority.split(":", 1)
        candidates = by_baseline.get(baseline, ())
        ranked = sorted(
            candidates,
            key=lambda row: (
                score_row(row, metric),
                parse_float(row.get("rectangle_count", "")),
                row["case_id"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked[:top_k], start=1):
            key = (baseline, row["case_id"])
            if key in seen_keys:
                continue
            archive, placements_member, board_id = split_case_id(row["case_id"])
            selected_row = dict(row)
            selected_row.update(
                {
                    "selection_priority": priority,
                    "selection_rank": str(rank),
                    "archive": archive,
                    "placements_member": placements_member,
                    "board_id": board_id,
                    "selection_score": f"{score_row(row, metric):.6f}",
                }
            )
            selected.append(selected_row)
            seen_keys.add(key)
    return tuple(selected)


def write_rows(rows: tuple[dict[str, str], ...], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "--priorities",
        nargs="+",
        default=list(DEFAULT_PRIORITIES),
        help="Selection rule formatted as baseline:metric.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "ablation_representative_cases.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.paired_comparison)
    selected = select_cases(rows, tuple(args.priorities), top_k=args.top_k)
    write_rows(selected, args.output)
    print(f"paired_comparison: {args.paired_comparison}")
    print(f"selected_cases: {len(selected)}")
    print(f"output: {args.output}")
    for row in selected:
        print(
            f"{row['selection_priority']:<48} board={row['board_id']:<6} "
            f"score={row['selection_score']:<10} baseline={row['baseline_method']}"
        )


if __name__ == "__main__":
    main()
