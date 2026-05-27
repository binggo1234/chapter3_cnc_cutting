from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_scalability import (  # noqa: E402
    ScalabilityRow,
    append_rows,
    count_planned_method_tasks,
    load_existing_rows,
    scalability_key,
    scalability_key_from_parts,
)
from run_chapter2_batch import batch_row_from_dict  # noqa: E402


def sample_row() -> ScalabilityRow:
    return ScalabilityRow(
        scenario="grid",
        size=10,
        method="process_aware_beam",
        repeat=0,
        support_policy="all_edges",
        min_support_count=1,
        min_support_ratio=0.75,
        min_area_normalized_support=0.0,
        adjacency_support_weight=1.0,
        topology_candidate_pool_size=None,
        beam_width=8,
        beam_candidate_pool_size=24,
        beam_max_expansions_per_node=48,
        beam_max_layer_expansions=56,
        beam_diversity_bucket_limit=1,
        beam_min_expansions_per_parent=0,
        beam_unstable_min_expansions_per_parent=2,
        beam_unstable_layer_expansion_multiplier=1.0,
        beam_unstable_layer_expansion_bonus=0,
        beam_completion_aware_prerank=False,
        beam_unstable_completion_focus_count=None,
        runtime_ms=12.5,
        rectangle_count=10,
        candidate_unit_count=40,
        selected_unit_count=40,
        action_count=80,
        air_move_distance=100.0,
        cutting_length=200.0,
        pierce_count=1,
        lift_count=1,
        safe_lift_count=0,
        safe_lift_distance=0.0,
        detour_count=0,
        detour_distance=0.0,
        travel_mode_cost=100.0,
        stability_penalty=0.0,
        hard_penalty=0.0,
    )


def test_scalability_rows_can_round_trip_for_resume(tmp_path: Path) -> None:
    output = tmp_path / "scalability.csv"
    row = sample_row()

    append_rows((row,), output)
    loaded = load_existing_rows(output)

    assert loaded == (row,)
    assert scalability_key(loaded[0]) == scalability_key(row)


def test_scalability_key_from_parts_matches_beam_row_defaults() -> None:
    args = argparse.Namespace(
        support_policy="all_edges",
        min_support_count=1,
        min_support_ratio=0.75,
        min_area_normalized_support=0.0,
        adjacency_support_weight=1.0,
        beam_width=None,
        beam_candidate_pool_size=None,
        beam_max_expansions_per_node=None,
        beam_max_layer_expansions=None,
        beam_diversity_bucket_limit=None,
        beam_min_expansions_per_parent=None,
        beam_unstable_min_expansions_per_parent=None,
        beam_unstable_layer_expansion_multiplier=None,
        beam_unstable_layer_expansion_bonus=None,
        beam_completion_aware_prerank=None,
        beam_unstable_completion_focus_count=None,
    )

    assert scalability_key_from_parts(
        "grid",
        10,
        0,
        "process_aware_beam",
        args,
    ) == scalability_key(sample_row())


def test_count_planned_method_tasks() -> None:
    assert count_planned_method_tasks((10, 50), 3, ("a", "b")) == 12


def test_batch_row_from_legacy_csv_uses_default_tool_event_gate() -> None:
    raw = {
        "archive": "a.zip",
        "case_name": "case",
        "placement_method": "rh",
        "seed": "seed_1000",
        "placements_member": "placements.csv",
        "board_id": "1",
        "support_policy": "all_edges",
        "min_support_count": "1",
        "min_support_ratio": "0.75",
        "min_area_normalized_support": "0.0",
        "adjacency_support_weight": "1.0",
        "method": "process_aware_beam_adaptive_polished",
        "topology_candidate_pool_size": "96",
        "beam_width": "8",
        "beam_candidate_pool_size": "24",
        "beam_max_expansions_per_node": "48",
        "beam_max_layer_expansions": "56",
        "beam_diversity_bucket_limit": "1",
        "beam_min_expansions_per_parent": "0",
        "beam_unstable_min_expansions_per_parent": "2",
        "beam_unstable_layer_expansion_multiplier": "1.0",
        "beam_unstable_layer_expansion_bonus": "",
        "runtime_ms": "1.0",
        "rectangle_count": "20",
        "rectangle_count_bin": "000_020",
        "candidate_unit_count": "80",
        "selected_unit_count": "80",
        "action_count": "100",
        "air_move_distance": "10.0",
        "cutting_length": "20.0",
        "pierce_count": "1",
        "lift_count": "1",
        "collision_penalty": "0.0",
        "boundary_penalty": "0.0",
        "stability_penalty": "0.0",
        "safe_lift_count": "0",
        "safe_lift_distance": "0.0",
        "detour_count": "0",
        "detour_distance": "0.0",
        "travel_mode_cost": "10.0",
        "hard_penalty": "0.0",
    }

    row = batch_row_from_dict(raw)

    assert row.tool_event_gate_enabled is True
    assert row.tool_event_min_travel_saving == 100.0
    assert row.tool_event_min_travel_saving_ratio == 0.02
    assert row.repeat_cut_policy == "hard"
    assert row.coverage_missing_segment_count == 0
    assert row.cut_repeated_segment_count == 0
    assert row.selected_composite_coverage_ratio == 0.0
