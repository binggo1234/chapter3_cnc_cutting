from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from cnc_cutting.models import PathMetrics  # noqa: E402
from cnc_cutting.optimizer import RoutePlan  # noqa: E402
from run_tool_event_gate_sensitivity import (  # noqa: E402
    CandidatePlan,
    GateDecisionRow,
    decision_rows_for_strategy,
    gate_requirement,
    load_existing_decision_rows,
    select_for_strategy,
    strategy_by_name,
    summarize_rows,
    write_dataclass_rows,
)


def plan(
    travel_mode_cost: float,
    pierce_count: int,
    lift_count: int,
    cutting_length: float = 1000.0,
) -> RoutePlan:
    return RoutePlan(
        selected_units=(),
        actions=(),
        metrics=PathMetrics(
            cutting_length=cutting_length,
            travel_mode_cost=travel_mode_cost,
            pierce_count=pierce_count,
            lift_count=lift_count,
        ),
    )


def test_gate_strategy_changes_selected_candidate() -> None:
    candidates = {
        "topology": CandidatePlan("topology", plan(12000.0, 1, 1), 10.0),
        "beam": CandidatePlan("beam", plan(10000.0, 1, 1), 20.0),
        "polished": CandidatePlan("polished", plan(9500.0, 2, 3), 30.0),
        "fallback_beam": CandidatePlan("fallback_beam", plan(10000.0, 1, 1), 0.0),
        "fallback_polished": CandidatePlan(
            "fallback_polished",
            plan(9500.0, 2, 3),
            0.0,
        ),
    }

    no_gate_selected, _ = select_for_strategy(
        candidates,
        strategy_by_name("no_gate"),
        fallback_margin=1000.0,
    )
    current_selected, _ = select_for_strategy(
        candidates,
        strategy_by_name("current"),
        fallback_margin=1000.0,
    )

    assert no_gate_selected.label == "polished"
    assert current_selected.label == "beam"


def test_summarize_rows_orders_known_strategies() -> None:
    from run_tool_event_gate_sensitivity import GateSensitivityRow

    common = {
        "archive": "a.zip",
        "case_name": "case",
        "placement_method": "rh",
        "seed": "seed_1000",
        "placements_member": "placements.csv",
        "board_id": "1",
        "gate_enabled": True,
        "min_travel_saving_per_extra_event": 100.0,
        "min_travel_saving_ratio_per_extra_event": 0.02,
        "min_machining_saving": 1e-9,
        "fallback_triggered": False,
        "selected_source": "beam",
        "estimated_runtime_ms": 10.0,
        "topology_runtime_ms": 1.0,
        "beam_runtime_ms": 2.0,
        "polish_runtime_ms": 3.0,
        "fallback_beam_runtime_ms": 0.0,
        "fallback_polish_runtime_ms": 0.0,
        "rectangle_count": 20,
        "rectangle_count_bin": "000_020",
        "candidate_unit_count": 80,
        "selected_unit_count": 80,
        "action_count": 100,
        "baseline_tool_event_count": 2,
        "target_tool_event_count": 2,
        "tool_event_delta": 0,
        "pierce_delta": 0,
        "lift_delta": 0,
        "safe_lift_delta": 0,
        "baseline_travel_mode_cost": 100.0,
        "target_travel_mode_cost": 90.0,
        "travel_mode_cost_delta": -10.0,
        "travel_mode_cost_reduction_pct": 10.0,
        "baseline_machining_cost": 200.0,
        "target_machining_cost": 190.0,
        "machining_cost_delta": -10.0,
        "machining_cost_reduction_pct": 5.0,
        "baseline_detour_count": 1,
        "target_detour_count": 1,
        "detour_delta": 0,
        "hard_penalty": 0.0,
        "stability_penalty": 0.0,
        "air_move_distance": 90.0,
        "cutting_length": 100.0,
    }
    rows = [
        GateSensitivityRow(strategy="current", **common),
        GateSensitivityRow(strategy="no_gate", **{**common, "gate_enabled": False}),
    ]

    summary = summarize_rows(rows)

    assert [row.strategy for row in summary] == ["no_gate", "current"]


def test_gate_requirement_reports_required_saving() -> None:
    baseline = plan(10000.0, 1, 1).metrics
    candidate = plan(9500.0, 2, 3).metrics

    passed, reason, required, travel_saving, machining_saving, extra_events = (
        gate_requirement(baseline, candidate, strategy_by_name("current").config)
    )

    assert passed is False
    assert reason == "insufficient_travel_saving"
    assert required == 600.0
    assert travel_saving == 500.0
    assert machining_saving == 500.0
    assert extra_events == 3


def test_decision_rows_mark_selected_and_protected_candidates() -> None:
    candidates = {
        "topology": CandidatePlan("topology", plan(12000.0, 1, 1), 10.0),
        "beam": CandidatePlan("beam", plan(10000.0, 1, 1), 20.0),
        "polished": CandidatePlan("polished", plan(9500.0, 2, 3), 30.0),
        "fallback_beam": CandidatePlan("fallback_beam", plan(10000.0, 1, 1), 0.0),
        "fallback_polished": CandidatePlan(
            "fallback_polished",
            plan(9500.0, 2, 3),
            0.0,
        ),
    }

    rows = decision_rows_for_strategy(
        Path("a.zip"),
        "case/rh/seed_1000/placements.csv",
        "1",
        strategy_by_name("current"),
        fallback_margin=1000.0,
        candidates=candidates,
    )

    selected = [row for row in rows if row.selected]
    protected = [row for row in rows if row.protected_candidate]

    assert [row.candidate_source for row in selected] == ["beam"]
    assert [row.candidate_source for row in protected] == ["beam"]
    assert any(row.gate_reason == "insufficient_travel_saving" for row in rows)


def test_load_existing_decision_rows_preserves_resume_state(tmp_path: Path) -> None:
    path = tmp_path / "decisions.csv"
    rows = [
        GateDecisionRow(
            archive="a.zip",
            case_name="case",
            placement_method="rh",
            seed="seed_1000",
            placements_member="placements.csv",
            board_id="1",
            strategy="current",
            candidate_source="beam",
            candidate_evaluated=True,
            selected=True,
            protected_candidate=True,
            gate_passed=True,
            gate_reason="protected_beam",
            required_travel_saving=0.0,
            travel_saving=0.0,
            machining_saving=0.0,
            extra_tool_events=0,
            baseline_tool_event_count=2,
            candidate_tool_event_count=2,
            baseline_travel_mode_cost=100.0,
            candidate_travel_mode_cost=100.0,
            baseline_machining_cost=200.0,
            candidate_machining_cost=200.0,
            baseline_detour_count=1,
            candidate_detour_count=1,
            hard_penalty=0.0,
            stability_penalty=0.0,
        )
    ]

    write_dataclass_rows(path, rows)
    loaded = load_existing_decision_rows(path)

    assert loaded == rows
    assert loaded[0].selected is True
    assert loaded[0].candidate_tool_event_count == 2
