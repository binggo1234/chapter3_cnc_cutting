from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from process_options import apply_experiment_preset, build_tool_event_gate_from_args  # noqa: E402


def base_args() -> argparse.Namespace:
    return argparse.Namespace(
        experiment_preset="paper-main",
        support_policy="top",
        min_support_count=1,
        min_support_ratio=0.0,
        min_area_normalized_support=0.0,
        adjacency_support_weight=0.0,
    )


def test_paper_main_preset_applies_stability_defaults() -> None:
    args = apply_experiment_preset(base_args(), argv=("--experiment-preset", "paper-main"))

    assert args.support_policy == "all_edges"
    assert args.min_support_count == 1
    assert args.min_support_ratio == 0.75
    assert args.min_area_normalized_support == 0.0
    assert args.adjacency_support_weight == 1.0


def test_explicit_stability_args_override_paper_main_preset() -> None:
    args = base_args()
    args.min_support_ratio = 0.5

    apply_experiment_preset(
        args,
        argv=(
            "--experiment-preset",
            "paper-main",
            "--min-support-ratio",
            "0.5",
        ),
    )

    assert args.support_policy == "all_edges"
    assert args.min_support_ratio == 0.5
    assert args.adjacency_support_weight == 1.0


def test_custom_preset_keeps_existing_values() -> None:
    args = base_args()
    args.experiment_preset = "custom"

    apply_experiment_preset(args, argv=())

    assert args.support_policy == "top"
    assert args.min_support_ratio == 0.0
    assert args.adjacency_support_weight == 0.0


def test_build_tool_event_gate_from_args() -> None:
    args = argparse.Namespace(
        disable_tool_event_gate=True,
        tool_event_min_travel_saving=250.0,
        tool_event_min_travel_saving_ratio=0.05,
        tool_event_min_machining_saving=1.0,
    )

    config = build_tool_event_gate_from_args(args)

    assert config.enabled is False
    assert config.min_travel_saving_per_extra_event == 250.0
    assert config.min_travel_saving_ratio_per_extra_event == 0.05
    assert config.min_machining_saving == 1.0
