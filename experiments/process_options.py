from __future__ import annotations

import argparse
import sys
from typing import Sequence

from cnc_cutting.local_search import REPEAT_CUT_POLICIES, validate_repeat_cut_policy
from cnc_cutting.models import CuttingProcessModel, EdgeRole, Layout
from cnc_cutting.optimizer import DEFAULT_TOOL_EVENT_GATE_CONFIG, ToolEventGateConfig
from cnc_cutting.process_model import build_process_model


SUPPORT_POLICIES = ("top", "all_edges", "none")
EXPERIMENT_PRESETS = ("custom", "paper-main")
PAPER_MAIN_STABILITY_PRESET = {
    "support_policy": "all_edges",
    "min_support_count": 1,
    "min_support_ratio": 0.75,
    "min_area_normalized_support": 0.0,
    "adjacency_support_weight": 1.0,
}
_STABILITY_OPTION_FIELDS = {
    "--support-policy": "support_policy",
    "--min-support-count": "min_support_count",
    "--min-support-ratio": "min_support_ratio",
    "--min-area-normalized-support": "min_area_normalized_support",
    "--adjacency-support-weight": "adjacency_support_weight",
}


def add_experiment_preset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-preset",
        choices=EXPERIMENT_PRESETS,
        default="custom",
        help=(
            "Named experiment preset. 'paper-main' applies the strict all-edge "
            "support model used for the main paper experiments; explicitly "
            "provided stability options override preset values."
        ),
    )


def add_stability_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--support-policy",
        choices=SUPPORT_POLICIES,
        default="top",
        help="Stability support model used by the process-aware evaluator.",
    )
    parser.add_argument("--min-support-count", type=int, default=1)
    parser.add_argument("--min-support-ratio", type=float, default=0.0)
    parser.add_argument("--min-area-normalized-support", type=float, default=0.0)
    parser.add_argument("--adjacency-support-weight", type=float, default=0.0)


def add_tool_event_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-tool-event-gate",
        action="store_true",
        help="Disable the adaptive route selector's extra-tool-event acceptance gate.",
    )
    parser.add_argument(
        "--tool-event-min-travel-saving",
        type=float,
        default=DEFAULT_TOOL_EVENT_GATE_CONFIG.min_travel_saving_per_extra_event,
        help="Minimum travel-cost saving required per extra tool event.",
    )
    parser.add_argument(
        "--tool-event-min-travel-saving-ratio",
        type=float,
        default=DEFAULT_TOOL_EVENT_GATE_CONFIG.min_travel_saving_ratio_per_extra_event,
        help="Minimum relative travel-cost saving required per extra tool event.",
    )
    parser.add_argument(
        "--tool-event-min-machining-saving",
        type=float,
        default=DEFAULT_TOOL_EVENT_GATE_CONFIG.min_machining_saving,
        help="Minimum machining-cost saving required when extra tool events are added.",
    )


def add_repeat_cut_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeat-cut-policy",
        choices=REPEAT_CUT_POLICIES,
        default="hard",
        help=(
            "How repeated cutting is handled by unit selection and process "
            "ranking: hard prioritizes zero repeats, soft folds repeat length "
            "into machining cost, off ignores repeat-specific penalties."
        ),
    )


def apply_experiment_preset(
    args: argparse.Namespace,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    preset_name = getattr(args, "experiment_preset", "custom")
    if preset_name == "custom":
        return args
    if preset_name != "paper-main":
        raise ValueError(f"unsupported experiment preset: {preset_name}")

    explicit_fields = _explicit_stability_fields(sys.argv[1:] if argv is None else argv)
    for field, value in PAPER_MAIN_STABILITY_PRESET.items():
        if field not in explicit_fields:
            setattr(args, field, value)
    return args


def repeat_cut_policy_from_args(args: argparse.Namespace) -> str:
    return validate_repeat_cut_policy(getattr(args, "repeat_cut_policy", "hard"))


def _explicit_stability_fields(argv: Sequence[str]) -> set[str]:
    fields: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        field = _STABILITY_OPTION_FIELDS.get(option)
        if field:
            fields.add(field)
    return fields


def build_process_model_from_args(
    layout: Layout,
    args: argparse.Namespace,
) -> CuttingProcessModel:
    if args.support_policy == "none":
        return build_process_model(
            layout,
            support_edge_role=None,
            min_remaining_support_count=args.min_support_count,
            min_remaining_support_length_ratio=args.min_support_ratio,
            min_area_normalized_support_length=args.min_area_normalized_support,
            adjacency_support_weight=args.adjacency_support_weight,
        )

    if args.support_policy == "all_edges":
        return build_process_model(
            layout,
            support_edge_roles=(
                EdgeRole.BOTTOM,
                EdgeRole.RIGHT,
                EdgeRole.TOP,
                EdgeRole.LEFT,
            ),
            min_remaining_support_count=args.min_support_count,
            min_remaining_support_length_ratio=args.min_support_ratio,
            min_area_normalized_support_length=args.min_area_normalized_support,
            adjacency_support_weight=args.adjacency_support_weight,
        )

    return build_process_model(
        layout,
        support_edge_role=EdgeRole.TOP,
        min_remaining_support_count=args.min_support_count,
        min_remaining_support_length_ratio=args.min_support_ratio,
        min_area_normalized_support_length=args.min_area_normalized_support,
        adjacency_support_weight=args.adjacency_support_weight,
    )


def build_tool_event_gate_from_args(args: argparse.Namespace) -> ToolEventGateConfig:
    return ToolEventGateConfig(
        enabled=not getattr(args, "disable_tool_event_gate", False),
        min_travel_saving_per_extra_event=getattr(
            args,
            "tool_event_min_travel_saving",
            DEFAULT_TOOL_EVENT_GATE_CONFIG.min_travel_saving_per_extra_event,
        ),
        min_travel_saving_ratio_per_extra_event=getattr(
            args,
            "tool_event_min_travel_saving_ratio",
            DEFAULT_TOOL_EVENT_GATE_CONFIG.min_travel_saving_ratio_per_extra_event,
        ),
        min_machining_saving=getattr(
            args,
            "tool_event_min_machining_saving",
            DEFAULT_TOOL_EVENT_GATE_CONFIG.min_machining_saving,
        ),
    )
