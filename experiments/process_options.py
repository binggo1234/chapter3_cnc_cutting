from __future__ import annotations

import argparse
import sys
from typing import Sequence

from cnc_cutting.models import CuttingProcessModel, EdgeRole, Layout
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
