from __future__ import annotations

import argparse

from cnc_cutting.models import CuttingProcessModel, EdgeRole, Layout
from cnc_cutting.process_model import build_process_model


SUPPORT_POLICIES = ("top", "all_edges", "none")


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
