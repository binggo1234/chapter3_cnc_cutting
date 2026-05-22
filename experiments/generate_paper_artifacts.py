from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = (
    "path_distance_local_search",
    "topology_process_aware",
    "process_local_search_multistart",
    "process_aware_beam",
    "process_aware_beam_adaptive",
    "process_aware_beam_polished",
    "process_aware_beam_adaptive_polished",
)
ABLATION_ORDER = (
    "full_process_aware_beam",
    "process_aware_beam_polished",
    "single_edges_only",
    "no_stability_guidance",
    "no_adjacency_support_guidance",
    "topology_no_beam",
    "process_local_search_multistart",
    "path_distance_baseline",
    "no_detour_operator",
    "no_safe_travel_modes",
)
METHOD_LABELS = {
    "path_distance_local_search": "Path-LS",
    "path_distance_baseline": "Path-LS",
    "topology_process_aware": "Process-aware topology",
    "topology_no_beam": "No beam",
    "process_local_search_multistart": "Multi-start process LS",
    "process_aware_beam": "Process-aware beam",
    "process_aware_beam_adaptive": "Adaptive beam",
    "process_aware_beam_polished": "Beam+process LS",
    "process_aware_beam_adaptive_polished": "Adaptive beam+LS",
    "full_process_aware_beam": "Full beam",
    "single_edges_only": "Single edges only",
    "no_stability_guidance": "No stability guidance",
    "no_adjacency_support_guidance": "No adjacency support",
    "no_detour_operator": "No detour",
    "no_safe_travel_modes": "No safe travel",
    "exact_process_dp": "Exact DP",
}
METHOD_COLORS = {
    "path_distance_local_search": "#CC79A7",
    "path_distance_baseline": "#CC79A7",
    "topology_process_aware": "#264653",
    "topology_no_beam": "#264653",
    "process_local_search_multistart": "#8E6C8A",
    "process_aware_beam": "#E69F00",
    "process_aware_beam_adaptive": "#0072B2",
    "process_aware_beam_polished": "#A6761D",
    "process_aware_beam_adaptive_polished": "#009E73",
    "full_process_aware_beam": "#E69F00",
    "single_edges_only": "#7B8794",
    "no_stability_guidance": "#D55E00",
    "no_adjacency_support_guidance": "#0072B2",
    "no_detour_operator": "#009E73",
    "no_safe_travel_modes": "#BE123C",
    "exact_process_dp": "#111827",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    }
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value == "":
        return default
    return float(value)


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def sort_key(method: str) -> tuple[int, str]:
    combined = METHOD_ORDER + ABLATION_ORDER
    if method in combined:
        return (combined.index(method), method)
    return (len(combined), method)


def fmt(value: float | str, digits: int = 2) -> str:
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_p(value: str | float) -> str:
    if value == "":
        return ""
    numeric = float(value)
    if numeric < 1e-4:
        return "<1e-4"
    return f"{numeric:.4f}"


def method_summary_table(rows: list[dict[str, str]], order: tuple[str, ...]) -> list[dict[str, str]]:
    by_method = {row["method"]: row for row in rows}
    table: list[dict[str, str]] = []
    for method in order:
        if method not in by_method:
            continue
        row = by_method[method]
        table.append(
            {
                "method": label(method),
                "n": row["n"],
                "machining_cost": fmt(number(row, "machining_cost_mean"), 1),
                "travel_mode_cost": fmt(number(row, "travel_mode_cost_mean"), 1),
                "hard_penalty": fmt(number(row, "hard_penalty_mean"), 2),
                "stability_penalty": fmt(number(row, "stability_penalty_mean"), 2),
                "safe_lift": fmt(number(row, "safe_lift_count_mean"), 2),
                "runtime_ms": fmt(number(row, "runtime_ms_mean"), 1),
            }
        )
    return table


def paired_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: sort_key(item["baseline_method"])):
        table.append(
            {
                "baseline": label(row["baseline_method"]),
                "cases": row["paired_cases"],
                "travel_reduction_pct": fmt(number(row, "travel_mode_cost_reduction_pct_mean"), 2),
                "machining_reduction_pct": fmt(number(row, "machining_cost_reduction_pct_mean"), 2),
                "stability_reduction": fmt(number(row, "stability_penalty_reduction_mean"), 2),
                "hard_delta": fmt(number(row, "hard_penalty_delta_mean"), 2),
                "process_win_rate": fmt(100.0 * number(row, "process_key_win_rate"), 1),
                "p_value": fmt_p(row.get("process_key_sign_test_p", "")),
            }
        )
    return table


def exact_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: sort_key(item["method"])):
        table.append(
            {
                "method": label(row["method"]),
                "n": row["n"],
                "travel_mode_cost": fmt(number(row, "travel_mode_cost_mean"), 1),
                "gap": fmt(number(row, "travel_mode_cost_gap_mean"), 1),
                "gap_ratio": fmt(number(row, "travel_mode_cost_gap_ratio_mean"), 3),
                "runtime_ms": fmt(number(row, "runtime_ms_mean"), 1),
                "hard_penalty": fmt(number(row, "hard_penalty_mean"), 2),
                "stability_penalty": fmt(number(row, "stability_penalty_mean"), 2),
            }
        )
    return table


def portfolio_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    source_labels = {
        "process_aware_beam_polished": "Beam+process LS",
        "process_aware_beam_adaptive": "Adaptive beam",
        "process_aware_beam": "Process-aware beam",
        "topology_process_aware": "Process-aware topology",
        "fallback_wide_beam_polished": "Wide beam+LS fallback",
        "fallback_wide_beam": "Wide beam fallback",
        "portfolio_fallback_candidate": "Portfolio-only fallback",
    }
    table: list[dict[str, str]] = []
    for row in rows:
        table.append(
            {
                "source": source_labels.get(row["source_label"], label(row["source_label"])),
                "n": row["n"],
                "share_pct": fmt(number(row, "share_pct"), 2),
                "travel_mode_cost": fmt(number(row, "travel_mode_cost_mean"), 1),
                "runtime_ms": fmt(number(row, "runtime_ms_mean"), 1),
            }
        )
    return table


def margin_sensitivity_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []
    for row in rows:
        if row["variant"] != "adaptive_polished":
            continue
        table.append(
            {
                "margin": fmt(number(row, "fallback_margin"), 0),
                "travel_mode_cost": fmt(number(row, "travel_mode_cost_mean"), 1),
                "machining_cost": fmt(number(row, "machining_cost_mean"), 1),
                "runtime_ms": fmt(number(row, "estimated_runtime_ms_mean"), 1),
                "fallback_trigger_pct": fmt(
                    number(row, "fallback_trigger_rate_pct"), 2
                ),
                "fallback_count": row["source_fallback_polished_count"],
            }
        )
    return table


def scalability_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []
    for row in rows:
        table.append(
            {
                "scenario": row["scenario"],
                "size": row["size"],
                "method": label(row["method"]),
                "n": row["n"],
                "travel_mode_cost": fmt(number(row, "travel_mode_cost_mean"), 1),
                "runtime_ms": fmt(number(row, "runtime_ms_mean"), 1),
                "hard_penalty": fmt(number(row, "hard_penalty_mean"), 2),
                "stability_penalty": fmt(number(row, "stability_penalty_mean"), 2),
            }
        )
    return table


def write_table(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("<", "$<$")
        .replace(">", "$>$")
    )


def write_latex_table(
    rows: list[dict[str, str]],
    output_path: Path,
    caption: str,
    latex_label: str,
) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(latex_escape(column.replace("_", " ")) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(str(row[column])) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def copy_optional_figure(source: Path, output_dir: Path, stem: str) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for suffix in (".pdf", ".png"):
        src = source.with_suffix(suffix)
        if not src.exists():
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{stem}{suffix}"
        shutil.copy2(src, destination)
        copied[suffix.lstrip(".")] = destination
    return copied


def plot_main_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
    ordered = [row for method in METHOD_ORDER for row in rows if row["method"] == method]
    labels = [label(row["method"]) for row in ordered]
    colors = [METHOD_COLORS.get(row["method"], "#999999") for row in ordered]
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.9))
    axes[0].bar(x, [number(row, "travel_mode_cost_mean") for row in ordered], color=colors)
    axes[0].set_title("Travel-mode cost")
    axes[0].set_ylabel("Mean cost")

    axes[1].bar(x, [number(row, "stability_penalty_mean") for row in ordered], color=colors)
    axes[1].set_title("Stability penalty")
    axes[1].set_ylabel("Mean penalty")

    axes[2].bar(x, [number(row, "runtime_ms_mean") for row in ordered], color=colors)
    axes[2].set_title("Runtime")
    axes[2].set_ylabel("Mean runtime (ms)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_main_method_summary")


def plot_ablation_pairs(rows: list[dict[str, str]], output_dir: Path) -> None:
    filtered = [
        row
        for row in rows
        if row["baseline_method"] not in {"process_aware_beam_polished"}
    ]
    labels = [label(row["baseline_method"]) for row in filtered]
    y = np.arange(len(filtered))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.5))
    axes[0].barh(
        y,
        [number(row, "machining_cost_reduction_pct_mean") for row in filtered],
        color="#E69F00",
    )
    axes[0].axvline(0.0, color="#111827", linewidth=0.7)
    axes[0].set_title("Machining reduction")
    axes[0].set_xlabel("Full vs. baseline (%)")

    axes[1].barh(
        y,
        [number(row, "stability_penalty_reduction_mean") for row in filtered],
        color="#009E73",
    )
    axes[1].axvline(0.0, color="#111827", linewidth=0.7)
    axes[1].set_title("Stability improvement")
    axes[1].set_xlabel("Penalty reduction")

    axes[2].barh(
        y,
        [100.0 * number(row, "process_key_win_rate") for row in filtered],
        color="#56B4E9",
    )
    axes[2].axvline(50.0, color="#111827", linewidth=0.7)
    axes[2].set_xlim(0.0, 105.0)
    axes[2].set_title("Process-key wins")
    axes[2].set_xlabel("Win rate (%)")

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_ablation_effects")


def plot_exact_gap(rows: list[dict[str, str]], output_dir: Path) -> None:
    filtered = [row for row in rows if row["method"] != "exact_process_dp"]
    labels = [label(row["method"]) for row in filtered]
    colors = [METHOD_COLORS.get(row["method"], "#999999") for row in filtered]
    x = np.arange(len(filtered))

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    ax.bar(x, [number(row, "travel_mode_cost_gap_mean") for row in filtered], color=colors)
    ax.set_title("Small-scale gap to exact DP")
    ax.set_ylabel("Mean travel-cost gap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_exact_gap")


def write_summary_markdown(
    output_path: Path,
    tables: dict[str, Path],
    figures: dict[str, Path],
) -> None:
    lines = [
        "# Chapter 3 Paper Artifacts",
        "",
        "This directory contains manuscript-ready tables and figures generated from completed CSV experiments.",
        "",
        "## Tables",
    ]
    for name, path in tables.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    lines.append("## Figures")
    for name, path in figures.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    lines.append("## Main Interpretation")
    lines.extend(
        [
            "- Distance-only path search is shorter but violates the process stability objective.",
            "- Heterogeneous cutting units reduce repeated cutting and lower total machining cost.",
            "- Process-aware beam search improves over topology-only and multi-start process local search on paired real boards.",
            "- Adaptive beam expansion provides a quality-prioritized robust variant for difficult boards.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-method-summary",
        type=Path,
        default=ROOT
        / "results"
        / "analysis_paper_main_margin1000_after_detour_intgraph_real_20_50"
        / "method_summary.csv",
    )
    parser.add_argument(
        "--main-paired-summary",
        type=Path,
        default=ROOT
        / "results"
        / "analysis_paper_main_margin1000_after_detour_intgraph_real_20_50"
        / "paired_summary.csv",
    )
    parser.add_argument(
        "--ablation-method-summary",
        type=Path,
        default=ROOT
        / "results"
        / "analysis_ablation_full_real_20_50"
        / "method_summary.csv",
    )
    parser.add_argument(
        "--ablation-paired-summary",
        type=Path,
        default=ROOT
        / "results"
        / "analysis_ablation_full_real_20_50"
        / "paired_summary.csv",
    )
    parser.add_argument(
        "--exact-summary",
        type=Path,
        default=ROOT / "results" / "exact_gap_small_real_final_summary.csv",
    )
    parser.add_argument(
        "--portfolio-summary",
        type=Path,
        default=ROOT
        / "results"
        / "adaptive_polished_selection_attribution_margin1000_after_detour_intgraph_summary.csv",
    )
    parser.add_argument(
        "--margin-summary",
        type=Path,
        default=ROOT
        / "results"
        / "adaptive_margin_sensitivity_after_detour_intgraph_real_20_50_summary.csv",
    )
    parser.add_argument(
        "--margin-figure",
        type=Path,
        default=ROOT
        / "figures"
        / "adaptive_margin_sensitivity_after_detour_intgraph_real_20_50"
        / "fig_adaptive_margin_sensitivity.pdf",
    )
    parser.add_argument(
        "--scalability-summary",
        type=Path,
        default=ROOT / "results" / "scalability_final_50_100_summary.csv",
    )
    parser.add_argument(
        "--scalability-figure",
        type=Path,
        default=ROOT
        / "figures"
        / "scalability_final_50_100"
        / "fig_final_scalability_summary.pdf",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ROOT / "results" / "paper_artifacts",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "paper_artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_methods = load_rows(args.main_method_summary)
    main_pairs = load_rows(args.main_paired_summary)
    ablation_methods = load_rows(args.ablation_method_summary)
    ablation_pairs = load_rows(args.ablation_paired_summary)
    exact_rows = load_rows(args.exact_summary) if args.exact_summary.exists() else []
    portfolio_rows = (
        load_rows(args.portfolio_summary) if args.portfolio_summary.exists() else []
    )
    margin_rows = load_rows(args.margin_summary) if args.margin_summary.exists() else []
    scalability_rows = (
        load_rows(args.scalability_summary)
        if args.scalability_summary.exists()
        else []
    )

    tables: dict[str, Path] = {}

    main_table = method_summary_table(main_methods, METHOD_ORDER)
    main_table_path = args.tables_dir / "table_main_method_summary.csv"
    write_table(main_table, main_table_path)
    write_latex_table(
        main_table,
        args.tables_dir / "table_main_method_summary.tex",
        "Main real-board comparison on 77 boards with 20-50 rectangles.",
        "tab:main-method-summary",
    )
    tables["main method summary"] = main_table_path

    main_pair_table = paired_table(main_pairs)
    main_pair_path = args.tables_dir / "table_main_paired_comparison.csv"
    write_table(main_pair_table, main_pair_path)
    write_latex_table(
        main_pair_table,
        args.tables_dir / "table_main_paired_comparison.tex",
        "Paired comparison using Adaptive beam+LS as the target method.",
        "tab:main-paired-comparison",
    )
    tables["main paired comparison"] = main_pair_path

    ablation_table = method_summary_table(ablation_methods, ABLATION_ORDER)
    ablation_table_path = args.tables_dir / "table_ablation_method_summary.csv"
    write_table(ablation_table, ablation_table_path)
    write_latex_table(
        ablation_table,
        args.tables_dir / "table_ablation_method_summary.tex",
        "Ablation summary on the same 77 real boards.",
        "tab:ablation-method-summary",
    )
    tables["ablation method summary"] = ablation_table_path

    ablation_pair_table = paired_table(ablation_pairs)
    ablation_pair_path = args.tables_dir / "table_ablation_paired_comparison.csv"
    write_table(ablation_pair_table, ablation_pair_path)
    write_latex_table(
        ablation_pair_table,
        args.tables_dir / "table_ablation_paired_comparison.tex",
        "Paired ablation effects using the full process-aware beam as the target method.",
        "tab:ablation-paired-comparison",
    )
    tables["ablation paired comparison"] = ablation_pair_path

    if exact_rows:
        exact_output = exact_table(exact_rows)
        exact_table_path = args.tables_dir / "table_exact_gap_summary.csv"
        write_table(exact_output, exact_table_path)
        write_latex_table(
            exact_output,
            args.tables_dir / "table_exact_gap_summary.tex",
            "Small-scale optimality gap against exact dynamic programming.",
            "tab:exact-gap-summary",
        )
        tables["exact gap summary"] = exact_table_path

    if portfolio_rows:
        portfolio_output = portfolio_table(portfolio_rows)
        portfolio_table_path = args.tables_dir / "table_portfolio_selection_summary.csv"
        write_table(portfolio_output, portfolio_table_path)
        write_latex_table(
            portfolio_output,
            args.tables_dir / "table_portfolio_selection_summary.tex",
            "Source attribution of the adaptive beam+LS portfolio.",
            "tab:portfolio-selection-summary",
        )
        tables["portfolio selection summary"] = portfolio_table_path

    if margin_rows:
        margin_output = margin_sensitivity_table(margin_rows)
        margin_table_path = args.tables_dir / "table_margin_sensitivity_summary.csv"
        write_table(margin_output, margin_table_path)
        write_latex_table(
            margin_output,
            args.tables_dir / "table_margin_sensitivity_summary.tex",
            "Fallback-margin sensitivity of the adaptive beam+LS portfolio.",
            "tab:margin-sensitivity-summary",
        )
        tables["margin sensitivity summary"] = margin_table_path

    if scalability_rows:
        scalability_output = scalability_table(scalability_rows)
        scalability_table_path = args.tables_dir / "table_scalability_summary.csv"
        write_table(scalability_output, scalability_table_path)
        write_latex_table(
            scalability_output,
            args.tables_dir / "table_scalability_summary.tex",
            "Scalability summary for final method candidates.",
            "tab:scalability-summary",
        )
        tables["scalability summary"] = scalability_table_path

    plot_main_summary(main_methods, args.figures_dir)
    plot_ablation_pairs(ablation_pairs, args.figures_dir)
    if exact_rows:
        plot_exact_gap(exact_rows, args.figures_dir)
    margin_figures = copy_optional_figure(
        args.margin_figure,
        args.figures_dir,
        "fig_adaptive_margin_sensitivity",
    )
    scalability_figures = copy_optional_figure(
        args.scalability_figure,
        args.figures_dir,
        "fig_final_scalability_summary",
    )

    figures = {
        "main method summary": args.figures_dir / "fig_main_method_summary.pdf",
        "ablation effects": args.figures_dir / "fig_ablation_effects.pdf",
    }
    if exact_rows:
        figures["exact gap"] = args.figures_dir / "fig_exact_gap.pdf"
    if "pdf" in margin_figures:
        figures["margin sensitivity"] = margin_figures["pdf"]
    if "pdf" in scalability_figures:
        figures["scalability"] = scalability_figures["pdf"]
    route_figure = args.figures_dir / "fig_route_large_gain_with_polish.png"
    if route_figure.exists():
        figures["representative route"] = route_figure
    write_summary_markdown(args.tables_dir / "paper_artifact_index.md", tables, figures)

    print(f"tables: {args.tables_dir}")
    print(f"figures: {args.figures_dir}")
    for path in tables.values():
        print(f"wrote: {path}")
    for path in figures.values():
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
