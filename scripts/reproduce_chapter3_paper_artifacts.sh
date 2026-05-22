#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-artifacts}"
export PYTHONPATH="src:experiments"

if [[ "$MODE" == "full" ]]; then
  python3 experiments/run_chapter2_batch.py \
    --max-members-per-archive 20 \
    --boards-per-member 3 \
    --min-rectangles 20 \
    --max-rectangles 50 \
    --methods path_distance_local_search topology_process_aware process_local_search_multistart process_aware_beam process_aware_beam_adaptive process_aware_beam_polished process_aware_beam_adaptive_polished \
    --experiment-preset paper-main \
    --task-timeout-seconds 120 \
	    --output results/paper_main_margin1000_after_detour_intgraph_real_20_50.csv \
	    --summary-output results/paper_main_margin1000_after_detour_intgraph_real_20_50_summary.csv \
	    --bin-summary-output results/paper_main_margin1000_after_detour_intgraph_real_20_50_bin_summary.csv \
	    --progress-output results/paper_main_margin1000_after_detour_intgraph_real_20_50_progress.csv \
    --resume

  python3 experiments/analyze_results.py \
	    --input results/paper_main_margin1000_after_detour_intgraph_real_20_50.csv \
	    --target-method process_aware_beam_adaptive_polished \
	    --baseline-methods path_distance_local_search topology_process_aware process_local_search_multistart process_aware_beam process_aware_beam_adaptive process_aware_beam_polished \
	    --output-dir results/analysis_paper_main_margin1000_after_detour_intgraph_real_20_50 \
	    --figures-dir figures/analysis_paper_main_margin1000_after_detour_intgraph_real_20_50

	  python3 experiments/analyze_portfolio_selection.py \
	    --input results/paper_main_margin1000_after_detour_intgraph_real_20_50.csv \
	    --target-method process_aware_beam_adaptive_polished \
	    --component-methods process_aware_beam_polished process_aware_beam_adaptive process_aware_beam topology_process_aware \
	    --unmatched-label fallback_wide_beam_polished \
	    --output results/adaptive_polished_selection_attribution_margin1000_after_detour_intgraph.csv \
	    --summary-output results/adaptive_polished_selection_attribution_margin1000_after_detour_intgraph_summary.csv

  python3 experiments/run_ablation.py \
    --max-members-per-archive 20 \
    --boards-per-member 3 \
    --min-rectangles 20 \
    --max-rectangles 50 \
    --variants full_process_aware_beam process_aware_beam_polished single_edges_only no_stability_guidance no_adjacency_support_guidance topology_no_beam process_local_search_multistart path_distance_baseline no_detour_operator no_safe_travel_modes \
    --experiment-preset paper-main \
    --task-timeout-seconds 120 \
	    --output results/ablation_after_detour_intgraph_real_20_50.csv \
	    --summary-output results/ablation_after_detour_intgraph_real_20_50_summary.csv \
	    --progress-output results/ablation_after_detour_intgraph_real_20_50_progress.csv \
    --resume

  python3 experiments/analyze_results.py \
	    --input results/ablation_after_detour_intgraph_real_20_50.csv \
    --target-method full_process_aware_beam \
    --baseline-methods process_aware_beam_polished single_edges_only no_stability_guidance no_adjacency_support_guidance topology_no_beam process_local_search_multistart path_distance_baseline no_detour_operator no_safe_travel_modes \
    --output-dir results/analysis_ablation_full_real_20_50 \
    --figures-dir figures/analysis_ablation_full_real_20_50

  python3 experiments/run_exact_gap.py \
    --experiment-preset paper-main \
    --max-members-per-archive 2 \
    --boards-per-member 2 \
    --min-rectangles 1 \
    --max-rectangles 3 \
    --max-exact-units 12 \
    --methods exact_process_dp process_aware_beam process_aware_beam_adaptive process_aware_beam_polished process_aware_beam_adaptive_polished process_local_search_multistart topology_process_aware \
    --output results/exact_gap_small_real_final.csv \
    --summary-output results/exact_gap_small_real_final_summary.csv

  python3 experiments/run_adaptive_margin_sensitivity.py \
    --experiment-preset paper-main \
    --max-members-per-archive 20 \
    --boards-per-member 3 \
    --min-rectangles 20 \
    --max-rectangles 50 \
    --margins 0 250 500 750 1000 1500 \
    --task-timeout-seconds 120 \
	    --output results/adaptive_margin_sensitivity_after_detour_intgraph_real_20_50.csv \
	    --summary-output results/adaptive_margin_sensitivity_after_detour_intgraph_real_20_50_summary.csv \
	    --figure-dir figures/adaptive_margin_sensitivity_after_detour_intgraph_real_20_50

  python3 experiments/run_scalability.py \
    --experiment-preset paper-main \
    --scenario clustered \
    --sizes 50 75 100 \
    --repeats 3 \
    --methods topology_process_aware process_aware_beam process_aware_beam_adaptive_polished \
    --task-timeout-seconds 120 \
    --output results/scalability_final_clustered_50_100.csv \
    --progress-output results/scalability_final_clustered_50_100_progress.csv \
    --resume

  python3 experiments/run_scalability.py \
    --experiment-preset paper-main \
    --scenario grid \
    --sizes 50 \
    --repeats 1 \
    --methods topology_process_aware \
    --task-timeout-seconds 120 \
    --output results/scalability_grid50_topology_probe.csv \
    --progress-output results/scalability_grid50_topology_probe_progress.csv \
    --resume

  python3 experiments/run_scalability.py \
    --experiment-preset paper-main \
    --scenario grid \
    --sizes 50 \
    --repeats 1 \
    --methods process_aware_beam \
    --task-timeout-seconds 120 \
    --output results/scalability_grid50_beam_probe.csv \
    --progress-output results/scalability_grid50_beam_probe_progress.csv \
    --resume

  python3 experiments/run_scalability.py \
    --experiment-preset paper-main \
    --scenario grid \
    --sizes 50 \
    --repeats 1 \
    --methods process_aware_beam_adaptive_polished \
    --task-timeout-seconds 120 \
    --output results/scalability_grid50_adaptive_polished_probe.csv \
    --progress-output results/scalability_grid50_adaptive_polished_probe_progress.csv \
    --resume

  python3 experiments/analyze_scalability_results.py \
    --inputs results/scalability_final_clustered_50_100.csv results/scalability_grid50_topology_probe.csv results/scalability_grid50_beam_probe.csv results/scalability_grid50_adaptive_polished_probe.csv \
    --summary-output results/scalability_final_50_100_summary.csv \
    --figure-dir figures/scalability_final_50_100
elif [[ "$MODE" != "artifacts" ]]; then
  echo "Usage: $0 [artifacts|full]" >&2
  exit 2
fi

python3 experiments/generate_paper_artifacts.py
python3 -m compileall -q src experiments tests

echo "Paper artifacts are ready in results/paper_artifacts and figures/paper_artifacts."
