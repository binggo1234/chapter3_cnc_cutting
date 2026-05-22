# Chapter 3 Paper Artifacts

This directory contains manuscript-ready tables and figures generated from completed CSV experiments.

## Tables
- main method summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_main_method_summary.csv`
- main paired comparison: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_main_paired_comparison.csv`
- statistical robustness: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_statistical_robustness.csv`
- ablation method summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_ablation_method_summary.csv`
- ablation paired comparison: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_ablation_paired_comparison.csv`
- exact gap summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_exact_gap_summary.csv`
- portfolio selection summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_portfolio_selection_summary.csv`
- tool-event gate summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_tool_event_gate_summary.csv`
- tool-event increase cases: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_tool_event_increase_cases.csv`
- tool-event gate sensitivity: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_tool_event_gate_sensitivity.csv`
- tool-event gate decisions: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_tool_event_gate_decisions.csv`
- board 340 route metrics: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/event_gate_board340_route_metrics.csv`
- margin sensitivity summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_margin_sensitivity_summary.csv`
- scalability summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/results/paper_artifacts/table_scalability_summary.csv`

## Figures
- main method summary: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_main_method_summary.pdf`
- ablation effects: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_ablation_effects.pdf`
- exact gap: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_exact_gap.pdf`
- margin sensitivity: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_adaptive_margin_sensitivity.pdf`
- scalability: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_final_scalability_summary.pdf`
- tool-event gate sensitivity: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_tool_event_gate_sensitivity.pdf`
- representative route: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_route_large_gain_with_polish.png`
- board 340 event-gate route: `/Users/binggo/Desktop/graduate/chapter3_cnc_cutting/figures/paper_artifacts/fig_event_gate_board340_route_comparison.png`

## Main Interpretation
- Distance-only path search is shorter but violates the process stability objective.
- Heterogeneous cutting units reduce repeated cutting and lower total machining cost.
- Process-aware beam search improves over topology-only and multi-start process local search on paired real boards.
- Event-gated adaptive beam+LS keeps process-aware beam as a protected candidate and accepts extra tool events only when they are justified by clear travel-cost and machining-cost savings.
