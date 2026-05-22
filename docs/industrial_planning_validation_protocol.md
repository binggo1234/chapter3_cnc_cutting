# Industrial planning-level validation protocol

This note defines how to present the Chapter 3 experiments when commercial CAM output
or physical cutting trials are unavailable.

## Validation Level

The current evidence should be described as **industrial planning-level validation**.
It validates CNC cutting route planning on real nesting layouts and process-aware
constraints, but it does not claim physical machine trial validation.

## What The Evidence Covers

- Real nesting layouts from three Chapter 2 industrial-style experiment archives.
- 77 real boards with 20-50 rectangular parts.
- Process-aware route feasibility through hard penalties, stability penalties, safe
  travel modes, detour modeling, and tool-event accounting.
- Paired comparisons on identical boards, not aggregate-only comparisons across
  different instances.
- Case-level route diagnostics and visual inspection for representative boards.

## What The Evidence Does Not Claim

- It is not a replacement for commercial CAM verification.
- It does not measure actual kerf quality, heat affected zone, vibration, or physical
  deformation.
- It does not claim superiority over a named commercial CAM package.
- It does not claim machine-controller-level cycle time; runtime is algorithmic planning
  time and route metrics are planning-level cost proxies.

## Suggested Paper Wording

> Since direct access to commercial CAM internals and physical machine trials was not
> available, the experiments are designed as an industrial planning-level validation.
> All methods are evaluated on the same real nesting layouts, under the same process
> model, and with paired board-wise comparisons. The evaluation focuses on route
> feasibility and planning quality, including hard-constraint violations, dynamic
> stability penalties, machining cost, travel-mode cost, tool events, detours, and
> runtime. Representative route diagnostics are further used to inspect whether the
> numerical gains correspond to process-reasonable cutting sequences.

## How To Use In Experiments Section

1. Put this statement before the metrics table.
2. Emphasize paired board-wise comparisons.
3. Report `hard_penalty = 0` and `stability_penalty = 0` as process feasibility
   indicators.
4. Use `Path-LS` as a cautionary baseline: it may shorten travel but violates process
   stability, showing why pure shortest-path optimization is insufficient.
5. Use board 340 as the interpretability case for benefit-justified tool-event increases.

## Minimum Paper Evidence Set

- `results/paper_artifacts/table_main_method_summary.csv`
- `results/paper_artifacts/table_main_paired_comparison.csv`
- `results/paper_artifacts/table_statistical_robustness.csv`
- `results/paper_artifacts/table_ablation_paired_comparison.csv`
- `results/paper_artifacts/table_tool_event_gate_sensitivity.csv`
- `results/paper_artifacts/table_tool_event_gate_decisions.csv`
- `figures/paper_artifacts/fig_event_gate_board340_route_comparison.png`
