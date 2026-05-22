# Claims

## C01: Event-gated Adaptive beam+LS is the Chapter 3 main algorithm
- **Statement**: The main algorithm should be reported as `Event-gated Adaptive beam+LS`, implemented by `process_aware_beam_adaptive_polished`, because it preserves `process_aware_beam` as a protected candidate while accepting extra tool events only when travel-cost and machining-cost savings justify them.
- **Status**: supported
- **Provenance**: user
- **Falsification criteria**: A later full 77-case run shows the method loses the protected-beam guarantee, violates hard/stability feasibility, or is dominated by the ungated variant under the thesis objective.
- **Proof**: [`results/adaptive_event_gate_protected_real_20x3_20_50.csv`, `results/paper_artifacts/table_main_paired_comparison.csv`, `results/paper_artifacts/table_tool_event_gate_summary.csv`]
- **Dependencies**: []
- **Tags**: main-algorithm, event-gate, thesis-chapter-3

## C02: Remaining tool-event increases are explainable tradeoffs
- **Statement**: The 6 cases where `Event-gated Adaptive beam+LS` increases tool events relative to `process_aware_beam` should be interpreted as benefit-justified tradeoffs, not algorithm failures.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Case-level diagnostics show the extra events are not accompanied by meaningful travel-cost, machining-cost, or detour reductions.
- **Proof**: [`docs/event_gate_tool_event_case_analysis.md`, `results/paper_artifacts/table_tool_event_increase_cases.csv`, `figures/paper_artifacts/fig_event_gate_board340_route_comparison.png`]
- **Dependencies**: [C01]
- **Tags**: tool-events, diagnostics, explanation

## C03: The current gate threshold is a balanced default
- **Statement**: The current tool-event gate is preferable to both no gate and stricter thresholds for the thesis main version because it sharply reduces extra-event cases while retaining more travel and machining benefit than the strict gate.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A broader sensitivity run shows a stricter or looser threshold improves both tool-event conservatism and travel/machining objectives.
- **Proof**: [`results/paper_artifacts/table_tool_event_gate_sensitivity.csv`, `figures/paper_artifacts/fig_tool_event_gate_sensitivity.pdf`]
- **Dependencies**: [C01]
- **Tags**: sensitivity, threshold, algorithm-default

## C04: Main paired gains are statistically robust on 77 real boards
- **Statement**: Event-gated Adaptive beam+LS has statistically robust paired improvements over process-aware topology, multi-start process local search, and ordinary process-aware beam on the current 77-board real dataset.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: Bootstrap confidence intervals for the main paired travel or machining reductions include zero, or paired sign tests become non-significant after correcting the input data.
- **Proof**: [`results/statistical_robustness_real_20_50.csv`, `results/paper_artifacts/table_statistical_robustness.csv`, `docs/statistical_robustness_summary.md`]
- **Dependencies**: [C01]
- **Tags**: statistics, robustness, paired-comparison

## C05: The evidence level should be framed as industrial planning-level validation
- **Statement**: Without commercial CAM output or physical cutting trials, the evidence should be framed as industrial planning-level validation rather than machine-trial validation.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Commercial CAM routes or physical cutting trials become available and are integrated into the experimental evidence.
- **Proof**: [`docs/industrial_planning_validation_protocol.md`, `results/paper_artifacts/paper_artifact_index.md`]
- **Dependencies**: [C01, C04]
- **Tags**: industrial-validation, limitations, experiment-design
