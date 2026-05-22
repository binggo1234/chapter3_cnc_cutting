# Algorithm Notes

## Event-Gated Adaptive Beam+LS

The current Chapter 3 main algorithm is implemented as `process_aware_beam_adaptive_polished` and should be described in the thesis as `Event-gated Adaptive beam+LS`.

Core behavior:

- Generate the process-aware topology candidate, ordinary process-aware beam candidate, polished beam candidate, and fallback polished candidate when needed.
- Treat ordinary `process_aware_beam` as a protected candidate.
- For unprotected candidates with extra tool events, require both machining-cost saving and a travel-cost saving threshold before they can beat the protected beam candidate.
- Keep the default gate enabled through experiment CLIs and reproduce scripts.

Primary evidence:

- `results/adaptive_event_gate_protected_real_20x3_20_50.csv`
- `results/analysis_adaptive_event_gate_protected_real_20x3_20_50/`
- `results/tool_event_gate_current_decisions_real_20_50.csv`
- `results/paper_artifacts/table_tool_event_gate_decisions.csv`
