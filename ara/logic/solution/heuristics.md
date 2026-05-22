# Heuristics

## H01: Protected Beam Candidate
- **Rationale**: Keeping `process_aware_beam` protected gives the adaptive portfolio a clear process baseline, so local search or fallback variants cannot win merely by improving path length at the cost of unjustified extra tool events.
- **Provenance**: user
- **Evidence**: `src/cnc_cutting/optimizer.py`, `tests/test_optimizer.py`

## H02: Benefit-Justified Extra Tool Events
- **Rationale**: Extra tool events are allowed only when they pass the configured travel-saving and machining-saving gate. This avoids a brittle hard ban while preserving process reasonableness.
- **Provenance**: ai-suggested
- **Evidence**: `results/paper_artifacts/table_tool_event_gate_sensitivity.csv`, `results/paper_artifacts/table_tool_event_gate_decisions.csv`

## H03: Decision Table for Explainability
- **Rationale**: The gate-decision table records selected candidates and rejected candidates, including the rejection reason and required versus achieved travel saving. This makes the remaining event-increase cases defensible in the thesis.
- **Provenance**: ai-executed
- **Evidence**: `results/tool_event_gate_current_decisions_real_20_50.csv`
