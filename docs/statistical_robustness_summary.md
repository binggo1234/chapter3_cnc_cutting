# Statistical robustness summary

Positive reduction values mean the target method is lower than the baseline.
Win/tie/loss counts use lower-is-better comparison; `process_key` uses the full lexicographic process objective.

| baseline | metric | n | reduction mean | 95% CI | win/tie/loss | sign-test p |
|---|---|---:|---:|---:|---:|---:|
| topology_process_aware | process_key | 77 | 2.85% | [1.83, 3.91] | 58/3/16 | <1e-4 |
| topology_process_aware | machining_cost | 77 | 2.85% | [1.87, 3.85] | 58/3/16 | <1e-4 |
| topology_process_aware | travel_mode_cost | 77 | 7.56% | [4.90, 10.30] | 58/3/16 | <1e-4 |
| topology_process_aware | stability_penalty | 77 | 0.00% | [0.00, 0.00] | 0/77/0 |  |
| topology_process_aware | hard_penalty | 77 | 0.00% | [0.00, 0.00] | 0/77/0 |  |
| topology_process_aware | tool_event_count | 77 | 11.14% | [9.47, 12.83] | 64/4/9 | <1e-4 |
| topology_process_aware | detour_count | 77 | -123.54% | [-187.61, -70.99] | 29/3/45 | 0.080507 |
| topology_process_aware | runtime_ms | 77 | -429.48% | [-488.38, -373.86] | 0/0/77 | <1e-4 |
| process_local_search_multistart | process_key | 77 | 2.79% | [1.78, 3.81] | 58/3/16 | <1e-4 |
| process_local_search_multistart | machining_cost | 77 | 2.79% | [1.77, 3.83] | 58/3/16 | <1e-4 |
| process_local_search_multistart | travel_mode_cost | 77 | 7.42% | [4.94, 10.23] | 58/3/16 | <1e-4 |
| process_local_search_multistart | stability_penalty | 77 | 0.00% | [0.00, 0.00] | 0/77/0 |  |
| process_local_search_multistart | hard_penalty | 77 | 0.00% | [0.00, 0.00] | 0/77/0 |  |
| process_local_search_multistart | tool_event_count | 77 | 11.08% | [9.34, 12.74] | 64/4/9 | <1e-4 |
| process_local_search_multistart | detour_count | 77 | -120.93% | [-187.19, -68.31] | 29/3/45 | 0.080507 |
| process_local_search_multistart | runtime_ms | 77 | -117.91% | [-137.96, -97.75] | 8/0/69 | <1e-4 |
| process_aware_beam | process_key | 77 | 1.34% | [0.80, 1.92] | 26/51/0 | <1e-4 |
| process_aware_beam | machining_cost | 77 | 1.34% | [0.82, 1.90] | 26/51/0 | <1e-4 |
| process_aware_beam | travel_mode_cost | 77 | 3.61% | [2.20, 5.18] | 26/51/0 | <1e-4 |
| process_aware_beam | stability_penalty | 77 | 0.00% | [0.00, 0.00] | 0/77/0 |  |
| process_aware_beam | hard_penalty | 77 | 0.00% | [0.00, 0.00] | 0/77/0 |  |
| process_aware_beam | tool_event_count | 77 | 0.76% | [0.08, 1.54] | 11/60/6 | 0.332306 |
| process_aware_beam | detour_count | 77 | -2.20% | [-14.70, 9.21] | 16/53/8 | 0.151590 |
| process_aware_beam | runtime_ms | 77 | -88.58% | [-108.27, -68.32] | 24/0/53 | 0.001263 |
