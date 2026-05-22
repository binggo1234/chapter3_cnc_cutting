# Event-gated Adaptive Beam+LS 的刀具事件案例分析

Baseline: `process_aware_beam`
Target: `process_aware_beam_adaptive_polished`

## 总体结果

- Paired cases: 77
- Tool-event decrease/tie/increase: 11/60/6
- Mean tool-event delta: -0.44
- Mean travel-mode cost reduction: 3.61%
- Mean machining cost reduction: 1.34%

## 刀具事件增加案例

| archive | seed | board | rectangles | event delta | travel reduction | machining reduction | detour delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| paper_extended_suite_20260321_continuous_run_only_20260324_0001.zip | seed_1004 | 149 | 21 | 3 | 8.62% | 2.90% | -43 |
| review_overnight_20260422.zip | seed_1000 | 340 | 30 | 3 | 17.82% | 6.02% | -27 |
| review_overnight_20260422.zip | seed_1001 | 340 | 30 | 3 | 17.82% | 6.02% | -27 |
| review_overnight_20260422.zip | seed_1002 | 340 | 30 | 3 | 17.82% | 6.02% | -27 |
| review_overnight_20260422.zip | seed_1003 | 340 | 30 | 3 | 17.82% | 6.02% | -27 |
| strong_baseline_overnight_20260407_001243.zip | seed_1004 | 149 | 21 | 3 | 8.62% | 2.90% | -43 |

## 解释口径

- 这些增加案例主要集中在 board `149` (2), board `340` (4)，不是互相独立的 6 种失败模式。
- 所有增加案例是否均降低通行代价：yes。
- 所有增加案例是否均降低加工总代价：yes。
- 最适合做代表图的是 board `340`，其 `travel_mode_cost` 降低 17.82%，`detour_count` 变化 -27。
- 论文中不宜把这些案例解释为算法失败；更合适的说法是：门控并不禁止所有额外刀具事件，而是只接受被明确加工收益证明的事件增加。
