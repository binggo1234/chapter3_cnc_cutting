# Paper-main 真实数据主实验小结

日期：2026-05-20

本轮实验使用 `--experiment-preset paper-main`，即：

- `support_policy = all_edges`
- `min_support_ratio = 0.75`
- `adjacency_support_weight = 1.0`
- `min_support_count = 1`

数据范围：

- 三个第二章真实排样归档；
- 每个归档抽取 3 个 `placements.csv`；
- 每个 `placements.csv` 抽取 3 张板；
- 过滤范围为 2-50 个矩形件；
- 共 27 张真实板材。

## 主方法对比

输出文件：

- `results/paper_main_real_3x3_upto50.csv`
- `results/paper_main_real_3x3_upto50_summary.csv`
- `results/paper_main_real_3x3_upto50_bin_summary.csv`
- `results/analysis_paper_main_real_3x3_upto50/`
- `figures/analysis_paper_main_real_3x3_upto50/`

关键结果：

- `process_aware_beam` 的平均 `stability_penalty` 为 0；
- `greedy` 与 `path_distance_local_search` 的平均 `stability_penalty` 均为 2.925926；
- `topology_process_aware` 的平均 `stability_penalty` 也为 0，但平均 `machining_cost` 高于 `process_aware_beam`；
- 相比 `topology_process_aware`，`process_aware_beam` 的平均 `machining_cost` 降低 6.289373%，成对胜率 81.4815%，符号检验 `p = 0.001514`；
- 相比 `greedy` 和 `path_distance_local_search`，`process_aware_beam` 的 `machining_cost` 更高，但稳定性显著更好。这说明本文主方法不是单纯路径最短算法，而是面向稳定加工约束的路径优化。

## 消融实验

输出文件：

- `results/ablation_paper_main_real_3x3_upto50.csv`
- `results/ablation_paper_main_real_3x3_upto50_summary.csv`
- `results/ablation_paper_main_real_3x3_upto50_key_table.csv`
- `results/analysis_ablation_paper_main_real_3x3_upto50/`
- `figures/ablation_paper_main_real_3x3_upto50/`
- `figures/analysis_ablation_paper_main_real_3x3_upto50/`

关键结果：

- 相比 `single_edges_only`，完整方法平均 `machining_cost` 降低 26.131726%，成对胜率 96.2963%。这支持共边、近共边和同线切割单元重构的贡献；
- 相比 `topology_no_beam`，完整方法平均 `machining_cost` 降低 6.289373%，成对胜率 81.4815%。这支持前缀束搜索相对单步拓扑启发式的贡献；
- 相比 `no_adjacency_support_guidance`，完整方法平均 `machining_cost` 降低 11.178464%，平均 `stability_penalty` 降低 0.407407。邻接支撑建模对严格支撑约束有贡献；
- 相比 `no_stability_guidance`，完整方法平均 `stability_penalty` 降低 4.111111，但平均 `machining_cost` 上升。这说明稳定性约束会带来路径代价，但该代价服务于加工可行性；
- 相比 `path_distance_baseline`，完整方法平均 `stability_penalty` 降低 2.925926，但平均 `machining_cost` 上升。这支持“路径距离优化不能替代工艺稳定性优化”的论文论点；
- `no_detour_operator` 和 `no_safe_travel_modes` 的收益不稳定，当前数据中不是最强贡献模块。后续可进一步检查真实数据是否较少触发低位绕行，或当前绕行策略是否仍需优化。

## 当前判断

本轮结果已经支持第三章主线：

1. 共边/近共边/同线切割单元重构显著降低实际加工总代价；
2. 稳定性引导会牺牲部分空移效率，但能显著降低释放不稳定风险；
3. 工艺感知前缀束搜索在满足稳定性约束的前提下，优于单步拓扑启发式；
4. 单纯路径距离优化在数值上路径更短，但不能满足严格 CNC 释放稳定性目标。

下一步应优先检查 `process_aware_beam` 在少数未胜过 `topology_process_aware` 的板件上失败的原因，并决定是否需要针对局部搜索或 beam 参数做二次优化。
