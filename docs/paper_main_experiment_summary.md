# Paper-main 真实数据主实验小结

日期：2026-05-22

## 最终默认配置

本轮第三章主实验使用 `--experiment-preset paper-main`，即：

- `support_policy = all_edges`
- `min_support_ratio = 0.75`
- `adjacency_support_weight = 1.0`
- `min_support_count = 1`
- `fallback_margin = 1000.0`
- `tool_event_gate = enabled`

本轮代码已将低位绕障内部图从 `Point` 字典/集合改为整数节点邻接表，并把绕障搜索从纯 Dijkstra 改为带曼哈顿启发的 A*。该优化不改变目标函数和可行性约束，只降低高密度布局下的几何路径评估开销。

数据范围：

- 三个第二章真实排样归档；
- 每个归档最多抽取 20 个 `placements.csv`；
- 每个 `placements.csv` 最多抽取 3 张板；
- 过滤范围为 20-50 个矩形件；
- 共 77 张真实板材。

最终主方法为 `process_aware_beam_adaptive_polished`，论文表述建议写为 `Event-gated Adaptive beam+LS`。它是一个带刀具事件门控的 adaptive portfolio：普通 `process_aware_beam` 作为受保护候选，局部精修或自适应候选只有在额外刀具事件被足够的通行代价和加工总代价收益证明时才进入最终比较。

## 主方法对比

输出文件：

- `results/adaptive_event_gate_protected_real_20x3_20_50.csv`
- `results/analysis_adaptive_event_gate_protected_real_20x3_20_50/`
- `results/adaptive_event_gate_protected_tool_event_summary.csv`
- `results/adaptive_event_gate_protected_tool_event_increase_cases.csv`
- `results/paper_artifacts/table_main_method_summary.csv`
- `results/paper_artifacts/table_main_paired_comparison.csv`
- `results/paper_artifacts/table_statistical_robustness.csv`
- `results/paper_artifacts/table_tool_event_gate_summary.csv`
- `results/paper_artifacts/table_tool_event_increase_cases.csv`
- `results/paper_artifacts/table_tool_event_gate_sensitivity.csv`
- `results/paper_artifacts/table_tool_event_gate_decisions.csv`
- `figures/paper_artifacts/fig_main_method_summary.pdf`

关键结果：

- `Event-gated Adaptive beam+LS` 平均 `travel_mode_cost = 9995.7`，平均 `machining_cost = 29816.0`，平均运行时间 `568.6 ms`；
- 该方法在 77 张板上平均 `hard_penalty = 0`，`stability_penalty = 0`；
- 相比 `topology_process_aware`，平均 `travel_mode_cost` 降低 `7.56%`，配对工艺目标胜率 `75.3%`，符号检验 `p < 1e-4`；
- 相比 `process_local_search_multistart`，平均 `travel_mode_cost` 降低 `7.42%`，配对工艺目标胜率 `75.3%`；
- 相比 `process_aware_beam`，平均 `travel_mode_cost` 降低 `3.61%`，平均 `machining_cost` 降低 `1.34%`；
- bootstrap 95% CI 均为正：相对 `topology_process_aware` 的 `travel_mode_cost` 降低区间为 `[4.90%, 10.30%]`，相对 `process_local_search_multistart` 为 `[4.94%, 10.23%]`，相对 `process_aware_beam` 为 `[2.20%, 5.18%]`；
- 相比 `process_aware_beam`，刀具事件在 11 个案例减少、60 个案例持平、6 个案例增加，平均减少 `0.44`；
- 相比纯路径距离局部搜索 `Path-LS`，本文方法路径成本更高，但将平均稳定性惩罚从 `5.09` 降为 `0`，说明本文优化目标不是单纯路径最短，而是满足 CNC 工艺稳定性的可加工路径。

## 刀具事件门控案例

输出文件：

- `docs/event_gate_tool_event_case_analysis.md`
- `results/paper_artifacts/table_tool_event_gate_summary.csv`
- `results/paper_artifacts/table_tool_event_increase_cases.csv`
- `results/paper_artifacts/table_tool_event_gate_sensitivity.csv`
- `results/paper_artifacts/table_tool_event_gate_decisions.csv`

关键结果：

- 77 个真实案例中，相对普通 beam 不增加刀具事件的案例为 71 个；
- 剩余 6 个案例主要来自 board `149` 和 board `340` 两类重复布局；
- board `149` 中刀具事件 `49 -> 52`，但 `travel_mode_cost` 降低 `8.62%`，detour `51 -> 8`；
- board `340` 中刀具事件 `69 -> 72`，但 `travel_mode_cost` 降低 `17.82%`，detour `37 -> 10`。

因此这 6 个案例不宜解释为算法失败，而应解释为“少量额外刀具事件被明确的绕行减少和加工代价收益证明”。正文可报告总体统计，补充材料或图注中展开 board `340`。

门控敏感性结果显示，当前门控不是简单禁止额外刀具事件，而是收益证明规则：`current` 策略得到 `11/60/6` 的减少/持平/增加分布，平均 `travel_mode_cost` 降低 `3.61%`；`strict` 策略只把增加案例从 6 个压到 4 个，但平均路径收益降到 `3.22%`。因此当前阈值是工艺保守性和路径收益之间更合适的折中。

## 消融实验

输出文件：

- `results/ablation_after_detour_intgraph_real_20_50.csv`
- `results/analysis_ablation_full_real_20_50/`
- `results/paper_artifacts/table_ablation_method_summary.csv`
- `results/paper_artifacts/table_ablation_paired_comparison.csv`
- `figures/paper_artifacts/fig_ablation_effects.pdf`

关键结果：

- 相比 `single_edges_only`，完整方法平均 `machining_cost` 降低 `27.69%`，证明共边、近共边和同线切割单元主要通过减少重复切割贡献收益；
- 相比 `no_stability_guidance`，完整方法将平均 `stability_penalty` 从 `6.71` 降为 `0`；
- 相比 `path_distance_baseline`，完整方法将平均 `stability_penalty` 从 `5.09` 降为 `0`；
- 相比 `no_adjacency_support_guidance`，完整方法平均 `travel_mode_cost` 降低 `23.94%`，并将平均稳定性惩罚从 `1.09` 降为 `0`；
- 相比 `topology_no_beam`，完整 beam 平均 `travel_mode_cost` 降低 `3.53%`；
- 相比 `no_detour_operator`，完整方法平均 `travel_mode_cost` 降低 `3.87%`；
- `no_safe_travel_modes` 虽然路径成本更短，但平均 `hard_penalty = 1.71`，不能作为可行工艺解。

## 精确解差距与规模实验

小规模 exact gap：

- 输出 `results/exact_gap_small_real_final_summary.csv`；
- `process_aware_beam_adaptive_polished` 在 7 个小规模实例上的平均 gap 为 `132.86`，平均运行时间 `57.4 ms`，`hard_penalty = 0`，`stability_penalty = 0`。

规模实验：

- 输出 `results/scalability_final_50_100_summary.csv`；
- clustered 50/75/100 件上，`Event-gated Adaptive beam+LS` 均保持 `hard_penalty = 0`、`stability_penalty = 0`；
- grid50 难例中，`topology_process_aware` 的 `travel_mode_cost = 9517.2`，`Event-gated Adaptive beam+LS` 降至 `7400.6`；绕障整数图优化后，`Event-gated Adaptive beam+LS` 运行时间从约 `13.3 s` 降至约 `5.6 s`。

## 代表性路线图

新生成的解释图：

- 主方法大幅收益案例：`figures/representative_route_cases_after_runtime_opt/main_adaptive_gain.png`
- 共边/切割单元消融：`figures/representative_route_cases_after_runtime_opt/shared_unit_ablation.png`
- 稳定性引导消融：`figures/representative_route_cases_after_runtime_opt/stability_guidance_ablation.png`
- 邻接支撑消融：`figures/representative_route_cases_after_runtime_opt/adjacency_support_ablation.png`
- 安全空移消融：`figures/representative_route_cases_after_runtime_opt/safe_travel_ablation.png`

对应 metrics 和逐动作 diagnostics 位于 `results/representative_route_cases_after_runtime_opt/`。

## 当前判断

现有结果已经能支撑第三章 SCI 主线：

1. 异构切割单元重构负责降低真实加工长度和重复切割；
2. 稳定性引导负责避免过早释放和低支撑离开；
3. 邻接支撑建模能把共边/近共边关系转化为工艺稳定性收益；
4. 安全空移和低位绕行负责把不可行空移转化为可加工路径；
5. 自适应 beam+LS portfolio 在保持 0 硬约束、0 稳定性惩罚的前提下，进一步降低工艺路径成本。

下一步代码重点应放在给敏感性实验补齐更细的进度输出、继续压缩 grid50 等高密度布局的最坏情况运行时间，以及把代表性案例图整理成论文图注和方法流程图。
