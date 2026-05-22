# Chapter 3 CNC Cutting Path Optimization

本目录用于存放毕业论文第三章“面向 CNC 加工约束的矩形件切割路径优化方法研究”的代码、实验数据和结果。

## 研究目标

第三章主要研究在已知矩形件排样结果的条件下，如何优化 CNC 切割过程中的刀具路径。核心关注：

- 减少空移距离；
- 减少入刀和抬刀次数；
- 识别真实共边、近共边和同线边段；
- 优化切割边段、切割单元、入刀点和切割方向；
- 检查修边安全区、刀具半径补偿和空移干涉；
- 支持增量路径代价评价；
- 为第四章排样-切割协同优化提供路径代价评价函数。

## 目录结构

- `src/cnc_cutting/`：核心算法代码；
- `data/sample_layouts/`：排样结果样例数据；
- `experiments/`：实验脚本；
- `results/`：实验输出结果；
- `docs/`：第三章算法说明、模型说明和实验记录；
- `tests/`：基础单元测试。

## 当前阶段

当前先建立第三章基础代码框架，包括：

- 矩形件与排样结果数据结构；
- 矩形轮廓、边段方向、线段相交等几何计算；
- 共边、近共边、同线边段关系识别；
- 基于修边量 5 mm、刀径 6 mm 的基础工艺参数表达；
- 空移干涉和有效加工区域检查；
- 全局与增量路径代价评价；
- 异构切割图基础构造；
- 覆盖单元选择、贪心路径基线、路径距离局部搜索基线、拓扑感知路径基线与局部搜索改进，用作后续对比实验。
- 零件释放状态、释放后空移干涉和可配置释放稳定性惩罚。
- 前缀状态缓存、候选池限制和 first-improvement 局部搜索，用于规模实验。
- 释放零件包围盒缓存、轴对齐矩形快速相交和候选方向缓存，用于降低大规模运行时间。
- 第二章输出的 `placements.csv`/zip 实验包导入，按 `board` 拆分为第三章单板切割输入。
- 低位空移与安全抬刀空移区分，将释放件穿越从单一硬碰撞扩展为可解释的 CNC 工艺代价。
- 低位绕行空移候选，用于在释放件干涉时优先尝试正交绕行，再回退到安全抬刀。
- 基于 `travel_mode_cost` 的空移模式成本比较，以及低位绕行路径缓存。
- 多支撑边释放稳定性模型，支持 `top`、`all_edges` 和 `none` 三类支撑策略，以及剩余支撑比例、面积归一化支撑强度参数；低支撑状态采用延迟处罚，连续完成零件释放不计为稳定性违规。
- 完成链上下文空移：空移动作会携带即将切割的单元信息，前往完成同一低支撑零件的空移不再被误判为放弃零件。
- 工艺感知前缀束搜索 `process_aware_beam`：在增量评价状态上保留多个候选前缀，联合考虑硬约束、动态释放稳定性、空移模式代价和拓扑转移得分。
- 束搜索层级扩展剪枝：通过 `max_layer_expansions` 先用低成本转移得分预筛候选，再做真实增量评价，当前真实数据默认约为 `7 * beam_width`。
- 束搜索前缀多样性保留：通过 `diversity_bucket_limit` 限制同一零件组、低支撑状态、释放数量和方向的同质前缀占满 beam，当前真实数据默认值为 1。
- 低支撑父前缀保底扩展：通过 `unstable_min_expansions_per_parent` 保证处于低支撑状态的前缀至少获得若干扩展名额，当前真实数据默认只保护低支撑父前缀，不强制普通父前缀扩展。
- 多低支撑零件状态保留：当多个零件同时处于低支撑状态时，继续加工其中任一低支撑零件不会误罚其他低支撑零件，只有转向完全无关零件才触发放弃惩罚。
- 低位绕障几何加速：绕障可见图使用轴对齐快速相交、行列障碍预筛和规范化缓存，减少前缀束搜索中的重复空移规划开销。
- 加工总代价指标：新增 `machining_cost = cutting_length + travel_mode_cost`，用于体现共边/同线切割单元减少实际切割长度的收益；前缀 beam 剪枝仍只使用硬约束、稳定性和空移状态，避免“先切短边”的前缀偏置。
- 结果证据链脚本：支持对批量实验 CSV 做逐板成对比较、分方法统计、运行时间比例、稳定性罚分减少量和按零件数分层的图表输出。
- 多方法路线对照图：支持在同一真实板件上并排可视化 `greedy`、路径局部搜索、工艺感知拓扑启发式和 `process_aware_beam` 的切割单元、空移模式、安全抬刀、绕行和稳定性事件。

普通 TSP 或最近邻路径只作为对比基线。第三章主算法将继续沿着“工艺感知异构切割图 + 拓扑感知候选动作 + 增量评价 + 前缀束搜索/局部搜索”的路线实现。
其中 `path_distance_local_search` 是面向论文对比新增的强路径基线：以近邻解为初始解，使用 swap、relocate 和方向受限 2-opt 优化真实通行代价，但不把释放稳定性作为优化目标。

## 运行示例

```bash
PYTHONPATH=src python3 experiments/run_greedy_route.py
```

```bash
PYTHONPATH=src python3 experiments/run_scalability.py
```

规模敏感性实验默认生成 `10, 50, 100, 200, 500` 件合成排样，并将结果写入 `results/scalability_results.csv`。
脚本支持 `process_aware_beam`、稳定性模型参数和 beam 参数覆盖；当前第二章真实数据最大板材约 45-46 件，50 件以上实验应视为合成压力测试。

```bash
PYTHONPATH=src python3 experiments/run_scalability.py --scenario clustered --output results/scalability_clustered_results.csv
```

`clustered` 场景生成带近共边通道的排样。若需要验证动态工艺状态对稳定性的影响，可运行：

```bash
PYTHONPATH=src python3 experiments/run_scalability.py --scenario clustered --process-aware-topology --sizes 10 20 50 --output results/scalability_clustered_process_aware.csv
```

合成 clustered 场景下运行 beam 主算法压力测试：

```bash
PYTHONPATH=src:experiments python3 experiments/run_scalability.py --scenario clustered --sizes 50 100 200 --support-policy all_edges --min-support-ratio 0.75 --adjacency-support-weight 1 --methods greedy topology_process_aware process_aware_beam --output results/scalability_clustered_beam.csv
```

使用第二章排样实验包中的真实排样结果运行第三章路径优化：

```bash
PYTHONPATH=src python3 experiments/run_chapter2_route.py --board-id 1
```

如需指定压缩包内某一个 `placements.csv`，先用脚本输出中的 `placements_member`，再通过 `--placements-member` 传入。导入时会从相邻的 `config_dump.json` 读取 `BOARD_W`、`BOARD_H`、`TRIM` 和 `TOOL_D`。

若需要启用更严格的多边支撑稳定性模型，可运行：

```bash
PYTHONPATH=src python3 experiments/run_chapter2_route.py --board-id 1 --support-policy all_edges --min-support-ratio 0.5
```

`run_chapter2_route.py`、`diagnose_chapter2_route.py`、`run_chapter2_batch.py` 和 `run_safe_lift_sensitivity.py` 均支持 `--support-policy`、`--min-support-count`、`--min-support-ratio`、`--min-area-normalized-support` 和 `--adjacency-support-weight`。

对真实排样结果生成逐动作诊断 CSV 和路径图：

```bash
PYTHONPATH=src python3 experiments/diagnose_chapter2_route.py --board-id 1 --method topology_local_search_process_aware --label-parts
```

诊断 CSV 会记录硬约束、空移模式、稳定性惩罚增量和低支撑零件状态，可用于解释算法为何选择连续完成某个零件。

对 `process_aware_beam` 生成逐层搜索诊断 CSV：

```bash
PYTHONPATH=src:experiments python3 experiments/diagnose_beam_search.py --board-id 1 --support-policy all_edges --min-support-ratio 0.75 --adjacency-support-weight 1
```

诊断 CSV 会记录每层输入 beam 数、候选扩展数、层级剪枝数、父前缀保底扩展数、去重数、多样性过滤数、保留 beam 数、当前最佳稳定性和通行代价，用于分析束搜索参数和剪枝策略。

对三个第二章实验包进行小批量真实数据对比：

```bash
PYTHONPATH=src python3 experiments/run_chapter2_batch.py
```

该命令默认从每个 zip 抽取 2 个 `placements.csv`，每个文件选 1 张零件数量较多的板，输出 `results/chapter2_batch_routes.csv` 和 `results/chapter2_batch_summary.csv`。
默认方法包含 `greedy`、`path_distance_local_search`、`topology`、`topology_process_aware`、`process_local_search_multistart`、`process_aware_beam`、`process_aware_beam_adaptive`、`process_aware_beam_polished`、`process_aware_beam_adaptive_polished`、`topology_local_search` 和 `topology_local_search_process_aware`。
批量明细 CSV 会记录 `process_aware_beam` 的 beam 参数，并额外输出按零件数量区间聚合的 bin summary，便于做规模分层分析。
`run_chapter2_batch.py`、`run_ablation.py` 和 `run_scalability.py` 会同步生成 `*_manifest.json`，记录命令行参数、Git commit、输入归档和输出文件元数据。若需要指定 manifest 路径，可使用 `--manifest-output`。
论文主实验可使用 `--experiment-preset paper-main`，该 preset 默认启用 `all_edges` 支撑模型、`min_support_ratio=0.75` 和 `adjacency_support_weight=1`；显式传入的稳定性参数会覆盖 preset。
`run_chapter2_batch.py`、`run_ablation.py`、`run_scalability.py` 和 `run_adaptive_margin_sensitivity.py` 会在每个方法/消融变体/规模任务/敏感性案例完成后立即追加写入明细 CSV，并同步生成 `*_progress.csv` 记录 started、completed、skipped 和 timed_out 事件。脚本默认在终端输出任务进度条；如需关闭，可使用 `--no-progress-bar`。若长实验中途终止，可在原命令中加入 `--resume` 继续运行，脚本会读取已有明细行并跳过已完成的任务组合；如需指定进度日志位置，可使用 `--progress-output`。长实验建议额外加入 `--task-timeout-seconds 600`，避免单个异常慢任务阻塞整轮实验。

基于真实数据批量结果生成论文图表：

```bash
PYTHONPATH=src python3 experiments/generate_chapter2_batch_figures.py
```

输出 `figures/fig_chapter2_batch_method_comparison.*` 和 `figures/fig_chapter2_batch_tradeoff_scatter.*`。

对最新批量结果生成逐板成对分析、统计表和图表：

```bash
PYTHONPATH=src:experiments python3 experiments/analyze_results.py --input results/chapter2_batch_expanded_3x3_upto50_multi_unstable_retained.csv --output-dir results/analysis_multi_unstable_retained --figures-dir figures/analysis_multi_unstable_retained
```

该脚本会输出 `method_summary.csv`、`paired_comparison.csv`、`paired_summary.csv` 和 `analysis_report.txt`。可通过 `--target-method process_aware_beam_adaptive_polished` 将质量优先的 adaptive beam+LS portfolio 作为目标方法。

对同一真实板件生成多方法路线对照图：

```bash
PYTHONPATH=src:experiments python3 experiments/visualize_route_comparison.py --support-policy all_edges --min-support-ratio 0.75 --adjacency-support-weight 1 --methods path_distance_local_search topology_process_aware process_aware_beam process_aware_beam_adaptive_polished --label-parts
```

该脚本会输出路线对照图、逐方法路线指标 CSV 和逐动作诊断 CSV。图中区分真实共边、近共边、同线链、单边切割、安全抬刀、低位绕行和稳定性事件，可用于解释算法改进来源。

运行核心创新点消融实验：

```bash
PYTHONPATH=src:experiments python3 experiments/run_ablation.py --max-members-per-archive 3 --boards-per-member 3 --min-rectangles 2 --max-rectangles 50 --support-policy all_edges --min-support-ratio 0.75 --adjacency-support-weight 1 --output results/ablation_real_3x3_upto50.csv --summary-output results/ablation_real_3x3_upto50_summary.csv
```

默认消融项包括：完整 `process_aware_beam`、仅单边段、关闭稳定性引导、关闭邻接支撑、关闭 beam、路径距离基线、关闭低位绕行、关闭安全空移模式。结果可继续用 `experiments/analyze_results.py` 做成对统计。

生成消融实验专用论文图和关键统计表：

```bash
PYTHONPATH=src:experiments python3 experiments/generate_ablation_figures.py
```

输出 `figures/ablation_real_3x3_upto50/fig_ablation_*.*` 和 `results/ablation_real_3x3_upto50_key_table.csv`。

基于当前 77 张真实板材主实验、消融实验和 exact-gap 结果生成论文可用表格与图：

```bash
scripts/reproduce_chapter3_paper_artifacts.sh artifacts
```

该命令只读取已有 CSV，输出 `results/paper_artifacts/table_*.csv`、`results/paper_artifacts/table_*.tex` 和 `figures/paper_artifacts/fig_*.*`，其中包含 adaptive portfolio 的来源归因表。 如需从头重跑主实验和消融实验，可使用：

```bash
scripts/reproduce_chapter3_paper_artifacts.sh full
```

`full` 模式会带 `--resume` 继续已有结果，并保留进度日志；运行时间明显更长。
完整重跑还会生成最终主算法的 exact-gap 小规模对比、`fallback_margin` 敏感性结果和 50/75/100 件规模实验，用于支撑 adaptive portfolio 的参数选择。当前最终默认 `fallback_margin=1000`，主实验文件名带 `margin1000_after_runtime_opt`。

单独运行阈值敏感性实验：

```bash
PYTHONPATH=src:experiments python3 experiments/run_adaptive_margin_sensitivity.py --experiment-preset paper-main --margins 0 250 500 750 1000 1500
```

输出 `results/adaptive_margin_sensitivity_after_runtime_opt_real_20_50*.csv` 和 `figures/adaptive_margin_sensitivity_after_runtime_opt_real_20_50/fig_adaptive_margin_sensitivity.*`。

自动筛选适合画路线对照图的代表性板件：

```bash
PYTHONPATH=src:experiments python3 experiments/select_ablation_cases.py --output results/ablation_representative_cases_after_runtime_opt.csv
```

默认读取 `results/analysis_ablation_full_real_20_50/paired_comparison.csv`。当前最终代表性案例输出为 `results/ablation_representative_cases_after_runtime_opt.csv`，其中包含每类消融最有解释力的 `archive`、`placements_member` 和 `board_id`。

对代表性消融案例生成路线对照图：

```bash
PYTHONPATH=src:experiments python3 experiments/visualize_ablation_case.py --selection-priority single_edges_only:machining_cost_reduction_pct --variants full_process_aware_beam single_edges_only --support-policy all_edges --min-support-ratio 0.75 --adjacency-support-weight 1 --label-parts
```

该脚本会输出消融案例路线图、逐方法指标 CSV 和逐动作诊断 CSV，用于把消融表格中的数值差异对应到具体切割路径。

生成安全抬刀固定成本敏感性实验：

```bash
PYTHONPATH=src python3 experiments/run_safe_lift_sensitivity.py
```

输出 `results/safe_lift_sensitivity*.csv` 和 `figures/fig_safe_lift_fixed_cost_sensitivity.*`。

生成释放稳定性支撑约束敏感性实验：

```bash
PYTHONPATH=src python3 experiments/run_support_sensitivity.py
```

默认扫描 `all_edges` 支撑策略下的多个 `min_support_ratio`。如需比较共边邻接支撑，可加入 `--adjacency-support-weights 0 1`。输出 `results/support_sensitivity*.csv`、`figures/fig_support_constraint_sensitivity_*.*`、`figures/fig_adjacency_support_ablation_*.*` 和 `figures/fig_adjacency_support_ablation_compact_*.*`。

生成工艺感知束搜索参数敏感性实验：

```bash
PYTHONPATH=src:experiments python3 experiments/run_beam_sensitivity.py
```

默认扫描多个 `beam_width`、层级扩展倍率和 `diversity_bucket_limit`，输出 `results/beam_sensitivity*.csv` 与 `figures/fig_process_aware_beam_sensitivity.*`，用于选择稳定性、通行代价和运行时间之间的折中参数。脚本会在每个参数组合前清空低位绕行缓存，避免后运行的配置因缓存预热获得不公平的时间优势。

后续真实数据优化路线记录在 `docs/real_data_optimization_roadmap.md`。

```bash
PYTHONPATH=src python3 experiments/generate_figures.py
```

图表会输出到 `figures/`，汇总表会输出到 `results/scalability_summary.csv`。
