# 第三章代码实现与 SCI 文章主线计划 v1

## 1. 当前修订原则

第三章代码不能只实现一个完整路径的全局评价器，也不能把边段、入刀点、方向和空移关系混在一个松散结构里。后续第四章如果要在排样搜索过程中频繁调用第三章路径代价，路径评价必须支持增量更新；第三章论文写作也必须从“工程流程”提升为“数学模型 + 拓扑感知启发式求解”。

因此，后续实现按以下原则收敛：

1. 路径评价支持增量评估；
2. 切割图采用明确的有向异构图；
3. 工艺状态维护已加工几何集合；
4. 干涉检测进入核心约束；
5. SCI 写作中建立数学规划模型，即使最终用启发式算法求解。

## 2. 代码架构修订

### 2.1 `models.py`

基础对象：

- `ToolConfig`：刀具直径、刀具半径、修边量、安全间距、空移距离度量方式；
- `Panel`：板材尺寸、有效加工区域；
- `RectanglePart`：矩形件坐标、尺寸、旋转状态、编号；
- `EdgeSegment`：矩形边段，包含起点、终点、方向、所属矩形、边类型；
- `DirectedCutSegment`：有向切割边段，表示可执行的切割动作；
- `CuttingState`：动态加工状态，记录已切边段、已加工区域、当前位置、当前方向、释放状态；
- `PathMetrics`：路径代价向量；
- `IncrementalMetricsState`：增量评价状态。

`CuttingState` 不能只是静态参数集合，必须保存动态几何状态。第一版可用已加工线段集合和简单空间索引实现，后续再升级为网格索引或四叉树。

### 2.2 `cutting_graph.py`

采用有向图：

`G = (V, E)`

顶点 `V`：

- 候选入刀点；
- 候选出刀点；
- 有向切割边段状态；
- 切割单元的起止状态。

边 `E` 分为两类：

- 实边 `cut_edge`：表示实际切割动作，权值包括切割长度、方向惩罚、入刀/出刀代价；
- 虚边 `travel_edge`：表示空移动作，权值为空移距离、越界/干涉惩罚、方向变化代价。

普通 TSP 只作为基线。本文方法优化的是带状态约束的有向异构切割图路径，不再把矩形件中心点当作唯一访问节点。

### 2.3 `relations.py`

识别三类边界关系：

- `shared_edge`：真实共边；
- `near_shared_edge`：近共边，重点用于刀具通道与连续切割链；
- `collinear_edge`：同线边段，用于连续访问与方向保持。

识别结果不仅用于可视化，还要直接影响候选动作生成、局部搜索邻域和路径代价。

### 2.4 `metrics.py`

必须支持两类评价：

1. 全局评价：输入完整路径，输出完整 `PathMetrics`；
2. 增量评价：输入当前 `IncrementalMetricsState` 和一个新动作，局部更新路径代价。

增量状态至少包含：

- 当前刀具位置；
- 当前刀具方向；
- 已切割边段集合；
- 已访问切割单元集合；
- 已加工几何集合；
- 入刀/抬刀计数；
- 空移距离累计；
- 转角惩罚累计；
- 干涉/越界惩罚累计；
- 稳定性惩罚累计。

这样第四章或后续前缀搜索只需评估局部扩展，不必每次全图重构。

### 2.5 `collision.py`

新增独立模块，用于物理可行性检查：

- 刀具中心线是否进入修边安全区；
- 空移线段是否穿越不可达区域；
- 空移轨迹是否与已释放零件/已加工区域发生干涉；
- 刀具通道宽度是否满足刀径 6 mm；
- 边界偏置后是否越界。

第一版实现可以用线段相交、包围盒和均匀网格；后续再考虑更复杂的多边形索引。

### 2.6 `optimizer.py`

局部搜索算子必须是工艺感知的：

- 状态保持 swap；
- 方向验证 2-opt；
- 切割链整体 relocate；
- 同线链 merge/split；
- 支撑边段延迟释放；
- 每次动作后进行入刀点重算、方向验证、状态依赖检查和干涉检测。

## 3. SCI 文章主线修订

### 3.1 Introduction

文章立意建议提升为：

传统 nesting 方法通常默认排样紧凑性与加工效率正相关，但在 CNC 板材切割中，这一假设并不总成立。由于刀具半径补偿、近共边刀具通道、连续切割拓扑和零件释放稳定性，紧凑排样可能并不具有最优可切削性。这种“排样几何结构”与“加工路径代价”之间的非对称解耦，是本文研究的核心动机。

可使用表述：

`asymmetric decoupling between nesting compactness and cutting path efficiency`

### 3.2 Problem Definition

必须建立数学闭环：

- 输入：排样布局 `L`，矩形件集合 `R`，刀具参数 `T`；
- 输出：切割动作序列 `A = (a_1, a_2, ..., a_m)`；
- 目标：最小化路径代价函数 `F(A | L, T)`；
- 约束：轮廓完整切割、边段唯一访问、刀具通道可行、入刀点合法、轨迹无干涉、修边安全区、释放稳定性。

即使最终采用启发式求解，也需要先给出 MIP 或状态图优化模型。

### 3.3 Methodology

方法部分不写“行业规则融合”，改写为：

`Topology-Aware Heuristic Operators`

这些算子包括：

- 空间聚类算子；
- 邻接兼容算子；
- 方向连续算子；
- 支撑稳定算子；
- 边界安全算子。

它们通过边界关系权重动态缩减局部搜索邻域，降低无效动作数量，并提升路径可行性。

### 3.4 Experiments

除路径质量对比和消融实验外，必须增加计算效率实验：

- 10 件；
- 50 件；
- 100 件；
- 200 件；
- 500 件。

输出耗时增长曲线，证明第三章算法具有作为第四章底层评价器的潜力。

## 4. 下一步代码实施顺序

第一步只写基础结构和可测试几何模块：

1. `models.py`：补全工艺参数、边段、有向切割边段、状态与增量评价对象；
2. `geometry.py`：边段、距离、方向、线段相交、边界偏置基础函数；
3. `relations.py`：共边、近共边、同线边段识别；
4. `collision.py`：修边区、刀具通道和基础干涉检测；
5. `metrics.py`：全局评价与增量评价框架。

这一阶段完成后，再进入 `cutting_graph.py` 和 `optimizer.py`。

## 5. 当前代码进展

截至当前版本，已经完成以下可运行模块：

1. `relations.py`：真实共边、近共边、同线边段识别；
2. `cutting_units.py`：从边界关系生成候选切割单元；
3. `cutting_graph.py`：构造包含切割边和空移边的有向异构切割图；
4. `metrics.py`：支持全局评价与增量评价；
5. `topology_operators.py`：拓扑感知候选动作生成，包括同零件连续、关系单元优先、方向连续惩罚；
6. `local_search.py`：支持 swap、relocate、方向受限 2-opt 的局部搜索；
7. `process_model.py`：从排样结果构造零件边段集合、零件多边形和支撑边信息；
8. `layout_generator.py`：生成可复现的合成矩形件排样，当前支持 `grid` 和 `clustered` 两类场景；
9. `io.py`：支持从第二章实验包中的 `placements.csv`/zip 导入真实排样结果，并按 `board` 拆分成单板 `Layout`；
10. `optimizer.py`：提供贪心、路径距离局部搜索、拓扑感知、工艺感知前缀束搜索、拓扑感知局部搜索和工艺感知路径规划入口。

当前实验脚本已经可以输出三类方法的基础对比：

- `greedy`；
- `path_distance_local_search`；
- `topology`；
- `process_aware_beam`；
- `topology_local_search`。

其中 `path_distance_local_search` 用于模拟经典路径优化基线：以近邻路径为初始解，通过 swap、relocate 和方向受限 2-opt 优化硬约束与真实通行代价，但不把动态释放稳定性作为优化目标。该基线用于区分“路径距离更短”和“加工过程更稳定”两类贡献。

`process_aware_beam` 是当前核心创新算法雏形：它不再像 `topology_process_aware` 一样每一步只选一个当前最优动作，而是在增量评价状态上保留多个候选前缀。每个前缀扩展时都检查动态释放状态、低支撑零件完成链、低位绕行/安全抬刀代价和拓扑连续性，再通过束宽控制搜索规模。

当前实现已加入 `max_layer_expansions` 层级扩展剪枝。该机制先基于父节点增量状态和拓扑转移得分对候选扩展做低成本预排序，只对最可能进入下一层 beam 的候选执行真实空移模式规划和增量评价，从而降低低位绕行可见图的重复计算开销。

当前真实数据默认层扩展上限采用约 `7 * beam_width`，并按零件规模切换束宽与候选池：20 件及以下保留更宽 beam，21-75 件转入中等配置。该设置来自真实板材 smoke 调参，当前比原 `12 * beam_width` 默认同时降低通行代价和运行时间。

当前实现也加入了 `diversity_bucket_limit` 前缀多样性约束。每层 beam 选择时，算法会限制同一当前零件组、低支撑状态、释放数量和方向的同质前缀数量，避免宽 beam 被局部相似路径占满。当前真实数据默认值设为 1。

当前实现进一步加入了低支撑父前缀保底扩展。与对所有父前缀平均保底不同，默认策略只对 `unstable_part_ids` 非空的父前缀保留额外候选，避免普通低质量前缀被强行保留，同时防止低支撑完成链在层级剪枝中过早消失。

低位绕障模块已进行几何级加速：绕障可见图连接使用轴对齐线段快速判定，水平/垂直候选边按行列预筛相关障碍物，网格节点合法性也按行预筛；绕障缓存对障碍物集合顺序做规范化，并可复用反向路径。当前实现进一步把绕障图内部结构改为整数节点邻接表，并将最短路搜索从纯 Dijkstra 改为带曼哈顿启发的 A*。该优化不改变目标函数，只降低 beam 前缀评估中的重复几何检测开销。

强对比实验已补充 `process_local_search_multistart`。该方法从最近邻、拓扑、工艺感知拓扑、横向/纵向 sweep 及其反向 sweep 多个初解出发，使用同一套工艺目标局部搜索，最后取字典序工艺代价最优路径。它用于模拟 VNS/ILS 类强启发式 baseline，避免只与弱贪心或单初解局部搜索对比。

正式数学模型已单独整理到 `docs/formal_model_and_strong_baselines.md`，其中给出了候选切割单元覆盖约束、有向排序变量、空移模式变量、释放件干涉约束、动态支撑稳定性约束、分层目标函数和状态图求解形式。该文件可作为后续 SCI 论文 Problem Definition 与 Methodology 的基础版本。

小规模最优性对照已补充 `src/cnc_cutting/exact_dp.py` 和 `experiments/run_exact_gap.py`。该 DP 只在默认 12 个被选切割单元以内运行，用于报告 beam、多启动局部搜索与 exact 解之间的 gap；超过上限会直接跳过，避免指数枚举拖垮实验。

新增 `process_aware_beam_polished` 和 `process_aware_beam_adaptive_polished` 作为 beam 后处理增强版本：先运行工艺感知前缀束搜索，再以 beam 输出为初解执行工艺目标局部搜索。局部搜索比较口径已统一为 `process_metric_key()`，避免为了减少刀具事件而牺牲显著的加工路径代价。

当前最终主方法为 `process_aware_beam_adaptive_polished`，默认 `fallback_margin = 1000.0`。该方法在 77 张 20-50 件真实板材上完成主实验，输出见 `results/paper_main_margin1000_after_detour_intgraph_real_20_50.csv` 和 `results/analysis_paper_main_margin1000_after_detour_intgraph_real_20_50/`。平均 `travel_mode_cost = 9788.2`，平均 `machining_cost = 29608.6`，平均运行时间约 `462.1 ms`，`hard_penalty = 0`，`stability_penalty = 0`。相对 `topology_process_aware`，平均 `travel_mode_cost` 降低约 `9.74%`；相对 `process_local_search_multistart` 降低约 `9.60%`；相对原 `process_aware_beam` 降低约 `5.55%`。因此论文中应将最终算法写成“工艺感知前缀束搜索 + 局部精修 + 自适应兜底 portfolio”，而不是单一 beam 或单一局部搜索。

最终 portfolio 归因结果见 `results/adaptive_polished_selection_attribution_margin1000_after_detour_intgraph_summary.csv`：`Beam+process LS` 被选中 44 例，占 `57.14%`；`Adaptive beam` 被选中 32 例，占 `41.56%`；`Wide beam+LS fallback` 只触发 1 例，占 `1.30%`。`fallback_margin` 敏感性实验见 `results/adaptive_margin_sensitivity_after_detour_intgraph_real_20_50_summary.csv`，当前选择 `1000.0` 是质量与兜底触发比例之间的折中。

全量消融实验已在同一批 77 张 20-50 件真实板材上完成，输出见 `results/ablation_after_detour_intgraph_real_20_50.csv` 和 `results/analysis_ablation_full_real_20_50/paired_summary.csv`。结果显示：去掉异构切割单元后，`single_edges_only` 虽然平均通行代价更低，但总加工代价比完整方法高约 `27.69%`，证明共边、近共边和同线链单元主要通过减少重复切割贡献收益；去掉稳定性引导或使用纯路径距离局部搜索会得到更短路径，但稳定性惩罚分别上升到 `6.71` 和 `5.09`；去掉邻接支撑引导后，完整方法平均通行代价降低约 `23.94%`，稳定性惩罚也从 `1.09` 降为 `0`；去掉绕行算子后，完整方法平均通行代价降低约 `3.87%`；去掉安全空移模式后，路径会更短但平均硬约束惩罚升至 `1.71`，因此不能作为可行工艺解。

批量实验 CSV 已记录 `process_aware_beam` 的关键参数，包括 `beam_width`、`beam_candidate_pool_size`、`beam_max_expansions_per_node`、`beam_max_layer_expansions`、多样性限制和低支撑父前缀保底参数；同时输出 `rectangle_count_bin` 和按规模区间聚合的 bin summary，便于后续论文实验按零件规模分层报告。

`run_scalability.py` 已扩展到 `process_aware_beam`，并支持稳定性模型参数与 beam 参数覆盖。当前三个第二章真实归档没有 50 件以上板材，因此 50/100/200 件结果来自合成 clustered 布局，仅用于算法复杂度和压力测试，不与真实数据结论混写。

已新增可选 `completion_aware_prerank` 策略，用于在低支撑前缀预剪枝阶段优先保留覆盖不稳定零件的候选。该策略对合成严格支撑压力测试略有帮助，但会降低当前真实数据默认路径质量，因此暂不作为默认算法。

多低支撑零件的状态保留语义已修正：若当前动作继续处理任一低支撑零件，则其他低支撑零件继续保留在状态中，不立即记为放弃；只有动作完全不关联低支撑集合时才触发放弃惩罚并清空上下文。这使算法能够处理多个零件同时进入低支撑完成链的情况，是当前大规模 clustered 压力测试能够归零稳定性的关键。

同时，`run_scalability.py` 已支持默认 `10, 50, 100, 200, 500` 件规模敏感性实验，并输出 CSV 文件：

- `runtime_ms`；
- `candidate_unit_count`；
- `selected_unit_count`；
- `action_count`；
- `air_move_distance`；
- `cutting_length`；
- `pierce_count`；
- `lift_count`；
- `stability_penalty`；
- `hard_penalty`。

局部搜索当前采用两级加速：

1. 前缀状态缓存：邻域解只从 `affected_start_index` 之后重新评价，未变化前缀直接复用；
2. 有界候选池：拓扑初解阶段只在当前刀位附近的候选单元中做拓扑评分，避免每一步扫描全部剩余单元；
3. `heapq.nsmallest` 候选筛选：候选池小于总单元数时避免完整排序；
4. 释放零件 bounds 缓存：释放零件多边形的包围盒只计算一次，后续空移干涉检测直接复用；
5. 轴对齐矩形快速相交：矩形零件使用 Liang-Barsky 风格的线段-矩形相交判断，避免反复执行通用多边形求交；
6. 正反向候选缓存：同一切割单元的 forward/reverse 候选在拓扑排序中复用。

规模实验中的 `topology_local_search` 使用 first-improvement 模式，以模拟后续第四章中作为底层快速评价器的使用方式。完整 best-improvement 仍保留在 `LocalSearchConfig(first_improvement=False)` 中。

当前实验场景分为三类：

1. `grid`：随机网格排样，主要用于复杂度和规模敏感性分析；
2. `clustered`：带近共边通道的排样，主要用于检验共边/近共边候选单元是否被识别；
3. `clustered + process-aware-topology`：在拓扑初解阶段显式考虑动态释放、空移干涉和支撑边稳定性，用于小规模工艺约束消融。

初步结果显示：单纯拓扑优先在 `clustered` 场景下并不必然优于贪心，因为它可能提前释放零件或增加释放后干涉；加入工艺状态感知后，稳定性惩罚可以显著下降。例如 50 件 clustered 算例中，`topology` 的 `stability_penalty` 可从 50 降至 0，但运行时间从约 62 ms 上升到约 1121 ms。这说明论文中应将其定位为“工艺约束增强模块”，并进一步研究空间索引与增量状态压缩。

经过当前性能优化后，50 件 clustered 工艺感知拓扑初解运行时间约为 283 ms，`topology_local_search` 约为 573 ms。500 件 grid 的 `topology_local_search` 约为 2312 ms，500 件 clustered 的 `topology_local_search` 约为 1344 ms。当前代码已经具备进一步做批量重复实验和误差棒统计的基础。

当前图表生成脚本：

- `experiments/generate_figures.py`

输出：

- `results/scalability_summary.csv`；
- `figures/fig_runtime_grid_fast.pdf/png`；
- `figures/fig_runtime_clustered_fast.pdf/png`；
- `figures/fig_clustered_process_aware_tradeoff.pdf/png`。

当前真实排样数据接入脚本：

- `experiments/run_chapter2_route.py`

该脚本可直接读取第二章输出的 zip 实验包，从相邻 `config_dump.json` 中获取板材尺寸、修边量和刀径，并从 `placements.csv` 中按 `board` 选择单张板进行第三章路径优化。初步 smoke test 已在 `strong_baseline_overnight_20260407_001243.zip` 中的 `data6/rh_mcts_ref/seed_1004/placements.csv` 上跑通。结果显示边界惩罚为 0，剩余硬约束主要来自释放件空移干涉，这将成为下一轮优化重点。

空移干涉模型已从单一硬碰撞升级为低位空移与安全抬刀空移两类。低位空移穿越释放件仍记录为潜在冲突，安全抬刀空移则将硬碰撞转化为 `safe_lift_count` 和 `safe_lift_distance`。在当前真实数据 smoke test 中，`collision_penalty` 已降为 0，保留的工艺代价表现为安全抬刀动作数量和安全抬刀距离。下一步应进一步加入绕行空移候选，使算法能在“绕行距离增加”和“安全抬刀成本增加”之间做选择。

当前已加入 `low_clearance_detour` 绕行空移模式。算法在发现低位直线空移会穿越释放件时，会先基于释放件包围盒生成正交低位绕行路径；若绕行不可行，再回退到安全抬刀。真实数据 smoke test 中，`safe_lift_count` 已降为 0，`detour_count` 和 `detour_distance` 成为新的可解释工艺代价。该模块后续需要继续优化运行时间，并引入机床时间模型来平衡绕行和安全抬刀。

进一步修订后，空移模式选择已由固定优先级改为毫米等价成本比较。`ToolConfig` 中的 `safe_lift_fixed_cost`、`safe_lift_travel_weight` 和 `detour_travel_weight` 控制安全抬刀与低位绕行之间的权衡，路径评价输出 `travel_mode_cost`。当前实现还加入了低位绕行缓存，工艺感知局部搜索在真实 smoke test 上的耗时从约 1 s 降至约 0.3 s。后续论文实验应把 `safe_lift_fixed_cost` 作为敏感性参数，验证不同机床参数下路径策略的变化。

真实排样批量实验脚本已新增：

- `experiments/run_chapter2_batch.py`
- `experiments/generate_chapter2_batch_figures.py`

批量脚本默认从三个第二章实验包各抽取 2 个 `placements.csv`，每个文件选择 1 张零件数量较多的板，批量运行 `greedy`、`path_distance_local_search`、`topology`、`topology_process_aware`、`process_aware_beam`、`topology_local_search` 和 `topology_local_search_process_aware`。当前 smoke batch 已能输出路径基线、拓扑基线和工艺感知方法的多指标结果。图表脚本生成 `figures/fig_chapter2_batch_method_comparison.*` 和 `figures/fig_chapter2_batch_tradeoff_scatter.*`。初步结果显示，路径距离局部搜索可以降低通行代价，但不能自动消除严格支撑约束下的稳定性惩罚；工艺感知方法能将平均稳定性惩罚降为 0，但会显著增加空移距离和运行时间；前缀束搜索在保持稳定性惩罚为 0 的同时显著降低工艺感知路径的通行代价，但运行时间更高。这为后续“工艺可行性-路径效率-搜索开销”提供了实验主线。

束搜索参数敏感性脚本已新增：

- `experiments/run_beam_sensitivity.py`
- `experiments/diagnose_beam_search.py`

`run_beam_sensitivity.py` 扫描 `beam_width`、层级扩展倍率和 `diversity_bucket_limit`，输出扩展节点数、运行时间、通行代价、稳定性惩罚和图表。脚本会在每个参数组合前清空低位绕行缓存，避免缓存预热造成运行时间比较不公平。初步 smoke 结果显示，束宽变大并不必然带来更低通行代价；当前小样本中严格多样性限制可进一步降低通行代价，但仍需在更大真实样本上验证。

`diagnose_beam_search.py` 输出逐层搜索诊断，包括输入 beam 数、候选扩展数、层级剪枝数、父前缀保底扩展数、重复前缀剪枝数、多样性过滤数、保留 beam 数、每层最佳稳定性和通行代价。该诊断用于解释束搜索在哪些层因为释放稳定性或候选剪枝发生质量变化。

安全抬刀固定成本敏感性脚本已新增：

- `experiments/run_safe_lift_sensitivity.py`

该脚本在第二章真实排样样本上扫描 `safe_lift_fixed_cost`，输出 `results/safe_lift_sensitivity.csv`、`results/safe_lift_sensitivity_summary.csv`，并生成 `figures/fig_safe_lift_fixed_cost_sensitivity.*`。当前小样本结果显示，当安全抬刀固定成本为 0 时，算法倾向直接使用安全抬刀；成本提高后，算法会转向低位绕行。这一结果可用于论文中论证“空移模式选择受机床过程参数驱动”，而不是固定经验规则。

释放稳定性支撑约束敏感性脚本已新增：

- `experiments/run_support_sensitivity.py`

该脚本扫描 `support_policy`、`min_support_ratio` 和 `adjacency_support_weight`，输出 `results/support_sensitivity.csv`、`results/support_sensitivity_summary.csv`，并生成 `figures/fig_support_constraint_sensitivity_*.*`、`figures/fig_adjacency_support_ablation_*.*` 和 `figures/fig_adjacency_support_ablation_compact_*.*`。当前扩展到 18 个真实板材样本后，普通拓扑路径的稳定性惩罚仍会随支撑比例阈值提高快速上升；工艺感知路径在中等约束强度下能明显压低稳定性惩罚，但会增加空移距离和空移模式成本。加入共边邻接支撑后，高支撑阈值下的残余稳定性惩罚进一步下降，且工艺感知路径的额外空移减少。这组结果可作为第三章“稳定性-路径效率折中”的核心参数敏感性实验。

下一步需要继续加强两部分：

1. 扩大 `support_policy`、`min_support_ratio` 和 `min_area_normalized_support` 的真实数据扫描规模，确定论文主实验采用的稳定性约束强度；
2. 扩大严格支撑模型下的真实批量实验，统计稳定性改善、空移变化和运行时间变化；
3. 比较 `adjacency_support_weight` 的不同取值，确定共边邻接支撑在论文主实验中的默认权重；
4. 进一步引入空间索引或 KD-tree，使候选池生成从排序筛选升级为近邻查询。

当前动态评价已经支持：

- 当某零件全部边段被切割后，将其标记为释放零件；
- 将释放零件多边形加入后续空移干涉检测；
- 若支撑边过早切割，或剩余支撑数量、剩余支撑长度比例、面积归一化支撑强度不足，则进入低支撑状态；
- 低支撑状态采用延迟处罚：连续完成零件释放不计罚，离开未释放零件或最终仍未释放时计入 `stability_penalty`；
- 空移动作携带下一切割单元的 `covered_segment_ids`，因此前往完成同一低支撑零件的空移不会被误判为离开该零件；
- 尚未释放的共边相邻零件可提供临时等效支撑长度，并通过 `adjacency_support_weight` 控制贡献强度；
- 诊断输出已记录稳定性惩罚增量和低支撑零件状态，可用于论文中解释完成链约束的作用机制；
- 比较路径优劣时，优先级为硬约束、稳定性、刀具事件、空移距离、转角惩罚、连续性奖励。

## 6. 当前最终图表与可复现入口

当前论文表格和总图统一由以下命令生成：

```bash
PYTHONPATH=src:experiments python3 experiments/generate_paper_artifacts.py
```

输出目录：

- `results/paper_artifacts/`
- `figures/paper_artifacts/`

完整重跑入口：

```bash
scripts/reproduce_chapter3_paper_artifacts.sh full
```

该脚本会重跑主实验、portfolio 归因、消融实验、exact gap、adaptive margin 敏感性和 50/75/100 件规模实验。若只根据已有 CSV 重建表格和图，运行：

```bash
scripts/reproduce_chapter3_paper_artifacts.sh artifacts
```

当前新增代表性路线图：

- `figures/representative_route_cases_after_runtime_opt/main_adaptive_gain.png`：最终 `Adaptive beam+LS` 相对路径基线、工艺拓扑和普通 beam 的代表性收益；
- `figures/representative_route_cases_after_runtime_opt/shared_unit_ablation.png`：共边/同线切割单元消融；
- `figures/representative_route_cases_after_runtime_opt/stability_guidance_ablation.png`：稳定性引导消融；
- `figures/representative_route_cases_after_runtime_opt/adjacency_support_ablation.png`：邻接支撑消融；
- `figures/representative_route_cases_after_runtime_opt/safe_travel_ablation.png`：安全空移模式消融。

对应 metrics 和逐动作 diagnostics 在 `results/representative_route_cases_after_runtime_opt/`。这些图适合用于论文 Methodology 或 Experiment 的解释性案例，不应替代 77 张板的统计表。
