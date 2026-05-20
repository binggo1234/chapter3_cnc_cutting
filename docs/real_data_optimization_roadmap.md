# 真实排样数据接入后的优化路线

本文第三章代码已经可以读取第二章输出的 `placements.csv`/zip 实验包，并将单张板排样结果转为切割路径优化输入。后续优化按以下顺序推进，避免只停留在可视化或单点调参。

## 1. 真实数据路径诊断与可视化

目标：

- 绘制单张板上的矩形件位置；
- 绘制切割动作、空移动作和触发干涉的空移段；
- 输出每一步动作的 `collision_penalty`、`boundary_penalty`、释放件数量和路径代价；
- 判断真实数据中的硬约束惩罚来自算法缺陷，还是来自当前工艺模型过严。

产出：

- 诊断 CSV；
- 路径诊断图；
- 真实算例中的主要冲突模式总结。

## 2. 空移干涉模型重构

当前模型将空移穿过已释放零件视为硬碰撞。真实 CNC 中，如果刀具抬到安全高度，部分跨越释放件的空移应被视为可行但有额外成本。

需要拆分为：

- 低位空移：穿越释放件时不可行或高惩罚；
- 安全抬刀空移：允许跨越释放件，但增加 Z 向动作、加工时间和刀具事件成本；
- 绕行空移：允许通过局部绕行降低碰撞，但增加空移距离。

当前进展：

- 已在 `TravelMode` 中区分 `low_clearance` 与 `safe_lift`；
- 第二章导入数据默认启用 `allow_safe_lift_over_released_parts`；
- 低位空移穿越释放件仍保留为 `low_clearance_collision_penalty`；
- 安全抬刀空移将硬碰撞转化为 `safe_lift_count` 和 `safe_lift_distance`；
- 真实算例 smoke test 中，原有 `collision_penalty` 已从 5/6/7 类硬惩罚转为 0，同时记录安全抬刀动作数量；
- 已新增 `experiments/run_safe_lift_sensitivity.py`，用于比较不同 `safe_lift_fixed_cost` 下的安全抬刀和低位绕行选择。

尚未完成：

- 安全抬刀的时间模型仍是简化的次数和距离指标，后续可加入 Z 向升降时间、机床速度和加减速参数。

## 3. 碰撞感知路径选择

在候选切割单元选择阶段提前评估：

- 空移是否穿过释放件；
- 是否应该绕行；
- 是否需要安全抬刀跨越；
- 是否应优先切相邻单元，以减少释放后穿越；
- 是否应推迟会导致大面积释放的支撑边。

目标是让 `topology` 和 `topology_local_search` 不只依赖几何邻近和共边关系，而是成为状态感知的 CNC 路径优化算法。

当前进展：

- 已加入 `low_clearance_detour` 空移模式；
- 当直线低位空移会穿越释放件时，先尝试基于释放件包围盒的正交低位绕行；
- 若绕行不可行，再回退到安全抬刀空移；
- 路径评价中已单独记录 `detour_count` 与 `detour_distance`；
- 已加入 `travel_mode_cost`，用毫米等价成本在低位绕行与安全抬刀之间选择；
- 已加入绕行路径缓存，避免局部搜索反复构造相同释放状态下的绕行可见图；
- 真实算例 smoke test 中，硬碰撞保持为 0，路径会混合使用低位绕行与安全抬刀。

尚未完成：

- 绕行搜索目前仍是网格化正交路径，后续可用更紧凑的可见图或 A*；
- `safe_lift_fixed_cost` 仍是经验参数，后续需要用机床 Z 向速度、空移速度和加减速参数校准；
- 后续批量实验需要比较不同 `safe_lift_fixed_cost` 下的敏感性。

## 4. 第二章真实排样结果批量实验

从三个第二章实验包中抽取多组真实排样：

- 不同数据集；
- 不同排样算法；
- 不同随机种子；
- 不同板号；
- 不同零件数量。

输出对比：

- 空移距离；
- 入刀/抬刀次数；
- 碰撞/越界惩罚；
- 稳定性惩罚；
- 运行时间；
- 不同规模下的增长趋势。

当前进展：

- 已新增 `experiments/run_chapter2_batch.py`；
- 已新增 `experiments/generate_chapter2_batch_figures.py`；
- 已新增 `path_distance_local_search` 强路径基线：使用近邻初始解与 swap/relocate/方向受限 2-opt，优化真实空移/通行代价，但不显式优化释放稳定性；
- 已新增 `process_aware_beam` 工艺感知前缀束搜索：在每个前缀上维护增量评价状态，使用动态释放稳定性、空移模式代价和拓扑转移得分筛选多个候选前缀；
- 已为 `process_aware_beam` 加入层级扩展剪枝 `max_layer_expansions`，先对候选前缀做低成本预排序，再执行昂贵的低位绕行/安全抬刀增量评价；
- 已为 `process_aware_beam` 加入前缀多样性保留 `diversity_bucket_limit`，限制同一零件组、低支撑状态、释放数量和方向的同质前缀占满 beam；
- 已为 `process_aware_beam` 加入低支撑父前缀保底扩展，默认只对 `unstable_part_ids` 非空的父前缀保留额外候选，不强制普通父前缀扩展；
- 已新增 `experiments/run_beam_sensitivity.py`，用于扫描 `beam_width`、层级扩展倍率和多样性限制，记录扩展节点数、运行时间、通行代价和稳定性惩罚；
- 已新增 `experiments/diagnose_beam_search.py`，用于导出 `process_aware_beam` 的逐层搜索诊断，包括候选扩展、层级剪枝、重复前缀剪枝、多样性过滤、低支撑前缀数量和每层最佳代价；
- 束搜索敏感性脚本会在每个参数组合前清空低位绕行缓存，避免缓存预热造成运行时间比较不公平。
- 默认从三个第二章 zip 中各抽取 2 个 `placements.csv`，每个文件选择 1 张零件数较多的板；
- 当前 smoke batch 共生成 6 个真实板材案例、30 行方法结果；
- 逐案例结果输出到 `results/chapter2_batch_routes.csv`；
- 按方法汇总结果输出到 `results/chapter2_batch_summary.csv`。
- 批量图表输出到 `figures/fig_chapter2_batch_method_comparison.*` 和 `figures/fig_chapter2_batch_tradeoff_scatter.*`。
- 安全抬刀成本敏感性结果输出到 `results/safe_lift_sensitivity*.csv`，图表输出到 `figures/fig_safe_lift_fixed_cost_sensitivity.*`。
- 支撑约束敏感性结果输出到 `results/support_sensitivity*.csv`，图表输出到 `figures/fig_support_constraint_sensitivity_*.*`、`figures/fig_adjacency_support_ablation_*.*` 和 `figures/fig_adjacency_support_ablation_compact_*.*`。
- 已将释放稳定性从单一 `top` 支撑边扩展为可配置模型，真实数据脚本支持 `top`、`all_edges` 和 `none` 三类支撑策略；
- 已支持 `min_support_count`、`min_support_ratio` 和 `min_area_normalized_support`，可用于后续稳定性约束敏感性实验；
- 低支撑状态已改为延迟处罚：若刀具连续完成该零件释放，则不计稳定性违规；若刀具离开该未释放零件或最终仍未释放，则计入 `stability_penalty`。
- 已加入完成链上下文空移：空移动作携带下一切割单元的 `covered_segment_ids`，用于区分“前往完成同一低支撑零件”和“离开低支撑零件”；
- 诊断 CSV 已输出 `stability_penalty_delta`、`unstable_parts_before/after` 和低支撑零件 id，便于论文解释低支撑状态与稳定性违规的区别。
- 已新增 `experiments/run_support_sensitivity.py`，用于扫描支撑策略和 `min_support_ratio` 对稳定性、空移距离、空移模式成本和运行时间的影响。
- 已加入邻接/共边临时支撑贡献：尚未释放的相邻零件可通过真实共边长度提供等效支撑，权重由 `adjacency_support_weight` 控制。

当前初步观察：

- 所有方法在真实批量样本中均保持 `hard_penalty = 0`；
- 工艺感知方法可将平均 `stability_penalty` 降到 0；
- 工艺感知方法的空移距离和运行时间明显高于普通 greedy/topology，说明后续需要继续优化工艺约束与路径效率的折中；
- 当前批量规模仍属于 smoke test，不能直接作为论文最终结论。
- 在当前小样本下，`safe_lift_fixed_cost = 0` 时算法倾向直接安全抬刀；当该成本提高到 100 mm 等价成本及以上时，算法会明显转向低位绕行，说明空移模式选择已经能响应机床工艺参数。
- 严格 `all_edges + min_support_ratio=0.5` 小样本中，普通拓扑存在稳定性惩罚，工艺感知方法可将 `stability_penalty` 降为 0；这说明完成链上下文对释放稳定性有效。
- 当前 3 个真实样本的支撑敏感性实验显示：普通拓扑的平均稳定性惩罚会随 `min_support_ratio` 提高快速上升；工艺感知方法在 0.25 和 0.5 下基本保持低稳定性惩罚，但在 0.75 这种过强约束下仍有残余惩罚。
- 当 `adjacency_support_weight=1` 时，共边邻接支撑可进一步降低高支撑阈值下的残余稳定性惩罚，并减少工艺感知路径为满足稳定性而产生的额外空移。
- 邻接支撑消融图显示：在 `min_support_ratio=0.75` 时，工艺感知方法的平均稳定性惩罚由约 2.67 降为 0，普通拓扑由约 19.33 降为 2.00；说明共边邻接支撑对严格稳定性约束具有显著缓解作用。
- 扩展到 18 个真实板材样本后，趋势仍然成立：`min_support_ratio=0.75` 时，工艺感知方法加入邻接支撑后的平均稳定性惩罚约为 0.06，普通拓扑由约 20.83 降至 4.44；工艺感知路径的平均空移也从约 7946 降至约 5747。
- 新增路径距离局部搜索基线后，当前 3 个真实样本 smoke 显示：`path_distance_local_search` 可将贪心的平均 `travel_mode_cost` 从约 1759.90 降至约 1693.07，但在严格支撑约束与邻接支撑下平均 `stability_penalty` 仍为约 2.67；工艺感知方法则可将该惩罚降至 0。这一对比可用于说明“路径距离优化”和“加工稳定性优化”不是同一个问题。
- 新增前缀束搜索后，当前 3 个真实样本 smoke 显示：在 `all_edges + min_support_ratio=0.75 + adjacency_support_weight=1` 下，`process_aware_beam` 的平均 `stability_penalty` 为 0，平均 `travel_mode_cost` 约 2814.15，明显低于 `topology_process_aware` 的约 4291.19 和 `topology_local_search_process_aware` 的约 4288.52；代价是平均运行时间提高到约 1.35 s。该结果说明束搜索正在改善“稳定性满足后路径仍过长”的问题。
- 加入层级扩展剪枝后，同一 3 个真实样本 smoke 中，默认 `process_aware_beam` 平均运行时间降至约 0.90 s，平均 `travel_mode_cost` 约 3193.67，`stability_penalty` 保持为 0；仍明显低于两个工艺感知贪心/局部搜索方法的约 4290 通行代价。
- 加入严格前缀多样性限制后，同一 3 个真实样本 smoke 中，默认 `process_aware_beam` 平均 `travel_mode_cost` 进一步降至约 3084.46，`stability_penalty` 保持为 0，平均运行时间约 0.92 s。
- 加入低支撑父前缀保底扩展后，同一 3 个真实样本 smoke 中，默认 `process_aware_beam` 平均 `travel_mode_cost` 进一步降至约 2723.47，`stability_penalty` 仍为 0，平均运行时间约 1.05 s。若同时强制普通父前缀保底扩展，通行代价反而升高，因此默认仅保护低支撑前缀。
- 加入低位绕障几何加速后，绕障可见图构造改为轴对齐快速相交、行列障碍物预筛和规范化缓存；同一 3 个真实样本 smoke 中，`process_aware_beam` 平均运行时间由约 1.04 s 降至约 0.44 s，`travel_mode_cost` 保持约 2723.47，`stability_penalty` 保持为 0。
- 进一步调参后，默认 `max_layer_expansions` 从 `12 * beam_width` 收缩为 `7 * beam_width`，并将小规模阈值从 25 件收紧为 20 件。当前 3 个真实样本 smoke 中，默认 `process_aware_beam` 平均 `travel_mode_cost` 降至约 2418.69，平均运行时间约 0.29 s，`stability_penalty` 为 0；20-50 件单板 smoke 中，默认 `process_aware_beam` 的 `travel_mode_cost` 为约 7465.62，低于 `topology_process_aware` 的约 13053.01。
- 扩大到 27 张 50 件以内真实板材后，`process_aware_beam` 平均 `stability_penalty` 保持为 0，平均 `travel_mode_cost` 约 5078.47，低于 `topology_process_aware` 的约 5971.70；相对工艺感知拓扑通行代价降低约 14.96%，运行时间约为 2.08 倍。路径距离局部搜索的平均 `travel_mode_cost` 约 3215.16，但 `stability_penalty` 仍为约 2.93，说明路径优化不能替代工艺稳定性优化。
- 27 张板材的规模分箱结果显示：20 件及以下样本中，`process_aware_beam` 的平均 `travel_mode_cost` 约 2446.53，低于 `topology_process_aware` 的约 3110.77；21-50 件样本中，`process_aware_beam` 的平均 `travel_mode_cost` 约 10436.78，低于 `topology_process_aware` 的约 11580.79。两个规模区间中 beam 的平均 `stability_penalty` 均为 0。
- 完整扫描当前三个第二章归档后，未发现 50 件以上真实板材；最大板材约 45-46 件。因此 50 件以上规模实验目前只能使用合成布局作为算法压力测试，不能标注为真实数据结论。
- 已修正多低支撑零件的评价语义：当多个零件同时处于低支撑状态时，继续加工其中任一低支撑零件不再视为“放弃”其他低支撑零件；只有转向完全无关零件时才记为放弃。该修正解决了 synthetic clustered 大规模场景下多不稳定零件互相误罚的问题。
- `run_scalability.py` 已扩展支持 `process_aware_beam`、真实通行代价字段、稳定性模型参数和 beam 参数覆盖。clustered 合成布局 50/100/200 件压力测试显示：在严格 `min_support_ratio=0.75` 下，`process_aware_beam` 和 `topology_process_aware` 均可将稳定性惩罚降为 0；`process_aware_beam` 的通行代价分别约为 5862.56、11654.84、28378.24，低于 `topology_process_aware` 的约 9112.89、17074.84、33396.29。
- 新增可选 `completion_aware_prerank` beam 策略：低支撑前缀的预剪枝优先保留覆盖更多不稳定零件的候选。该策略在 synthetic clustered 100 件严格支撑场景下可将稳定性惩罚从 23 降到 21、并降低通行代价，但会损害 27 张真实板材的默认通行代价，因此当前仅作为压力测试开关，默认关闭。
- 初步束宽敏感性 smoke 显示，`beam_width`、`max_layer_expansions` 和 `diversity_bucket_limit` 对路径质量并非单调影响。当前小样本中 `diversity_bucket_limit=1` 优于关闭多样性和限制为 2，但仍需扩大样本后再固定最终参数。
- 单板 beam 诊断 smoke 显示：29 层搜索中层级预剪枝累计 3872 次，重复前缀剪枝 386 次，多样性过滤 20 次，低支撑父前缀保底扩展触发 72 次，后期最多同时保留 8 条低支撑前缀。这说明当前主要搜索压缩来自层级预剪枝，保底扩展主要在后半段低支撑状态中发挥作用。
- 已新增 `machining_cost = cutting_length + travel_mode_cost`，用于评估 CNC 总加工代价。该指标解决了仅看空移代价时低估共边/同线切割单元的问题：单边段消融虽然平均 `travel_mode_cost` 更低，但会重复切割大量轮廓边。
- 已新增 `experiments/run_ablation.py` 和 `experiments/analyze_results.py` 的消融统计链路。27 张 50 件以内真实板材中，完整 `process_aware_beam` 相比 `single_edges_only` 的平均 `machining_cost` 降低约 26.13%，逐板胜率约 96.30%；相比 `topology_no_beam` 的平均 `machining_cost` 降低约 6.29%，稳定性同为 0；相比 `no_stability_guidance` 和 `path_distance_baseline`，完整方法牺牲部分空移代价，但分别减少约 4.11 和 2.93 的平均稳定性惩罚。
- 已修正加工代价进入搜索的位置：完整路线比较使用 `machining_cost`，但前缀 beam 剪枝不使用累计 `cutting_length`，避免同一深度下“先切短边”被错误奖励。

下一步：

- 扩大 `max-members-per-archive` 与 `boards-per-member`；
- 按 `rectangle_count` 分层统计；
- 增加最终论文图表的误差棒策略、显著性检验和参数敏感性分析；
- 扩大 `support_policy` 与 `min_support_ratio` 扫描规模，确定第三章主实验采用的稳定性约束强度；
- 扩大严格支撑模型下的真实数据批量实验，统计稳定性改善是否在不同排样算法和不同零件规模中保持。
- 继续扩大 `run_beam_sensitivity.py` 的真实数据样本，验证当前 `7 * beam_width` 默认层扩展和 20 件小规模阈值是否在不同排样算法、不同零件数量区间中保持稳定。
- 下一轮应聚焦两个方向：一是继续在真实 45 件以内板材扩大样本与显著性统计；二是针对 synthetic 50 件以上压力测试，设计更强的多不稳定零件完成链机制，而不是简单扩大 beam 宽度。
- 基于 `diagnose_beam_search.py` 的逐层结果，定位低支撑前缀集中出现的深度，并判断是否需要针对这些层动态放宽候选扩展或增强完成链优先级。

## 5. 反推第三章 SCI 创新点

等真实数据实验形成稳定结果后，再收敛文章创新点：

- 面向真实排样结果的异构切割图建模；
- 共边/近共边/同线边段驱动的切割单元重构；
- 释放状态与空移干涉耦合的动态代价评价；
- 支持安全抬刀、绕行和延迟释放的状态感知路径优化；
- 作为第四章排样-切割协同优化底层评价器的增量计算框架。
