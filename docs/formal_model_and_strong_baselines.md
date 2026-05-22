# 第三章正式数学模型与强对比实验设计

## 1. 问题定位

第三章不把 CNC 矩形件切割简化为“零件中心点 TSP”，而是将给定排样结果转化为带动态工艺状态的异构切割单元排序问题。输入为第二章得到的单张板材排样，输出为一条可执行的切割动作序列，目标是在保证轮廓完整、刀具可达、释放件无干涉和零件稳定性的条件下，降低空移、抬刀、安全抬刀、绕行和方向变化等综合加工代价。

该定义对应代码中的主线：

- `cutting_units.py`：从矩形边段关系构造候选切割单元；
- `metrics.py`：全局评价与增量评价；
- `process_model.py`：释放件、支撑边、邻接支撑和稳定性状态；
- `travel.py`：低位空移、低位绕行和安全抬刀；
- `local_search.py`：路径局部搜索、强 baseline 和工艺感知前缀束搜索；
- `optimizer.py`：统一规划入口。

## 2. 集合与参数

给定单张板材排样：

`L = (P, R, T)`

其中：

- `P = [0, W] x [0, H]` 为板材区域；
- `R = {1, 2, ..., n}` 为矩形件集合；
- `Q_i` 为第 `i` 个矩形件的多边形区域；
- `S_i` 为第 `i` 个矩形件的原始边段集合；
- `S = union_i S_i` 为所有待切割原始边段；
- `T` 为刀具与机床参数，包括修边量 `m`、刀径 `d_t`、刀具半径 `r_t = d_t / 2`、安全间距、低位空移权重、绕行权重和安全抬刀固定成本。

根据边界关系识别结果，构造候选切割单元集合：

`U = U_single union U_shared union U_near union U_collinear`

其中：

- `U_single`：单边切割单元；
- `U_shared`：真实共边切割单元；
- `U_near`：近共边刀具通道切割单元；
- `U_collinear`：同线连续切割链单元。

每个切割单元 `u in U` 覆盖一组原始边段：

`C(u) subseteq S`

若切割单元可反向执行，则生成两个有向候选：

`D(u) = {u^+, u^-}`

全部有向候选记为：

`D = union_{u in U} D(u)`

对任意有向候选 `d in D`，定义：

- `u(d)`：对应切割单元；
- `e(d)`：入刀点；
- `o(d)`：出刀点；
- `vec(d)`：切割方向；
- `C(d) = C(u(d))`：覆盖的原始边段集合；
- `l_cut(d)`：切割长度。

## 3. 决策变量

设最多执行 `K = |S|` 个切割单元位置。由于共边、近共边和同线链可以一次覆盖多个原始边段，实际使用位置数可小于 `K`。

二元变量：

`x_{k,d} = 1`

表示第 `k` 个切割位置选择有向候选 `d`。

`v_d = 1`

表示有向候选 `d` 被选入最终路径。

`a_k = 1`

表示第 `k` 个切割位置处于启用状态。

`y_{k,d,e} = 1`

表示第 `k` 个位置执行候选 `d`，第 `k+1` 个位置执行候选 `e`，即存在从 `o(d)` 到 `e(e)` 的空移转移。

`q_{k,d,e,m} = 1`

表示转移 `(d,e)` 采用空移模式 `m`，其中：

`m in M = {low, detour, safe_lift}`

状态变量：

- `p_{k,s}`：进入第 `k` 个位置前，原始边段 `s` 是否已经完成；
- `r_{k,i}`：进入第 `k` 个位置前，矩形件 `i` 是否已经完全释放；
- `b_{k,i}`：进入第 `k` 个位置前，矩形件 `i` 的剩余支撑强度；
- `h_{k,i}`：矩形件 `i` 是否处于低支撑未完成状态。

## 4. 硬约束

### 4.1 原始轮廓完整覆盖

每条原始矩形边段必须被覆盖且只被覆盖一次：

`sum_{d in D: s in C(d)} v_d = 1, for all s in S`

这使真实共边、近共边和同线链可以替代多个单边动作，但不能造成漏切或重复切割。

同一物理切割单元最多选择一个方向：

`sum_{d in D(u)} v_d <= 1, for all u in U`

有向候选选择和位置变量一致：

`sum_{k=1}^{K} x_{k,d} = v_d, for all d in D`

每个启用位置只能选择一个有向切割候选：

`sum_{d in D} x_{k,d} = a_k, for k = 1, ..., K`

启用位置必须连续：

`a_{k+1} <= a_k, for k = 1, ..., K-1`

### 4.2 转移一致性

相邻位置的转移变量满足：

`y_{k,d,e} <= x_{k,d}`

`y_{k,d,e} <= x_{k+1,e}`

`y_{k,d,e} >= x_{k,d} + x_{k+1,e} - 1`

转移必须选择唯一空移模式：

`sum_{m in M} q_{k,d,e,m} = y_{k,d,e}`

### 4.3 加工边界可达

刀具中心线必须位于考虑修边量和刀具半径后的有效区域：

`gamma(a) subseteq Pominus(m + r_t)`

其中 `gamma(a)` 表示切割或空移动作的刀具中心轨迹，`Pominus(.)` 表示板材区域的内缩可达域。

若 `gamma(a)` 超出该区域，则产生硬惩罚：

`H_boundary(A) > 0`

正式实验中，所有可行方法应满足：

`H_boundary(A) = 0`

### 4.4 释放件干涉

进入第 `k` 个位置前，已经完全切断的零件集合为：

`R_k^rel = {i | r_{k,i} = 1}`

低位空移不能穿越已释放零件：

`gamma_{k,d,e,low} cap (union_{i in R_k^rel} Q_i) = empty`

若低位直线空移不可行，可以选择低位绕行 `detour`；若绕行仍不可行，则选择安全抬刀 `safe_lift`。因此空移可行性不是简单的欧氏距离，而是状态相关的模式选择：

`c_travel(k,d,e) = min_{m in M_k(d,e)} c_m(k,d,e)`

其中 `M_k(d,e)` 为在当前释放状态下可用的空移模式集合。

### 4.5 动态释放与稳定性

原始边段完成状态：

`p_{k,s} = sum_{h < k} sum_{d in D: s in C(d)} x_{h,d}`

零件释放状态：

`r_{k,i} = 1 iff p_{k,s} = 1, for all s in S_i`

设 `B_i subseteq S_i` 为第 `i` 个矩形件的支撑边集合，`l_s` 为边段长度，`A_i` 为第 `i` 件面积，`N_i(k)` 为第 `k` 步仍未释放且与 `i` 邻接的零件集合。支撑强度定义为：

`b_{k,i} = sum_{s in B_i} l_s (1 - p_{k,s}) + alpha sum_{j in N_i(k)} adj_{i,j}`

其中 `alpha` 为邻接支撑权重，`adj_{i,j}` 为相邻零件对 `i` 提供的等效支撑长度。

稳定性阈值可写为：

`b_{k,i} >= tau_i`

`tau_i = rho sum_{s in B_i} l_s`

或者面积归一化形式：

`b_{k,i} / sqrt(A_i) >= eta`

当前代码采用“低支撑完成链”逻辑：若某零件进入低支撑状态，但后续动作连续完成该零件直到释放，则不立即惩罚；若算法离开该低支撑零件或最终仍未释放，则计入稳定性惩罚：

`Phi_stab(A) = sum_k sum_i abandon_{k,i} + final_unfinished_unstable_i`

这比简单的“任意时刻低支撑即不可行”更贴近连续切割工艺，因为它允许短时间的连续释放过程。

## 5. 目标函数

本文不采用单纯欧氏空移最短作为目标，而采用分层工艺目标。路径动作序列记为：

`A = (a_1, a_2, ..., a_m)`

总代价向量为：

`F_proc(A) = (H(A), Phi_stab(A), C_mach(A), N_tool(A), C_travel(A), D_air(A), Theta(A), -R_cont(A))`

其中：

- `H(A) = H_boundary(A) + H_collision(A)`：硬约束惩罚；
- `Phi_stab(A)`：动态稳定性惩罚；
- `C_mach(A) = C_cut(A) + C_travel(A)`：切割长度与模式加权空移代价之和；
- `N_tool(A)`：入刀、抬刀、安全抬刀等刀具事件数；
- `C_travel(A)`：模式加权空移代价；
- `D_air(A)`：原始空移距离；
- `Theta(A)`：方向变化惩罚；
- `R_cont(A)`：连续切割奖励。

算法比较两个路径 `A` 与 `A'` 时采用字典序：

`A' better than A iff F_proc(A') <_lex F_proc(A)`

也可写成大权重等价形式：

`min M_1 H(A) + M_2 Phi_stab(A) + M_3 C_mach(A) + M_4 N_tool(A) + C_travel(A) + w_a D_air(A) + w_theta Theta(A) - w_r R_cont(A)`

其中：

`M_1 >> M_2 >> M_3 >> M_4 >> 1`

主算法最终选择口径对应 `process_metric_key()`；beam 前缀筛选对应 `process_state_metric_key()`。刀具事件不是简单越少越好，而是在 `process_metric_key()` 中位于 `machining_cost` 之后，并由 event gate 额外限制：非保护候选若增加刀具事件，必须同时给出足够的通行代价收益和加工总代价收益，才能进入最终比较。最终主方法的完整理论表述见 `docs/formal_theory_event_gated_beam_ls.md`。

## 6. 状态图求解形式

可将问题写为动态状态图最短路。

状态节点：

`X_k = (D_k, z_k, dir_k, P_k, R_k^rel, R_k^unstable, F_k)`

其中：

- `D_k`：已经选择的切割候选集合；
- `z_k`：当前刀具位置；
- `dir_k`：当前刀具方向；
- `P_k`：已加工边段集合；
- `R_k^rel`：已释放零件集合；
- `R_k^unstable`：低支撑未完成零件集合；
- `F_k`：累计代价向量。

从 `X_k` 扩展一个候选切割单元 `d` 时，增量更新：

`X_{k+1} = Update(X_k, d, L, T)`

该更新同时计算：

- 转移空移模式；
- 释放件干涉；
- 支撑强度变化；
- 低支撑完成链；
- 入刀/抬刀/安全抬刀；
- 方向变化与连续性。

精确求解该状态图会产生组合爆炸。本文主算法 `process_aware_beam` 使用前缀束搜索近似求解：每层只保留若干个代价向量优秀且状态多样的前缀，并对低支撑前缀设置保底扩展，避免连续完成链被过早剪枝。

## 7. 强对比实验矩阵

为了避免只和弱 baseline 对比，第三章实验应采用以下层次。

### B0 Greedy

最近邻贪心路径。作用是给出最基础的路径构造下界参照，不能作为主要说服力来源。

代码入口：

`plan_greedy_route`

### B1 Path-LS

路径距离局部搜索。以最近邻为初解，使用 swap、relocate 和方向受限 2-opt，优化硬约束、空移和路径代价，但不显式利用动态稳定性作为搜索引导。

作用：证明“几何路径短”不等于“CNC 加工过程稳定”。

代码入口：

`plan_path_distance_local_search_route`

### B2 Process-aware topology

工艺感知拓扑贪心。每一步基于动态释放状态和拓扑关系选择当前最优切割单元，但不保留多个候选前缀。

作用：模拟文献中常见的规则驱动/图启发式方法，检验 beam 是否真正改善了“稳定性满足后路径仍过长”的问题。

代码入口：

`plan_topology_route(..., process_aware=True)`

### B3 Multi-start process local search

本次新增强 baseline。它从多种确定性初解出发：

- 最近邻初解；
- 拓扑关系初解；
- 工艺感知拓扑初解；
- 横向 sweep 初解；
- 纵向 sweep 初解；
- 反向横向/纵向 sweep 初解。

每个初解都使用同一套工艺目标局部搜索，最后取字典序代价最优结果。

作用：它比单一局部搜索更接近文献中 VNS/ILS 类强启发式对照。如果 `process_aware_beam` 相对它仍能保持稳定性并降低通行代价，论文证据会明显增强。

代码入口：

`plan_process_local_search_multistart_route`

实验方法名：

`process_local_search_multistart`

### B4 Ablation

消融实验用于拆分本文贡献：

- `single_edges_only`：去掉异构切割单元，只保留单边切割；
- `no_stability_guidance`：去掉稳定性搜索引导；
- `no_adjacency_support_guidance`：去掉邻接支撑；
- `topology_no_beam`：去掉前缀束搜索；
- `no_detour_operator`：去掉低位绕行；
- `no_safe_travel_modes`：去掉绕行与安全抬刀模式。

当前 77 张 20-50 件真实板材消融结果可支持以下写作结论：

- `single_edges_only` 的平均通行代价较低，但总加工代价比完整方法高约 27.69%，说明异构切割单元的核心价值是减少重复切割和工艺动作，而不是单纯缩短空移；
- `no_stability_guidance` 和 `path_distance_baseline` 会明显降低路径距离，但分别带来 6.71 和 5.09 的平均稳定性惩罚，完整方法在工艺字典序目标下配对胜率均为 100%；
- `no_adjacency_support_guidance` 的平均稳定性惩罚为 1.09，且完整方法平均通行代价降低约 23.94%，说明邻接支撑不仅提高稳定性，也帮助 beam 避免低质量前缀；
- `topology_no_beam` 和 `process_local_search_multistart` 均保持稳定性为 0，但完整方法平均通行代价分别降低约 7.73% 和 7.59%，配对胜率均为 77.92%；
- `no_detour_operator` 仍可行但安全抬刀更多，完整方法平均通行代价降低约 3.87%；`no_safe_travel_modes` 平均硬约束惩罚升至 1.71，说明安全空移模式是可行性组件，而不是单纯的性能调参。

### B5 Small-scale exact DP

小规模 exact DP 不作为大规模基线，而作为最优性锚点。它在给定异构切割单元集合后，对切割单元顺序和可逆方向进行动态规划枚举，并保留动态释放、支撑稳定、空移模式和增量代价状态。

使用限制：

- 默认最多 `12` 个被选切割单元；
- 超过上限直接拒绝运行；
- 只用于报告小规模 optimality gap，不用于 20 件以上真实主实验。

代码入口：

`exact_process_dp_order`

实验脚本：

`experiments/run_exact_gap.py`

### B6 Beam + process local search polishing

该方法不是替代 `process_aware_beam` 的新贡献，而是一个后处理增强版本。它先运行工艺感知前缀束搜索，再以 beam 结果为初解执行工艺目标局部搜索。局部搜索的比较口径已统一为 `process_metric_key()`，避免为了减少少量刀具事件而接受显著更长的加工路径。

用途：

- 检查 beam 的剩余误差是否可以由局部邻域修复；
- 在主算法和强局部搜索 baseline 之间提供一个中间版本；
- 作为后续论文中的可选增强或消融项，而不是替代原始 beam。

实验方法名：

`process_aware_beam_polished`

当前 77 张 20-50 件真实板材配对结果显示：该方法相对工艺感知拓扑的通行代价平均降低约 8.46%，相对多启动工艺局部搜索平均降低约 8.32%，配对胜率均为 77.92%；相对原 `process_aware_beam` 的平均通行代价降低约 0.72%，但 84.42% 样本与 beam 工艺目标持平。因此它更适合作为“后处理增强/稳健性检查”，不宜写成独立主贡献。

## 8. 当前已落地的代码变化

新增强 baseline：

- `src/cnc_cutting/local_search.py`
  - `sweep_unit_order`
  - `multistart_process_initial_orders`
  - `process_local_search_multistart_order`
- `src/cnc_cutting/optimizer.py`
  - `plan_process_local_search_multistart_route`

新增小规模最优性对照：

- `src/cnc_cutting/exact_dp.py`
  - `ExactDPConfig`
  - `exact_process_dp_order`
- `src/cnc_cutting/optimizer.py`
  - `plan_exact_process_dp_route`
- `experiments/run_exact_gap.py`
  - 输出 exact gap、exact 扩展节点数、保留状态数和相对 exact 的通行代价差距。

新增 beam 后处理增强：

- `process_aware_beam_polished_search_order`
- `plan_process_aware_beam_polished_route`
- 已注册到批量实验、消融、规模实验、路线可视化和 exact-gap 脚本。

实验脚本已注册：

- `experiments/run_chapter2_batch.py`
- `experiments/run_ablation.py`
- `experiments/run_scalability.py`
- `experiments/visualize_route_comparison.py`
- `experiments/analyze_results.py`
- `experiments/generate_chapter2_batch_figures.py`
- `experiments/run_support_sensitivity.py`

测试已覆盖：

- 多启动强 baseline 能返回完整路径；
- sweep 初解按指定主轴排序；
- 多启动在不进行迭代时能从所有初解中选出最优工艺代价；
- 统一 optimizer 入口能返回正确增量评价指标。

## 9. 下一步实验建议

下一轮主实验不要只报告平均运行时间。建议优先报告：

1. `process_aware_beam` vs `process_local_search_multistart` 的配对差异；
2. `process_aware_beam_polished` 是否能稳定降低 beam 的 residual gap；
3. 稳定性惩罚是否保持为 0；
4. 在稳定性均为 0 的样本上，通行代价和安全抬刀/绕行次数是否下降；
5. 代表性样本路径图，说明 beam 的优势来自全局前缀选择，而不是只来自共边单元数量；
6. 小规模样本使用 `run_exact_gap.py` 报告 exact DP gap。若 beam 在极小规模上不如 exact 或多启动局部搜索，应如实写成规模/搜索宽度限制，而不是过度包装成所有场景最优。
