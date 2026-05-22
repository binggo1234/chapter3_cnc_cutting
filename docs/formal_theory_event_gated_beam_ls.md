# Event-gated Adaptive Beam+LS 理论表达稿

日期：2026-05-22

本文档用于把当前第三章主算法 `process_aware_beam_adaptive_polished` 转化为论文方法部分可直接使用的理论表述。核心定位是二维板材 CNC 切割的 CAM planning-level toolpath optimization，而不是机床动力学、切削力或五轴插补控制。因此理论贡献应围绕真实排样下的切割单元覆盖、动态工艺状态、工艺可行路径排序和事件门控选择展开。

## 1. 对 Gemini 建议的取舍

Gemini 的总体判断是正确的：不能把本文方法写成普通集合覆盖或几何 TSP，而应写成带状态约束的序列决策问题。其关于形式化定义、词典序目标、命题和复杂度的四段式建议可以吸收。

需要收敛的地方有两点：

- 不宜宣称本文解决五轴奇异点、颤振、切削力或表面粗糙度问题。这些不属于当前代码与实验边界。
- 不宜把绕行加速写成严格的复杂度阶跃，例如从 `O(N_poly^2)` 降到 `O(N_bbox log N_bbox)`。当前实现确实使用释放件包围盒、行列区间筛选、整数邻接表、A* 和缓存来降低实际评估开销，但严格上界应按候选网格节点和可见边数量表达。

建议论文中的一句定位：

> We study CAM planning-level toolpath optimization for two-dimensional nested panel cutting, where process feasibility is enforced at the routing-planning stage through dynamic release, support, and safe travel-state evaluation.

## 2. 状态约束序列决策问题

给定单张板材排样：

`L = (P, R, T)`

其中 `P` 是板材可加工区域，`R = {1, ..., n}` 是矩形件集合，`T` 是刀具与工艺参数。每个矩形件 `i` 具有原始边段集合 `S_i`，所有待加工边段为：

`S = union_i S_i`

候选切割单元集合为：

`U = U_single union U_shared union U_near union U_collinear`

其中 `single` 表示单边切割，`shared` 表示真实共边切割，`near` 表示近共边通道切割，`collinear` 表示同线连续链切割。每个切割单元 `u in U` 覆盖一个原始边段子集：

`C(u) subseteq S`

若单元可逆，则构造两个有向候选 `d = (u, +)` 与 `d = (u, -)`。切割路线不是静态集合选择，而是有序动作序列：

`A = (d_1, d_2, ..., d_m)`

在第 `k` 步前，工艺状态定义为：

`s_k = (z_k, dir_k, P_k, R_k^rel, R_k^unstable, B_k, M_k)`

其中：

- `z_k` 是当前刀具位置；
- `dir_k` 是当前刀具方向；
- `P_k subseteq S` 是已加工边段集合；
- `R_k^rel` 是已完全释放的零件集合；
- `R_k^unstable` 是低支撑但尚未完成释放的零件集合；
- `B_k` 记录各零件剩余支撑强度；
- `M_k` 是累计路径评价向量。

执行有向切割单元 `d_k` 后，状态通过动态工艺模型更新：

`s_{k+1} = Update(s_k, d_k; L, T)`

该更新同时处理：

- 从 `z_k` 到 `entry(d_k)` 的空移模式选择；
- 已释放零件的低位空移干涉；
- 低位绕行或安全抬刀回退；
- 当前切割覆盖的原始边段；
- 零件是否释放；
- 低支撑零件是否被连续完成；
- 入刀、抬刀、安全抬刀、方向变化和连续切割奖励。

因此，本文问题不是普通 TSP，而是状态约束动态路由问题：

`min_A F(A), subject to Coverage(A) = S and H(A) = 0`

其中 `F(A)` 由动态状态转移累积得到。

## 3. 覆盖约束与异构切割单元

最终路线必须对每条原始边段完成一次覆盖：

`sum_{u in U: s in C(u)} x_u = 1, for all s in S`

同一物理切割单元至多选择一个方向：

`sum_{d in D(u)} v_d <= 1, for all u in U`

其中 `D(u)` 为单元 `u` 的有向候选集合。

这一定义体现本文第一个贡献：算法不是在原始边段上排序，而是先把重复切割可被合并的边界关系转化为异构候选切割单元。共边、近共边和同线链单元降低的是真实加工长度和重复加工动作，而不只是几何空移距离。

## 4. 工艺可行性与动态稳定性

硬约束惩罚定义为：

`H(A) = H_boundary(A) + H_collision(A)`

其中 `H_boundary` 表示刀具中心轨迹越出考虑修边量和刀具半径后的可加工区域，`H_collision` 表示低位空移穿越已释放零件。

对相邻切割单元 `d_k` 和 `d_{k+1}`，空移并非固定欧氏线段，而是在当前释放状态下选择：

`m_k in {low_clearance, low_clearance_detour, safe_lift}`

对应代价为：

`c_travel(s_k, d_{k+1}) = min_{m in M(s_k)} c_m(z_k, entry(d_{k+1}))`

若低位直线空移不穿越释放件，则使用 `low_clearance`；若低位直线不可行，则尝试 `low_clearance_detour`；若绕行不可行或成本不优，则回退到 `safe_lift`。因此，空移代价是状态相关的，不是静态点间距离。

动态稳定性由剩余支撑与邻接支撑共同决定。设零件 `i` 的支撑边集合为 `B_i`，第 `k` 步前未切支撑长度为：

`support_i(k) = sum_{s in B_i, s notin P_k} length(s)`

邻接零件提供的等效支撑为：

`adj_i(k) = alpha * sum_{j in N_i(k)} adj_{ij}`

综合支撑强度为：

`b_i(k) = support_i(k) + adj_i(k)`

若 `b_i(k)` 低于阈值，零件进入低支撑集合 `R_k^unstable`。当前代码采用低支撑连续完成链逻辑：零件短暂进入低支撑状态并不立即判为失败；只有当算法离开该零件、或最终仍未完成释放时，才计入稳定性惩罚：

`Phi_stab(A) = sum_k abandon_unstable(k) + final_unfinished_unstable`

这比“任意时刻低支撑即惩罚”的规则更适合连续轮廓释放过程。

## 5. 词典序工艺目标

当前主算法最终比较使用 `process_metric_key()`，其目标向量为：

`F_proc(A) = (H(A), Phi_stab(A), C_mach(A), N_tool(A), C_travel(A), D_air(A), Theta(A), -R_cont(A))`

其中：

- `C_mach(A) = C_cut(A) + C_travel(A)` 是切割长度与模式加权空移代价之和；
- `N_tool(A)` 是入刀、抬刀和安全抬刀事件数；
- `C_travel(A)` 是低位空移、低位绕行和安全抬刀的模式加权通行代价；
- `D_air(A)` 是原始空移距离；
- `Theta(A)` 是方向变化惩罚；
- `R_cont(A)` 是连续切割奖励。

路径比较采用词典序：

`A' better than A iff F_proc(A') <_lex F_proc(A)`

等价地，可写作满足层级权重的加权目标：

`J(A) = M_1 H(A) + M_2 Phi_stab(A) + M_3 C_mach(A) + M_4 N_tool(A) + C_travel(A) + w_a D_air(A) + w_theta Theta(A) - w_r R_cont(A)`

其中：

`M_1 >> M_2 >> M_3 >> M_4 >> 1`

这一定义要在论文中明确解释：在 CAM planning-level 场景中，边界可达、释放件干涉和零件稳定性是工艺硬边界；加工效率是硬边界满足后的软优化。因此 Path-LS 可能获得更短路径，但若产生稳定性惩罚，则不能作为可加工路线。

## 6. Event-gated protected selection

Event-gated Adaptive Beam+LS 的最终候选集包括：

- `topology_process_aware`：廉价工艺拓扑参考；
- `process_aware_beam`：受保护 beam 候选；
- `beam + process LS`：以 beam 为初解的工艺局部精修；
- `wide beam + LS fallback`：仅在风险触发时运行的宽 beam 兜底。

其中 `process_aware_beam` 是 protected plan。任何非保护候选如果相对其他候选引入额外刀具事件，必须满足收益证明：

`Delta N_tool > 0 => Delta C_mach > epsilon and Delta C_travel >= Delta N_tool * max(lambda_abs, lambda_rel * C_travel^ref)`

当前默认参数为：

- `lambda_abs = 100.0`
- `lambda_rel = 0.02`
- `epsilon = 1e-9`

换言之，候选不能仅凭微小路径收益替代普通 beam；额外刀具事件必须被足够的通行代价收益和加工总代价收益共同证明。最终只在通过门控的候选与 protected beam 中按 `F_proc` 选择。

这也是本文相对普通 adaptive portfolio 更强的地方：不是简单“谁指标更好选谁”，而是把工艺保守性作为候选准入条件。

## 7. 算法伪代码

### Algorithm 1: Heterogeneous cutting-unit construction

输入：排样 `L`、刀具参数 `T`

输出：候选切割单元集合 `U`

1. 从每个矩形件提取四条原始边段，得到 `S`。
2. 构造所有 `single_edge` 单元，保证每条原始边段至少有一个候选覆盖。
3. 检测真实共边关系，构造 `shared_edge` 单元。
4. 检测满足刀具通道宽度的近共边关系，构造 `near_shared_channel` 单元。
5. 检测同线连续关系，构造 `collinear_chain` 单元。
6. 返回 `U_single union U_shared union U_near union U_collinear`。

### Algorithm 2: Process-aware prefix beam search

输入：候选单元 `U`、板材 `P`、刀具参数 `T`、工艺模型 `G_proc`

输出：beam 路线 `A_beam`

1. 初始化 beam 为一个空前缀状态 `s_0`。
2. 对深度 `k = 0, ..., |U|-1`：
3. 对 beam 中每个前缀，按工艺状态和转移代价预排序剩余候选。
4. 若存在低支撑未完成零件，优先扩展能继续完成该零件的候选。
5. 对有限候选执行 `Update(s_k, d; L, T)`，得到新状态和增量代价。
6. 对新前缀去重，并按 `process_state_metric_key()` 排序。
7. 使用多样性桶与低支撑父节点保底扩展，保留下一层 beam。
8. 返回最终层中 `F_proc` 最优的完整路线。

### Algorithm 3: Event-gated Adaptive Beam+LS

输入：候选单元 `U`、工艺模型、门控参数

输出：最终路线 `A*`

1. 计算 `A_topology`。
2. 计算受保护候选 `A_beam`。
3. 以 `A_beam` 为初解执行工艺局部搜索，得到 `A_polish`。
4. 构造候选集 `C = {A_topology, A_beam, A_polish}`。
5. 若当前最优候选相对拓扑参考未形成足够收益，运行宽 beam 并局部精修，加入 `C`。
6. 对 `C` 中所有非保护候选执行 event gate。
7. 在通过门控的候选与 `A_beam` 中，按 `F_proc` 返回最优路线。

## 8. 可用于论文的命题

### Proposition 1: Coverage feasibility of heterogeneous units

若 `U` 包含所有原始边段对应的 `single_edge` 单元，且最终选择满足覆盖约束 `sum_{u: s in C(u)} x_u = 1`，则输出路线完整覆盖所有原始轮廓边段，不会产生漏切。共边、近共边和同线链单元只作为覆盖替代项进入选择，不破坏原始轮廓覆盖完备性。

证明思路：对任意 `s in S`，`single_edge(s)` 保证候选存在性；覆盖等式保证被选覆盖次数为 1；方向变量只改变执行方向，不改变覆盖集合 `C(u)`。

### Proposition 2: Monotonicity of process-aware local refinement

工艺局部搜索以 `F_proc` 为接受准则，只有当邻域解 `A'` 满足 `F_proc(A') <_lex F_proc(A)` 时才替换当前解。因此，搜索过程中的目标向量严格单调下降，并在有限邻域集合上终止于局部工艺帕累托最优解。

证明思路：swap、relocate 和 2-opt 产生的候选数量有限；每次接受都使词典序目标严格变小；有限排列空间中不存在无限严格下降链。

### Proposition 3: Protected event-gated dominance bound

设 `A_b` 为 protected beam 候选，`A_c` 为任意非保护候选。若 `A_c` 相对某候选 `A_r` 增加 `Delta N_tool > 0` 个刀具事件，且不满足：

`Delta C_mach > epsilon and Delta C_travel >= Delta N_tool * max(lambda_abs, lambda_rel * C_travel(A_r))`

则 `A_c` 不能进入最终选择集合并替代 protected beam。

该性质保证算法不会为了微小通行代价下降而系统性增加抬刀、入刀或安全抬刀事件。实验中当前门控得到相对普通 beam 的刀具事件减少、持平、增加分布为 `11/60/6`，说明该规则保留了少量收益充分的例外，同时抑制了无约束 adaptive portfolio 中的事件膨胀。

## 9. 复杂度表达

设：

- `N = |U|` 为候选切割单元数；
- `B` 为 beam width；
- `K` 为每个前缀扩展的候选数上限；
- `L` 为路线深度，通常 `L <= N`；
- `Q` 为局部搜索每轮检查的邻域数量；
- `I` 为局部搜索迭代次数；
- `V_d, E_d` 为一次低位绕行构造出的正交绕行图节点数和边数。

候选切割单元构造主要由矩形边段关系检测决定。由于每个矩形件提供 4 条边，原始边段数为 `O(n)`；若采用直接关系检测，最坏情况可写为 `O(n^2)`，但实际工业排样中通过方向、坐标和重叠条件筛选会显著减少有效比较。

Process-aware beam search 的主要复杂度可写为：

`O(L * B * K * C_eval)`

其中 `C_eval` 是一次前缀状态扩展的增量评价成本，包括空移模式选择、释放件更新、支撑状态更新和必要时的绕行求解。

对低位绕行，当前实现从释放件包围盒生成正交候选网格，并在整数邻接表上运行带曼哈顿启发的 A*。单次绕行可表达为：

`O(V_d log V_d + E_d)`

另外，代码使用：

- 释放件包围盒而非完整多边形反复求交；
- 按行/列生成阻塞区间；
- 反向路径缓存；
- 障碍物集合规范化缓存；
- 最大障碍物数与最大网格节点数保护。

因此论文中更稳妥的说法是：这些结构降低了重复几何检测和高密度案例中的实际扩展成本，而不是宣称一个脱离实现前提的绝对复杂度阶跃。

局部搜索精修复杂度为：

`O(I * Q * C_delta)`

其中 `C_delta` 是受影响片段的增量重评估成本。代码中会尝试复用前缀状态并在后缀状态可重接时复用后缀代价，以避免每个邻域都从头评价完整路线。

Event-gated adaptive portfolio 至多执行常规 beam、一次 beam 后局部精修和一次宽 beam 后局部精修，因此总体复杂度仍是少数 beam/LS 运行的常数倍；其工程意义在于只在风险样本上触发宽 beam，而非对所有样本使用最大搜索预算。

## 10. SCI 写作边界

可以强调：

- 真实工业排样数据上的 paired board-wise validation；
- 0 硬约束惩罚与 0 稳定性惩罚；
- 相比 topology、multi-start LS 和 ordinary beam 的统计显著收益；
- 事件门控保证 adaptive 候选不会无约束增加刀具事件；
- 代码、复现实验脚本、统计表和代表性路线图。

不应声称：

- 已经优于商业 CAM；
- 已经经过真实机床切削验证；
- 改善了表面粗糙度、切削力、颤振或轮廓误差；
- 可直接推广到五轴曲面加工或实时 CNC 插补。

如果投 RCIM、CAD、JMS、EAAI 或 CIE，建议表述为 industrial planning-level validation。若转投机床物理或切削力方向期刊，则必须补充真实加工实验、测量仪器和物理质量指标。
