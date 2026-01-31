# V8implementationTODO.md

> 目标：在 **不继续“堆 GNN 超参”** 的前提下，把“跨变量依赖”做成**可控、可解释、可退化为无害**的结构性增益。
>
> V7 结论（你已跑完）非常明确：  
> - **GAT / Token-Attn** 整体收益不稳定，几乎没有形成一致增益。  
> - **GCN（sym norm, L2）** 在 *weather / flotation* 上能拉开一点差距，但对 *grinding* 会带来明显退化。  
> 这说明：问题不是“GNN 不行”，而是 **图（A）不够可靠 + 传播（message passing）缺少“安全阀”**。  
>
> 因此 V8 主线：**稳定图 prior + 动态图 residual + 置信路由 +（可选）输出端图校正**  
> 让模型在“图可靠时用图、不可靠时退化为 baseline”，避免 grinding 这种被错边污染。

---

## 全局约定（所有 Step 都要做）

### 数据集与评估
- 数据集：`ETTm1 / weather / flotation / grinding`（与 V6/V7 保持一致）
- 指标：`MSE / MAE`
- 额外日志（必须保留，便于判断“图到底有没有学到”）：
  - `entropy_mean, topk_mass, topk_overlap, l1_adj_diff`
  - `gate_mean, alpha_mean`
  - **新增**：`dyn_vs_prior_overlap`（或 L1 距离）——衡量动态图与 prior 的一致性

### 训练与对照
- 每个 step 至少包含：
  - 1 个 **baseline 对照**（不改动其余模块）
  - 2 个以内的超参点（不要扩大到 sweep）
- 每个 step 的“通过/失败”判据（建议）：
  - weather / flotation 至少一个有提升（或两个都小幅提升）
  - grinding 不允许出现明显退化（经验阈值：MSE +0.02 以上就要警惕）

---

# Step1 — 引入「稳定图 prior」：让 base graph 变得有意义

## 核心想法
V7 里动态图很可能出现“近似均匀 + 随机 topk”的行为（或受 gate 抑制），导致结构学习不充分；  
先给一个 **可解释、可复现** 的 prior 图，让 GNN 有可靠骨架。

## 设计
新增 `A_prior`（静态、dataset-level），来源推荐两种之一：
1) **相关系数图**（首选，最省事）：对训练集每个变量做标准化后，计算 `|Pearson|` 或 `|Spearman|`  Wasserstein Distance 相关矩阵  
2) **低频/趋势相关图**（可选）：对每个变量先做移动平均或趋势抽取，再算相关（更贴近“稳定结构”）

`A_prior` 处理：
- 对每个节点保留 topK 邻居（含/不含 self-loop 由实现决定，但要一致）
- 归一化方式先用 **sym norm**：`D^{-1/2} A D^{-1/2}`（与 V7 GCN-best 方向一致）
- 保存为 `prior_graph.npy`（或放进 dataset cache）

## Implementation TODO
- [ ] 新增脚本：`tools/build_prior_graph.py`
  - 输入：dataset 名称、train split、corr_method、topk
  - 输出：`A_prior`（float32, shape [C,C]）+ 元信息（topk 列表）
- [ ] dataloader 侧：加载 `A_prior`（每个 dataset 一个）
- [ ] model 侧：支持 `base_graph_type = {identity, prior}`
  - 当 `prior`：`A_base = A_prior`（替代原先“无意义”的 base graph / identity）
- [ ] 日志：额外打印 `prior_sparsity`, `prior_entropy_mean`

## 需要跑的实验（少量超参点）
- **S1.1** `corr=pearson_abs`, `topk = min(8, C-1)`
- **S1.2** `corr=spearman_abs`, `topk = min(8, C-1)`
- （可选第三点，只有当 S1.1/S1.2 两个都不稳定时才做）  
  **S1.3** `corr=pearson_abs`, `topk = min(12, C-1)`

## 预期结果
- 图统计上：`A_prior` 不会出现“每段都随机”的现象；可解释性强
- 性能上：weather / flotation **有机会**小幅提升；同时 grinding 不应显著变差（因为先只替换 base，不强行放大传播）

---

# Step2 — 用「稳定的 GCN」承载跨变量传播（先做 seasonal 分支）

## 核心想法
V7 里 GCN（sym）在部分数据集有效，说明“传播算子”更关键；  
先把 **季节项 seasonal** 作为跨变量依赖的承载点（trend 分支在 V6 里更容易出问题，先别碰）。

## 设计
- 在 seasonal encoder 输出（或 graph-mixer 输入）处引入 **Residual-GCN**：
  - `X' = X + g * GCN(X, A_use)`  
  - `A_use` 在 Step2 里先用 `A_prior`（稳定）
- 关键：`g` 不要直接放开到很大，**保持“安全退化”**：
  - `g` 用 trainable scalar（或每变量），初始化为 0（或接近 0），训练自己决定是否需要图

## Implementation TODO
- [ ] 新增模块：`modules/gcn.py`
  - 支持 `norm = sym`（必须）与 `row`（可选对照）
  - 支持 `layers = 1/2`
- [ ] 将 GCN 插入 seasonal 路径：
  - 推荐位置：`season_latent` 形成后、进入最终预测 head 前
- [ ] 增加 `g_scale`（trainable），默认 init=0
- [ ] 日志：`g_scale_mean`（或分布），确保它不是训练后仍为 0

## 需要跑的实验（少量超参点）
- **S2.1** `gcn_layers=1`, `norm=sym`, `g_init=0`
- **S2.2** `gcn_layers=2`, `norm=sym`, `g_init=0`

## 预期结果
- 如果 weather / flotation 的提升来自“稳定传播算子”，此 step 应该能复现/接近 V7 的收益
- grinding 若仍退化，说明“图可靠性”仍不足，需要 Step3 的路由安全阀

---

# Step3 — 动态图 residual + 置信路由：图不可信时自动回退 prior

## 核心想法
不要强迫模型在所有样本/所有段都用动态图；  
**动态图只做 residual**，并由置信度决定“动态 vs prior”的占比。

## 设计
- 保留原动态图 `A_dyn(t)`（按 segment / 样本计算）
- 定义 `conf(t)`：衡量 `A_dyn(t)` 是否可靠  
  推荐用“与 prior 的一致性”而不是纯熵：
  - `conf = overlap(topk(A_dyn), topk(A_prior))`  或  `conf = 1 - L1(A_dyn - A_prior)`
- 路由系数（参考你现有设计文档的 routing 思路）：
  - **确定性**：`alpha = (1 - conf)^gamma`，`gamma=2`（0 参数，先跑）
  - **可学习仿射**：`alpha = sigmoid(w*(1-conf) + b)`（参数很少，但更灵活）
- 融合图：
  - `A_mix = (1 - alpha) * A_dyn + alpha * A_prior`
- 传播用 `A_mix`，并保留 Step2 的 `g_scale`（最终仍可退化到 0）

### 训练稳定性
- 继续使用 `warmup_epochs`：前 1 个 epoch 强制 `g_scale=0`（或 `alpha=1`），避免早期乱图污染

## Implementation TODO
- [ ] 新增 `conf_metric`：
  - `overlap_topk`（首选，简单好用）
  - `l1_distance`（备选）
- [ ] 新增 `routing_mode = {deterministic, affine_learned}`
- [ ] 实现 `A_mix` 并替换 Step2 中的 `A_use`
- [ ] 日志：`conf_mean, alpha_mean, dyn_vs_prior_overlap`

## 需要跑的实验（少量超参点）
- **S3.1** `routing=deterministic(gamma=2)`, `warmup=1`
- **S3.2** `routing=affine_learned`, `warmup=1`
- （可选，如果 deterministic 过于保守）  
  **S3.3** `routing=deterministic(gamma=1)`, `warmup=1`

## 预期结果
- grinding 的退化应显著缓解（因为低置信时会回退 prior）
- weather / flotation 仍能保留（或扩大）收益
- 图统计上：不同数据集会出现明显差异（例如 grinding 的 `alpha_mean` 更大、更依赖 prior）

---

# Step4（可选但推荐）— 输出端「Graph Residual Correction」：把跨变量依赖用在“修正项”上

> 如果 Step3 仍无法在四个数据集都稳住，这是最“安全”的最后一招：  
> **不要在表征里混图，而是在输出上做小幅校正**。

## 核心想法
- 主干先给出 `y_base`（强 baseline，稳定）
- 图模块只预测一个 `Δy`（校正项），并由一个很小的系数 `β` 控制：
  - `y = y_base + β * Δy`
- `β` 初始化为 0，保证一开始完全等价于 baseline

## 设计
- `Δy` 的输入特征建议用 seasonal 输出或最后一层 seasonal latent（更贴近可交换信息）
- `Δy` 的 GNN 使用 **1-layer GCN + A_mix（来自 Step3）**
- `β` 用 trainable scalar（或 per-var），init=0

## Implementation TODO
- [ ] 新增 `GraphCorrectionHead`
- [ ] 将其接在最终输出前（只修正 seasonal 更安全）
- [ ] 日志：`beta_mean`, `delta_y_norm`

## 需要跑的实验（少量超参点）
- **S4.1** `correct_on=season_only`, `beta_init=0`
- **S4.2** `correct_on=full(season+trend)`, `beta_init=0`（仅当 S4.1 有收益再做）

## 预期结果
- 风险最小：若图无效，`β` 会学到接近 0，模型退化为 baseline
- 若图有效（weather / flotation），会出现稳定的提升且不伤 grinding

---

## 交付物与最终决策方式

### 每个 step 的产出
- 实验记录：`exp_id, mse/mae, 图统计, 训练曲线`
- 关键对比表：baseline vs step best
- 失败时也要记录：失败原因通常能从 `conf/alpha/g/beta` 的学习状态看出来

### V8 最终选择规则（建议）
- 优先选择 **Step3 best**（它解决“可靠性”根因）
- 如果 Step3 仍不稳定，则选择 **Step4 best**（输出端校正，最安全）

