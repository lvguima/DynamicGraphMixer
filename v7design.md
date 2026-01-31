# v7design.md

## 目标
在 v6 的结论基础上，我们不再纠结于 `graph_rank/graph_scale/topk/base_alpha/gate_init` 这类“微调型”参数，而是把注意力放到 **跨变量交互算子（message passing / interaction operator）本身的结构升级**：

- **方案 A：GAT 化（稀疏注意力图）**
- **方案 B：显式注意力（token-wise cross-var MHA）**
- **方案 C：GCN 化（归一化图卷积）**

这三个方案都保持 B5 的“好骨架”（TCN + EMA 双流 + season 走跨变量、trend 走线性 head），只替换/增强 **GraphMixer** 这一段。

---

## 1. v6 结果复盘：为什么要从结构下手

### 1.1 baseline 行为与信号
v6 Step0 的 BASE 里，`gate_mean≈0.0025`（接近 `sigmoid(-6)`），意味着跨变量传播长期处于“几乎关闭”的状态；同时 `alpha_mean≈0.000335`（接近 `sigmoid(-8)`），base graph mixing 也几乎不参与。换句话说：
- **我们名义上有图传播，但实际有效跨变量注入很弱**（至少从日志统计上看）。

（例：ETTm1 BASE 的 `gate_mean=0.002445, alpha_mean=0.000335`；weather BASE 的 `gate_mean=0.002500, alpha_mean=0.000336`；grinding BASE 的 `gate_mean=0.002481, alpha_mean=0.000335`。）

### 1.2 “调参”带来的变化非常有限/不稳定
v6 Step1/2/3/4 的结论（仅列关键现象）：

- **gate_init 从 -6 提到 -2** 会把 `gate_mean` 拉到 ~0.11 左右，但收益不稳定：
  - flotation 有明显提升（MSE 从 ~0.786 降到 ~0.777）
  - grinding 也有改善（MSE 从 ~4.249 降到 ~4.216）
  - 但 ETTm1 反而回退（0.3165 → 0.3175）
  - weather 的 MAE 还变差（0.2298 → 0.2309）

- **confidence gate / routing** 能把 `alpha_mean` 拉到 ~0.5（表示大量依赖 base 或进行路由），但整体收益仍小且不稳（例如 C3/C4 只有个别数据集略有改善）。

- **trend 分支上做图传播**（Step3）整体是负收益，尤其 grinding 明显变差（MAE 从 ~0.698 到 ~0.724）。

- **multi-scale graph**（Step4）对多数数据集没有稳定提升。

综上：继续围绕现有 GraphMixer 的参数（scale/topk/gate/routing）“打补丁”，大概率只能获得 1e-3 量级、甚至不稳定的收益。

### 1.3 更强证据：图相关组件对性能贡献很小
组件消融总结里，去掉 `SMGP / gate / base graph / sparsify` 在多数数据集上的变化都非常小；甚至 **graph-disabled** 的组合实验也能接近 BASE（ETTm1/weather/flotation 基本贴合）。这说明：
- 当前图传播链路的可用增益很有限
- 可能真正瓶颈不在“图怎么调”，而在 **传播算子太弱/太容易注入噪声**

---

## 2. v7 总体原则（所有方案共用）

### 2.1 保留的“主干正确性”
我们不推翻 B5 的整体结构：
- TemporalEncoder 仍使用 **TCN**（消融显示把 TCN 换成 linear 会明显掉点）
- Dual-stream 仍使用 **EMA decomposition**：`y_hat = ForecastGraph(season) + TrendHead(trend)`
- **跨变量交互只作用在 season 分支**（v6 的 trend graph 传播是负收益）

### 2.2 Map-Value Decoupling（保留 v3 的核心直觉）
继续遵循：
- 用 stationarized 表征估计关系（map / QK）
- 用原始表征做传播（value / V）

直觉：非平稳会污染“相似度/相关性”的估计，但预测又不能丢掉趋势/尺度信息。

### 2.3 最少但足够的超参
每个方案控制在 **2–3 个关键超参**，避免再掉进“扫参深坑”。

### 2.4 评价与对比方式
- 数据集与 v6 对齐：`ETTm1 / weather / flotation / grinding`
- 先用 **单 seed 快速筛选**，再对最优候选做 **2 seeds 复核**
- 主要指标：MSE/MAE；辅助观察：gate/entropy/topk_overlap/adj_diff

---

## 3. 方案 A：Segment-level Sparse GAT（S-GAT）

### 3.1 设计理念
当前 GraphMixer 是：
- 先用低秩 learner 得到 `A_t`
- 再做 `mixed = A_t @ X`（本质是“线性加权平均”）

它的问题是：
- adjacency 的表达能力有限（低秩 + row-softmax）
- message function 过于线性（相当于一层、无条件的 mixing）

**GAT 的核心**：边权不是直接由低秩参数化，而是由节点特征的注意力计算得到；多头注意力能表达多种关系子空间。

### 3.2 结构与公式（按 segment）
对每个 segment k：
- 从 map 分支得到节点表示 `z_map ∈ R^{B×C×D}`（例如 segment 内 token 平均）
- 从 val 分支取 `H_val_seg ∈ R^{B×C×S×D}`（S 为该 segment 的 token 数）

计算多头注意力得到 adjacency：
- `Q = z_map W_Q`, `K = z_map W_K`
- `score_{ij} = (Q_i · K_j) / sqrt(d_h)`
- (可选) 加入 base graph 作为 bias：`score += β * log(A_base + ε)`
- (可选) top-k sparsify：每行只保留 k 个最大 score
- `A_attn = softmax(score, dim=j)`

传播：
- 对 segment 内所有 token 共享 `A_attn`：
  - `M = A_attn @ H_val_seg`（在变量维做矩阵乘）
- 残差注入：`H_out_seg = H_val_seg + γ ⊙ Dropout(M)`

其中 `γ` 建议做成 **可学习 residual scale**（可 per-var / per-head），并配合 warmup（前若干 epoch 强制 γ=0）。

### 3.3 关键超参（建议范围）
- `gat_heads ∈ {4, 8}`
- `gat_topk ∈ {6, 12}`（小 C 数据集 6 就够；想更稳可 12）
- `residual_scale_init ∈ {0.0, 0.1}`（比 gate_init 更直观）

### 3.4 预期
- `A_entropy` 不再长期接近 uniform（更有结构差异）
- 在 grinding / flotation 这类需要跨变量信息的数据集上，应该更容易获得 >1e-3 级别的稳定增益

---

## 4. 方案 B：Token-wise Cross-Variable Attention（T-CVAttn）

### 4.1 设计理念
GAT 仍然是“先学 A，再用 A 混合 token”。这在非平稳场景可能仍会有“错边注入”的风险。

因此第二个方案更激进：
- **不显式构建 adjacency**
- 直接在每个 token 上做跨变量的 multi-head attention（把它视为“变种 iTransformer / cointegrated attention”但局部化到 segment/patch）

优势：
- attention 的输出就是融合后的表示，不需要显式 A 的低秩约束
- token-wise 能捕捉 **时间局部** 的跨变量耦合（比 segment 平均更细）

### 4.2 结构与公式
对每个 token n（或每个 segment 内 token）：
- `H_map[:, :, n, :]` 作为 Q/K
- `H_val[:, :, n, :]` 作为 V

做跨变量注意力（变量维长度为 C）：
- `Y_n = MHA(Q=H_map_n, K=H_map_n, V=H_val_n)`

残差注入：
- `H_out_n = H_val_n + γ ⊙ Proj(Y_n)`

为了避免过大计算量：
- 默认只在 **season branch** 做
- segment 内 token 数较大时，可选 top-k attention（实现上相当于对 attention weights 做稀疏化）

### 4.3 关键超参（建议范围）
- `attn_heads ∈ {4, 8}`
- `attn_topk ∈ {None, 12}`（C≤21 时 dense 也能跑；C 更大再启用 topk）
- `residual_scale_init ∈ {0.0, 0.1}`

### 4.4 预期
- 更容易在 **ETTm1/weather** 这种小中通道数据上看到收益（因为 token-wise 更细，避免 segment pooling 损失）
- 但计算量比 S-GAT 更大，需要注意显存

---

## 5. 方案 C：Normalized GCN Propagation（N-GCN）

### 5.1 设计理念
GCN 的价值在于：
- 用 **归一化邻接** 做谱意义上的平滑/滤波
- 通过 `XW` + 非线性，让 message passing 不再是“纯 mixing”

这更像“结构化的低通滤波 + learnable feature transform”。

### 5.2 结构与公式
仍然按 segment 学动态邻接 `A_dyn`（可以复用现有 low-rank learner，也可直接用方案 A 的 attention adjacency）。

构造归一化邻接（两种二选一）：
- **row-norm**：`Â = RowNorm(A_dyn + I)`
- **sym-norm**：`Â = D^{-1/2}(A_sym + I)D^{-1/2}`, `A_sym = (A_dyn + A_dyn^T)/2`

GCN layer（1 层版本）：
- `M = Â @ H_val_seg`
- `H_out_seg = H_val_seg + γ ⊙ Dropout( σ( M W ) )`

可选 2 层（谨慎，避免 oversmoothing）：
- `H1 = σ(Â @ H W1)`
- `H2 = Â @ H1 W2`
- 残差：`H_out = H + γ ⊙ H2`

### 5.3 关键超参（建议范围）
- `gcn_layers ∈ {1, 2}`
- `gcn_norm ∈ {row, sym}`
- `residual_scale_init ∈ {0.0, 0.1}`

### 5.4 预期
- 比 attention 更“稳”，在某些噪声很强的数据上可能更鲁棒
- 但如果真正需要的是非线性选择性路由，GCN 可能不如 GAT/Attention

---

## 6. v7 成功标准（务实）

我们把 v7 当作最后三枪：每个方案都应该有“可验证的价值”。建议标准：

1) 至少有一个方案在 **≥2 个数据集**上相对 v6 BASE 获得稳定提升：
- `ΔMSE ≤ -0.001` 或 `ΔMAE ≤ -0.001`（单 seed 先看趋势，双 seed 再确认）

2) 不允许以明显不稳定为代价：
- 训练 loss 不应频繁爆炸
- `A_entropy` / attention 分布不应长期接近 uniform（否则等价无图）

3) 结构可解释：
- S-GAT / T-CVAttn 的 attention weights 能在可视化上表现“非随机”的跨变量模式

---

## 7. 我们不做什么（为了聚焦）
- 不再继续 OFAT 扫 `graph_rank/graph_scale/topk/base_alpha_init`（v6+消融已证收益边际很小）
- 不再把跨变量传播加到 trend 分支（v6 Step3 是负收益）
- 不再引入更多“第三路稳定流/复杂 routing”把实验矩阵做爆

---

## 8. 交付物
- `v7implementationTODO.md`：逐 step 的实施与实验计划（含必要的超参小扫）

