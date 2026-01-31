# v7implementationTODO.md

> 本文档是“按图施工”的执行清单：每一步实现什么、跑哪些验证、预期看到什么。
> 默认沿用 v6 的数据集与训练脚本体系（`graph_logs/.../stats.csv`），并确保所有新模块可退化为 B5。

---

## Step 0 — 基线锁定与对齐（1 次性）

### 0.1 目标
把 v7 的所有对比建立在同一基线上：**B5 (v6 Step0 BASE)**。

### 0.2 实施
1. 固定 config 为 B5：
   - TCN encoder
   - EMA decomp (`decomp_alpha=0.1`)
   - graph_scale=16
   - gate_init=-6
   - base_alpha_init=-8
   - adj_topk=6
2. 跑 4 个数据集：`ETTm1 / weather / flotation / grinding`
3. 导出最后一行指标到 `v7_baseline_check.md`（可选）。

### 0.3 验收（允许轻微浮动）
- ETTm1: MSE≈0.3172, MAE≈0.3575
- weather: MSE≈0.1728, MAE≈0.2301
- flotation: MSE≈0.7861, MAE≈0.6228
- grinding: MSE≈4.249, MAE≈0.699

如果偏差很大（>0.003 MSE），先排查：seed / data split / lr / epoch / AMP。

---

## Step 1 — 抽象统一接口：GraphMixerV7

### 1.1 目标
把“跨变量交互算子”抽象成可插拔模块，三种方案通过一个开关切换：

- `graph_mixer_type = baseline | gat_seg | attn_token | gcn_norm`

并确保：
- `baseline` 与当前 B5 完全一致
- 其他 type **shape/日志/训练流程不破坏**

### 1.2 代码改动点（建议）
- `models/DynamicGraphMixer.py`
  - 在原 GraphMix 处替换为 `GraphMixerV7.forward(...)`
- `models/modules/graph_mixer_v7.py`（新增）
  - `BaselineGraphMixer`（封装旧逻辑，便于一致性）
  - `GATSegMixer`
  - `TokenCVAttnMixer`
  - `NormGCNMixer`
- `configs/*.yaml`
  - 新增 v7 参数段：
    - `graph_mixer_type`
    - `gat_heads, gat_topk, gat_bias_base`
    - `attn_heads, attn_topk(optional)`
    - `gcn_layers, gcn_norm`
    - `residual_scale_init, warmup_epochs`

### 1.3 快速单元验证（必须做）
- forward shape：输入 `[B,L,C]` 输出 `[B,pred_len,C]`
- backward：loss 能正常反传（尤其 attention 模块）
- `baseline` 的输出与旧代码逐元素 diff（允许浮点误差 1e-6 级）

### 1.4 预期结果
- 还没追求提升，只要求 **可跑、可回归、可记录**

---

## Step 2 — 方案 A：S-GAT（segment-level sparse GAT）

### 2.1 核心实现
**输入**（与现有 GraphLearner 对齐）：
- `H_map`: map 分支 token，shape `[B,C,N,D]`
- `H_val`: value 分支 token，shape `[B,C,N,D]`
- segment 切分：按 `graph_scale` 得到多个 segment

**每个 segment k：**
1. `z_map = mean(H_map[:,:,seg,:], dim=token)` → `[B,C,D]`
2. 多头注意力打分：
   - `Q=z_map Wq, K=z_map Wk`
   - `score = (Q K^T)/sqrt(dh)` → `[B,heads,C,C]`
3. (可选) base graph bias：
   - `score += beta * log(A_base + eps)`（beta 可学习或常数 1）
4. top-k sparsify（行内）：保留 k 个最大 score，其余置 -inf
5. softmax 得 `A_attn`
6. 传播：
   - `M = A_attn @ H_val_seg`（对每个 token 广播）
7. 残差注入：
   - `H_out_seg = H_val_seg + rs * dropout(M)`

其中 `rs` 是 residual scale（建议 per-var 或标量），并支持 warmup：前 `warmup_epochs` 强制 `rs=0`。

### 2.2 实验验证（小扫即可）
先跑小通道与中通道：`ETTm1, weather`（单 seed）

- Sweep-1（最小）：
  - `gat_heads`: 4 vs 8
  - `gat_topk`: 6 vs 12
  - 固定 `graph_scale=16`, `warmup=1`, `residual_scale_init=0.0`

总共 2×2=4 组（每个数据集 4 次）。

### 2.3 判定与下一步
- 若某组合在 ETTm1 或 weather 上出现 **稳定的 ΔMSE ≤ -0.001**，记为候选。
- 若都无提升：
  - 允许再试一次 `residual_scale_init=0.1`（只试最优 head/topk 组合）

### 2.4 预期结果
- `A_entropy` 不应一直接近 `log(C)`（完全均匀）
- 相比 baseline：gate 不重要了（可保留旧 gate 但默认关闭），核心看 residual scale 是否学到非零

---

## Step 3 — 方案 B：T-CVAttn（token-wise cross-variable attention）

### 3.1 核心实现
对每个 token（或每个 segment 内 token）做跨变量 MHA：

- reshape：`H_* : [B,C,N,D] → [B*N, C, D]`
- `MHA(Q=H_map, K=H_map, V=H_val)`（attention 在变量维 C 上）
- 输出 reshape 回 `[B,C,N,D]`
- 残差：`H_out = H_val + rs * proj(attn_out)`

这里的 `proj` 是一个小线性层（可选），用于稳定尺度。

### 3.2 实验验证（小扫）
先跑 `ETTm1, weather`（单 seed）：

- Sweep-1：
  - `attn_heads`: 4 vs 8
  - `warmup_epochs`: 0 vs 1

如果 C 很小（ETTm1）8 heads 可能不必要，但我们允许它作为对照。

可选补充（只做 1 个对照，不要扩张）：
- `token_attn_topk`: None vs 6（做稀疏 attention，节省算力，同时检验是否更稳）

### 3.3 判定
- 若 token-wise attention 在 ETTm1/Weather 至少一个数据集上带来 ≥1e-3 的提升，并且训练稳定，则进入最终候选池。
- 若 attention 明显不稳定：优先把 warmup 打开，或把 `rs` 初始化调小（0→0.05）。

### 3.4 预期结果
- 相比 S-GAT：
  - 更容易捕捉“时变的跨变量依赖”
  - 但计算更重（O(N·C^2)），需要注意显存

---

## Step 4 — 方案 C：N-GCN（normalized GCN propagation）

### 4.1 核心实现
保留现有 low-rank GraphLearner 得到 `A_dyn`（或复用 S-GAT 的 `A_attn` 也可，但先不混线）。

GCN 层：
- `A_hat = A_dyn + I`
- 归一化：
  - `row`: `A_norm = A_hat / A_hat.sum(-1, keepdim=True)`
  - （可选）`sym`: `A_norm = D^{-1/2} A_hat D^{-1/2}`
- 更新：`H1 = act( A_norm @ (H_val W1) )`
- 残差：`H_out = H_val + rs * dropout(H1)`

如果 `gcn_layers=2`：再堆一层（但要加残差，避免过平滑）。

### 4.2 实验验证（小扫）
跑 `ETTm1, weather` 单 seed：

- Sweep-1：
  - `gcn_layers`: 1 vs 2
  - `gcn_norm`: row vs sym

总共 4 组。

### 4.3 判定
- 若 2-layer 出现退化（常见），直接保留 1-layer。
- 若 sym_norm 不稳/不提升，保留 row_norm。

### 4.4 预期结果
- 比 attention 更稳
- 但上限可能不如 attention（因为表达力较弱）

---

## Step 5 — 扩展到四个数据集 + 最终筛选

### 5.1 实施
对每个方案选出“在 ETTm1/weather 上最好的一组超参”，然后在 4 个数据集全跑（单 seed）：
- ETTm1
- weather
- flotation
- grinding

### 5.2 评分方式（务实）
- 计算 4 数据集的平均 rank（或平均相对改善）
- 同时检查是否出现“灾难性退化”（任一数据集 MSE 恶化 > 0.01）

### 5.3 预期结果
- 至少 1 个方案在 flotation/grinding 上能吃到结构收益
- 若某方案只在一个数据集提升但其他都不动，可接受，但需要解释该数据集确实更依赖 cross-var

---

## Step 6 — 最终复核（2 seeds）+ 固化版本

### 6.1 实施
对 Step5 的最佳方案做 2 seeds 复核：
- 只跑 2 个代表性数据集：`grinding + weather`（一个工业噪声，一个公开基准）
- 或者跑 4 个都做 2 seeds（如果算力允许）

### 6.2 验收
- 2 seeds 平均仍有提升（至少一个指标 MSE/MAE 明确下降）
- 方差不应显著变大

### 6.3 固化
- 把最佳方案写入默认 config：`v7_best.yaml`
- 输出一份 `v7_results_summary.md`（包含：baseline vs 三方案最佳 vs v7 best）

---

## 失败兜底与快速调整（不扩大实验矩阵）

- 如果训练不稳定：
  1) 开启 warmup=1
  2) residual_scale_init 调小（0→0.05）
  3) attention heads 减半

- 如果显存超：
  1) token-wise attention 改为 segment-level（只作为临时 sanity）
  2) 开启 topk attention

- 如果所有方案都无提升：
  - 结论应当是：当前数据集/设定下，跨变量交互收益不足，主性能来自 temporal encoder + dual-stream；后续应把精力投入到更强的时间建模或更好的非平稳归一化（而不是继续折腾图）。
