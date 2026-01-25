# DynamicGraphMixer v2 实施与实验 TODO（implementationTODOv2.md）

> 这份 TODO 是“可以照着一步一步跑”的版本路线图。  
> 你要求：每个大推进下的小实验要尽量全面——本文对关键模块（门控/路由/正则/多尺度）给出了较完整的实验矩阵与验收标准。

---

## 总原则（强约束）

1. **Anchor baseline 不得丢**  
   每个版本都必须能通过配置开关退化为当前最优 baseline（例如 M8-5），并复现同量级指标。

2. **一次只引入一个关键机制**  
   每个里程碑只引入 1 个“主变量”（例如只改 gate，不同时改 graph learner）。

3. **每个结论至少 3 seeds**  
   - 快速筛选：1 seed（early stop + 少量 epoch）  
   - 最终确认：至少 3 seeds（固定训练策略）  
   以避免把随机波动当成提升。

4. **日志必须对齐**  
   每次实验输出：
   - test MSE/MAE（主指标）
   - graph stats（entropy/topk_mass/adj_diff）
   - gate/alpha 的统计（mean/std/分位数）
   - 最好保存 2~3 段的 adjacency heatmap（便于直观诊断）

---

## 0) v2.0：冻结 baseline + 补齐可观测性（强烈建议先做）

### 0.1 代码任务
- [ ] 在 `DynamicGraphMixer.Model._mix_segments()` 里额外记录：
  - `adj_raw`（sparsify 前）的 entropy、topk_mass
  - `adj`（sparsify 后）的 entropy、topk_mass
  - `adj_diff = mean(|A_t - A_{t-1}|)`  
- [ ] 在 `GraphMixer` 或 `DynamicGraphMixer` 里记录：
  - `gate_value = sigmoid(gate_param)` 的均值/方差/分位数（若 gate_mode!=none）
  - `alpha_value = sigmoid(graph_base_alpha)`（若 base_mode=mix）
- [ ] 输出到 `graph_logs/<exp_id>/stats.csv`（与你已有格式兼容即可）

> 验收：关掉新增日志（或仅做 detach 记录）不应改变指标（数值误差允许 1e-4 量级）。

### 0.2 实验
- [ ] **E0-1**：复现当前最优配置（推荐以 M8-5 为准）1 seed  


> 验收：mean MSE 与历史最优差异不超过 0.002；std 合理（你可自定义阈值）。

---

## 1) v2.1：只做“门控粒度 & 初始化”的系统性消融（不改结构）

> 目的：在引入“置信度门控”之前，先把 **静态 gate 的上限** 跑清楚。  
> 当前你只验证了 `gate_mode=scalar`，但 `per_var/per_token` 可能更强。

### 1.1 实验维度（建议固定其他参数为 baseline）
固定：
- `graph_source=content_mean`
- `graph_base_mode=mix`
- `adj_sparsify=topk, adj_topk=6`
- `graph_scale=8`
- `graph_rank=8`

#### (A) gate_mode 扫描
- [ ] **E1-A**：`gate_mode ∈ {none, scalar, per_var, per_token}`

#### (B) gate_init 扫描（每个 gate_mode 都做）
- [ ] **E1-B**：`gate_init ∈ {-8, -6, -4, -2, 0}`  
  解释：sigmoid 后约为 {0.0003, 0.0025, 0.018, 0.12, 0.5}

#### (C) graph_scale×gate 的交互（小网格）
- [ ] **E1-C**：对最好的 gate_mode 额外扫 `graph_scale ∈ {1, 2, 4, 8, 16}`  
  目的：看 gate 是否在更频繁更新图时更重要。

### 1.2 每组实验的输出与诊断
- gate 的最终值是否趋近 0（说明图混合在该设置下整体有害）
- gate 的 var 是否过大（尤其 per_token 可能造成过拟合）
- 邻接 entropy 与 gate 的相关性（静态 gate 也可能“补偿”过密图）

### 1.3 验收标准
- 选出一个 “静态 gate 最佳配置” 作为 v2.2 的 base setting
- 至少 2 seeds 确认该提升不是偶然（最终 3 seeds 再确认）

---

## 2) v2.2：实现并验证“置信度门控”（关键里程碑，实验要更密）

> 核心假设：**图越不确定（熵越高/抖动越大），跨变量传播越应该弱。**  
> 目标：用动态 gate 替代静态 gate，让模型“该混就混，不该混就别混”。

### 2.1 代码改动（最小侵入实现）

#### 2.1.1 新增 gate_mode
在 `modules/mixer.py::GraphMixer` 支持：
- `gate_mode="conf_scalar"`：每个 segment 一个标量 gate
- `gate_mode="conf_per_var"`：每个变量一个 gate（shape `[B,C,1,1]`）

实现方式建议：
- `GraphMixer.forward(adj, x, token_offset)` 内部计算置信度（从 adj 得到）
- 将 gate 作为 broadcast 张量与 `mixed` 相乘

#### 2.1.2 置信度度量（至少实现 2 个）
- `entropy`（推荐默认）
- `adj_diff`（需要在 DynamicGraphMixer 里把 prev_adj 或 diff 传入 GraphMixer，或在 _mix_segments 外计算 gate）

你也可以先只做 entropy（最简），adj_diff 作为第二阶段加。

#### 2.1.3 置信度到 gate 的映射（先做 2 种）
- **G1 确定性**：`g = conf^gamma`
- **G2 可学习仿射**：`g = sigmoid(w*conf + b)`  
  参数量很小，容易稳定。

#### 2.1.4 conf 计算点（必须可选）
- `gate_conf_source=pre_sparsify`（推荐）
- `gate_conf_source=post_sparsify`

> 因为 top-k 会改变熵，建议默认 pre_sparsify。

#### 2.1.5 warmup（强烈建议）
- `gate_warmup_epochs`：前 k 个 epoch 固定 `g≈0`（等价不混），之后再启用动态 gate  
  防止训练早期“乱图污染表征”。

---

### 2.2 实验矩阵（尽量全面，但可按两阶段跑）

#### 阶段 1：快速筛选（1 seed + 较少 epoch）
固定其它为 v2.1 最佳静态配置。

- [ ] **E2-1**：gate_mode ∈ {conf_scalar, conf_per_var}
- [ ] **E2-2**：conf_metric=entropy，mapping ∈ {G1, G2}
- [ ] **E2-3**：`gate_conf_source ∈ {pre_sparsify, post_sparsify}`
- [ ] **E2-4**（仅 G1）：`gamma ∈ {0.5, 1, 2, 4}`
- [ ] **E2-5**（仅 G2）：`(w_init, b_init)` 组合  
  建议：w_init ∈ {2,4,8}，b_init ∈ {-4,-2,0}  
- [ ] **E2-6**：`gate_warmup_epochs ∈ {0, 1, 3, 5}`

筛选目标：找到 2~3 个最有希望的组合。

#### 阶段 2：严格确认（≥3 seeds）
- [ ] **E2-Final**：对阶段 1 最优 2~3 个组合跑 3 seeds，报告 mean±std

---

### 2.3 诊断与验收
你应该重点看下面三条是否同时成立：

1) **性能提升**：mean MSE 优于 v2.1 的最佳静态 gate  
2) **gate 与 entropy 呈负相关**：entropy 高 → gate 小（至少整体趋势如此）  
3) **不会坍缩为全 0**：若 gate 长期接近 0，说明图混合在该设置下无贡献，需要回到图生成/稀疏化层面找原因

---

## 3) v2.3：实现“置信度路由 + 有效 base 正则”（第二关键里程碑）

> 目标：当动态图不可信时，让模型回退到 base graph（或稳定图），避免错误动态边污染。  
> 同时把 base_reg 从“无效 L1”换成“结构性约束”。

### 3.1 代码任务

#### 3.1.1 base_reg 修正（必须）
在 `DynamicGraphMixer._mix_segments()` 里替换 base_reg：

- `graph_base_reg_type=offdiag_l1`：
  $$\|A_{base}\odot(1-I)\|_1$$
- `graph_base_reg_type=entropy`：
  $$\frac{1}{C}\sum_i H(A_{base,i})$$
- `graph_base_reg_type=diag_prior`：
  $$-\frac{1}{C}\sum_i A_{base,ii}$$

并引入 `graph_base_reg_lambda`。

#### 3.1.2 动态 alpha（置信度路由）
新增：
- `graph_base_alpha_mode=static|conf`
- 若 conf：`alpha = sigmoid(w_alpha*(1-conf)+b_alpha)`  
  conf 可以用 v2.2 已实现的 entropy-based conf（复用即可）

> 这里建议：先做 **alpha 为 scalar**，稳定后再做 per_var。

---

### 3.2 实验矩阵（较全面）

#### (A) static vs conf alpha
- [ ] **E3-A1**：alpha_mode=static（当前实现）
- [ ] **E3-A2**：alpha_mode=conf（scalar）
- [ ] **E3-A3**：alpha_mode=conf（per_var，可选）

#### (B) base_reg 类型与强度
对 (A) 的最优 alpha_mode，扫：
- [ ] **E3-B1**：reg_type ∈ {none, offdiag_l1, entropy, diag_prior}
- [ ] **E3-B2**：reg_lambda ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}

#### (C) 与 gate 的组合方式（关键）
- [ ] **E3-C1**：只用 conf gate（v2.2 best）
- [ ] **E3-C2**：只用 conf alpha（不启用 conf gate）
- [ ] **E3-C3**：conf gate + conf alpha（同时启用）  
  观察是否互补（有时会重复抑制，导致传播过弱）

> 每个组合至少 2 seeds；最终最优跑 3 seeds。

---

### 3.3 验收标准
- 相比 v2.2 最优至少再有可测提升（或同等指标但方差更小/更稳）
- base graph 可解释性变强：对角质量、top-k 边更稳定（从日志可见）

---

## 4) v2.4：多尺度双图融合（把 stable-stream 用到“该用的地方”）

> 这一阶段不是必须，但很可能带来提升，尤其在更强非平稳/工况切换数据上。  
> 核心思想：stable-stream 不适合高频更新图，但适合提供长尺度“结构骨架”。

### 4.1 实现建议（保持可退化）
新增一张“稳定图”：
- `A_stable`：由 `graph_source=stable_stream` 在更粗尺度 `graph_scale_stable` 上生成  
- 与 `A_dyn`（content_mean）融合：
  - convex mix：`A = (1-β)A_dyn + βA_stable`
  - 或把 A_stable 当 base：`A = (1-α)A_dyn + αA_stable`

β/α 可以静态可学习，也可以由置信度决定。

### 4.2 实验矩阵（建议按两阶段）

#### 阶段 1：稳定图单独验证（找靠谱设置）
- [ ] **E4-1**：stable_feat_type ∈ {detrend, diff, none}
- [ ] **E4-2**：stable_window ∈ {8, 16, 32}
- [ ] **E4-3**：graph_scale_stable ∈ {8, 16, 32}
- [ ] **E4-4**：stable_share_encoder ∈ {False, True}（优先 False）
- [ ] **E4-5**：stable_detach ∈ {False, True}

#### 阶段 2：融合策略
在阶段 1 最优 stable 设置上：
- [ ] **E4-6**：fuse_mode ∈ {dyn+stable(convex), dyn+stable(as_base)}
- [ ] **E4-7**：β_init ∈ {-6, -4, -2}（sigmoid 后很小→保证可退化）
- [ ] **E4-8**：conf routing：低置信度时提高 β（或 α）

> 这一阶段每个候选至少 2 seeds；最终最优 3 seeds。

---

## 5) v2.5：Lag-aware Graph Mixing（偏研究向，但可能是“工业杀手锏”）

> 如果你后续目标是工业过程（滞后影响强），这一块值得做成论文亮点。  
> 但建议放在 gate/route 稳定后做。

### 5.1 最小实现（先别做太复杂）
实现 `LagGraphMixer`：
- 输入 `x: [B,C,N,D]`
- 对 lag=0..L-1，取 `x_shifted`（时间维右移）
- 用共享 `A` 或少量参数化的 `A_l` 做：
  $$\text{mixed}=\sum_{\ell} w_\ell\cdot (A\cdot x_{t-\ell})$$
- `w` 用 softmax，保证可解释且稳定。

### 5.2 实验矩阵
- [ ] **E5-1**：L_lag ∈ {2, 4, 8}
- [ ] **E5-2**：share_adj ∈ {True, False(每 lag 一张图，先不推荐)}
- [ ] **E5-3**：w_init ∈ {uniform, bias_to_small_lag}
- [ ] **E5-4**：与 conf gate/route 的组合（是否需要更强抑制）

---

## 6) v2.6：补全“可发表级”实验套件（最终打磨）

当你选定 v2 主线最优配置后，建议做这些“论文必须的消融/泛化”：

- [ ] 多 horizon：`pred_len ∈ {96, 192, 336, 720}`（至少 2~3 个）
- [ ] 多数据集：ETTm2/ETTh1/ETTh2 等（至少 2 个）
- [ ] 复杂度：记录 params、吞吐、显存
- [ ] 可解释性：给出若干工况片段的图变化与 gate 变化曲线（非常加分）

---

## 附：建议的版本命名与记录格式

- 代码版本：`v2.x.y`（x=里程碑，y=小修复）
- 每个实验设置写入：
  - `--graph_log_exp_id`：例如 `v2.2_E2-4_seed2021`
  - 保存 `config.json`（或命令行）到同目录
- 每个里程碑结束输出一张总表：mean±std（3 seeds）+ 关键 graph 指标对比

---

## 最后的路线建议（优先级）

如果只选一条最可能继续提升的主线：

1. **v2.2 置信度门控（重点做深）**
2. **v2.3 置信度路由 + base 正则**
3. 之后再考虑 v2.4 多尺度融合、v2.5 滞后图

这样推进最稳、最容易把贡献写清楚，也更符合你“单步推进 + 版本管理 + 充分实验”的原则。

---
