# DynamicGraphMixer v2 设计文档（model_design_v2.md）

> 本文档基于当前 **DynamicGraphMixer superset 实现** 与 **ETTm1(96→96) 的实验日志** 总结出“已验证有效的设计”，并提出下一阶段（v2）可控、可退化、可逐步验证的升级路线。  
> 目标：在 **不破坏现有最优配置** 的前提下，引入更强的“稳健图路由”与“可解释结构”，进一步提升性能与泛化稳定性。

---

## 1. 背景：非平稳预测在动态图视角下的核心矛盾

在多变量预测中，非平稳不仅表现为单变量均值/方差漂移，更关键的是：

- **跨变量耦合结构会随时间/阶段/工况变化**（图结构随时间变）。
- 若把跨变量混合做得过强或过密，**错误边会把噪声/无关信息注入表征**，造成系统性退化。
- 若为了抑制伪相关而过度“平稳化”，又会造成 **幅值信息被抹掉**，输出过平滑。

因此我们需要一个“能变、但不乱变”的动态图骨干：  
**能在需要时使用跨变量传播，不确定时自动减弱传播**；同时具备稳定骨架（base graph）来兜底。

---

## 2. v1（当前实现）复盘：结构与已验证结论

### 2.1 当前 superset 模型结构（代码一致）

现有 `DynamicGraphMixer.py` 的 forward 可以抽象为：

1) **Tokenizer（可选 patch）**  
输入 `x_enc ∈ R^{B×L×C}`  
输出 token 序列 `h_time ∈ R^{B×C×N×D}`（`N=L` 或 `N≈L/patch_len`）

2) **TemporalEncoder（变量内时间编码）**  
`TemporalEncoderWrapper` 支持 TCN / Transformer（当前最优基本来自 TCN）。

3) **Graph tokens 选择**（估图输入来源）  
- `graph_source="content_mean"`：直接用 `h_time`  
- `graph_source="stable_stream"`：用 `StableFeature` 变换输入（detrend/diff），经（共享或独立）encoder 得到 `h_graph`

4) **分段动态图生成**  
把 token 维按 `graph_scale` 分段（每段长度 `scale`），段内用平均池化：
$$
z_t = \text{MeanPool}(h_{\text{graph}}[:, :, \text{seg}, :]) \in R^{B\times C\times D}
$$
低秩图生成器输出：
$$
A_t=\text{RowSoftmax}\left(\frac{U_tV_t^\top}{\sqrt{r}}\right)\in R^{B\times C\times C}
$$

5) **Base graph（可选）**  
当前实现是静态可学习 `A_{base}= \text{RowSoftmax}(W_{base})`，并用一个标量 `α = σ(α_{logit})` 混合：
$$
A_t \leftarrow (1-\alpha)A_t + \alpha A_{base}
$$

6) **Adj sparsify（可选 top-k）**  
对每行保留 top-k，再行归一化。

7) **GraphMixer（跨变量传播 + 残差）**  
$$
\text{mixed} = A_t \cdot X,\quad Y = X + g\odot \text{Dropout}(\text{mixed})
$$
其中 gate `g` 目前是 **静态可学习参数**：`none / scalar / per_var / per_token`。

8) **ForecastHead**  
Flatten token 维后线性映射到 `pred_len`。

> 以上流程与实现严格一致（`_mix_segments / _select_graph_tokens / GraphMixer` 等）。  

---

### 2.2 关键实验结论（只引用已发生的结果）

基于实验日志（ETTm1, 96→96）我们已经确认：

- **Gated mixing 是“决定性增益项”**：在 stable-stream 设置下从 0.3466 → 0.3357（M7）。  
- **Top-k 稀疏化带来小幅但相对稳定的收益**（M8 系列）。  
- **当前最优配置来自 content_mean 估图 + base graph + gate + topk**（M8-5）。  
- **Stable-stream 估图对 `graph_scale` 更敏感**：在粗尺度更新时效果明显更好；细尺度更新会明显退化（例如 M8-6）。  
- **Token-level 的 StableFeature（对 latent token 做 detrend/diff）显著退化**（M5 系列）。  
- **在 TCN 下，patch 路径（无论 encode-then-pool 或 token_first）都不如 point-token 基线**（M9 系列）。  
- **base_l1 当前实现形式“几乎无意义”**：因为 `A_base` 是行归一化概率矩阵，`mean(|A_base|)` 基本是常数 $$1/C$$，需要重新设计正则形式。

---

## 3. v2 的核心目标与原则

v2 的总体目标不是“堆更多模块”，而是把 v1 已验证有效的机制推进到更稳健的形态：

1. **让跨变量传播强度从“静态参数”升级为“按置信度自适应”**  
当图不确定 / 抖动 / 近似均匀时，自动减小传播；当图稳定且尖锐时，允许更强传播。

2. **让 base graph 从“固定兜底”升级为“按不确定性路由”**  
动态图可信时偏向动态；动态图不可信时偏向 base（或稳定图）。

3. **把 stable-stream 的价值用在“长尺度结构”上**  
而不是让 stable-stream 在每个 token 上频繁更新图，避免被短期噪声诱导。

4. **保持可退化性与可消融性**  
任何新增机制必须满足：开关关闭时严格回到当前最优 baseline（M8-5 等价）。

---

## 4. v2 提案：Confidence-Calibrated Dynamic Graph Mixer（CC-DGM）

v2 的核心创新点（主线）是 **“置信度校准的动态图传播”**，包括两层：

- **(A) 置信度门控（Confidence Gate）**：控制“传播量”  
- **(B) 置信度路由（Confidence Routing）**：控制“用哪张图传播”（动态 vs base / stable）

### 4.1 图置信度度量（Graph Confidence）

对一个行归一化邻接矩阵 $A\in R^{B\times C\times C}$，定义每个节点的行熵：

$$
H_i = -\sum_{j=1}^{C} A_{ij}\log(A_{ij}+\epsilon)
$$

归一化后得到 $[0,1]$ 区间的“非置信度”：

$$
\tilde H_i = \frac{H_i}{\log C}
$$

定义置信度：

$$
c_i = 1-\tilde H_i
$$

此外可选再引入两个辅助信号（更稳健）：

- **Top-k mass**：每行 top-k 权重之和（尖锐度）
- **Temporal stability**：与上一段图的差异
$$
d = \|A_t-A_{t-1}\|_1
$$

最终可用线性组合得到综合置信度（先做归一化）：
$$
c = \lambda_1 c^{entropy} + \lambda_2 c^{topk} + \lambda_3 c^{stable}
$$

> 重要细节：置信度最好在 **sparsify 之前** 计算（否则 top-k 会人为降低熵，让置信度失真）。

---

### 4.2 置信度门控（控制“传播强度”）

当前 GraphMixer 的 gate 是静态可学习参数。v2 提出：

#### 方案 G1：确定性门控（0 参数，最稳）

$$
g_i = (c_i)^\gamma,\quad \gamma>0
$$

当图接近均匀（低置信度）时 $g_i\to 0$，当图很尖锐（高置信度）时 $g_i\to 1$。

#### 方案 G2：可学习仿射门控（2 参数/每变量）

$$
g_i = \sigma(w\cdot c_i + b)
$$

优势：让模型自己学习“什么时候该相信图”。

#### 方案 G3：轻量 MLP（可选，表达更强）

$$
g_i=\sigma(\text{MLP}([c_i, \|z_i\|, \text{var}(z_i), ...]))
$$

> v2 推荐先做 G2（可学习但足够简单），并做 warmup（前若干 epoch 固定 g≈0，避免早期错误图污染）。

---

### 4.3 置信度路由：动态/稳定骨架的自适应融合

当前 base graph 混合系数 α 是全局标量，v2 把它升级为 **由置信度决定**：

#### 方案 R1：动态 α（scalar 或 per-var）

$$
\alpha = \sigma(w_\alpha\cdot (1-c) + b_\alpha)
$$

- 当 $c$ 低（不确定）→ α 高：更多依赖 base graph（兜底）
- 当 $c$ 高（确定）→ α 低：更多依赖动态图（自适应）

然后：
$$
A_{mix}=(1-\alpha)A_{dyn}+\alpha A_{base}
$$

这比“只 gate mixed feature”更本质：它在结构层面就避免把不可信动态边注入进去。

#### 方案 R2：稳定图作为“base 的 data-driven 版本”（多尺度两图）

从 stable-stream 产生一个 **更粗尺度** 的图 $A_{stable}$（例如 `graph_scale_stable=16/32`），然后做三方融合：

$$
A_{mix} = \beta_1 A_{dyn}+\beta_2 A_{stable}+\beta_3 A_{base},\quad \sum\beta=1
$$

其中 $\beta$ 可以是固定超参、可学习参数，或由置信度决定。

---

### 4.4 Base graph 正则：从“无效 L1”升级为“结构性约束”

由于 $A_{base}$ 行归一化，`mean(|A_base|)` 基本常数。v2 建议使用以下可控正则：

- **off-diagonal L1（鼓励稀疏/保守传播）**
$$
\mathcal{L}_{off} = \|A_{base}\odot (1-I)\|_1
$$

- **entropy penalty（鼓励 base 更尖锐）**
$$
\mathcal{L}_{ent} = \frac{1}{C}\sum_{i}H(A_{base,i})
$$

- **diag prior（鼓励保留自信息）**
$$
\mathcal{L}_{diag} = -\frac{1}{C}\sum_i A_{base,ii}
$$

这些正则都能产生真实梯度，且语义明确。

---

## 5. v2 可选增强方向（第二阶段再做）

### 5.1 多时滞图（Lag-aware Graph Tensor）

工业过程常见滞后影响。把单一 $A_t$ 扩展为 $A_{t,\ell}$：

$$
\text{mixed}_t = \sum_{\ell=0}^{L_{lag}-1} A_{t,\ell}\cdot X_{t-\ell}
$$

为了避免参数爆炸，建议从“共享结构 + 滞后权重核”开始：

$$
A_{t,\ell}=w_\ell\odot A_t,\quad \sum_\ell w_\ell=1
$$

### 5.2 原型图字典（Graph Prototypes）

当工况离散切换明显时，用 K 个原型图：

$$
A_t=\sum_{k=1}^{K}\pi_{t,k}A^{(k)},\quad \pi_t=\text{softmax}(g(z_t))
$$

---

## 6. 配置接口建议（v2 新增/调整）

为保持兼容性，所有新增参数必须有默认值，使得默认行为复现现有 baseline。

### 6.1 置信度门控相关
- `gate_mode`: `none|scalar|per_var|per_token|conf_scalar|conf_per_var`
- `gate_conf_metric`: `entropy|topk_mass|adj_diff|combo`
- `gate_conf_source`: `pre_sparsify|post_sparsify`
- `gate_conf_gamma`: (用于 G1)
- `gate_conf_w_init`, `gate_conf_b_init`: (用于 G2)
- `gate_warmup_epochs`: 前若干 epoch 固定 gate≈0

### 6.2 Base 路由/正则
- `graph_base_alpha_mode`: `static|conf`
- `graph_base_reg_type`: `none|offdiag_l1|entropy|diag_prior`
- `graph_base_reg_lambda`: 正则权重

### 6.3 图生成校准
- `graph_temp`: softmax 温度（>0）
- `graph_temp_learnable`: 是否可学习温度

### 6.4 多尺度稳定图（可选）
- `graph_use_stable_base`: bool
- `graph_scale_stable`: stable 图更新尺度
- `graph_fuse_mode`: `dyn+base|dyn+stable|dyn+stable+base`
- `graph_fuse_beta_init`: 初始融合权重

---

## 7. 评估指标与可解释性输出

除 MSE/MAE 外，v2 必须持续记录以下（用于定位收益来源）：

- 邻接熵（pre/post sparsify）
- top-k mass（固定 k）
- 邻接变化率 $\|A_t-A_{t-1}\|_1$
- gate 分布（均值/方差/分位数）
- α 分布（如果启用 conf routing）
- base graph 的对角质量、top-k 结构（可解释）

这些日志能直接解释“性能提升来自更好的图还是来自更谨慎的传播”。

---

## 8. 最小可发表主线（建议优先做）

如果我们要快速把 v2 打磨成一个“论文可写”的核心贡献，我建议优先把资源集中在：

1) **Confidence Gate（G2）**：动态图不确定时自动减弱传播  
2) **Confidence Routing（R1）**：动态图不可信时回退到 base graph  
3) **有效 base 正则（offdiag/diag/entropy）**：让 base graph 真正成为“结构骨架”而不是无意义参数

这三点叠加，具有清晰动机、明确实现、强可解释性，且能严格消融验证。

---
