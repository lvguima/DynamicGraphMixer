# Model Design：DynamicGraphMixer vNext（以 V1 为 Anchor Baseline 的逐步增强设计）

> 本文档目的：以 **V1 当前实现** 为可复现的性能锚点（anchor baseline），在不破坏可退化性的前提下，逐步吸收 V2 的洞见，形成一个“可开关、可消融、可解释”的非平稳多变量预测主干。
>
> - Baseline 代码：`DynamicGraphMixer_v1.py`
> - 对照实现：`DynamicGraphMixer_v2.py` + `StableFeat_v2.py` + `TemporalEncoder_v2.py`
> - 设计原则来源：`method_desing_v1.md` 与 `method_desingv2.md`（P1~P4）

---

## 1. 任务与核心假设

### 1.1 任务定义
输入多变量序列 $$X \in \mathbb{R}^{B\times L\times C}$$，输出未来 $$H$$ 步预测：
$$\hat{Y} \in \mathbb{R}^{B\times H\times C_{out}}.$$

我们重点关心“非平稳”的一个工业化视角：  
**跨变量耦合结构随时间/工况变化**（而不仅是单变量均值/方差漂移）。

### 1.2 设计目标
1. **动态图（dynamic coupling graph）**：学习随时间片更新的 $$A_t\in\mathbb{R}^{C\times C}$$，驱动跨变量信息传播。
2. **抗伪相关（anti-spurious）**：图更新频率慢于时间步（长尺度生成图），减少短期噪声诱导的错误边。
3. **不过度平稳化**：即便结构学习使用稳定特征，传播仍以“内容特征”为主，避免幅值信息被抹掉。
4. **可退化到 V1**：任何增强项都必须可以通过配置开关关掉，使模型严格退化为 V1（便于逐步推进与版本管理）。

---

## 2. V1 Baseline 结构复盘（以实现为准）

### 2.1 模块划分
V1 可以拆成 5 个概念模块：

1) **Temporal Encoder（变量内时间编码）**  
对每个变量独立编码，得到时间表征：
$$H = f_{\text{time}}(X) \in \mathbb{R}^{B\times C\times N\times d}.$$

- 当 `use_patch=False`：$$N=L$$  
- 当 `use_patch=True` 且 encoder 不自带 patch（如 TCN）：先点级编码再做 patch pooling 得到 $$N=\text{num_tokens}$$

2) **Graph Update（按 segment 更新图）**  
把 token 维按 `graph_scale` 分段，段内做均值池化得到图条件特征：
$$z_s = \text{MeanPool}(H_{[:, :, \text{seg } s, :]}) \in \mathbb{R}^{B\times C\times d}.$$

3) **Low-Rank Graph Generator（低秩估图）**  
用双投影构造 logits：
$$U = z_s W_u,\quad V = z_s W_v,$$
$$S = \frac{1}{\sqrt{r}} UV^\top,$$
$$A_s = \text{RowSoftmax}(S).$$

4) **Graph Mixing（跨变量传播）**  
对该 segment 内所有 token 做传播并残差回加：
$$H'_s = H_s + A_s H_s.$$

5) **Forecast Head（多步线性头）**  
对每个变量把 token 展平，用共享线性层输出 $$H$$ 步：
$$\hat{Y} = \text{Head}(\text{Flatten}(H')),$$
其中 Head 在实现里是 `Linear(d_model * num_tokens -> pred_len)`，对变量维共享。

### 2.2 V1 的“为什么稳”
- 单流（content-only），梯度路径简单：预测损失直接监督时间编码器与图混合后的表征。
- 图条件特征来自同一条内容流（`z_t = mean(h_seg)`），不会出现“稳定流/内容流目标冲突”。

---

## 3. V2 引入的关键思想与潜在风险点

V2 相对 V1 的新增点：

1) **StableFeat + 双流**：构造 $$X_{stab}$$，并得到 $$H_{stab}$$ 专用于估图；$$H_{cont}$$ 用于传播与预测。
2) **share_encoder**：默认稳定流与内容流共享 temporal encoder 参数。
3) **Base Graph**：引入全局的 $$A_{base}$$（通过 `graph_base_logits`）并与动态图相加后行归一化。
4) **Patch 路径变化**：当 `use_patch=True` 时，V2 采用“先 patch embed，再编码”（token-first），而 V1 对 TCN 是“先点级编码，再 patch pooling”（encode-first）。

风险直觉（用于指导后续的逐步实验，而非直接定论）：
- **共享编码器可能引入梯度冲突**：stable 流的目标不等同于内容流的目标。
- **StableFeat 若定义不当会把估图信号推向高频噪声**：导致错误边被强传播放大。
- **Base graph 若过密**：可能形成“平均搅拌”，造成过平滑与性能下降。
- **token-first TCN 与 encode-first TCN 的等价性不成立**：信息路径变化很大。

---

## 4. vNext 的总体设计：一个“可退化”的 Superset 架构

我们建议把下一阶段统一为一个 superset 模型：**默认配置严格等价 V1**，打开开关后逐个引入增强项。

### 4.1 总体前向（概念版）
1) Token 化（两种路径二选一）：
- **V1 路径（默认）**：encode-first，然后 patch pooling（仅对 conv/TCN）
- **token-first（实验开关）**：先 patch embed 再编码（适合 transformer/SSM，也可用于对照）

2) 得到内容表征：
$$H_{cont} \in \mathbb{R}^{B\times C\times N\times d}.$$

3) （可选）构造稳定表征用于估图：
$$H_{stab} = f_{stab}(X)\ \text{or}\ f_{stab}(T_{cont}),$$
并可在长尺度下采样：
$$\bar{H}_{stab} \in \mathbb{R}^{B\times C\times N_{long}\times d}.$$

4) 图生成（piecewise constant）：
- 对每个 long-segment 生成 $$A_k$$，然后广播到对应的 token 段。

5) 图混合（内容传播）：
$$H' = H_{cont} + g \odot (A H_{cont}),$$
其中 $$g$$ 为可选门控（默认 $$g=1$$，退化到 V1）。

6) 预测头输出 $$\hat{Y}$$。

### 4.2 配置开关（关键：保证可退化）
建议把增强项都做成开关，并保证“关掉 = V1”：

- `graph_source = {content_mean (V1), stable_stream}`  
- `stable_feat_type = {none, detrend, diff, ...}`  
- `stable_share_encoder = {false (默认建议), true}`  
- `stable_detach = {true/false}`（当 share 时用于避免梯度冲突）
- `graph_base_mode = {none, mix, add}`（默认 none）
- `graph_base_alpha_init = 0`（确保初始等价 V1）
- `mix_gate = {none, scalar, per_var, per_token}`（默认 none）
- `adj_sparsify = {none, topk}`（默认 none）
- `patch_mode = {v1_encode_then_pool, token_first}`（默认 v1_encode_then_pool）

---

## 5. 关键模块设计（建议的“最小可用”实现选择）

### 5.1 Tokenizer / Patch（Patch 模块的统一语义）
我们明确区分两种 patch 模式：

**模式 A：V1 encode-first（默认）**  
- 适用：TCN 等卷积类 encoder  
- 路径：$$X \xrightarrow{\text{TCN}} H_{point} \xrightarrow{\text{MeanPool-unfold}} H_{token}$$  
- 优点：先在点级捕捉局部模式，再汇聚到 token，通常更稳。

**模式 B：token-first（实验开关）**  
- 适用：Transformer/SSM，或对照试验  
- 路径：$$X \xrightarrow{\text{PatchEmbed}} T \xrightarrow{\text{Encoder}} H$$  
- 风险：patch 内细粒度信息在进入 encoder 前被压缩，可能损失预测关键结构。

> 结论：vNext 默认保留 V1 的 TCN 路径；token-first 作为明确的消融对照。

### 5.2 StableFeat（稳定特征用于估图）
我们把 StableFeat 视为“结构学习的抗伪相关视角”，而不是对外部输入的强预处理。两条实现路线：

- **点级 StableFeat（已实现，V2）**：对 $$X$$ 做 detrend/diff 得到 $$X_{stab}$$。  
- **token 级 StableFeat（建议新增）**：对 $$T_{cont}$$ 沿 token 维做移动平均/差分，得到 $$T_{stab}$$。  
  - 直觉：token 级更偏长尺度，更贴合“图在长尺度估”的原则。

> vNext 推荐：先保留点级 StableFeat 作为复现与对照；随后新增 token 级版本用于更稳的估图。

### 5.3 Graph Learner：低秩 + 可选稀疏化 + base/residual
#### (1) 动态残差图（V1）
$$\Delta A_k = \text{RowSoftmax}(S_k).$$

#### (2) Base + Residual（P3）
建议把 base 图显式参数化为 row-stochastic：
$$A_{base} = \text{RowSoftmax}(B).$$

最终图用 convex mixing（保持 row-stochastic）：
$$A_k = (1-\alpha)\Delta A_k + \alpha A_{base},\quad \alpha\in[0,1].$$

- 关键工程约束：**初始化 $$\alpha=0$$**，即可严格退化到 V1。
- 可选：$$\alpha$$ 设为可学习标量（sigmoid 参数化）或训练 schedule（先 0，后逐步升）。

#### (3) 稀疏化（可解释 + 抗噪）
当变量数较大或伪相关明显时，可加入 top-k：
- 对每行只保留 top-k logits，再 softmax；
- 或对 $$A_k$$ 做 hard mask + renorm。

> 注意：稀疏化要与 base 图兼容（例如 base 图也做 top-k，或只对 residual 做 top-k）。

### 5.4 Graph Mixing：从“强注入”到“可控注入”
V1 的 mixing 是：
$$H' = H + A H.$$

vNext 建议提供可选门控（默认关闭）：
$$H' = H + g \odot (A H),$$

可选的 $$g$$：
- **scalar gate**：每层一个标量（最稳）
- **per-variable gate**：$$g\in\mathbb{R}^{B\times C\times 1\times 1}$$
- **per-token gate**：$$g\in\mathbb{R}^{B\times C\times N\times 1}$$（更强但更易过拟合）

门控的输入建议优先用稳定流或图不确定性指标（例如 adjacency entropy）。

### 5.5 图正则（与 V1 兼容的最小集合）
保持 V1 已有的 time-smooth：
$$\mathcal{L}_{smooth}=\sum_k \|A_k - A_{k-1}\|_1.$$

新增建议（后续逐步开关验证）：
- **稀疏/L1**：鼓励更少的有效边（尤其 base 图）
- **熵正则**：控制过于均匀的行（避免“平均搅拌”）
- **度约束**：限制出度过大（可选）

---

## 6. 一句话版本：我们要做什么

1) 保持 V1 的“单流动态图 mixer”作为可复现锚点；  
2) 把 V2 的洞见（P1 stable-for-graph、P3 base+residual、P2 长尺度估图）做成 **逐个可开关** 的增强项；  
3) 每次只打开一个开关做严格消融，保证任何时候都能退回 V1。

---

## 7. 论文/报告友好的命名建议（可选）

为了更容易写论文/汇报，我们可以把 vNext 总体称为：

**BDGM：Bridge Dynamic Graph Mixer**  
- Bridge：结构学习与内容传播分离（P1）  
- Dynamic Graph：按段更新的 $$A_k$$（P2）  
- Mixer：跨变量传播主干（与 V1 一致）

