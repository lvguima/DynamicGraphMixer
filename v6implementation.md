
---

## 0. 背景与当前结论（用于统一认知）

### 0.1 目前 B5 baseline 的关键结构点（你后续所有改动都要围绕它“可回退”）

* B5 输出：`y_hat = ForecastGraph(season) + TrendHead(trend)`，且 **Dual-Stream sum 在 B5 永远开启**。
* EMA 分解：`trend_t = alpha * x_t + (1-alpha)*trend_(t-1)`，`season_t = x_t - trend_t`；**season 走图预测，trend 走线性头（并且 trend_head_share=1）**。
* 图学习只用 `content_mean`；SMGP 默认 `ma_detrend(window=16)`；map 分支默认不 detach（梯度会回传）。
* 动态图：按 `graph_scale` 分段、段内 token 均值池化，低秩图学习 `A = softmax(UV^T/sqrt(rank))`；并与 base 图混合 `A=(1-α)A_dyn+αA_base`；top-k 稀疏化固定 `adj_topk=6`；gate 是 per-var，且 **gate_init = -6（非常弱）**。

### 0.2 消融结果给出的“最关键洞见”

1. **图相关组件（SMGP / gate / base graph / sparsify）对多数数据集指标影响极小**：

   * ETTm1：去掉 gate/base/SMGP/sparsify 变化都在 1e-3 量级；甚至 graph-disabled（把图整条链路关掉）还更好（MSE -0.001662）。
   * weather：同样几乎不变，graph-disabled 还略好（MSE -0.000203）。
   * flotation/grinding：图关掉略差/略差（幅度也很小）。
2. **时间编码（TCN）是绝对主力**：把 TCN 换线性（B1_no_TCN_linear）所有数据集显著退化，grinding 退化尤其大。
3. **Dual-Stream 是“看数据集”的**：ETTm1/weather 去掉 Dual-Stream 明显变差；但 flotation 去掉反而更好（-0.016975）。
4. 结合 v3 设计文档里的诊断：v2 里经常出现 `gate_mean` 接近 0，意味着模型倾向于“不做跨变量传播”，并且原因很可能是 **非平稳导致的伪相关（spurious regression）污染了图权重估计**，训练动力学就把 gate 压回 0 来“自保”。

**一句话洞见：** 你现在的模型性能主要来自 **TCN +（部分数据集）Dual-Stream**，而 GNN/GraphMix 之所以消融“几乎没影响”，很可能不是“图没用”，而是“图被模型学会了不用”（gate/α 很小 + 图估计不可靠）。你要提升性能，核心不是再微调 `graph_rank/topk`，而是让“跨变量传播”变成 **可靠（图更准）、可控（错了能退回）、可被激活（gate 不贴 0）**。

---

## 1. 总目标与验收口径（每一步都围绕它）

### 1.1 总目标

把 GNN 这条链路推进到：

* 在 **高维/强跨变量依赖**数据集上能带来**可见收益**（不再是 1e-3 小数点级别抖动），并且
* 日志里能看到：`gate_mean` 不再长期贴近 0，图的熵/一致性等统计具有可解释变化。
---

## 2. 推进路线总览（从低风险到结构性升级）

我建议你按下面 6 步推进（每步都可以独立提交、独立写消融）：

1. **Step 0：基线复现 + 图链路“用没用”诊断（先把问题看清）**
2. **Step 1：低风险“让图更稳”的参数/训练策略（graph_scale / gate_init）**
3. **Step 2：把“可控传播”补齐：warmup + 置信 gate + 置信路由（dyn/base）**
4. **Step 3：把图放到更该放的地方：Trend/低频分支也做跨变量传播**
5. **Step 4：多尺度图（multi-scale graph_scale）做成“免调参鲁棒化”**
6. **Step 5：收敛成 paper-ready 的 v4：完整对照、消融、报告与默认配置**

下面逐步展开。

---

## Step 0：基线复现 + 诊断（必须做）

### 0.1 实施内容

A. **冻结并记录 B5 baseline 配置**（作为所有改动的可回退锚点）

* 确认与你当前 paper-ready 的 B5 一致：TCN、EMA dual-stream、SMGP=ma_detrend(16)、graph_rank=8、graph_scale=8、topk=6（固定）、base mix、gate(per_var, init=-6)。

B. **把“图到底有没有在工作”变成可视化事实**
利用你已有的日志机制（`stats.csv`）：

* 已有：`A_entropy`, `A_overlap/adj_diff`, `gate_mean`, `base_alpha` 等；并会把 MSE/MAE append 到同一个文件里。
* v3 建议额外确认：`map_std_mean`, `map_mean_abs`、以及 `E_trend/E_season`（dual-stream 的能量比）。

> 如果你当前日志已经包含这些项，就直接画图；没有就补齐字段输出（不改训练逻辑，只加 logging）。

### 0.2 实验计划

* 数据集：建议保留你已跑通的 **ETTm1（小通道）、Weather（中通道）、Grinding/Flotation（业务集）**。这些在你的消融里已有对照，便于回归检查。
* 运行：B5 baseline 单 seed（与 sweep 基线一致）+ 打开 `graph_log_interval`。

### 0.3 预期结果（你应该看到什么）

* 如果确实如 v3 推断：

  * `gate_mean` 长期接近 0（或大部分变量接近 0），说明模型在“主动不用图”。
  * `A_entropy` 可能偏高、`adj_diff` 波动大（图不稳/不可信），从而推动 gate 变小（自保机制）。
* 如果 `gate_mean` 不小但指标仍无提升：

  * 说明图在“搅动”，但对预测未提供信息（可能建图源不对、传播对象不对——这会直接指向 Step 3：trend 分支建图/传播）。

### 0.4 通过/失败决策

* **通过（常见）**：`gate_mean` 很小 → 进入 Step 1/2 让图变得“可被启用”。
* **失败（也有价值）**：`gate_mean` 不小但没收益 → 直接跳到 Step 3（把图搬到 trend/低频）。

---

## Step 1：低风险“让图更稳/更可用”的改动（先吃最容易的收益）

你的 sweep 明确说：整体敏感性低，但 `graph_scale` 是最一致的可用旋钮；`graph_rank=4` 风险很大。

### 1.1 实施内容

A. **把 graph_scale 从“固定 8”推进到“更合理的默认/候选”**

* 在 sweep 里：`graph_scale=16` 在 3/7 数据集是最优，并且平均 delta 最好；`graph_scale=4` 在 ETTm1 最好。
* 因此先做最小改动：只开放一个“低风险默认”：

  * 默认候选：`graph_scale=16`（跨数据集更稳）
  * 对 ETTm1 额外跑 `graph_scale=4` 做 sanity（因为它在 sweep 里赢得更明显）。

B. **验证“更激进 gate_init”是否真的能把图激活**
v3 计划里明确建议：在引入 SMGP 后，尝试把 `gate_init` 从很小的 -6 提到 -2（更强传播）来观察能否提升且保持稳定。

* 你当前 B5 gate_init=-6。
* 我建议只做两档：`-6`（基线） vs `-2`（更激进），避免扫太多。

### 1.2 实验计划（最小矩阵）

* F0：B5 baseline（scale=8, gate=-6）
* F1：scale=16（gate=-6）
* F2：scale=16 + gate=-2
* （ETTm1 额外）F1b：scale=4（gate=-6）

### 1.3 预期结果

* 指标：在小/中通道数据集上可能提升很有限（你现有消融就暗示了这一点），但不应明显回退。
* 诊断：

  * scale 变大后，`A_entropy` 可能更低/更稳（段内 pooling 更“长尺度”，关系估计更鲁棒）；
  * gate=-2 后，`gate_mean` 应显著上升（不再贴 0）。

### 1.4 通过/失败决策

* **若 gate=-2 带来训练不稳或明显退化**：不要继续硬提 gate，进入 Step 2（先做 warmup + 置信路由，把风险兜住）。
* **若 scale=16 稳定略优**：把它作为新的“默认 scale”，进入 Step 2/3 做结构升级。
* **若任何设置都几乎不动**：说明需要更强的结构性改变（Step 3/4）。

---

## Step 2：把“可控传播”补齐（让图敢用、用得对）

你现状最缺的是：**当动态图不可信时，模型如何系统性地退回安全方案**。目前只有静态 base mix + 静态可学习 gate，而 gate_init 又极小，很容易“全程不用”。

v2 设计文档给了一套非常工程友好的方案：

* 置信 gate（用图置信度调节传播强度）
* warmup（前 k 个 epoch 强制不传播，避免早期乱图污染）
* confidence routing（用置信度决定更偏 dyn 还是更偏 base）
* base 正则建议改为“向单位阵靠拢”，让 base 成为保守兜底骨架。

### 2.1 实施内容（按优先级）

A. **Gate warmup（训练稳定性保险丝）**

* 增加 `gate_warmup_epochs=k`：前 k 个 epoch 强制 `g=0`，之后再启用门控传播。
* 这一步几乎不会让性能变差（最坏也就是前期等价“无图”），但能显著减少“早期噪声图把表征搞坏”的风险。

B. **Confidence Gate：让传播强弱随“图可信度”变化**

* v2 提供两种映射：

  * `g_i = (c_i)^γ`（0 参数，快速验证）
  * `g_i = σ(w c_i + b)`（推荐主线，参数少且稳）
* 置信度 `c` 的来源：先用你已有的 `A_entropy`（行熵/平均熵）做度量（熵低→更确定）。

  * 注意 v3 的提醒：**“图变尖锐（低熵）不等价于更正确”**，所以 confidence gate 的价值在于“减小错边风险”，而不是宣称熵能代表正确性。

C. **Confidence Routing：在结构层面决定用 dyn 还是用 base**

* v2 的标量路由：

  * `α_t = σ(wα · (1 - c̄_t) + bα)`
  * `A_t = (1-α_t)A_dyn,t + α_t A_base`
  * 图越不确定（置信度低）→ α 越大 → 更多依赖 base（兜底）。

D. **Base graph 正则改造：让 A_base 真正成为“保守骨架”**

* 你当前 loss 只对 A_base 做 L1 稀疏正则（容易学成奇怪的稀疏模式，不一定“保守”）。
* v2 推荐先上一个主方案：`L_base = ||A_base - I||_F^2`，语义非常明确：默认更像单位阵（以自环为主），只在必要时学少量跨变量边。

### 2.2 实验计划（严格 on/off，控制变量）

按 v2 的“快速验证准则”做：先 on/off，再极端点。

* C0：Step1 最优配置（例如 scale=16, gate=-6 或 -2）
* C1：+ warmup（k=3 或 5，任选一个固定值）
* C2：C1 + confidence gate（先用 0 参数 `pow`，再上 affine）
* C3：C2 + confidence routing（α_t 随置信变化）
* C4：C3 + base_reg = l2_to_identity（替换/叠加原 base_l1）

### 2.3 预期结果

* **诊断层面（你几乎必定会看到）**

  * warmup 让早期 `adj_diff` 更可控（不至于一上来乱飞）；
  * confidence routing 让 `base_alpha` 不再是“固定小值”，而是随着图不确定性动态变化（更符合“保守兜底”的设计目标）。
* **指标层面**

  * 小通道（ETTm1）可能不一定提升，但**不应明显回退**；
  * 中/大通道或业务集若存在跨变量依赖，应该开始出现“可见增益”，并且 `gate_mean` 不再长期贴 0（这也是 v3 的成功标准之一）。

### 2.4 通过/失败决策

* 如果 **诊断很好看（gate/α 变得有意义）但指标不动**：说明图“能用了”，但“传播对象/建图源”可能不对 → 进入 Step 3（trend 分支/低频图）。
* 如果 **指标提升但不稳定**：优先调 warmup、base_reg 强度，而不是继续加复杂度。

---

## Step 3：把图放到更该放的地方（Trend/低频也要跨变量）

这一点是我认为你“下一段性能跃迁”的最大机会：
你现在的图传播主要作用在 **season（残差）** 上，而 **trend** 直接走共享线性头（trend_head_share=1）。
但很多跨变量关系（尤其是共涨共跌、共趋势、长期联动）更可能存在于 **低频/趋势** 而不是高频残差里。

### 3.1 实施内容（从最小改动开始）

我建议按 T0→T2 递进，每一步都能回退。

**T0：Trend 分支也加一层 GraphMix（最小可行改动）**

* 流程：trend（或 trend 的 encoder 表征）也走一遍 GraphMix，再喂给 TrendHead。
* 好处：

  * 不改变最终 “season + trend” 的加和结构；
  * 只是把 trend 的跨变量联动显式建模。
* 你只需要复用现有 GraphLearner/GraphMix 模块（或单独一套参数）。

**T1：Trend 分支做 SMGP 的 map-value 解耦建图（更贴合 v3 叙事）**

* v3 的核心叙事是：用 stationarized representation 学 A_t，但传播用 original value（类似更稳的 Q/K 生成 map，再乘回原 V）。
* 你可以对 trend 分支也做同样的 Stationarize(map)（ma_detrend / ema_detrend / diff1），但传播仍用原 trend value。

**T2：季节/趋势分支使用“不同图/不同 base”的双图结构（更强但更重）**

* A_trend 和 A_season 分开学，分别有自己的 base 与 gate；
* 最后仍然按原公式相加（不改输出定义）。

### 3.2 实验计划（最小矩阵）

* D0：Step2 最优（只在 season 做图）
* D1：D0 + T0（trend 也做 GraphMix）
* D2：D1 +（trend 分支也启用 SMGP map/value）

### 3.3 预期结果

* 如果你的数据集存在明显的长期跨变量关系：

  * D1/D2 应该比 D0 更容易出现“稳定的可见增益”，同时 trend 分支的 `gate_mean` 不再贴 0。
* 如果仍然不提升：

  * 说明跨变量关系可能不是“同一尺度/同一类型”，需要 Step 4 的多尺度图去匹配。

---

## Step 4：多尺度图（multi-scale graph_scale）做成“免调参鲁棒化”

你 sweep 的现象非常典型：

* `graph_scale=16` 在多个数据集最好，但 ETTm1 更偏好 `graph_scale=4`；整体对 rank/topk 很不敏感。
  这说明：**跨变量依赖可能存在多尺度**，单一 graph_scale 很难“一把梭”赢所有数据集。

### 4.1 实施内容（推荐的工程实现）

做一个 **Multi-Scale GraphMix**，尺度集合先固定 `{4, 8, 16}`（不扫参）：

* 对每个 scale s：

  * 按 s 分段 → 段内 mean pooling 得到 z_s → 学 A_s → 消息 m_s = A_s @ x
* 融合方式（从稳到强）：

  1. **固定平均**：m = mean_s(m_s)（0 参数，先跑通）
  2. **可学习权重**：m = Σ_s softmax(w)_s · m_s（极少参数，建议主线）
  3. **按变量权重**：w ∈ R^{C×S}（更强，但注意过拟合）

这一步做完，你就能把 “graph_scale 选择题”变成 “模型自己学怎么用尺度”。

### 4.2 实验计划

* MS0：Step3 最优（单尺度）
* MS1：+ multi-scale（固定平均）
* MS2：+ multi-scale（learnable weights）

### 4.3 预期结果

* 指标：相比单尺度，**跨数据集稳定性更好**（少量数据集不再需要特判 scale=4/16）。
* 诊断：不同尺度的权重会呈现“可解释分化”（比如某些数据集偏向长尺度 16，某些偏向短尺度 4），这也是 paper 里很好讲的点。

---

## Step 5：收敛成 v4（paper-ready）并补齐消融叙事

### 5.1 你最终需要交付的“v4 最小故事线”

建议把贡献收敛成 2~3 个清晰点（避免组件太碎）：

* **(C1) Map-Value Decoupling 的 SMGP**：在图权重估计层面做去非平稳，而不是盲目归一化或把 gate 压到 0。
* **(C2) 可控传播（Confidence Gate + Routing + Warmup）**：图不可信时结构层面回退到保守骨架，降低伪相关污染风险。
* **(C3) Trend/Low-frequency Graph**（如果 Step 3 生效）：把跨变量传播放到更有“长期关系”的分支上（这往往是性能跃迁点）。

### 5.2 最终消融表建议（保持小而硬）

以你现有 A/B 消融风格扩展：

* Base：B5（原始）
* * Step1（scale=16）
* * Step2（warmup+conf routing+base reg）
* * Step3（trend graph）
* * Step4（multi-scale graph）

并保留 1~2 个关键 off：

* 关掉 routing（只留 conf gate）
* 关掉 trend graph（只 season graph）

### 5.3 预期结果

* 最终你应该能从“图模块消融几乎没影响”推进到：

  * 在至少一个你关心的大通道/强耦合数据集上出现 ≥0.002 级别的稳定收益（v3 的建议口径）。
  * 日志显示 gate/α 的曲线具有可解释行为（模型何时相信图、何时回退 base）。

---



* [ ] Step0：跑 B5 baseline + 打开图日志，画 `gate_mean/A_entropy/adj_diff/base_alpha` 曲线。
* [ ] Step1：跑 `graph_scale=16`；跑 `gate_init=-2`（至少在一个你认为跨变量强的数据集上）。

* [ ] Step2：实现 `gate_warmup_epochs` + confidence gate + confidence routing + base_reg=l2_to_identity。
* [ ] Step3：trend 分支加入 GraphMix（先 T0）。
* [ ] Step4：multi-scale graph_scale（4/8/16）+ learnable fusion。

---