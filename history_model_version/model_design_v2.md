# DynamicGraphMixer v2 设计文档（model_design_v2.md）

> v2 的目标：在保持当前最优 baseline（ETTm1 96→96，MSE≈0.3334）不被破坏的前提下，**更快、更稳** 地判断“一个新增组件到底有没有用”，并围绕已验证有效的方向继续提升性能与稳定性。  
> 核心主线：**Confidence-Calibrated Graph Propagation**（用“图置信度”校准跨变量传播强度与路由）。

---

## 1. 我们已经确认的事实（来自现有实验与代码结构）

结合你当前的 `DynamicGraphMixer` superset 实现，forward 的关键链路是：

1) `TemporalEncoder`（TCN/Transformer）先做变量内时间编码，得到  
$$h_{time}\in \mathbb{R}^{B\times C\times N\times D}$$

2) 分段（segment）生成动态图：每段用均值池化得到  
$$z_t=\text{MeanPool}(h_{graph}[:,:,seg,:])\in \mathbb{R}^{B\times C\times D}$$  
再用低秩图生成器得到  
$$A_t=\text{RowSoftmax}(U_tV_t^\top/\sqrt{r})\in \mathbb{R}^{B\times C\times C}$$

3) 用图进行跨变量传播（GraphMixer，残差式）  
$$\text{mixed}=A_t\cdot X,\quad Y=X+g\odot \text{Dropout}(\text{mixed})$$

在实验上你已经看到：

- **gate（跨变量传播强度可控）是决定性增益项**：开启 gate 后提升明显。
- **top-k 稀疏化一般带来小幅稳定收益**（属于“微调型增益”）。
- **stable-stream 估图只有在合适的“尺度”上才有效**（尺度不匹配会明显掉点）。
- **token-level stable（对 latent token 做 detrend/diff）在当前实现形态下会明显退化**（先不作为 v2 主线）。
- **patch 在 TCN 场景下目前不占优**（且会改变“动态图更新次数/段数”，导致不可比）。

另外一个重要实现细节：你现在 `base_l1 = mean(|A_base|)` 在 $$A_{base}=\text{RowSoftmax}(\cdot)$$ 的情况下几乎是常数（梯度很弱/无效），因此 v2 必须更换 base 的正则形式。

---

## 2. v2 的核心问题：动态图的“风险”来自传播强度

在你的架构里，动态图的主要风险不是“图估得不够尖锐”，而是：

> 错边一旦被用于传播（残差注入），会把无关变量信息持续注入表征，造成系统性退化。

这解释了为什么 **gate 能显著提升**：它让“跨变量传播”从高风险模块变成可控的增量能力。

因此 v2 的主线非常明确：

1) **让 gate 从静态参数升级为“按图置信度自适应”**（Confidence Gate）  
2) **让 base graph 从静态混合升级为“按置信度路由”**（Confidence Routing）  
3) **修正 base 的正则，使其真正成为稳定骨架，而不是自由噪声参数**

---

## 3. v2 模型提案：CC-DGM（Confidence-Calibrated Dynamic Graph Mixer）

### 3.1 结构总览（保持可退化）

v2 的 forward 仍遵循你现有 superset 的组织方式，只在 “A 的使用策略” 上增强：

1) 生成动态图：得到 $$A_{dyn,t}$$  
2) 计算置信度：得到 $$c_t$$（标量或每变量）  
3) 路由融合：得到最终用于传播的图  
$$A_t = (1-\alpha_t)\,A_{dyn,t} + \alpha_t\,A_{base}$$
4) 传播门控：  
$$Y = X + g_t \odot \left(A_t\cdot X\right)$$  
（其中 $$g_t$$ 由置信度决定）

**可退化要求（强约束）**  
- 关闭 v2 开关时，必须严格退化为你当前最优配置（例如现有的 `gate_mode=scalar`、`alpha` 静态可学习等）。

---

## 4. 图置信度（Confidence）：只用一个“足够强且便宜”的信号

为了让 v2 快速验证有效性，**建议只从 entropy 开始**（够强、计算便宜、可解释）。

给定行归一化邻接 $$A\in \mathbb{R}^{B\times C\times C}$$，定义每个节点的行熵：

$$H_i=-\sum_{j=1}^{C}A_{ij}\log(A_{ij}+\epsilon)$$

归一化后得到 $$[0,1]$$ 的“非置信度”：

$$\tilde H_i = \frac{H_i}{\log C}$$

置信度：

$$c_i = 1-\tilde H_i$$

> 实用建议：置信度最好在 **top-k sparsify 之前** 计算（否则 top-k 会人为降低熵，导致置信度偏高）。

---

## 5. Confidence Gate：用置信度控制“传播强度”

### 5.1 两个足够的门控形式（不做花哨）

#### G0：静态 gate（现有基线）
$$g=\sigma(\theta)$$  
这是你已经验证有效的形式（例如 `scalar gate`）。

#### G1：确定性置信度门控（0 参数，快速验证用）
$$g_i = (c_i)^\gamma,\quad \gamma>0$$  
解释：图越确定（熵越低），传播越强；反之越弱。

#### G2：可学习仿射门控（推荐主线）
$$g_i = \sigma(w\cdot c_i + b)$$  
优点：参数很少、稳定、易消融。

### 5.2 门控粒度（建议只保留两档）
- `conf_scalar`：每段一个 gate（快、稳，先做）
- `conf_per_var`：每变量一个 gate（更强，作为第二步）

### 5.3 训练稳定性：warmup
为避免训练早期“乱图污染表征”，建议提供：

- `gate_warmup_epochs`：前 k 个 epoch 强制 $$g=0$$（等价不混合），之后再启用置信度门控。

---

## 6. Confidence Routing：用置信度决定“用动态图还是用 base”

相比只在特征层面 gate，路由是在结构层面减少错边风险：

### 6.1 标量路由（先做，最稳）
$$\alpha_t=\sigma(w_\alpha\cdot (1-\bar c_t)+b_\alpha)$$  
其中 $$\bar c_t$$ 可取每变量置信度的均值（或中位数）。

最终图：
$$A_t=(1-\alpha_t)A_{dyn,t}+\alpha_t A_{base}$$

解释：
- 图不确定（置信度低）→ $$\alpha_t$$ 大 → 更多依赖 base（稳定骨架）
- 图确定（置信度高）→ $$\alpha_t$$ 小 → 使用动态图（自适应）

### 6.2 重要：v2 先不引入“stable 图三方融合”
稳定图（stable-stream）是一个潜力方向，但它会引入额外超参（window/scale/share/detach），容易拖慢验证速度。  
v2 先把 **Confidence Gate + Routing** 做扎实；稳定图作为 v2.4 的可选扩展。

---

## 7. Base graph：修正正则，使其成为“保守骨架”

### 7.1 Base 的角色定位
- **保守兜底**：当动态图不可信时，至少不把表征搅乱。
- **慢变结构**：允许学习少量稳定跨变量关系，但不应过密、过激进。

### 7.2 推荐的 base 正则（只保留一个主方案，便于快速验证）
最推荐先上一个直观且梯度明确的正则：

#### R-base：向单位阵靠拢（保守兜底）
$$\mathcal{L}_{base}=\|A_{base}-I\|_F^2$$

优点：
- 梯度明确、语义明确（更保守、更稳定）。
- 不会像“强制对角=1”那样绝对刚性，仍允许学到必要的跨变量边。

> 如果后续发现 base 过于保守，再考虑加入 entropy 正则或放松该项。

---

## 8. 配置接口（v2 新增，默认不改变 baseline 行为）

### 8.1 Confidence Gate
- `gate_mode`: `none | scalar | per_var | conf_scalar | conf_per_var`
- `gate_conf_metric`: `entropy`（v2 先只做这个）
- `gate_conf_source`: `pre_sparsify`（默认）
- `gate_conf_map`: `pow | affine`
- `gate_conf_gamma`: (pow)
- `gate_conf_w_init, gate_conf_b_init`: (affine)
- `gate_warmup_epochs`: 默认 0（但实验建议 3）

### 8.2 Confidence Routing
- `graph_base_alpha_mode`: `static | conf`
- `graph_base_alpha_w_init, graph_base_alpha_b_init`

### 8.3 Base 正则
- `graph_base_reg_type`: `none | l2_to_identity`
- `graph_base_reg_lambda`: 例如 `0, 1e-3, 1e-2`

---

## 9. v2 的“快速验证”准则（写在设计里，指导实验）

为了避免实验过密，v2 的验证遵循：

1) **先做 on/off**：组件开 vs 关（最少 1 个对照）  
2) **再做极端点**：只测 2~3 个离散取值（间距大）  
3) **只要不稳定就先加 warmup**：先稳住训练，再谈花样  
4) **单组件通过阈值才进入 3-seed 确认**：  
   - 快速筛选：1 seed（少 epoch）  
   - 通过阈值（例如 MSE 至少改善 0.001）→ 再做 3 seeds

---

## 10. v2 主线贡献点（论文表达友好）

如果 v2 成功，方法叙事可以非常清晰：

- **动态图结构学习 + 置信传播校准**：  
  “learn graph” 之外引入 “calibrate propagation”  
- **结构层路由**：动态图不可信时回退稳定骨架，显著降低伪相关污染风险  
- **可解释性自然增强**：gate/alpha 的曲线直接解释“模型何时相信图”

---
