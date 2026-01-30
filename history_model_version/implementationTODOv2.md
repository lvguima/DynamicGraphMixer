# DynamicGraphMixer v2 实施与实验 TODO（implementationTODOv2.md）

> 这份 TODO 按“尽快验证组件是否有效”的原则重写：  
> - 去掉过密网格；每步只保留 **最关键的对照** 与 **少量间距较大的参数点**。  
> - 每个里程碑默认只跑 **1 seed + 少 epoch** 做筛选；达到阈值再做 **3 seeds** 严格确认。  
> - 每次只改一个关键开关，确保可回退。

---

## Anchor Baseline（必须固定）

以你当前最优配置作为锚点（示例：ETTm1 96→96，MSE≈0.3334）：

- `graph_source=content_mean`
- `graph_scale=8`
- `graph_rank=8`
- `graph_base_mode=mix`，`graph_base_alpha_init=-8`（保持你目前习惯的“早期几乎不依赖 base”）
- `adj_sparsify=topk`，`adj_topk=6`
- `gate_mode=scalar`，`gate_init=-4`

> v2 的所有新增开关关闭后，必须复现此 baseline（误差≤0.002）。

---

## 0) v2.0：补齐观测 + 冻结基线（只做必要工作）

### 0.1 代码任务（只加“必需日志”）
- [ ] 记录 `A_raw`（sparsify 前）的 entropy（均值/分位数）
- [ ] 记录 `A`（sparsify 后）的 entropy（均值/分位数）
- [ ] 记录 `adj_diff = mean(|A_t - A_{t-1}|)`（段间变化）
- [ ] 记录 gate/alpha：`mean/std/p50/p90`

> 验收：只加 detach 的统计，不应改变指标。

### 0.2 实验（很少）
- [ ] **E0-1**：Anchor baseline 复现（1 seed）
- [ ] **E0-2**：Anchor baseline 3 seeds（用于后续判断“真提升”）

---

## 1) v2.1：静态 gate 的“最小确认”（只做 2~3 个对照）

> 目的：确认 v2 后续对比基准（static gate）选哪一种更稳。  
> 不做大扫：因为你已经证明 scalar gate 有效。

### 1.1 实验（1 seed 快筛）
固定其它参数 = Anchor baseline。

- [ ] **E1-1**：`gate_mode=none`（验证 gate 的必要性）
- [ ] **E1-2**：`gate_mode=scalar, gate_init=-4`（Anchor baseline）
- [ ] **E1-3**：`gate_mode=per_var, gate_init=-4`（只做这一档粒度扩展）

> 通过条件：若 `per_var` 比 `scalar` 明显更好（例如 MSE 改善 ≥0.001），后续 v2 以 per_var 作为静态对照；否则继续用 scalar。

### 1.2 严格确认（可选）
- [ ] 对 E1-2 与 E1-3 的更优者跑 3 seeds，得到 mean±std。

---

## 2) v2.2：Confidence Gate（主线 1）——少量实验快速判断“值不值”

> 核心问题：**动态置信度门控是否能稳定超过最佳静态 gate？**  
> v2.2 只做 entropy 置信度（不引入更多 conf 信号，避免复杂化）。

### 2.1 代码任务（最小实现）
新增 gate 模式（保持可退化）：
- [ ] `gate_mode=conf_scalar`：每段 1 个 gate
- [ ] `gate_mode=conf_per_var`：每变量 1 个 gate

置信度：
$$c_i = 1 - \frac{H(A_{i,:})}{\log C}$$

映射（只保留两种）：
- [ ] `gate_conf_map=pow`：$$g=c^\gamma$$（0 参数）
- [ ] `gate_conf_map=affine`：$$g=\sigma(wc+b)$$（推荐）

训练稳定性：
- [ ] `gate_warmup_epochs`：实现前 k 个 epoch 固定 `g=0`（默认 0）

### 2.2 实验设计（筛选阶段：1 seed，epoch 可减半）
固定其它参数 = v2.1 的最佳静态 gate 配置（scalar 或 per_var）。

只做 4 个关键对照：

- [ ] **E2-1（核心）**：`conf_scalar + affine`，`warmup=3`
- [ ] **E2-2**：`conf_per_var + affine`，`warmup=3`
- [ ] **E2-3**：`conf_per_var + pow(gamma=2)`，`warmup=3`
- [ ] **E2-4（稳定性检查）**：对 E2-2 的配置做 `warmup=0`（看是否训练不稳/掉点）

> 参数间距说明：pow 只测一个中等值 `gamma=2`（先不扫）；warmup 只测 `{0,3}` 两点。

### 2.3 通过条件（决定是否进入 v2.3）
满足任一即可进入 v2.3：
- 性能：相比最佳静态 gate，MSE 改善 ≥0.001（1 seed 下就能看到趋势）
- 稳定性：gate 不坍缩到全 0，且 `entropy↑ => gate↓` 的相关趋势在日志里可见

### 2.4 严格确认（3 seeds）
- [ ] 只对 **最优的 1 个 conf gate 配置** 跑 3 seeds，报告 mean±std，并与静态 gate 对照。

---

## 3) v2.3：Confidence Routing + Base 正则（主线 2）——结构层减少错边风险

> 目的：当动态图不可信时，**在结构层面回退到 base graph**，而不是只靠 gate 在特征层硬压。  
> 同时修正 base_reg（当前 base_l1 近似无效）。

### 3.1 代码任务（最小实现）
- [ ] `graph_base_alpha_mode=conf`：  
  $$\alpha_t=\sigma(w_\alpha(1-\bar c_t)+b_\alpha)$$  
  $$A_t=(1-\alpha_t)A_{dyn,t}+\alpha_t A_{base}$$
- [ ] `graph_base_reg_type=l2_to_identity`：  
  $$\mathcal{L}_{base}=\|A_{base}-I\|_F^2$$  
  并支持 `graph_base_reg_lambda`

> 默认 `graph_base_alpha_mode=static`、`graph_base_reg_lambda=0`，保证可退化。

### 3.2 实验（筛选阶段：1 seed）
固定其它参数 = v2.2 的最优 conf gate 配置。

只做“3+2”个实验：

#### (A) routing 是否值得做（3 个）
- [ ] **E3-1**：只用 conf gate（关闭 conf alpha）【对照】
- [ ] **E3-2**：只用 conf alpha（用静态 gate 对照 v2.1 最优）【隔离 routing 的贡献】
- [ ] **E3-3**：conf gate + conf alpha（两者同时开）【预期最优】

#### (B) base_reg 强度（2 个，间距大）
在 E3-3 基础上：
- [ ] **E3-4**：`graph_base_reg_lambda=1e-3`
- [ ] **E3-5**：`graph_base_reg_lambda=1e-2`

> 说明：先不测更多值。若 1e-2 明显过强（base 退化成接近 I），再补一个 3e-3 即可。

### 3.3 通过条件
- E3-3 相比 E3-1（只 gate）有稳定提升，或至少 **方差更小/训练更稳**
- base_reg 不导致明显掉点，同时 base 的对角占比上升、过密边减少（从日志可见）

### 3.4 严格确认（3 seeds）
- [ ] 只对 “routing 最优 + base_reg 最优” 这一组做 3 seeds，输出 mean±std。

---

## 4) v2.4（可选）：把 stable-stream 当“低频结构骨架”（只做极少实验）

> 这一步不是 v2 必做项。只有当 v2.3 已经稳定、且你怀疑“结构还不够稳健”时再做。  
> 核心点：stable-stream 只在 **更粗尺度** 上估图，避免高频伪相关。

### 4.1 实现建议（最小改动）
- 生成一张稳定图 `A_stable`（用 stable_stream，且 `graph_scale_stable` 更大）
- 把 `A_stable` 当“另一种 base”参与 routing（先不做三方复杂融合）：
  $$A_t=(1-\alpha_t)A_{dyn,t}+\alpha_t A_{stable}$$

### 4.2 实验（1 seed，2 个点就够）
基于 v2.3 最优配置：

- [ ] **E4-1**：`A_stable`：detrend，`stable_window=16`，`graph_scale_stable=16`
- [ ] **E4-2**：同上但 `graph_scale_stable=32`

> 通过条件：若任一配置相对 v2.3 改善 ≥0.001 或显著降低方差，再考虑做更细的 stable_window 微调；否则停。

---

## 5) v2.5（研究向可选）：Lag-aware Mixing 的“1 次试探”

> 只有当 v2.3/v2.4 已经接近饱和，且你希望进一步贴近工业滞后耦合时再做。  
> 先用最简单可控版本验证“方向是否有价值”。

### 5.1 最小实验（1 seed）
- [ ] **E5-1**：`L_lag=2`，共享同一张图 $$A_t$$，学习 softmax 权重 $$w_0,w_1$$  
  $$\text{mixed}_t=w_0(A_tX_t)+w_1(A_tX_{t-1})$$
- [ ] **E5-2**：若 E5-1 有收益，再试 `L_lag=4`（不再扫更多）

---

## 6) 你每一步“该不该继续”的决策规则（避免无意义加密实验）

- 若某里程碑最优配置在 1 seed 下都 **没有改善趋势**（<0.001），先停止并回到上一层分析日志（看是 gate/alpha 全 0、还是 entropy 过高、还是图过于均匀）。
- 只有当 1 seed 下出现稳定优势，再进入 3-seed 严格确认。
- v2 的主线优先级：  
  **v2.2（conf gate） > v2.3（conf routing + base_reg） > v2.4（stable 低频骨架） > v2.5（滞后图）**

---

## 记录格式建议（保持轻量但可追溯）

- 每次实验保存：命令行 / config、metrics、graph_stats（entropy/gate/alpha）
- 每个里程碑只保留一个“冠军配置”，进入下一步，避免配置爆炸

---
