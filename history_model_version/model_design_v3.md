# DynamicGraphMixer v3 — Stationary-Map Graph + (Optional) EMA Dual-Stream

> 核心目标：突破 v2 在 ETTm1(96→96) 上“靠近 0 的 gate”所暴露出的性能瓶颈，让 **跨变量信息传播真正变得“可用且可控”**，而不是一味把 gate 压到 0。

---

## 0. 背景回顾：v2 暴露出的瓶颈是什么？

**现象**  
- 在 v2 多数最优/次优配置中，`gate_mean` 常常极小（接近 0），说明模型在训练后倾向于“基本不做跨变量传播”，更多依赖 temporal encoder 自己解决问题。  
- `entropy-based confidence gate` 的效果不稳定甚至变差，说明“图变尖锐(低熵)”并不等价于“图更正确”。

**推断**  
- 不是“gate/alpha 设计不够复杂”，而是：**图权重的估计在非平稳场景下仍然容易带来 spurious regression**，导致训练动力学把 gate 压回 0（安全但收益有限）。

---

## 1. v3 核心假设

> **把“图权重（interaction map）”的学习从非平稳值域里拉出来，但传播的 value 仍保留非平稳信息**。

直觉：  
- 非平稳主要体现在局部均值/尺度漂移、趋势变化等；这些会污染 “变量之间相似度/相关性”的估计。  
- 但预测时又不能把趋势/尺度信息完全丢掉（否则会出现系统性偏差）。  

因此我们提出 **Map-Value Decoupling**：

- 用 *stationarized* 表征去估计图（A_t）：更可靠的“关系图”。  
- 用 *original* 表征作为传播的 value：保留趋势/尺度等信息。

这类似于“用更稳的 Q,K 产生 attention map，再乘回原始 V”的范式。

---

## 2. 方法总览：Stationary-Map Graph Propagation (SMGP)

给定 temporal encoder 的输出 token 序列 `H`：

- `H_val = H`：用于传播（value）。
- `H_map = Stationarize(H)`：用于图学习（map）。

然后：

1) **GraphLearner(H_map) → A_t**  
2) **GraphMix(H_val, A_t) → H_out**

### 2.1 Stationarize(·)：v3 计划支持的几种方式（轻量可插拔）

我们只在 *map 分支* 上做 stationarize：

- `none`：退化为 v2（对齐基线）
- `ma_detrend(w)`：滑动平均 detrend（窗口 w）
- `diff1`：一阶差分（pad 回长度 L）
- `ema_detrend(alpha)`：EMA trend + residual（trend=EMA，map 用 residual）

> 经验上（当前实验结果）：
> - **ETTm1**：`ma_detrend(window=16)` 表现最好（优于 ema 系列与 diff1）
> - **Weather**：`ema_detrend(alpha=0.1)` 略优
> 若无先验数据集特性，先用 `ma_detrend(16)`，再试 `ema_detrend(0.3)` 作为对照。

### 2.2 GraphLearner：不改主结构，只改输入

保持 v2 的 low-rank / top-k / row-softmax 等结构，但输入从 `H_val` 换成 `H_map`。

### 2.3 GraphMix：传播 value（H_val）而不是 map

核心仍是残差传播：

- `H_out = H_val + gate ⊙ (A_t @ H_val)`  
- gate 仍可用 scalar / per_var / per_token（与 v2 对齐）

---

## 3. 可选增强：EMA Dual-Stream Decomposition (DS)

如果 SMGP 在 ETTm1 上仍然提升有限（但在大通道数据集更有效），我们再启用 **Dual-Stream**：

- 输入 X → EMA 分解：`X_trend = EMA(X)`, `X_season = X - X_trend`
- trend 走 **线性头**（如 DLinear 风格：Linear(seq_len→pred_len)）
- season 走 **DynamicGraphMixer(SMGP)**（更接近 stationary，利于跨变量建图）
- 最终预测：`Y = Y_trend + Y_season`

> 该模块独立可开关，避免 v3 一开始引入过多自由度。

---

## 4. 关键超参与 CLI 设计（建议）

### 4.1 SMGP 相关
- `--graph_map_norm {none,ma_detrend,diff1,ema_detrend}`
- `--graph_map_window 16`（用于 ma_detrend）
- `--graph_map_alpha 0.3`（用于 ema_detrend）
- `--graph_map_detach 0/1`（可选：是否阻断 map 分支梯度；默认 0）

### 4.2 Dual-Stream 相关（可选）
- `--decomp_mode {none,ema}`
- `--decomp_alpha 0.3`
- `--trend_head {none,linear}`（默认 linear）
- `--trend_head_share 1`（trend head 是否跨变量共享参数；默认 1）

---

## 5. 诊断与可解释性（建议增加日志）

为验证“graph 真正在用”，建议额外打印：

- `gate_mean / sat_low / sat_high`（已有）
- `A_entropy`（已有）
- `A_overlap / adj_diff`（已有）
- **map 分支统计**：`map_std_mean`, `map_mean_abs`
- **trend/season 能量比**（Dual-Stream 时）：`E_trend / E_season`

---

## 6. 最小实验矩阵（单 seed，适中即可）

我们把 v3 分成两步走：

### Step-A：先验证 SMGP 是否能让图更可用
- F0：v2 best（复现）
- F1：SMGP（ETTm1：`ma_detrend(window=16)`；Weather：`ema_detrend(alpha=0.1)`）
- F2：F1 + 更激进的 `gate_init`（比如 -2），看是否能提升并保持稳定  
（F2 建议只在大通道数据集上跑）
### Step-B：如果 Step-A 在小数据集收益有限，再加 Dual-Stream
- F3：Dual-Stream(EMA α=0.3) + season 走 SMGP

数据集建议（实验量适中）：
- 小通道：ETTm1 (C=7) 96→96（快速 sanity & 回归检查）
- 中通道：Weather (C=21) 96→96（检验中等维度）
- 大通道：Traffic 或 Electricity 96→96（验证跨变量传播价值）
---

## 7. 预期结果与成功标准（建议）

- **Traffic/Electricity**：F1 相对 F0 有可见提升（MSE 或 MAE ≥ 0.002 级别），并且 `gate_mean` 不再长期贴近 0。  
- **ETTm1/Weather**：不回退即可；如果能小幅提升更好。  
- 如果 F1 有效但 F2 无效：说明 map 更可靠了，但强传播仍会引入噪声，可考虑更强的正则/平滑而不是继续提 gate。

---

## 8. 贡献点总结（写 paper 的角度）

- Map-Value 解耦：用 stationarized representation 学关系图，但传播保留非平稳 value。  
- 在动态图框架下，将“去非平稳”定位为 **图权重估计问题**，而不是简单做归一化/门控。  
- 可插拔：SMGP 不强绑 backbone；Dual-Stream 作为可选扩展。



