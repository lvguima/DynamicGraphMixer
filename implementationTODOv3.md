# Implementation TODO v3

> 目标：实现 v3 的 SMGP（Stationary-Map Graph Propagation）为主线；Dual-Stream EMA 分解作为可选第二步。  
> 约束：不做 online adaptation；实验量适中；不做 multi-seeds。

---

## 0. Anchor Baseline（v2 best）

- 先复现 v2 best 的脚本（保持与 exp_resultsv2 里的 best 配置一致）
- 固定 seed（单次即可）

输出至少包含：
- MSE / MAE
- gate_mean / alpha_mean
- ent / overlap / adj_diff

---

## 1. v3.0 — SMGP: Map-Value Decoupling for Graph

### 1.1 代码改动点（建议最小侵入）

**(A) GraphLearner 输入解耦**
- 当前：GraphLearner 用 `H`（value features）构造 A_t  
- v3：GraphLearner 改为用 `H_map = Stationarize(H)` 构造 A_t  
- GraphMix 仍然对 `H_val=H` 做传播

**(B) 新增 Map Normalizer 模块**
- 文件：建议放到 `models/layers/` 或 `models/modules/`
- 接口：
  - `forward(H: Tensor[B,C,L,D]) -> H_map`
- 支持：
  - none
  - ma_detrend(window)
  - diff1
  - ema_detrend(alpha)

**(C) CLI / config**
新增 args：
- `graph_map_norm`
- `graph_map_window`
- `graph_map_alpha`
- `graph_map_detach`

> 默认：`graph_map_norm=none`（与 v2 完全对齐，方便回归测试）

### 1.2 单元测试 / 回归测试（强烈建议）
- 当 `graph_map_norm=none` 时：
  - A_t 计算必须与 v2 完全一致（数值误差 < 1e-6 级别）
- `diff1`：
  - pad 策略一致（前面补 0 或复用第一步差分）
- `ema_detrend`：
  - trend 初值对齐（s0=x0）

---

## 2. v3.1 — 实验：验证 SMGP

> 单 seed，实验量适中即可。

### 2.1 Experiments
- F0：v2 best（复现）
- F1：E0 + `graph_map_norm=ema_detrend`，`graph_map_alpha=0.3`
- F2：E1 + `gate_init=-2`（可选，只在大通道数据集跑）

### 2.2 数据集建议（两档）
- 小通道：ETTm1 96→96
- 大通道：Traffic 或 Electricity 96→96

### 2.3 观察点
- F1 是否能让 `gate_mean` 显著上升（比如从 0.002 变到 >0.01）
- A_entropy 是否更稳定（不是越低越好，而是与性能提升是否相关）
- 大通道数据集是否有更显著收益

---

## 3. v3.2（可选）— Dual-Stream EMA Decomposition

> 仅当 SMGP 在 ETTm1 提升有限、但我们认为瓶颈在“趋势/季节混叠导致图不敢用”时再做。

### 3.1 代码改动点
- 在模型最前加入：
  - `X_trend = EMA(X, alpha)`
  - `X_season = X - X_trend`
- Trend head：
  - 简单 `nn.Linear(seq_len, pred_len)`（共享参数）
- Season branch：
  - 复用现有 DynamicGraphMixer（开启 SMGP）

输出：
- `Y = Y_trend + Y_season`

新增 args：
- `decomp_mode {none,ema}`
- `decomp_alpha`
- `trend_head {none,linear}`
- `trend_head_share`

### 3.2 最小实验
- F3：Dual-Stream(ema α=0.3) + SMGP(ema_detrend α=0.3)

---

## 4. 日志与可解释性（建议）

新增打印：
- map 分支：`map_std_mean`, `map_mean_abs`
- Dual-Stream：`E_trend`, `E_season`, `E_ratio`

---

## 5. 退出条件 / 决策规则（建议）

- 若 F1 在大通道数据集明显提升，优先深挖 SMGP（可能再做 graph smoothing / 正则）。  
- 若 F1 没提升但 Dual-Stream（E3）有效，则主线转向“分解 + season 用图”。  
- 若两者都无效：说明瓶颈不在 graph 权重估计，可能要转向 frequency normalization / probabilistic / Koopman 等更大范式。

