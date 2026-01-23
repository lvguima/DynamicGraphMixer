# Implementation TODO：以 V1 为 Baseline 的逐步推进路线（版本化 + 实验化）

> 目标：把当前 V1 作为性能锚点（anchor），用“**一次只改一个关键因素**”的方式推进到 vNext（BDGM superset），并为每一步建立清晰的版本与实验记录。

---

## 0. 统一规范（先做，避免后面返工）

### 0.1 实验可复现规范
- 固定随机种子：`seed`, `torch.backends.cudnn.deterministic`, `benchmark=False`
- 记录关键超参：`seq_len/pred_len/enc_in/d_model/e_layers/dropout/graph_scale/graph_rank/use_patch/patch_len/patch_stride`
- 保存：训练日志、验证指标曲线、最终 test 指标、最佳 epoch、耗时、显存

### 0.2 版本管理规范（建议）
- `v1.0`：当前 V1 原样（可复现）
- `v1.1`：仅增加日志/可视化（模型行为不变）
- `v1.2`：仅重构模块（保持数值一致）
- `v1.3+`：每引入一个新机制 + 对应消融实验

每个版本都必须满足：
- **能够通过配置完整退化到 v1.0 行为**
- “退化配置”在同一随机种子下指标差异处于可接受浮动范围（建议先允许极小浮动，再逐步做到 bitwise/近似一致）

---

## 1. Milestone 1：锁定 V1 baseline（必须完成）

### 1.1 复现实验（ETTm1 / 你当前对比设置）
- 目标：在当前代码与训练脚本上复现你已有的 V1 指标（例如你在 `analysis.md` 中记录的那组 MSE/MAE）。
- 输出物：
  - `configs/v1_baseline.yaml`
  - `runs/v1_baseline/{seed}/metrics.json`
  - 训练曲线图（loss、val mse/mae）

### 1.2 最小单元测试（强烈建议）
给这些函数写 shape test（不需要大框架，pytest 也行）：
- `TemporalEncoder` 输入/输出形状（TCN / Transformer / SSM）
- `_apply_patch_pooling` 的 token 数是否正确
- `_mix_segments` 的拼接 token 数是否等于输入
- `LowRankGraphGenerator` 输出是否 row-stochastic：`adj.sum(-1)≈1`

验收标准：
- 同一 config、同一 seed，V1 输出形状与指标稳定。
- 测试通过。

---

## 2. Milestone 2：只加“观测能力”，不改模型（v1.1）

> 目的：我们需要知道动态图到底学成什么样，否则后续改进像盲人摸象。

### 2.1 增加图统计日志
每个 epoch（或每 N step）记录：
- adjacency 行熵：$$H(A_i)=-\sum_j A_{ij}\log A_{ij}$$（越低越尖锐）
- top-k mass：每行 top-k 权重和（衡量稀疏性）
- `||A_t - A_{t-1}||_1`（对应 smooth reg 的量级）
- adjacency 的均值/方差、最大边权的分布

### 2.2 可视化（离线保存即可）
- 随机抽样 batch 的若干个 segment，画 heatmap（或保存 numpy）
- 画每个变量的“主要邻居”（top-k 边）

验收标准：
- 日志/可视化可以稳定生成，不影响训练。
- 不改变模型数值行为（同 config 指标基本一致）。

---

## 3. Milestone 3：模块化重构为 “Superset 外壳”（v1.2，行为不变）

> 目的：把 v1 的逻辑拆成可插拔组件，但**默认配置不变**。

建议拆分文件（示例）：
- `modules/tokenizer.py`：负责 patch 逻辑（encode-first pooling / token-first embedding）
- `modules/temporal.py`：TemporalEncoderWrapper（统一输出 `[B,C,N,D]`）
- `modules/graph_learner.py`：LowRankGraphLearner（输出 adjacency）
- `modules/mixer.py`：GraphMixer（线性 mixing / gated mixing）
- `modules/head.py`：ForecastHead

验收标准：
- `v1.2` 在退化配置下与 `v1.0` 指标对齐（允许很小浮动）。
- 代码更清晰，后续增强只需替换/打开模块开关。

---

## 4. Milestone 4：引入 P1（stable-for-graph）但先做“最安全版本”（v1.3）

> 关键原则：**先消除梯度冲突风险，再谈效果**。

### 4.1 先把 stable 流接入，但默认不启用
新增 config：
- `graph_source: content_mean | stable_stream`
- `stable_feat_type: none | detrend | diff`
- `stable_share_encoder: false`（默认 false）
- `stable_detach: false`（默认 false）

默认配置：
- `graph_source=content_mean`（完全等价 V1）
- stable 流即便存在也不参与图生成

### 4.2 实验 4A：stable_stream + identity（只改变“用哪条流估图”）
对照组：
1) V1：`graph_source=content_mean`
2) stable_stream：`graph_source=stable_stream, stable_feat_type=none, stable_share_encoder=false`

观察点：
- 指标是否提升/下降
- adjacency 熵是否变化（是否更尖锐/更均匀）
- 是否出现训练不稳定（loss 抖动）

### 4.3 实验 4B：StableFeat 扫描（detrend/diff + window）
在 `stable_stream` 下扫描：
- detrend window：`3, 5, 9, patch_len`
- diff：是否需要额外平滑（先不加）

验收标准：
- 至少找到一个 stable 设置 **不显著差于 V1**（否则说明当前 stable 定义不适配，需要改 stable 的“层级/尺度”）。

---

## 5. Milestone 5：把 StableFeat 从“点级”推进到“token 级”（v1.4）

> 动机：token 级更贴合“长尺度估图”的 anti-spurious 原则。

### 5.1 新增 `StableFeatToken`
输入：`T_cont: [B,C,N,D]`  
输出：`T_stab: [B,C,N,D]`

建议先实现两种：
- `token_detrend`：沿 N 做移动平均再相减（对每个 embedding dim）
- `token_diff`：沿 N 做一阶差分 + 前补零

### 5.2 实验 5A：点级 vs token 级 stable 的对照
固定 `graph_scale`、encoder、patch 设置：
- 点级 stable：`X -> StableFeat -> Encoder -> z`
- token 级 stable：`X -> (Encoder/Tokenizer) -> StableFeatToken -> z`

验收标准：
- token 级 stable 至少在一个设置上优于点级 stable，或表现更稳（方差更小、训练更平滑）。

---

## 6. Milestone 6：引入 P3（base + residual graph）但保证“可退化”（v1.5）

### 6.1 Base 图的参数化（推荐）
- base logits：`B`，base adjacency：`A_base = RowSoftmax(B)`
- 最终图：$$A_k=(1-\alpha)\Delta A_k+\alpha A_{base}$$
- 其中 $$\alpha=\sigma(a)$$，并把 `a` 初始化成一个很小值（例如对应 $$\alpha\approx 0$$）。

这样：
- `alpha=0` 时严格等价 V1
- 训练后模型可自动学到 base/residual 的分工

### 6.2 实验 6A：学习 base 是否有益
对照：
- no-base（V1）
- base-mix（alpha learnable，init≈0）
- base-mix + L1(base)（鼓励稀疏）

观察点：
- base 图是否收敛到稳定结构（可解释）
- residual 图是否更“专注于变化”
- 指标与稳定性变化

验收标准：
- base 引入后不导致系统性退化；若退化，优先检查：
  - alpha 是否过快变大（需要 schedule/约束）
  - base 是否过密（需要稀疏或 top-k）

---

## 7. Milestone 7：引入“可控传播”（Gated Mixing）（v1.6）

> 动机：当图估错时，V1 的强注入会污染表征。门控可以把风险变成可学习的强度。

### 7.1 实现建议（从最简单开始）
- `gate_mode=scalar`：每层一个标量 gate（sigmoid）
- 公式：$$H' = H + g\,(A H)$$

后续再升级：
- `gate_mode=per_var` 或 `per_token`
- gate 的输入可用 stable 流或 adjacency 熵

### 7.2 实验 7A：gate 是否提升鲁棒性
对照：
- baseline mixing（V1）
- scalar gate（init g≈1 或更小，两种都试）

验收标准：
- 至少在“容易掉点”的配置下 gate 能减少退化幅度（鲁棒性提升）。

---

## 8. Milestone 8：稀疏化（Top-k）与 anti-spurious 组合（v1.7）

### 8.1 实现
- `adj_sparsify=topk`
- `k in {4, 8, 16}`

### 8.2 实验矩阵（建议）
固定 encoder 与 patch，扫：
- graph_scale（更新频率）：`1, 2, 4, 8`
- topk：`none, 8, 16`
- stable：`content_mean vs token_stable`

验收标准：
- 找到一个组合能显著提升或至少稳定超过 V1；
- 同时 adjacency 更稀疏、解释更直观。

---

## 9. Milestone 9：Patch 路径消融（v1.8）

> 目的：把 V1 的 encode-first 与 V2 的 token-first 作为可控开关，明确到底谁更适合你的数据/encoder。

实现开关：
- `patch_mode=v1_encode_then_pool`（默认）
- `patch_mode=token_first`

实验：
- 在 TCN 下重点对照（因为差异最大）
- 在 Transformer/SSM 下对照（可能 token-first 更自然）

验收标准：
- 形成明确结论：在你的设置里，patch 应该放在 encoder 前还是后。

---

## 10. Milestone 10：更强结构（可选，最后做）

### 10.1 交替堆叠（Temporal ↔ Graph）
把“TemporalEncoder → GraphMix”做成 block 并堆叠多次，而不是只在末端做一次跨变量 mixing。

### 10.2 多时滞图（P4）
实现思路（简化版）：
- 生成 $$A_{k,\ell}$$（少量 lag，如 1~3）
- mixing：对 token 维做 shift，再用对应 lag 的图传播后求和

这些属于后期增强项，建议在前面 P1/P3/P2 跑通并稳定后再做。

---

## 11. 最终交付物（你后面写论文/报告会用到）

- 一套统一模型（superset），可通过 config 复现：
  - V1 anchor
  - P1 stable-for-graph
  - P3 base+residual
  - anti-spurious（长尺度 + top-k）
  - gated mixing
- 一张消融表：每次只打开一个开关
- 图可视化与解释性案例（base 图 vs residual 图随时间变化）

