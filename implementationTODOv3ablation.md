# implementationTODOv3ablation.md
> 目标：用**尽量少的实验**回答一个关键问题：在 v3 的 Dual‑Stream 框架下，性能提升主要来自哪里？  
> 方式：只做 3 组 ablation（每组 1 个对照），并且尽量 **不改代码，只改运行参数**。  
> 建议：先只在 **ETTm1 (96→96)** 跑完 4 个实验；如果结论清晰，再可选在 Weather 复核一次。

---

## 0) 先回答你：这 3 组 ablation 要不要改模型？

**结论：基本不需要改模型，直接改运行参数就可以。**  
前提：你现在的 v3 已经支持这些 CLI 开关（你在 v3 文档里也已经列导致这些 args）：
- SMGP：`--graph_map_norm / --graph_map_window / --graph_map_alpha`  
- Dual‑Stream：`--decomp_mode / --decomp_alpha / --trend_head / --trend_head_share`  
- Graph mixing：`--gate_mode / --gate_init`  

> 需要特别注意的一点：**不要用 `gate_mode=none` 来“关闭图传播”**。  
> 在当前 GraphMixer 的实现中，`gate_mode=none` 只是“没有 gate”，但 mixing 仍然是 `x + A@x` 的**全强度传播**。  
> 要想“几乎关闭传播”，最省事的方式是把 `gate_init` 设得很小（很负），让 `sigmoid(gate)` 近似 0。

（如果你想做严格意义上的“完全不计算图且不做传播”的 ablation，可以后续再加一个 `--disable_graph` 开关；但不是必须。）

---

## 1) 实验公共设置（保持不变）

### 1.1 固定训练配置
- **只跑单 seed**（你当前的 v3 约定）
- 训练 epoch / patience / batch_size / lr 等尽量保持与你 v3 最优一致
- **所有 ablation 只改一个因素**（其余参数完全复用 baseline）

### 1.2 统一日志
每次 run 都记录并保留：
- MSE / MAE
- gate_mean
- A_entropy / overlap / adj_diff（如果你已打印）
- Dual‑Stream 的：E_trend / E_season / E_ratio（如果你已打印）

> 如果你当前代码会把图日志写到 `--graph_log_dir`，请给每个实验设置唯一的 `--graph_log_exp_id`，避免覆盖。

---

## 2) Baseline：Dual‑Stream + SMGP（只需要跑 1 次）

这就是你 v3 当前 ETTm1 的 best config（后续 3 组 ablation 都只改一个参数）。

### 2.1 ETTm1 baseline（记为 A0）
（下面命令模板里，除了 ExpID 相关字段，其它尽量与你现有 best 保持一致。）

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id DGmix_v3_DS_SMGP_ETTm1_96_96_A0 \
  --model DynamicGraphMixer \
  --data ETTm1 --root_path ./datasets --data_path ETTm1.csv \
  --features M --target OT --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 128 --d_ff 256 \
  --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 \
  --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 \
  --graph_rank 8 --graph_scale 8 --graph_smooth_lambda 0 \
  --adj_sparsify topk --adj_topk 6 \
  --graph_base_mode mix --graph_base_alpha_init -8 --graph_base_l1 0.001 \
  --gate_mode per_var --gate_init -6 \
  --graph_map_norm ma_detrend --graph_map_window 16 \
  --decomp_mode ema --decomp_alpha 0.1 \
  --trend_head linear --trend_head_share 1 \
  --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 \
  --graph_log_dir ./graph_logs --graph_log_exp_id A0
```

> A0 结果应接近你已记录的 DualStream+SMGP best（MSE≈0.3176, MAE≈0.3579）。  
> 如果差异很大（> 1e-3），先不要跑 ablation，先排查 seed / 数据划分 / 版本代码是否一致。

---

## 3) Ablation‑1：Dual‑Stream 下“图传播是否真的有贡献？”

### 3.1 目的
隔离：**在 Dual‑Stream 里，season 分支的动态图传播是否真的带来提升**，还是主要提升来自 trend head + 分解本身。

### 3.2 做法（不改代码）
保持 A0 所有参数不变，只把 gate 近似置零（传播几乎关闭）：
- `--gate_mode per_var` 不变
- `--gate_init` 改为一个极小值，比如 **-20**（也可 -16/-24，但只测一个就够）

#### A1：DS+SMGP + no‑graph‑prop（近似）
```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id DGmix_v3_DS_SMGP_ETTm1_96_96_A1_noGraphProp \
  --model DynamicGraphMixer \
  --data ETTm1 --root_path ./datasets --data_path ETTm1.csv \
  --features M --target OT --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 128 --d_ff 256 \
  --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 \
  --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 \
  --graph_rank 8 --graph_scale 8 --graph_smooth_lambda 0 \
  --adj_sparsify topk --adj_topk 6 \
  --graph_base_mode mix --graph_base_alpha_init -8 --graph_base_l1 0.001 \
  --gate_mode per_var --gate_init -20 \
  --graph_map_norm ma_detrend --graph_map_window 16 \
  --decomp_mode ema --decomp_alpha 0.1 \
  --trend_head linear --trend_head_share 1 \
  --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 \
  --graph_log_dir ./graph_logs --graph_log_exp_id A1
```

### 3.3 判读
- 如果 A1 ≈ A0（差 < 1e-3）：说明 **Dual‑Stream 的主要增益来自分解/趋势头**，图传播贡献很小（至少在 ETTm1 上）。
- 如果 A1 明显更差：说明 **图传播对 season 分支有正贡献**（值得继续研究图/传播强度）。
- 同时看 `gate_mean`：A1 理论上应接近 0（~ 2e‑9 量级）。

---

## 4) Ablation‑2：Dual‑Stream 下“SMGP 是否必要？”

### 4.1 目的
隔离：**图更稳（SMGP）到底有没有带来实际预测收益**，还是 Dual‑Stream 本身已经足够把 season 做得接近平稳，使 SMGP 变成可有可无。

### 4.2 做法（不改代码）
保持 A0 参数不变，只把：
- `--graph_map_norm` 从 `ma_detrend` 改成 `none`（退化回 v2 的 map/value 同源）

#### A2：DS + graph_map_norm=none
```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id DGmix_v3_DS_noSMGP_ETTm1_96_96_A2 \
  --model DynamicGraphMixer \
  --data ETTm1 --root_path ./datasets --data_path ETTm1.csv \
  --features M --target OT --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 128 --d_ff 256 \
  --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 \
  --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 \
  --graph_rank 8 --graph_scale 8 --graph_smooth_lambda 0 \
  --adj_sparsify topk --adj_topk 6 \
  --graph_base_mode mix --graph_base_alpha_init -8 --graph_base_l1 0.001 \
  --gate_mode per_var --gate_init -6 \
  --graph_map_norm none \
  --decomp_mode ema --decomp_alpha 0.1 \
  --trend_head linear --trend_head_share 1 \
  --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 \
  --graph_log_dir ./graph_logs --graph_log_exp_id A2
```

### 4.3 判读
- A2 明显更差：SMGP 在 Dual‑Stream 中仍然重要（说明“图估计”仍是瓶颈的一部分）。
- A2 与 A0 接近：说明 Dual‑Stream 已经足够“去非平稳”，SMGP 只是轻微正则/更稳但收益小。
- 重点看图统计：A2 的 `adj_diff` 是否明显上升、entropy 是否下降，以及性能是否同步变化。

---

## 5) Ablation‑3：Dual‑Stream 下“更强传播是否可行？”（gate_init 只测 1 个更强点）

### 5.1 目的
验证一个关键假设：  
> Dual‑Stream 把 season 变得更接近平稳后，是否允许把 gate 从初始化附近抬起来（例如从 -6 → -4）而不引入明显负效应？

这直接回答你在 v3 设计里提出的核心目标：“让跨变量传播变得可用且可控”。

### 5.2 做法（不改代码）
保持 A0 参数不变，只把：
- `--gate_init -6` 改为 `--gate_init -4`

#### A3：DS+SMGP + stronger gate
```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id DGmix_v3_DS_SMGP_gate-4_ETTm1_96_96_A3 \
  --model DynamicGraphMixer \
  --data ETTm1 --root_path ./datasets --data_path ETTm1.csv \
  --features M --target OT --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 128 --d_ff 256 \
  --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 \
  --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 \
  --graph_rank 8 --graph_scale 8 --graph_smooth_lambda 0 \
  --adj_sparsify topk --adj_topk 6 \
  --graph_base_mode mix --graph_base_alpha_init -8 --graph_base_l1 0.001 \
  --gate_mode per_var --gate_init -4 \
  --graph_map_norm ma_detrend --graph_map_window 16 \
  --decomp_mode ema --decomp_alpha 0.1 \
  --trend_head linear --trend_head_share 1 \
  --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 \
  --graph_log_dir ./graph_logs --graph_log_exp_id A3
```

### 5.3 判读
- 如果 A3 优于 A0：说明 **Dual‑Stream + SMGP 已经把图/传播“驯化到可以更强注入”**，这条路线值得继续深挖（例如再尝试 -3 或引入 confidence gate）。
- 如果 A3 明显更差：说明图仍不够可靠，强传播仍会放大错误边；下一步优先不是继续增大 gate，而是改“图的可靠性”（例如更强的结构先验/更合理的 top‑k、或更强的分解尺度）。

---

## 6) 结果记录模板（建议你直接贴到 exp_resultsv3.md 末尾）

| ExpID | Dataset | Variant | Key change vs A0 | MSE | MAE | gate_mean | ent | overlap | adj_diff | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A0 | ETTm1 | DS+SMGP | baseline | 0.316922 | 0.357374 | 0.002444 | 1.790298 | 0.859375 | 0.037647 | |
| A1 | ETTm1 | DS+SMGP | gate_init=-20 (≈no graph prop) | 0.317695 | 0.357663 | 0.000000 | 1.791756 | 0.852557 | 0.035835 | |
| A2 | ETTm1 | DS | graph_map_norm=none (no SMGP) | 0.317203 | 0.357607 | 0.002589 | 1.471901 | 0.855154 | 0.096311 | |
| A3 | ETTm1 | DS+SMGP | gate_init=-4 (stronger) | 0.317263 | 0.357626 | 0.017667 | 1.790259 | 0.858888 | 0.037957 | |
| A0 | Weather | DS+SMGP | baseline | 0.172784 | 0.230185 | 0.002489 | 1.791758 | 0.339191 | 0.060142 | |
| A1 | Weather | DS+SMGP | gate_init=-20 (≈no graph prop) | 0.172877 | 0.230136 | 0.000000 | 1.791716 | 0.338975 | 0.059616 | |
| A2 | Weather | DS | graph_map_norm=none (no SMGP) | 0.172877 | 0.230142 | 0.002493 | 1.790260 | 0.522781 | 0.043244 | |
| A3 | Weather | DS+SMGP | gate_init=-4 (stronger) | 0.173066 | 0.230903 | 0.017852 | 1.791758 | 0.327462 | 0.061329 | |

---

## 7) （可选）Weather 复核（同样 4 个实验）

如果你想确认结论在中等通道数据上是否一致，可以按 A0/A1/A2/A3 原样再跑一遍 Weather：
- A0w：graph_map_norm=ema_detrend, graph_map_alpha=0.1
- 其它只改对应 ablation 的那一个参数

> Weather 的 best config 在你记录里也明确是：`decomp_alpha=0.1, share=1, graph_map_norm=ema_detrend(alpha=0.1), gate_init=-6`。

---

## 8) 你跑完后把什么发我（越省事越好）

你只需要把下面两样发我即可：
1) 上面那张表填好的 4 行（A0/A1/A2/A3）  
2) 每个实验对应的 gate_mean / ent / adj_diff（你日志里应该能直接看到）

我就能非常清楚地给出：
- Dual‑Stream 的主收益来源（分解 vs 图传播）
- SMGP 在 DS 中是否必要
- DS 是否真的让“更强传播”变得可行（gate 能不能抬起来）

