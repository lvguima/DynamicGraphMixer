# exp_resultsv3 (template)

> 约定：单 seed 即可（用户要求不做 multi-seeds）。  
> 指标：MSE / MAE（与 v2 对齐），并记录 gate/alpha/entropy 等图统计用于诊断。

锚点：python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 
--enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 4 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_log_exp_id E0-1 --graph_source content_mean --stable_level point --stable_feat_type detrend --stable_window 16 --graph_base_mode mix --graph_base_alpha_init -8 --graph_base_l1 0.001 --gate_mode scalar --gate_init -4 --adj_sparsify topk --adj_topk 4

mse:0.3333682119846344, mae:0.36996880173683167

---

## Common Setup
- task: long_term_forecast
- seq_len / pred_len: 96 / 96（先对齐快速实验）
- 训练超参：尽量与 v2 best 对齐
- 记录：MSE, MAE, gate_mean, alpha_mean, ent, overlap, adj_diff

---

## Table Columns
| ExpID | Dataset | seq→pred | Variant | Key Args | MSE | MAE | gate_mean | alpha_mean | ent | overlap | adj_diff | Notes |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|

---

## Step-A: SMGP (Stationary-Map Graph)

| ExpID | Dataset | seq→pred | Variant | Key Args | MSE | MAE | gate_mean | alpha_mean | ent | overlap | adj_diff | Notes |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F0 | ETTm1 | 96→96 | v2_best | gate=per_var(-6), graph_scale=8 | 0.333418 | 0.369808 | 0.002545 | 0.000312 | 1.177287 | 0.891315 | 0.126304 | baseline |
| F1 | ETTm1 | 96→96 | SMGP | ema_detrend alpha=0.1 | 0.333550 | 0.369766 | 0.002520 | 0.000329 | 1.554796 | 0.868588 | 0.096559 | slightly worse vs F0 |
| F1 | ETTm1 | 96→96 | SMGP | ema_detrend alpha=0.3 | 0.333121 | 0.369539 | 0.002512 | 0.000333 | 1.737902 | 0.856331 | 0.064133 | best among ETTm1 SMGP |
| F1 | ETTm1 | 96→96 | SMGP | ema_detrend alpha=0.5 | 0.333334 | 0.369575 | 0.002506 | 0.000336 | 1.790040 | 0.855235 | 0.039008 | small gain, more smoothing |
| F1 | ETTm1 | 96→96 | SMGP | diff1 | 0.333347 | 0.369688 | 0.002503 | 0.000336 | 1.790132 | 0.853774 | 0.039205 | similar to ema a0.5 |
| F1 | ETTm1 | 96→96 | SMGP | ma_detrend window=16 | 0.332923 | 0.369483 | 0.002509 | 0.000335 | 1.778012 | 0.860390 | 0.046645 | best overall on ETTm1 |
| F1 | Weather | 96→96 | SMGP | ema_detrend alpha=0.1 | 0.182167 | 0.240396 | 0.002545 | 0.000336 | 1.791359 | 0.438799 | 0.051301 | best on Weather |
| F1 | Weather | 96→96 | SMGP | ema_detrend alpha=0.3 | 0.182395 | 0.240501 | 0.002565 | 0.000336 | 1.791755 | 0.341274 | 0.059749 | slightly worse |
| F1 | Weather | 96→96 | SMGP | ema_detrend alpha=0.5 | 0.182253 | 0.240390 | 0.002569 | 0.000336 | 1.791759 | 0.330141 | 0.060793 | similar to a0.1 |
| F1 | Weather | 96→96 | SMGP | diff1 | 0.182353 | 0.240450 | 0.002567 | 0.000336 | 1.791759 | 0.311418 | 0.062492 | worst overlap |
| F1 | Weather | 96→96 | SMGP | ma_detrend window=16 | 0.182485 | 0.240459 | 0.002591 | 0.000336 | 1.791755 | 0.348945 | 0.059308 | slightly worse |
| F0 | Traffic | 96→96 | v2_best | (copy v2 best args) |  |  |  |  |  |  |  | baseline |
| F1 | Traffic | 96→96 | SMGP | graph_map_norm=ema_detrend, graph_map_alpha=0.3 |  |  |  |  |  |  |  |  |

### (Optional) Stronger propagation
| ExpID | Dataset | seq→pred | Variant | Key Args | MSE | MAE | gate_mean | alpha_mean | ent | overlap | adj_diff | Notes |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F2 | ETTm1 | 96→96 | SMGP+strong_gate | F1 + gate_init=-2 | 0.336202 | 0.374268 | 0.114171 | 0.000370 | 1.750082 | 0.845495 | 0.057989 | worse on ETTm1 (gate too strong) |

---

## Step-B (Optional): Dual-Stream EMA

| ExpID | Dataset | seq→pred | Variant | Key Args | MSE | MAE | gate_mean | alpha_mean | ent | overlap | adj_diff | Notes |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F3 | ETTm1 | 96→96 | DualStream+SMGP | decomp_alpha=0.1 + ma_detrend(16) | 0.316988 | 0.357506 | 0.002444 | 0.000335 | 1.790245 | 0.859050 | 0.037898 | best in ETTm1, trend ratio≈0.879 |
| F3 | ETTm1 | 96→96 | DualStream+SMGP | decomp_alpha=0.3 + ma_detrend(16) | 0.326613 | 0.362494 | 0.002469 | 0.000329 | 1.731072 | 0.858157 | 0.055875 | trend ratio≈0.967 |
| F3 | ETTm1 | 96→96 | DualStream+SMGP | decomp_alpha=0.5 + ma_detrend(16) | 0.329261 | 0.364531 | 0.002480 | 0.000326 | 1.721187 | 0.858604 | 0.057997 | trend ratio≈0.987 (over-smooth) |
| F3 | Weather | 96→96 | DualStream+SMGP | decomp_alpha=0.1 + ema_detrend(0.1) | 0.172776 | 0.230077 | 0.002481 | 0.000336 | 1.791741 | 0.367154 | 0.057241 | best in Weather, trend ratio≈0.876 |
| F3 | Weather | 96→96 | DualStream+SMGP | decomp_alpha=0.3 + ema_detrend(0.1) | 0.178982 | 0.237261 | 0.002496 | 0.000336 | 1.791627 | 0.289705 | 0.064538 | worse, trend ratio≈0.906 |
| F3 | Weather | 96→96 | DualStream+SMGP | decomp_alpha=0.5 + ema_detrend(0.1) | 0.184784 | 0.246073 | 0.002504 | 0.000335 | 1.791744 | 0.289894 | 0.064627 | worse, trend ratio≈0.981 |
