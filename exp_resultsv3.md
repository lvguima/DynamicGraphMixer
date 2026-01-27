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
| F1 | ETTm1 | 96→96 | SMGP | graph_map_norm=ema_detrend, graph_map_alpha=0.3 | 0.333296 | 0.369657 | 0.002511 | 0.000333 | 1.737882 | 0.856047 | 0.064255 | slight gain, gate_mean unchanged |
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
| F3 | ETTm1 | 96→96 | DualStream+SMGP | decomp_mode=ema,decomp_alpha=0.3 + SMGP |  |  |  |  |  |  |  |  |
| F3 | Traffic | 96→96 | DualStream+SMGP | decomp_mode=ema,decomp_alpha=0.3 + SMGP |  |  |  |  |  |  |  |  |
