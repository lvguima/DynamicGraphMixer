# DynamicGraphMixer v2 Experiment Results (exp_resultsv2.md)

## Purpose
- Track v2 experiments defined in `implementationTODOv2.md`.
- Keep results reproducible with configs, seeds, and log paths.
- Record both metrics (MSE/MAE) and graph diagnostics.

## Logging conventions
- Command template (example):
  - `python run.py ... --graph_log_interval N --graph_log_exp_id <id>`
- Graph stats output:
  - `graph_logs/<exp_id>/stats.csv`
  - `graph_logs/<exp_id>/epochXXX_stepYYYYY/adj_heatmap_segK.png`
- Results output:
  - `results/<setting>/metrics.npy`
  - `result_long_term_forecast.txt` (append)
- Stats fields (high-level):
  - post: entropy/topk_mass/topk_overlap/adj_diff/diag/offdiag
  - raw: entropy/topk_mass/topk_overlap/adj_diff + conf stats
  - conf: conf_mean/std/p10/p50/p90
  - gate/alpha: mean/std/quantiles/saturation
  - base: entropy/topk_mass/diag/offdiag

## v2.0 - Baseline freeze + observability

### Code tasks
- [ ] Log adj_raw entropy/topk_mass/topk_overlap pre-sparsify
- [ ] Log adj entropy/topk_mass/topk_overlap post-sparsify
- [ ] Log adj_diff mean(|A_t - A_{t-1}|)
- [ ] Log conf stats from raw entropy
- [ ] Log gate and alpha stats (mean/std/quantiles/saturation)
- [ ] Log base graph entropy/topk_mass/diag/offdiag (if enabled)
- [ ] Save stats to `graph_logs/<exp_id>/stats.csv`

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0-1 | DynamicGraphMixer_TCN_ETTm1_96_96 | `python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 4 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_log_exp_id E0-1 --graph_source content_mean --stable_level point --stable_feat_type detrend --stable_window 16 --graph_base_mode mix --graph_base_alpha_init -8 --graph_base_l1 0.001 --gate_mode scalar --gate_init -4 --adj_sparsify topk --adj_topk 4` | mse:0.33411359786987305, mae:0.3708299994468689 | ent=1.0821, topk=1.000, overlap=0.9200, l1=0.1118, mean=0.1429, var=0.0386, max=0.9754, diag=0.0930, off=0.1512 | raw_ent=1.4077, raw_topk=0.9058, raw_overlap=0.9030, raw_l1=0.0866, raw_mean=0.1429, raw_var=0.0289, raw_max=0.9691, raw_diag=0.1024, raw_off=0.1496 | conf_mean=0.2766, conf_std=0.2507, p10=0.0206, p50=0.2014, p90=0.6773 | gate_mean=0.01876, std=0.00000, p10=0.01876, p50=0.01876, p90=0.01876, sat_low=1.00, sat_high=0.00 | alpha_mean=0.000304, std=0.000000, p10=0.000304, p50=0.000304, p90=0.000304, sat_low=1.00, sat_high=0.00 | base_ent=1.9458, base_topk=0.7193, base_diag=0.1433, base_off=0.1428 | graph_logs/E0-1/stats.csv | baseline M8-4 (ETTm1 96->96) |


### Summary
- Result: M8-4 baseline run (ETTm1 96->96) mse=0.33437901735305786, mae=0.37103986740112305
- Decision:

---

## v2.1 - Static gate sweep (no structural change)

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-A |  | gate_mode in {none, scalar, per_var, per_token} |  |  |  |  |  |  |  |  |  |
| E1-B |  | gate_init in {-8, -6, -4, -2, 0} |  |  |  |  |  |  |  |  |  |
| E1-C |  | graph_scale in {1, 2, 4, 8, 16} |  |  |  |  |  |  |  |  |  |

### Summary
- Best static gate config:
- Next step:

---

## v2.2 - Confidence gate (dynamic gate)

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2-1 |  | gate_mode in {conf_scalar, conf_per_var} |  |  |  |  |  |  |  |  |  |
| E2-2 |  | mapping in {G1, G2} |  |  |  |  |  |  |  |  |  |
| E2-3 |  | gate_conf_source in {pre_sparsify, post_sparsify} |  |  |  |  |  |  |  |  |  |
| E2-4 |  | gamma in {0.5, 1, 2, 4} |  |  |  |  |  |  |  |  |  |
| E2-5 |  | w_init in {2, 4, 8}, b_init in {-4, -2, 0} |  |  |  |  |  |  |  |  |  |
| E2-6 |  | gate_warmup_epochs in {0, 1, 3, 5} |  |  |  |  |  |  |  |  |  |
| E2-Final |  | best 2-3 configs, 3 seeds | mean= , std= |  |  |  |  |  |  |  |  |

### Summary
- Best conf gate config:
- Gate vs entropy trend:
- Decision:

---

## v2.3 - Confidence routing + base regularization

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E3-A |  | alpha_mode in {static, conf} |  |  |  |  |  |  |  |  |  |
| E3-B |  | base_reg in {none, offdiag_l1, entropy, diag_prior} with lambda sweep |  |  |  |  |  |  |  |  |  |
| E3-C |  | conf gate vs conf alpha vs both |  |  |  |  |  |  |  |  |  |

### Summary
- Best routing + reg config:
- Interpretation notes:

---

## v2.4 - Multi-scale stable graph fusion

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4-1 |  | stable_feat_type in {detrend, diff, none} |  |  |  |  |  |  |  |  |  |
| E4-2 |  | stable_window in {8, 16, 32} |  |  |  |  |  |  |  |  |  |
| E4-3 |  | graph_scale_stable in {8, 16, 32} |  |  |  |  |  |  |  |  |  |
| E4-4 |  | stable_share_encoder and stable_detach sweep |  |  |  |  |  |  |  |  |  |
| E4-5 |  | fuse_mode and beta_init sweep |  |  |  |  |  |  |  |  |  |

### Summary
- Best stable fusion config:
- Decision:

---

## v2.5 - Lag-aware graph mixing

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5-1 |  | L_lag in {2, 4, 8} |  |  |  |  |  |  |  |  |  |
| E5-2 |  | share_adj in {True, False} |  |  |  |  |  |  |  |  |  |
| E5-3 |  | w_init in {uniform, bias_to_small_lag} |  |  |  |  |  |  |  |  |  |
| E5-4 |  | interactions with conf gate/route |  |  |  |  |  |  |  |  |  |

### Summary
- Best lag-aware config:
- Decision:

---

## v2.6 - Publication-ready experiment suite

### Experiments
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Adj stats (post) | Adj stats (raw) | Conf stats | Gate stats | Alpha stats | Base stats | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E6-1 |  | multi-horizon pred_len in {96, 192, 336, 720} |  |  |  |  |  |  |  |  |  |
| E6-2 |  | multi-dataset in {ETTm2, ETTh1, ETTh2} |  |  |  |  |  |  |  |  |  |
| E6-3 |  | complexity: params/throughput/memory |  |  |  |  |  |  |  |  |  |
| E6-4 |  | interpretability cases |  |  |  |  |  |  |  |  |  |

### Summary
- Final v2 config:
- Key ablations:
- Final decision:
