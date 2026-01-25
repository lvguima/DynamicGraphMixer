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
| E1-A | DynamicGraphMixer_TCN_ETTm1_96_96_E1A_none | gate_mode=none, gate_init=-4 |  | ent=1.4945, topk=0.9631, overlap=0.9016, l1=0.0778, mean=0.1429, var=0.0169, max=0.9888, diag=0.1956, off=0.1341 | raw_ent=1.5368, raw_topk=0.9489, raw_overlap=0.9016, raw_l1=0.0750, raw_mean=0.1429, raw_var=0.0161, raw_max=0.9887, raw_diag=0.1942, raw_off=0.1343 | conf_mean=0.2102, conf_std=0.1351, p10=0.0629, p50=0.1872, p90=0.3880 | gate=none | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7368, base_diag=0.1529, base_off=0.1412 | `graph_logs/v2_1_E1-A_gate_none/stats.csv` | last log step |
| E1-A | DynamicGraphMixer_TCN_ETTm1_96_96_E1A_scalar | gate_mode=scalar, gate_init=-4 |  | ent=1.3143, topk=0.9487, overlap=0.8912, l1=0.1098, mean=0.1429, var=0.0314, max=0.9998, diag=0.0995, off=0.1501 | raw_ent=1.3924, raw_topk=0.9143, raw_overlap=0.8912, raw_l1=0.1022, raw_mean=0.1429, raw_var=0.0295, raw_max=0.9997, raw_diag=0.0988, raw_off=0.1502 | conf_mean=0.2845, conf_std=0.2668, p10=0.0191, p50=0.1983, p90=0.7039 | gate_mean=0.01882, std=0.00000, p10=0.01882, p50=0.01882, p90=0.01882, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1430, base_off=0.1428 | `graph_logs/v2_1_E1-A_gate_scalar/stats.csv` | last log step |
| E1-A | DynamicGraphMixer_TCN_ETTm1_96_96_E1A_per_var | gate_mode=per_var, gate_init=-4 |  | ent=1.3264, topk=0.9490, overlap=0.8922, l1=0.1049, mean=0.1429, var=0.0306, max=0.9998, diag=0.0990, off=0.1502 | raw_ent=1.4040, raw_topk=0.9148, raw_overlap=0.8922, raw_l1=0.0975, raw_mean=0.1429, raw_var=0.0287, raw_max=0.9997, raw_diag=0.0974, raw_off=0.1504 | conf_mean=0.2785, conf_std=0.2615, p10=0.0188, p50=0.1958, p90=0.6899 | gate_mean=0.01812, std=0.00095, p10=0.01711, p50=0.01806, p90=0.01904, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1430, base_off=0.1428 | `graph_logs/v2_1_E1-A_gate_per_var/stats.csv` | last log step |
| E1-A | DynamicGraphMixer_TCN_ETTm1_96_96_E1A_per_token | gate_mode=per_token, gate_init=-4 |  | ent=1.3346, topk=0.9464, overlap=0.8894, l1=0.1056, mean=0.1429, var=0.0302, max=0.9998, diag=0.0976, off=0.1504 | raw_ent=1.4153, raw_topk=0.9103, raw_overlap=0.8894, raw_l1=0.0977, raw_mean=0.1429, raw_var=0.0282, raw_max=0.9997, raw_diag=0.0971, raw_off=0.1505 | conf_mean=0.2727, conf_std=0.2648, p10=0.0172, p50=0.1807, p90=0.6879 | gate_mean=0.01804, std=0.00080, p10=0.01712, p50=0.01791, p90=0.01924, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1430, base_off=0.1428 | `graph_logs/v2_1_E1-A_gate_per_token/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_none_init-8 | gate_mode=none, gate_init=-8 |  | ent=1.4941, topk=0.9632, overlap=0.9017, l1=0.0778, mean=0.1429, var=0.0169, max=0.9882, diag=0.1953, off=0.1341 | raw_ent=1.5365, raw_topk=0.9488, raw_overlap=0.9017, raw_l1=0.0750, raw_mean=0.1429, raw_var=0.0161, raw_max=0.9882, raw_diag=0.1939, raw_off=0.1343 | conf_mean=0.2104, conf_std=0.1351, p10=0.0631, p50=0.1872, p90=0.3881 | gate=none | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7367, base_diag=0.1529, base_off=0.1412 | `graph_logs/v2_1_E1-B_none_init-8/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_none_init-6 | gate_mode=none, gate_init=-6 |  | ent=1.4933, topk=0.9633, overlap=0.9016, l1=0.0778, mean=0.1429, var=0.0170, max=0.9886, diag=0.1955, off=0.1341 | raw_ent=1.5356, raw_topk=0.9490, raw_overlap=0.9016, raw_l1=0.0750, raw_mean=0.1429, raw_var=0.0161, raw_max=0.9885, raw_diag=0.1941, raw_off=0.1343 | conf_mean=0.2109, conf_std=0.1355, p10=0.0630, p50=0.1877, p90=0.3887 | gate=none | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7368, base_diag=0.1529, base_off=0.1412 | `graph_logs/v2_1_E1-B_none_init-6/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_none_init-4 | gate_mode=none, gate_init=-4 |  | ent=1.4949, topk=0.9630, overlap=0.9017, l1=0.0778, mean=0.1429, var=0.0169, max=0.9887, diag=0.1955, off=0.1341 | raw_ent=1.5374, raw_topk=0.9486, raw_overlap=0.9017, raw_l1=0.0750, raw_mean=0.1429, raw_var=0.0161, raw_max=0.9886, raw_diag=0.1941, raw_off=0.1343 | conf_mean=0.2099, conf_std=0.1351, p10=0.0629, p50=0.1869, p90=0.3882 | gate=none | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7368, base_diag=0.1529, base_off=0.1412 | `graph_logs/v2_1_E1-B_none_init-4/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_none_init-2 | gate_mode=none, gate_init=-2 |  | ent=1.4607, topk=0.9661, overlap=0.8995, l1=0.0788, mean=0.1429, var=0.0188, max=0.9937, diag=0.1953, off=0.1341 | raw_ent=1.5006, raw_topk=0.9530, raw_overlap=0.8995, raw_l1=0.0763, raw_mean=0.1429, raw_var=0.0180, raw_max=0.9934, raw_diag=0.1936, raw_off=0.1344 | conf_mean=0.2289, conf_std=0.1500, p10=0.0658, p50=0.2014, p90=0.4309 | gate=none | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7368, base_diag=0.1529, base_off=0.1412 | `graph_logs/v2_1_E1-B_none_init-2/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_none_init0 | gate_mode=none, gate_init=0 |  | ent=1.4936, topk=0.9632, overlap=0.9016, l1=0.0779, mean=0.1429, var=0.0170, max=0.9885, diag=0.1955, off=0.1341 | raw_ent=1.5359, raw_topk=0.9490, raw_overlap=0.9016, raw_l1=0.0751, raw_mean=0.1429, raw_var=0.0161, raw_max=0.9885, raw_diag=0.1940, raw_off=0.1343 | conf_mean=0.2107, conf_std=0.1352, p10=0.0632, p50=0.1876, p90=0.3882 | gate=none | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7368, base_diag=0.1529, base_off=0.1412 | `graph_logs/v2_1_E1-B_none_init0/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_scalar_init-8 | gate_mode=scalar, gate_init=-8 |  | ent=1.2281, topk=0.9553, overlap=0.8898, l1=0.1215, mean=0.1429, var=0.0372, max=0.9997, diag=0.0961, off=0.1506 | raw_ent=1.3013, raw_topk=0.9246, raw_overlap=0.8898, raw_l1=0.1145, raw_mean=0.1429, raw_var=0.0353, raw_max=0.9997, raw_diag=0.0948, raw_off=0.1509 | conf_mean=0.3313, conf_std=0.2803, p10=0.0300, p50=0.2526, p90=0.7921 | gate_mean=0.00036, std=0.00000, p10=0.00036, p50=0.00036, p90=0.00036, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00033, std=0.00000, p10=0.00033, p50=0.00033, p90=0.00033, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9459, base_topk=0.7146, base_diag=0.1428, base_off=0.1429 | `graph_logs/v2_1_E1-B_scalar_init-8/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_scalar_init-6 | gate_mode=scalar, gate_init=-6 |  | ent=1.1826, topk=0.9607, overlap=0.8903, l1=0.1276, mean=0.1429, var=0.0399, max=0.9998, diag=0.0972, off=0.1505 | raw_ent=1.2485, raw_topk=0.9343, raw_overlap=0.8903, raw_l1=0.1216, raw_mean=0.1429, raw_var=0.0382, raw_max=0.9997, raw_diag=0.0962, raw_off=0.1506 | conf_mean=0.3584, conf_std=0.2817, p10=0.0404, p50=0.2928, p90=0.8222 | gate_mean=0.00265, std=0.00000, p10=0.00265, p50=0.00265, p90=0.00265, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9459, base_topk=0.7165, base_diag=0.1427, base_off=0.1429 | `graph_logs/v2_1_E1-B_scalar_init-6/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_scalar_init-4 | gate_mode=scalar, gate_init=-4 |  | ent=1.3202, topk=0.9483, overlap=0.8906, l1=0.1098, mean=0.1429, var=0.0311, max=0.9998, diag=0.0998, off=0.1500 | raw_ent=1.3990, raw_topk=0.9135, raw_overlap=0.8906, raw_l1=0.1021, raw_mean=0.1429, raw_var=0.0291, raw_max=0.9997, raw_diag=0.0992, raw_off=0.1501 | conf_mean=0.2811, conf_std=0.2650, p10=0.0195, p50=0.1933, p90=0.6989 | gate_mean=0.01881, std=0.00000, p10=0.01881, p50=0.01881, p90=0.01881, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1430, base_off=0.1428 | `graph_logs/v2_1_E1-B_scalar_init-4/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_scalar_init-2 | gate_mode=scalar, gate_init=-2 |  | ent=1.5594, topk=0.9333, overlap=0.9001, l1=0.0718, mean=0.1429, var=0.0161, max=0.9995, diag=0.1211, off=0.1465 | raw_ent=1.6529, raw_topk=0.8941, raw_overlap=0.9001, raw_l1=0.0638, raw_mean=0.1429, raw_var=0.0140, raw_max=0.9995, raw_diag=0.1201, raw_off=0.1466 | conf_mean=0.1506, conf_std=0.1611, p10=0.0194, p50=0.1023, p90=0.3250 | gate_mean=0.11438, std=0.00000, p10=0.11438, p50=0.11438, p90=0.11438, sat_low=0.00000, sat_high=0.00000 | alpha_mean=0.00033, std=0.00000, p10=0.00033, p50=0.00033, p90=0.00033, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9442, base_topk=0.7323, base_diag=0.1483, base_off=0.1419 | `graph_logs/v2_1_E1-B_scalar_init-2/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_scalar_init0 | gate_mode=scalar, gate_init=0 |  | ent=1.5805, topk=0.9529, overlap=0.8912, l1=0.0692, mean=0.1429, var=0.0125, max=0.9405, diag=0.1602, off=0.1400 | raw_ent=1.6348, raw_topk=0.9343, raw_overlap=0.8912, raw_l1=0.0656, raw_mean=0.1429, raw_var=0.0114, raw_max=0.9403, raw_diag=0.1611, raw_off=0.1398 | conf_mean=0.1599, conf_std=0.0989, p10=0.0499, p50=0.1442, p90=0.2931 | gate_mean=0.47851, std=0.00000, p10=0.47851, p50=0.47851, p90=0.47851, sat_low=0.00000, sat_high=0.00000 | alpha_mean=0.00032, std=0.00000, p10=0.00032, p50=0.00032, p90=0.00032, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7372, base_diag=0.1527, base_off=0.1412 | `graph_logs/v2_1_E1-B_scalar_init0/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_per_var_init-8 | gate_mode=per_var, gate_init=-8 |  | ent=1.2286, topk=0.9548, overlap=0.8880, l1=0.1194, mean=0.1429, var=0.0372, max=0.9997, diag=0.0936, off=0.1511 | raw_ent=1.3022, raw_topk=0.9236, raw_overlap=0.8880, raw_l1=0.1122, raw_mean=0.1429, raw_var=0.0353, raw_max=0.9997, raw_diag=0.0924, raw_off=0.1513 | conf_mean=0.3308, conf_std=0.2826, p10=0.0286, p50=0.2524, p90=0.7981 | gate_mean=0.00035, std=0.00002, p10=0.00032, p50=0.00035, p90=0.00036, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00033, std=0.00000, p10=0.00033, p50=0.00033, p90=0.00033, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9459, base_topk=0.7146, base_diag=0.1428, base_off=0.1429 | `graph_logs/v2_1_E1-B_per_var_init-8/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_per_var_init-6 | gate_mode=per_var, gate_init=-6 |  | ent=1.1885, topk=0.9596, overlap=0.8899, l1=0.1251, mean=0.1429, var=0.0396, max=0.9998, diag=0.0948, off=0.1509 | raw_ent=1.2555, raw_topk=0.9324, raw_overlap=0.8899, raw_l1=0.1190, raw_mean=0.1429, raw_var=0.0378, raw_max=0.9997, raw_diag=0.0940, raw_off=0.1510 | conf_mean=0.3548, conf_std=0.2839, p10=0.0388, p50=0.2823, p90=0.8265 | gate_mean=0.00254, std=0.00013, p10=0.00239, p50=0.00256, p90=0.00267, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9459, base_topk=0.7164, base_diag=0.1427, base_off=0.1429 | `graph_logs/v2_1_E1-B_per_var_init-6/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_per_var_init-4 | gate_mode=per_var, gate_init=-4 |  | ent=1.3282, topk=0.9474, overlap=0.8899, l1=0.1062, mean=0.1429, var=0.0305, max=0.9998, diag=0.0969, off=0.1505 | raw_ent=1.4071, raw_topk=0.9122, raw_overlap=0.8899, raw_l1=0.0985, raw_mean=0.1429, raw_var=0.0286, raw_max=0.9997, raw_diag=0.0966, raw_off=0.1506 | conf_mean=0.2769, conf_std=0.2664, p10=0.0178, p50=0.1856, p90=0.6941 | gate_mean=0.01812, std=0.00095, p10=0.01712, p50=0.01806, p90=0.01904, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1430, base_off=0.1428 | `graph_logs/v2_1_E1-B_per_var_init-4/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_per_var_init-2 | gate_mode=per_var, gate_init=-2 |  | ent=1.5347, topk=0.9382, overlap=0.9017, l1=0.0727, mean=0.1429, var=0.0175, max=0.9996, diag=0.1214, off=0.1464 | raw_ent=1.6228, raw_topk=0.9023, raw_overlap=0.9017, raw_l1=0.0656, raw_mean=0.1429, raw_var=0.0155, raw_max=0.9995, raw_diag=0.1198, raw_off=0.1467 | conf_mean=0.1661, conf_std=0.1664, p10=0.0253, p50=0.1158, p90=0.3545 | gate_mean=0.11412, std=0.00386, p10=0.11094, p50=0.11321, p90=0.11837, sat_low=0.00000, sat_high=0.00000 | alpha_mean=0.00033, std=0.00000, p10=0.00033, p50=0.00033, p90=0.00033, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9441, base_topk=0.7323, base_diag=0.1483, base_off=0.1419 | `graph_logs/v2_1_E1-B_per_var_init-2/stats.csv` | last log step |
| E1-B | DynamicGraphMixer_TCN_ETTm1_96_96_E1B_per_var_init0 | gate_mode=per_var, gate_init=0 |  | ent=1.5651, topk=0.9553, overlap=0.8844, l1=0.0717, mean=0.1429, var=0.0133, max=0.8658, diag=0.1621, off=0.1396 | raw_ent=1.6192, raw_topk=0.9371, raw_overlap=0.8844, raw_l1=0.0682, raw_mean=0.1429, raw_var=0.0123, raw_max=0.8657, raw_diag=0.1627, raw_off=0.1396 | conf_mean=0.1679, conf_std=0.1035, p10=0.0509, p50=0.1499, p90=0.3127 | gate_mean=0.48035, std=0.00387, p10=0.47528, p50=0.48238, p90=0.48418, sat_low=0.00000, sat_high=0.00000 | alpha_mean=0.00032, std=0.00000, p10=0.00032, p50=0.00032, p90=0.00032, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9427, base_topk=0.7372, base_diag=0.1527, base_off=0.1412 | `graph_logs/v2_1_E1-B_per_var_init0/stats.csv` | last log step |
| E1-C | DynamicGraphMixer_TCN_ETTm1_96_96_E1C_scalar_scale1 | gate_mode=scalar, gate_init=-4, graph_scale=1 |  | ent=1.1859, topk=0.9627, overlap=0.9116, l1=0.1081, mean=0.1429, var=0.0395, max=0.9998, diag=0.1027, off=0.1495 | raw_ent=1.2444, raw_topk=0.9398, raw_overlap=0.9116, raw_l1=0.1031, raw_mean=0.1429, raw_var=0.0381, raw_max=0.9997, raw_diag=0.1023, raw_off=0.1496 | conf_mean=0.3605, conf_std=0.2807, p10=0.0441, p50=0.2949, p90=0.8149 | gate_mean=0.01899, std=0.00000, p10=0.01899, p50=0.01899, p90=0.01899, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1431, base_off=0.1428 | `graph_logs/v2_1_E1-C_scalar_scale1/stats.csv` | last log step |
| E1-C | DynamicGraphMixer_TCN_ETTm1_96_96_E1C_scalar_scale2 | gate_mode=scalar, gate_init=-4, graph_scale=2 |  | ent=1.2685, topk=0.9553, overlap=0.9118, l1=0.0970, mean=0.1429, var=0.0341, max=0.9998, diag=0.1050, off=0.1492 | raw_ent=1.3374, raw_topk=0.9264, raw_overlap=0.9118, raw_l1=0.0910, raw_mean=0.1429, raw_var=0.0324, raw_max=0.9997, raw_diag=0.1047, raw_off=0.1492 | conf_mean=0.3127, conf_std=0.2717, p10=0.0264, p50=0.2389, p90=0.7342 | gate_mean=0.01892, std=0.00000, p10=0.01892, p50=0.01892, p90=0.01892, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7234, base_diag=0.1431, base_off=0.1428 | `graph_logs/v2_1_E1-C_scalar_scale2/stats.csv` | last log step |
| E1-C | DynamicGraphMixer_TCN_ETTm1_96_96_E1C_scalar_scale4 | gate_mode=scalar, gate_init=-4, graph_scale=4 |  | ent=1.3005, topk=0.9508, overlap=0.9045, l1=0.1010, mean=0.1429, var=0.0323, max=0.9998, diag=0.0999, off=0.1500 | raw_ent=1.3757, raw_topk=0.9181, raw_overlap=0.9045, raw_l1=0.0941, raw_mean=0.1429, raw_var=0.0304, raw_max=0.9997, raw_diag=0.1004, raw_off=0.1499 | conf_mean=0.2931, conf_std=0.2678, p10=0.0216, p50=0.2127, p90=0.7089 | gate_mean=0.01884, std=0.00000, p10=0.01884, p50=0.01884, p90=0.01884, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00030, std=0.00000, p10=0.00030, p50=0.00030, p90=0.00030, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1431, base_off=0.1428 | `graph_logs/v2_1_E1-C_scalar_scale4/stats.csv` | last log step |
| E1-C | DynamicGraphMixer_TCN_ETTm1_96_96_E1C_scalar_scale8 | gate_mode=scalar, gate_init=-4, graph_scale=8 |  | ent=1.2837, topk=0.9523, overlap=0.8931, l1=0.1112, mean=0.1429, var=0.0336, max=0.9997, diag=0.0982, off=0.1503 | raw_ent=1.3581, raw_topk=0.9205, raw_overlap=0.8931, raw_l1=0.1042, raw_mean=0.1429, raw_var=0.0317, raw_max=0.9997, raw_diag=0.0969, raw_off=0.1505 | conf_mean=0.3021, conf_std=0.2707, p10=0.0255, p50=0.2169, p90=0.7442 | gate_mean=0.01881, std=0.00000, p10=0.01881, p50=0.01881, p90=0.01881, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7236, base_diag=0.1430, base_off=0.1428 | `graph_logs/v2_1_E1-C_scalar_scale8/stats.csv` | last log step |
| E1-C | DynamicGraphMixer_TCN_ETTm1_96_96_E1C_scalar_scale16 | gate_mode=scalar, gate_init=-4, graph_scale=16 |  | ent=1.3233, topk=0.9511, overlap=0.8856, l1=0.1183, mean=0.1429, var=0.0300, max=0.9997, diag=0.0937, off=0.1510 | raw_ent=1.3938, raw_topk=0.9188, raw_overlap=0.8856, raw_l1=0.1106, raw_mean=0.1429, raw_var=0.0283, raw_max=0.9996, raw_diag=0.0940, raw_off=0.1510 | conf_mean=0.2837, conf_std=0.2554, p10=0.0154, p50=0.2118, p90=0.6612 | gate_mean=0.01854, std=0.00000, p10=0.01854, p50=0.01854, p90=0.01854, sat_low=1.00000, sat_high=0.00000 | alpha_mean=0.00031, std=0.00000, p10=0.00031, p50=0.00031, p90=0.00031, sat_low=1.00000, sat_high=0.00000 | base_ent=1.9455, base_topk=0.7235, base_diag=0.1427, base_off=0.1429 | `graph_logs/v2_1_E1-C_scalar_scale16/stats.csv` | last log step |

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
