# Experiment Log

## Run Registry
| Run ID | Date | Task | Dataset | Model | Seed | Key hparams | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DynamicGraphMixer_TCN_ETTm1_96_96 | 2026-01-23 | long_term_forecast | ETTm1 | DynamicGraphMixer | 2021 | seq_len=96; pred_len=96; enc_in=7; d_model=128; e_layers=2; graph_scale=8; graph_rank=8; use_patch=false; graph_smooth_lambda=0 | M1 baseline |

## M1 Baseline Reproduction (v1.0)
| Exp ID | Run ID | Config or Args | Metrics (MSE/MAE) | Best epoch | Time | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1-1 | DynamicGraphMixer_TCN_ETTm1_96_96 | `python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2` | mse=0.3592156768, mae=0.4024800062 |  |  | checkpoints/<setting>/checkpoint.pth; results/<setting>/metrics.npy; results/<setting>/pred.npy; results/<setting>/true.npy; test_results/<setting>/*.pdf; result_long_term_forecast.txt | |

## M2 Graph Observability (v1.1)
| Exp ID | Run ID | Settings | Adj entropy (mean) | Topk mass (k=) | L1 adj diff | Adj mean/var/max | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2-1 | long_term_forecast_DynamicGraphMixer_TCN_ETTm1_96_96_DynamicGraphMixer_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0 | log stats per epoch or per N steps |  |  |  |  | `graph_logs\long_term_forecast_DynamicGraphMixer_TCN_ETTm1_96_96_DynamicGraphMixer_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0\stats.csv` | mse=0.35914286971092224, mae=0.40239110589027405 |

| Exp ID | Run ID | Segments sampled | Heatmap or numpy path | Top-k neighbors path | Notes |
| --- | --- | --- | --- | --- | --- |
| M2-2 | |  |  |  | |

## M3 Superset Refactor (v1.2)
| Exp ID | Run ID | Notes | Metrics (MSE/MAE) |
| --- | --- | --- | --- |
| M3-1 | long_term_forecast_DynamicGraphMixer_TCN_ETTm1_96_96_DynamicGraphMixer_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0 | modularization only; V1 parity check | mse=0.3591, mae=0.4022 |

## M4 Stable Stream for Graph (v1.3)
### M4A Stable stream identity
| Exp ID | Run ID | graph_source | stable_feat_type | stable_share_encoder | stable_detach | Metrics (MSE/MAE) | Adj stats | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4A-1 | python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source content_mean --graph_log_exp_id M4A-1| content_mean | none | false | false | mse:0.3593936562538147, mae:0.4026050567626953 | ent=1.599, topk=0.933, var=0.0135, max=0.940 | l1_adj=0.067 | |
| M4A-2 | python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source stable_stream --stable_feat_type none --graph_log_exp_id M4A-2| stable_stream | none | false | false | mse:0.3684585392475128, mae:0.3986046016216278 | ent=1.362, topk=0.966, var=0.0254, max=0.999 | l1_adj=0.089 | |

### M4B StableFeat sweep
| Exp ID | Run ID | graph_source | stable_feat_type | window | Metrics (MSE/MAE) | Adj stats | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4B-1 | python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source stable_stream --stable_feat_type none --stable_feat_type detrend --stable_window 3 --graph_log_exp_id M4B-1| stable_stream | detrend | 3 |  mse:0.5010863542556763, mae:0.5178493857383728 | ent=1.866, topk=0.820, var=0.0034, max=0.672 | l1_adj=0.054 | |
| M4B-2 | python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source stable_stream --stable_feat_type none --stable_feat_type detrend --stable_window 5 --graph_log_exp_id M4B-2| stable_stream | detrend | 5 | mse:0.4693937301635742, mae:0.4908128082752228 | ent=1.817, topk=0.833, var=0.0058, max=0.939 | l1_adj=0.070 | |
| M4B-3 | python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source stable_stream --stable_feat_type none --stable_feat_type detrend --stable_window 9 --graph_log_exp_id M4B-3| stable_stream | detrend | 9 | mse:0.44306108355522156, mae:0.4706106185913086 | ent=1.395, topk=0.937, var=0.0263, max=0.973 | l1_adj=0.120 | |
| M4B-4 |python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source stable_stream --stable_feat_type none --stable_feat_type detrend --stable_window 16 --graph_log_exp_id M4B-4 | stable_stream | detrend | patch_len = 16 | mse:0.3473997116088867, mae:0.38508477807044983 | ent=1.053, topk=0.970, var=0.0461, max=1.000 | l1_adj=0.129 | |
| M4B-5 |python -u run.py --task_name long_term_forecast --is_training 1 --model_id DynamicGraphMixer_TCN_ETTm1_96_96 --model DynamicGraphMixer --data ETTm1 --root_path ./datasets --data_path ETTm1.csv --features M --target OT --freq t --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 256 --enc_in 7 --dec_in 7 --c_out 7 --batch_size 64 --train_epochs 15 --patience 3 --use_norm 1 --graph_scale 8 --graph_rank 8 --graph_smooth_lambda 0 --temporal_encoder tcn --tcn_kernel 3 --tcn_dilation 2 --graph_log_interval 200 --graph_log_topk 5 --graph_log_num_segments 2 --graph_log_dir ./graph_logs --graph_source stable_stream --stable_feat_type none --stable_feat_type diff --graph_log_exp_id M4B-5| stable_stream | diff |  | mse:0.3598678708076477, mae:0.3946714997291565 | ent=1.237, topk=0.956, var=0.0349, max=0.983 | l1_adj=0.132 | |

## M5 Token-Level StableFeat (v1.4)
| Exp ID | Run ID | Stable type | Settings | Metrics (MSE/MAE) | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| M5-1 | | point |  |  |  | |
| M5-2 | | token |  |  |  | |

## M6 Base and Residual Graph (v1.5)
| Exp ID | Run ID | base_mode | alpha init | base L1 | Metrics (MSE/MAE) | Adj stats | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M6-1 | | none |  |  |  |  | |
| M6-2 | | mix (learnable) | ~0 |  |  |  | |
| M6-3 | | mix (learnable) | ~0 | on |  |  | |

## M7 Gated Mixing (v1.6)
| Exp ID | Run ID | gate_mode | gate init | Metrics (MSE/MAE) | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| M7-1 | | none |  |  |  | |
| M7-2 | | scalar | ~0 or small |  |  | |

## M8 Top-k Sparsify and Anti-spurious (v1.7)
| Exp ID | Run ID | graph_scale | topk | stable source | Metrics (MSE/MAE) | Adj stats | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M8-1 | | 1 | none | content_mean |  |  | |
| M8-2 | | 1 | 8 | content_mean |  |  | |
| M8-3 | | 2 | 8 | content_mean |  |  | |
| M8-4 | | 4 | 8 | content_mean |  |  | |
| M8-5 | | 8 | 8 | content_mean |  |  | |
| M8-6 | | 1 | 16 | stable_stream |  |  | |

## M9 Patch Path Ablation (v1.8)
| Exp ID | Run ID | encoder | patch_mode | use_patch | Metrics (MSE/MAE) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| M9-1 | | tcn | v1_encode_then_pool | true |  | |
| M9-2 | | tcn | token_first | true |  | |
| M9-3 | | transformer or ssm | v1_encode_then_pool | true |  | |
| M9-4 | | transformer or ssm | token_first | true |  | |
