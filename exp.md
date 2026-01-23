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
| M2-1 | | log stats per epoch or per N steps |  |  |  |  | logs or csv | |

| Exp ID | Run ID | Segments sampled | Heatmap or numpy path | Top-k neighbors path | Notes |
| --- | --- | --- | --- | --- | --- |
| M2-2 | |  |  |  | |

## M4 Stable Stream for Graph (v1.3)
### M4A Stable stream identity
| Exp ID | Run ID | graph_source | stable_feat_type | stable_share_encoder | stable_detach | Metrics (MSE/MAE) | Adj stats | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4A-1 | | content_mean | none | false | false |  |  |  | |
| M4A-2 | | stable_stream | none | false | false |  |  |  | |

### M4B StableFeat sweep
| Exp ID | Run ID | graph_source | stable_feat_type | window | Metrics (MSE/MAE) | Adj stats | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4B-1 | | stable_stream | detrend | 3 |  |  |  | |
| M4B-2 | | stable_stream | detrend | 5 |  |  |  | |
| M4B-3 | | stable_stream | detrend | 9 |  |  |  | |
| M4B-4 | | stable_stream | detrend | patch_len |  |  |  | |
| M4B-5 | | stable_stream | diff |  |  |  |  | |

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
