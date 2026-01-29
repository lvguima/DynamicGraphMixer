# Ablation Summary (No Traffic)

Date: 2026-01-28

This note summarizes the B0B5 ablation design and results for all datasets except traffic.
All runs are single-seed and use the same base training setup unless noted.

## Experiment Design (B0B5)

- B0: Temporal encoder + head only (no DS, no SMGP, graph propagation off)
  - decomp_mode=none, trend_head=none, graph_map_norm=none, gate_init=-20
- B1: Graph propagation only (no DS, no SMGP)
  - decomp_mode=none, trend_head=none, graph_map_norm=none, gate_init=-6
- B2: SMGP + graph propagation (no DS)
  - decomp_mode=none, trend_head=none, graph_map_norm=ma_detrend (window=16), gate_init=-6
- B3: Dual-Stream only (no SMGP, graph propagation off)
  - decomp_mode=ema (alpha=0.1), trend_head=linear (share=1), graph_map_norm=none, gate_init=-20
- B4: Dual-Stream + graph propagation (no SMGP)
  - decomp_mode=ema (alpha=0.1), trend_head=linear (share=1), graph_map_norm=none, gate_init=-6
- B5: Dual-Stream + SMGP + graph propagation (full)
  - decomp_mode=ema (alpha=0.1), trend_head=linear (share=1), graph_map_norm=ma_detrend (window=16), gate_init=-6

## Common Settings

- task: long_term_forecast
- seq_len / label_len / pred_len: 96 / 48 / 96
- features: M (multivariate -> multivariate)
- target: OT
- temporal encoder: TCN (kernel=3, dilation=2)
- model dims: d_model=128, e_layers=2, d_ff=256
- training: batch_size=64, train_epochs=15, patience=3, use_norm=1
- graph: graph_rank=8, graph_scale=8, adj_topk=6 (topk), base graph mix enabled

## Results (MSE / MAE)

| Dataset | B0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| ETTh1 | 0.387886 / 0.401452 | 0.387961 / 0.401546 | 0.387969 / 0.401574 | 0.385145 / 0.399467 | 0.385097 / 0.399374 | 0.385140 / 0.399422 |
| ETTh2 | 0.352351 / 0.399924 | 0.351702 / 0.399710 | 0.351661 / 0.399566 | 0.328037 / 0.382540 | 0.327766 / 0.382315 | 0.327830 / 0.382373 |
| ETTm1 | 0.333091 / 0.369501 | 0.333400 / 0.369683 | 0.333554 / 0.369756 | 0.317740 / 0.357811 | 0.317110 / 0.357558 | 0.316962 / 0.357464 |
| ETTm2 | 0.215485 / 0.300579 | 0.214565 / 0.300251 | 0.215121 / 0.300504 | 0.187743 / 0.282131 | 0.187690 / 0.282083 | 0.187899 / 0.282242 |
| weather | 0.182292 / 0.240482 | 0.182245 / 0.240386 | 0.182428 / 0.240520 | 0.172885 / 0.230177 | 0.172964 / 0.230596 | 0.172848 / 0.230176 |
| national_illness | 3.322825 / 1.291248 | 3.323074 / 1.291307 | 3.322889 / 1.291283 | 4.843631 / 1.672288 | 4.843742 / 1.672311 | 4.843680 / 1.672290 |
| flotation | 0.769143 / 0.619316 | 0.769185 / 0.619329 | 0.769111 / 0.619267 | 0.786324 / 0.622870 | 0.786544 / 0.622961 | 0.786076 / 0.622775 |
| grinding | 26.385504 / 2.190284 | 26.296850 / 2.198079 | 26.459135 / 2.195739 | 4.250288 / 0.699211 | 4.250654 / 0.699502 | 4.249634 / 0.699089 |

## Per-dataset Takeaways

- ETTh1/ETTh2/ETTm1/ETTm2/weather/grinding: Dual-Stream (B3B5) yields clear gains vs non-DS (B0B2).
- national_illness and flotation: Dual-Stream hurts; non-DS settings are better.
- SMGP and graph propagation effects are small across all completed datasets (B1「B0, B2「B1; B4「B3, B5「B4).

## Overall Conclusion (So Far)

- Dual-Stream is the only component that consistently changes performance, but it is not universally helpful.
- SMGP and graph propagation show minimal impact on these datasets under current settings.
- Traffic (large-channel) is pending and may be the critical case to validate graph-related components.
