# Component Ablation Summary (A/B + Graph-Disabled)

This document merges:
- Single-component ablations (A/B series).
- The combined graph-disabled experiment (SMGP + propagation + mixing + sparsify off).

Base (GBASE) values are taken from `sweep_ablation.md` when available.

## ETTm1
- Base (GBASE): MSE=0.317662, MAE=0.357933

| Variant | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|
| A1_no_dualstream | 0.332469 | 0.369091 | +0.014807 | +0.011158 |
| A2_no_SMGP | 0.317314 | 0.357638 | -0.000348 | -0.000295 |
| A3_gate_off | 0.317399 | 0.357573 | -0.000263 | -0.000360 |
| A4_no_base_graph | 0.317365 | 0.357708 | -0.000297 | -0.000225 |
| A5_no_sparsify | 0.316954 | 0.357444 | -0.000708 | -0.000489 |
| B1_no_TCN_linear | 0.344195 | 0.371374 | +0.026533 | +0.013441 |
| B2_trend_only | 0.405746 | 0.412257 | +0.088084 | +0.054324 |
| GDIS_graph_disabled | 0.316000 | 0.356980 | -0.001662 | -0.000953 |

## weather
- Base (GBASE): MSE=0.172699, MAE=0.229743

| Variant | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|
| A1_no_dualstream | 0.182338 | 0.240254 | +0.009639 | +0.010511 |
| A2_no_SMGP | 0.172855 | 0.230114 | +0.000156 | +0.000371 |
| A3_gate_off | 0.172916 | 0.230135 | +0.000217 | +0.000392 |
| A4_no_base_graph | 0.172848 | 0.230173 | +0.000149 | +0.000430 |
| A5_no_sparsify | 0.172859 | 0.230099 | +0.000160 | +0.000356 |
| B1_no_TCN_linear | 0.194764 | 0.252387 | +0.022065 | +0.022644 |
| B2_trend_only | 0.208985 | 0.272936 | +0.036286 | +0.043193 |
| GDIS_graph_disabled | 0.172496 | 0.229368 | -0.000203 | -0.000375 |

## flotation
- Base (GBASE): MSE=0.786082, MAE=0.622778

| Variant | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|
| A1_no_dualstream | 0.769107 | 0.619271 | -0.016975 | -0.003507 |
| A2_no_SMGP | 0.786573 | 0.622971 | +0.000491 | +0.000193 |
| A3_gate_off | 0.786346 | 0.622887 | +0.000264 | +0.000109 |
| A4_no_base_graph | 0.786165 | 0.622791 | +0.000083 | +0.000013 |
| A5_no_sparsify | 0.786148 | 0.622813 | +0.000066 | +0.000035 |
| B1_no_TCN_linear | 0.809212 | 0.637784 | +0.023130 | +0.015006 |
| B2_trend_only | 1.039547 | 0.731307 | +0.253465 | +0.108529 |
| GDIS_graph_disabled | 0.787030 | 0.622920 | +0.000948 | +0.000142 |

## grinding
- Base (GBASE): MSE=4.239915, MAE=0.698634

| Variant | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|
| A1_no_dualstream | N/A | N/A | N/A | N/A |
| A2_no_SMGP | 4.251087 | 0.699515 | +0.011172 | +0.000881 |
| A3_gate_off | 4.249457 | 0.699139 | +0.009542 | +0.000505 |
| A4_no_base_graph | 4.249100 | 0.699120 | +0.009185 | +0.000486 |
| A5_no_sparsify | 4.250048 | 0.699244 | +0.010133 | +0.000610 |
| B1_no_TCN_linear | 5.069052 | 0.864069 | +0.829137 | +0.165435 |
| B2_trend_only | 6.634008 | 1.044827 | +2.394093 | +0.346193 |
| GDIS_graph_disabled | 4.244438 | 0.697960 | +0.004523 | -0.000674 |

## Summary Observations

- Dual-Stream removal (A1_no_dualstream) hurts ETTm1 and weather but helps flotation; grinding A1 is missing.
- Removing SMGP / gate / base graph / sparsify changes metrics only slightly in most datasets.
- Linear temporal encoder (B1_no_TCN_linear) degrades all datasets, especially grinding.
- Trend-only (B2_trend_only) fails across all datasets.
- The graph-disabled combined run closely matches the base in ETTm1/weather/flotation, and is slightly worse on grinding.