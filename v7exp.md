# v7exp.md

## Step0 Baseline (B5)

| Dataset | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | map_mean_abs | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | 0.316886 | 0.357298 | 0.916860 | 0.126411 | 0.878832 | 1.791759 | 0.833373 | 0.856161 | 0.035400 | 0.002453 | 0.000336 | 0.082554 | 6 | 7 |
| weather | 0.172768 | 0.229623 | 0.764476 | 0.127317 | 0.857235 | 1.791759 | 0.833352 | 0.347262 | 0.059321 | 0.002495 | 0.000335 | 0.052466 | 6 | 21 |
| flotation | 0.785747 | 0.622590 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | 4.244823 | 0.698541 | 1.099435 | 0.074838 | 0.936269 | 1.791699 | 0.833493 | 0.516094 | 0.071209 | 0.002478 | 0.000335 | 0.046253 | 6 | 12 |

## Step1 Baseline (GraphMixerV7 = baseline)

| Dataset | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | map_mean_abs | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | 0.317247 | 0.357619 | 0.916860 | 0.126411 | 0.878832 | 1.791759 | 0.833376 | 0.856697 | 0.035170 | 0.002454 | 0.000336 | 0.082300 | 6 | 7 |
| weather | 0.172783 | 0.229756 | 0.764476 | 0.127317 | 0.857235 | 1.791759 | 0.833353 | 0.347708 | 0.059323 | 0.002495 | 0.000335 | 0.052720 | 6 | 21 |
| flotation | 0.785775 | 0.622599 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | 4.244899 | 0.698490 | 1.099435 | 0.074838 | 0.936269 | 1.791700 | 0.833492 | 0.515573 | 0.071196 | 0.002478 | 0.000335 | 0.046232 | 6 | 12 |

## Step2 S-GAT Sweep (best per dataset)

| Dataset | Best exp_id | mse | mae | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | v7_step2_gat_ETTm1_H4_K12_L1 | 0.323591 | 0.364114 | 1.945881 | 0.714750 | 0.862857 | 0.000259 | 0.000335 | 0.077858 | 0.876916 | 0.128858 | 0.871882 |
| weather | v7_step2_gat_weather_H4_K12_L1 | 0.172366 | 0.230349 | 2.652599 | 0.408391 | 0.380298 | 0.031578 | 0.000335 | 0.055124 | 0.942074 | 0.396860 | 0.703600 |
| flotation | v7_step2_gat_flotation_H4_K6_L2 | 0.782341 | 0.624838 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | v7_step2_gat_grinding_H4_K12_L1 | 4.265688 | 0.704722 | 2.484906 | 0.416703 | 0.529271 | 0.000009 | 0.000335 | 0.040003 | 0.884143 | 0.029665 | 0.967537 |

## Step3 Token-Attn Sweep (best per dataset)

| Dataset | Best exp_id | mse | mae |
| --- | --- | --- | --- |
| ETTm1 | v7_step3_attn_ETTm1_H8_W0_L1 | 0.318460 | 0.357827 |
| weather | v7_step3_attn_weather_H8_W0_L1 | 0.172186 | 0.230027 |
| flotation | v7_step3_attn_flotation_H8_W1_L1 | 0.787332 | 0.624011 |
| grinding | v7_step3_attn_grinding_H8_W1_L1 | 4.266300 | 0.704834 |

## Step4 GCN Sweep (best per dataset)

| Dataset | Best exp_id | mse | mae | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | v7_step4_gcn_ETTm1_L1_Nrow | 0.319897 | 0.356824 | 1.791759 | 0.833390 | 0.863929 | 0.033097 | 0.000335 | 0.081429 | 0.919558 | 0.135729 | 0.871382 |
| weather | v7_step4_gcn_weather_L2_Nsym | 0.171304 | 0.227934 | 1.788497 | 0.835252 | 0.387054 | 0.054985 | 0.000334 | 0.049353 | 0.779450 | 0.128585 | 0.858392 |
| flotation | v7_step4_gcn_flotation_L2_Nsym | 0.777328 | 0.622753 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | v7_step4_gcn_grinding_L1_Nsym | 4.298116 | 0.729698 | 1.791759 | 0.833340 | 0.514583 | 0.074394 | 0.000335 | 0.042733 | 0.794432 | 0.038293 | 0.954014 |
