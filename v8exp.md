# v8exp.md

## Step0 Baseline (B5)

| Dataset | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | map_mean_abs | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | 0.316886 | 0.357298 | 0.916860 | 0.126411 | 0.878832 | 1.791759 | 0.833373 | 0.856161 | 0.035400 | 0.002453 | 0.000336 | 0.082554 | 6 | 7 |
| weather | 0.172768 | 0.229623 | 0.764476 | 0.127317 | 0.857235 | 1.791759 | 0.833352 | 0.347262 | 0.059321 | 0.002495 | 0.000335 | 0.052466 | 6 | 21 |
| flotation | 0.785747 | 0.622590 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | 4.244823 | 0.698541 | 1.099435 | 0.074838 | 0.936269 | 1.791699 | 0.833493 | 0.516094 | 0.071209 | 0.002478 | 0.000335 | 0.046253 | 6 | 12 |

## Step1 Prior Graph Sweep (topk=6, best per dataset)

| Dataset | Best exp_id | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | map_mean_abs | prior_sparsity | dyn_vs_prior_overlap | dyn_vs_prior_l1 | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | v8_step1_prior_ETTm1_mi_k6 | 0.316670 | 0.357124 | 0.916860 | 0.126411 | 0.878832 | 1.791759 | 0.833395 | 0.878036 | 0.016037 | 0.002460 | 0.000349 | 0.082413 | 0.857143 | 0.885714 | 0.055634 | 6 | 7 |
| weather | v8_step1_prior_weather_mi_k6 | 0.172732 | 0.229718 | 0.764476 | 0.127317 | 0.857235 | 1.791759 | 0.833453 | 0.859702 | 0.009003 | 0.002461 | 0.000366 | 0.052741 | 0.285714 | 0.980952 | 0.029904 | 6 | 21 |
| flotation | v8_step1_prior_flotation_mi_k6 | 0.785694 | 0.622555 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | v8_step1_prior_grinding_pearson_abs_k6 | 4.244828 | 0.698469 | 1.099435 | 0.074838 | 0.936269 | 1.791691 | 0.833532 | 0.789740 | 0.025747 | 0.002490 | 0.000341 | 0.046252 | 0.500000 | 0.883333 | 0.031538 | 6 | 12 |

## Step2 Seasonal GCN (prior=mi, topk=6, best per dataset)

| Dataset | Best exp_id | gcn_layers | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | g_scale_mean | map_mean_abs | prior_sparsity | dyn_vs_prior_overlap | dyn_vs_prior_l1 | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | v8_step2_season_gcn_ETTm1_mi_k6_L1 | 1 | 0.318758 | 0.357954 | 0.919558 | 0.135729 | 0.871382 | 1.791758 | 0.833452 | 0.871875 | 0.025874 | 0.002457 | 0.000350 | -0.056467 | 0.084157 | 0.857143 | 0.914286 | 0.056989 | 6 | 7 |
| weather | v8_step2_season_gcn_weather_mi_k6_L2 | 2 | 0.172441 | 0.230802 | 0.861675 | 0.382304 | 0.692676 | 1.791759 | 0.833453 | 0.866756 | 0.008170 | 0.002463 | 0.000362 | -0.056579 | 0.053564 | 0.285714 | 0.961905 | 0.029904 | 6 | 21 |
| flotation | v8_step2_season_gcn_flotation_mi_k6_L2 | 2 | 0.776968 | 0.622577 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | v8_step2_season_gcn_grinding_mi_k6_L1 | 1 | 4.300831 | 0.730693 | 0.794432 | 0.038293 | 0.954014 | 1.791756 | 0.833399 | 0.815990 | 0.018412 | 0.002478 | 0.000334 | -0.018950 | 0.043294 | 0.500000 | 0.950000 | 0.024189 | 6 | 12 |
