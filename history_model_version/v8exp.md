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

## Step3 Routing (prior=mi, topk=6, best per dataset)

| Dataset | Best exp_id | routing | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | routing_conf_mean | routing_alpha_mean | map_mean_abs | prior_sparsity | dyn_vs_prior_overlap | dyn_vs_prior_l1 | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | v8_step3_routing_ETTm1_mi_k6_affine | affine | 0.316557 | 0.357036 | 0.916860 | 0.126411 | 0.878832 | 1.752280 | 0.869836 | 1.000000 | 0.000013 | 0.002488 | 0.000335 | 1.000000 | 0.512030 | 0.082834 | 0.857143 | 1.000000 | 0.025657 | 6 | 7 |
| weather | v8_step3_routing_weather_mi_k6_det_g2 | det_g2 | 0.173101 | 0.231315 | 0.754258 | 0.106719 | 0.876049 | 1.645262 | 0.907315 | 0.999970 | 0.001505 | 0.002429 | 0.000335 | 0.291791 | 0.503885 | 0.050937 | 0.285714 | 1.000000 | 0.012468 | 6 | 21 |
| flotation | v8_step3_routing_flotation_mi_k6_affine | affine | 0.785597 | 0.622523 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | v8_step3_routing_grinding_mi_k6_det_g2 | det_g2 | 4.243982 | 0.698492 | 1.099435 | 0.074838 | 0.936269 | 1.789339 | 0.845232 | 0.998646 | 0.000927 | 0.002478 | 0.000335 | 0.558594 | 0.197115 | 0.046233 | 0.500000 | 0.983333 | 0.018059 | 6 | 12 |

## Step4 Graph Correction (prior=mi, topk=6, best per dataset)

| Dataset | Best exp_id | correction | mse | mae | E_trend | E_season | E_ratio | entropy_mean | topk_mass | topk_overlap | l1_adj_diff | gate_mean | alpha_mean | routing_conf_mean | routing_alpha_mean | beta_mean | delta_y_norm | map_mean_abs | prior_sparsity | dyn_vs_prior_overlap | dyn_vs_prior_l1 | segments | num_vars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | v8_step4_correction_ETTm1_mi_k6_season_L1 | season_L1 | 0.317796 | 0.357644 | 0.840986 | 0.133137 | 0.863326 | 1.791757 | 0.833611 | 0.869554 | 0.031694 | 0.002446 | 0.000335 | 1.000000 | 0.000000 | -0.000032 | 0.758261 | 0.079414 | 0.857143 | 0.828571 | 0.071664 | 6 | 7 |
| weather | v8_step4_correction_weather_mi_k6_full_L2 | full_L2 | 0.171781 | 0.228842 | 0.820046 | 0.169240 | 0.828927 | 1.648507 | 0.906422 | 0.997321 | 0.001542 | 0.002515 | 0.000335 | 0.299996 | 0.492848 | 0.014040 | 28.901339 | 0.086824 | 0.285714 | 1.000000 | 0.012632 | 6 | 21 |
| flotation | v8_step4_correction_flotation_mi_k6_full_L2 | full_L2 | 0.766786 | 0.615852 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| grinding | v8_step4_correction_grinding_mi_k6_season_L2 | season_L2 | 4.297673 | 0.733752 | 0.806709 | 0.033892 | 0.959681 | 1.789576 | 0.844705 | 0.998906 | 0.000877 | 0.002476 | 0.000335 | 0.569481 | 0.187620 | 0.000288 | 1.183417 | 0.043664 | 0.500000 | 1.000000 | 0.018205 | 6 | 12 |
