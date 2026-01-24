# DynamicGraphMixer v2 Experiment Results (exp_resultsv2.md)

## Purpose
- Track v2 experiments defined in `implementationTODOv2.md`.
- Keep results reproducible with configs, seeds, and log paths.
- Record both metrics (MSE/MAE) and graph diagnostics.

## Current code notes (baseline context)
- Graphs are built per segment: mean pool -> LowRankGraphLearner -> optional base mix -> optional topk.
- GraphMixer gate is static: `none|scalar|per_var|per_token` (sigmoid on a learnable logit).
- Graph logging currently stores post-sparsify adjacencies per segment when enabled.
- Base L1 regularization uses mean(|A_base|) and is effectively constant.

## Baseline reference (anchor)
- Dataset/task:
- Model: DynamicGraphMixer
- Baseline config (M8-5 equivalent):
  - graph_source=content_mean
  - graph_base_mode=mix
  - gate_mode=scalar
  - adj_sparsify=topk, adj_topk=6
  - graph_scale=8
  - graph_rank=8
  - temporal_encoder=tcn
  - use_patch=false
- Expected metrics (from history):
  - MSE:
  - MAE:
- Notes:

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
  - gate/alpha: mean/std/quantiles/saturation
  - base: entropy/topk_mass/diag/offdiag

## Global record fields (fill per experiment)
| Field | Value |
| --- | --- |
| Date | |
| Exp ID | |
| Setting | |
| Seed(s) | |
| Dataset | |
| Seq/Pred | |
| Model | |
| Key config delta | |
| MSE / MAE | |
| Graph stats (raw/post entropy/topk_mass/topk_overlap/adj_diff/diag/offdiag) | |
| Conf stats (conf_mean/std/p10/p50/p90) | |
| Gate stats (mean/std/p10/p50/p90/sat_low/sat_high) | |
| Alpha stats (mean/std/p10/p50/p90/sat_low/sat_high) | |
| Base stats (entropy/topk_mass/diag/offdiag) | |
| Log paths | |
| Notes | |

---

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
#### E0-1 Baseline reproduction (1 seed)
| Field | Value |
| --- | --- |
| Exp ID | |
| Seed | |
| Config delta | |
| MSE / MAE | |
| Notes | |

#### E0-2 Baseline 3 seeds
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Seeds | |
| Mean MSE / MAE | |
| Std MSE / MAE | |
| Notes | |

#### E0-3 Baseline with graph_log enabled
| Field | Value |
| --- | --- |
| Exp ID | |
| Seed | |
| Log interval | |
| Log path | |
| Notes | |

### Summary
- Result:
- Decision:

---

## v2.1 - Static gate sweep (no structural change)

### Experiments
#### E1-A gate_mode sweep (none/scalar/per_var/per_token)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best gate_mode | |
| MSE / MAE | |
| Notes | |

#### E1-B gate_init sweep (per gate_mode)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best gate_init | |
| MSE / MAE | |
| Notes | |

#### E1-C graph_scale x best gate_mode
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best graph_scale | |
| MSE / MAE | |
| Notes | |

### Summary
- Best static gate config:
- Next step:

---

## v2.2 - Confidence gate (dynamic gate)

### Experiments (stage 1: quick scan)
#### E2-1 gate_mode (conf_scalar/conf_per_var)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best mode | |
| MSE / MAE | |
| Notes | |

#### E2-2 mapping (G1 vs G2)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best mapping | |
| MSE / MAE | |
| Notes | |

#### E2-3 gate_conf_source (pre/post sparsify)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best source | |
| MSE / MAE | |
| Notes | |

#### E2-4 gamma sweep (G1)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best gamma | |
| MSE / MAE | |
| Notes | |

#### E2-5 w_init/b_init sweep (G2)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best (w_init, b_init) | |
| MSE / MAE | |
| Notes | |

#### E2-6 warmup sweep
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best warmup | |
| MSE / MAE | |
| Notes | |

### Experiments (stage 2: 3 seeds)
#### E2-Final best 2-3 configs
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Seeds | |
| Mean MSE / MAE | |
| Std MSE / MAE | |
| Notes | |

### Summary
- Best conf gate config:
- Gate vs entropy trend:
- Decision:

---

## v2.3 - Confidence routing + base regularization

### Experiments
#### E3-A alpha mode (static vs conf)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best alpha mode | |
| MSE / MAE | |
| Notes | |

#### E3-B base_reg type and lambda
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best reg type | |
| Best lambda | |
| MSE / MAE | |
| Notes | |

#### E3-C gate vs alpha combination
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best combo | |
| MSE / MAE | |
| Notes | |

### Summary
- Best routing + reg config:
- Interpretation notes:

---

## v2.4 - Multi-scale stable graph fusion

### Experiments
#### E4-1 stable_feat_type
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best stable_feat_type | |
| MSE / MAE | |
| Notes | |

#### E4-2 stable_window
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best stable_window | |
| MSE / MAE | |
| Notes | |

#### E4-3 graph_scale_stable
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best graph_scale_stable | |
| MSE / MAE | |
| Notes | |

#### E4-4 stable_share_encoder / stable_detach
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best setting | |
| MSE / MAE | |
| Notes | |

#### E4-5 fuse_mode + beta_init
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best fuse_mode | |
| Best beta_init | |
| MSE / MAE | |
| Notes | |

### Summary
- Best stable fusion config:
- Decision:

---

## v2.5 - Lag-aware graph mixing

### Experiments
#### E5-1 lag length
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best L_lag | |
| MSE / MAE | |
| Notes | |

#### E5-2 shared vs per-lag adj
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best share_adj | |
| MSE / MAE | |
| Notes | |

#### E5-3 w_init
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best w_init | |
| MSE / MAE | |
| Notes | |

#### E5-4 interactions with conf gate/route
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Best combo | |
| MSE / MAE | |
| Notes | |

### Summary
- Best lag-aware config:
- Decision:

---

## v2.6 - Publication-ready experiment suite

### Experiments
#### Multi-horizon (pred_len 96/192/336/720)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Results | |
| Notes | |

#### Multi-dataset (ETTm2/ETTh1/ETTh2 ...)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Results | |
| Notes | |

#### Complexity (params/throughput/memory)
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Results | |
| Notes | |

#### Interpretability cases
| Field | Value |
| --- | --- |
| Exp ID(s) | |
| Results | |
| Notes | |

### Summary
- Final v2 config:
- Key ablations:
- Final decision:
