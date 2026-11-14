# Independent virtual-species recorder effort-offset challenge

**Development simulation, NOT independent empirical ecological validation.**

Observed presence process is habitat intensity MULTIPLIED by an
independently generated recorder-effort process. Training background
samples describe environmental availability, NOT true absence.
True ecological intensity is known only to the simulator and used
only for final post-fit scoring. Test landscapes and observer effort
never enter estimator training or inner stage selection.

Hypotheses were set before running this new simulation suite:
an explicit known-effort IPP offset should reduce habitat/effort
confounding under biased observation, while nearly matching
the naive formulation under constant effort. A deliberately
imperfect independently generated effort proxy measures fragility.
An effort-weighted CatBoost control evaluates whether any gain
is simply generic class-background sampling correction.

Three independent random seeds × three geographically separate
outer folds × two virtual habitats × two observation-effort
regimes = 36 paired conditions, 13 models = 468 prediction
records. A different training-only geographical block selects
the number of native forest correction stages. Spatial-fold
scores share much of their training data and are correlated.

## Ecological oracle Spearman correlation (higher better)

| Habitat | Recorder bias | Native neural naive | Native neural known q | Gradient naive | Gradient known q | Gradient proxy q | MaxEnt naive | CatBoost effort-weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ecotone | biased | 0.8122 | 0.6723 | 0.8103 | 0.6829 | 0.8206 | 0.8561 | 0.8429 |
| ecotone | uniform | 0.7904 | 0.7904 | 0.7965 | 0.7965 | 0.7965 | 0.8977 | 0.8815 |
| interaction | biased | 0.2991 | 0.3506 | 0.3074 | 0.3842 | 0.4529 | 0.6678 | 0.8760 |
| interaction | uniform | 0.6775 | 0.6775 | 0.7045 | 0.7045 | 0.7045 | 0.9067 | 0.9089 |

## Paired oracle suitability effects, all cases

| Habitat | Recorder bias | Estimator contrast | Mean oracle rho gain | Wins/cases |
| --- | --- | --- | ---: | ---: |
| ecotone | biased | CatBoost_effort_weighted vs CatBoost_naive | +0.0108 | 3/9 |
| ecotone | biased | GLM_true vs GLM_naive | -0.1836 | 1/9 |
| ecotone | biased | IPP_gradient_proxy_hidden vs IPP_gradient_naive_hidden | +0.0103 | 6/9 |
| ecotone | biased | IPP_gradient_true_hidden vs CatBoost_effort_weighted | -0.1599 | 2/9 |
| ecotone | biased | IPP_gradient_true_hidden vs IPP_gradient_naive_hidden | -0.1273 | 3/9 |
| ecotone | biased | IPP_gradient_true_hidden vs IPP_gradient_true_random | +0.0122 | 4/9 |
| ecotone | biased | IPP_gradient_true_hidden vs IPP_gradient_true_raw | +0.0057 | 3/9 |
| ecotone | biased | IPP_gradient_true_hidden vs IPP_neural_true | +0.0107 | 3/9 |
| ecotone | biased | IPP_gradient_true_hidden vs MaxEnt_lqph | -0.1732 | 1/9 |
| ecotone | biased | IPP_neural_proxy vs IPP_neural_naive | +0.0017 | 6/9 |
| ecotone | biased | IPP_neural_true vs IPP_neural_naive | -0.1400 | 3/9 |
| ecotone | uniform | CatBoost_effort_weighted vs CatBoost_naive | +0.0000 | 0/9 |
| ecotone | uniform | GLM_true vs GLM_naive | +0.0000 | 0/9 |
| ecotone | uniform | IPP_gradient_proxy_hidden vs IPP_gradient_naive_hidden | +0.0000 | 0/9 |
| ecotone | uniform | IPP_gradient_true_hidden vs CatBoost_effort_weighted | -0.0850 | 5/9 |
| ecotone | uniform | IPP_gradient_true_hidden vs IPP_gradient_naive_hidden | +0.0000 | 0/9 |
| ecotone | uniform | IPP_gradient_true_hidden vs IPP_gradient_true_random | -0.0050 | 3/9 |
| ecotone | uniform | IPP_gradient_true_hidden vs IPP_gradient_true_raw | -0.0064 | 2/9 |
| ecotone | uniform | IPP_gradient_true_hidden vs IPP_neural_true | +0.0061 | 5/9 |
| ecotone | uniform | IPP_gradient_true_hidden vs MaxEnt_lqph | -0.1012 | 2/9 |
| ecotone | uniform | IPP_neural_proxy vs IPP_neural_naive | +0.0000 | 0/9 |
| ecotone | uniform | IPP_neural_true vs IPP_neural_naive | +0.0000 | 0/9 |
| interaction | biased | CatBoost_effort_weighted vs CatBoost_naive | +0.1478 | 9/9 |
| interaction | biased | GLM_true vs GLM_naive | +0.0259 | 3/9 |
| interaction | biased | IPP_gradient_proxy_hidden vs IPP_gradient_naive_hidden | +0.1455 | 7/9 |
| interaction | biased | IPP_gradient_true_hidden vs CatBoost_effort_weighted | -0.4918 | 0/9 |
| interaction | biased | IPP_gradient_true_hidden vs IPP_gradient_naive_hidden | +0.0768 | 6/9 |
| interaction | biased | IPP_gradient_true_hidden vs IPP_gradient_true_random | +0.0312 | 5/9 |
| interaction | biased | IPP_gradient_true_hidden vs IPP_gradient_true_raw | +0.0229 | 5/9 |
| interaction | biased | IPP_gradient_true_hidden vs IPP_neural_true | +0.0335 | 6/9 |
| interaction | biased | IPP_gradient_true_hidden vs MaxEnt_lqph | -0.2837 | 1/9 |
| interaction | biased | IPP_neural_proxy vs IPP_neural_naive | +0.1384 | 7/9 |
| interaction | biased | IPP_neural_true vs IPP_neural_naive | +0.0516 | 6/9 |
| interaction | uniform | CatBoost_effort_weighted vs CatBoost_naive | +0.0000 | 0/9 |
| interaction | uniform | GLM_true vs GLM_naive | +0.0000 | 0/9 |
| interaction | uniform | IPP_gradient_proxy_hidden vs IPP_gradient_naive_hidden | +0.0000 | 0/9 |
| interaction | uniform | IPP_gradient_true_hidden vs CatBoost_effort_weighted | -0.2043 | 0/9 |
| interaction | uniform | IPP_gradient_true_hidden vs IPP_gradient_naive_hidden | +0.0000 | 0/9 |
| interaction | uniform | IPP_gradient_true_hidden vs IPP_gradient_true_random | +0.0182 | 5/9 |
| interaction | uniform | IPP_gradient_true_hidden vs IPP_gradient_true_raw | +0.0133 | 5/9 |
| interaction | uniform | IPP_gradient_true_hidden vs IPP_neural_true | +0.0270 | 6/9 |
| interaction | uniform | IPP_gradient_true_hidden vs MaxEnt_lqph | -0.2021 | 0/9 |
| interaction | uniform | IPP_neural_proxy vs IPP_neural_naive | +0.0000 | 0/9 |
| interaction | uniform | IPP_neural_true vs IPP_neural_naive | +0.0000 | 0/9 |

## Important interpretation safeguards

- The detection effort is **known by oracle in simulation**. Real
  opportunistic datasets rarely provide this precise independent
  surface; inaccurate proxy effort can make correction worse.
- Applying an offset is a standard point-process idea and alone
  is NOT a new algorithm. The potential contribution requires
  native objective-aligned C++ forests, sampling-effort robustness,
  matched raw/random-feature attribution and real-world external
  ecological improvements beyond DeepMaxent and MaxEnt.
- The research simulation has been generated during model
  development: even if it favors NeuralForest, it cannot
  establish universal superiority or independent confirmation.
- Full generalization requires genuine independent survey
  detection histories, spatially/temporally held-out surveys,
  uncertainty of effort proxies and equal-budget spatial tuning.
- Presence-background AUC can be artificially high because
  recorder bias is predictable; Oracle habitat correlation is
  the PRIMARY known-truth metric here.
- Scores are relative habitat intensity, NEVER calibrated
  species occupancy, pathogen infection or abundance estimates.

See all_runs.csv, paired_effect.csv, paired_summary.csv,
stage_selections.csv and group_means.csv for every metric,
timing and unfavorable case.
