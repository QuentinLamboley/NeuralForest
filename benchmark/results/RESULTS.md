# Vampire benchmark: audited NeuralForest

All times are measured wall-clock values on GitHub-hosted CPU
runners with mathematical libraries constrained to one thread.
Selection time includes internal hyperparameter and expert search.
Final fit time measures the separately trained final estimator.
Prediction time includes evaluation inference on the held-out set.
The three stages are also available independently in summary.csv.

The experiment covers 19 scenarios, ten seeds and 31 standard methods
plus five PU-specific predictions in three presence/background scenarios.
The matched two-restart teacher shares the fit and selection cost
of PUStable: it is not a separate independently trained algorithm.
Successful prediction records: 6040 / 6040.
Independent model fits: 6010; the additional
PUStableTeacher prediction shares its fitted model with PUStable.
Seeds 401 and 503 remain confirmation seeds from the v0.6 audit.
A second shifted simulator population is evaluated independently
without using any of its labels during fitting or model selection.
This is synthetic transfer validation, not real-world external validation.

## Overall ranking for each scenario

| Scenario | Metric | Top three (ten-seed mean) | NeuralForest adaptatif | Rank | Selection (s) | Final fit (s) | Prediction (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| correlated_30 | auc | Voting_Heterogeneous 0.796, GradientBoosting 0.795, Voting_RF_CatBoost 0.794 | 0.790 | 8 | 0.61 | 0.24 | 0.047 |
| distinct_0 | auc | Voting_RF_ExtraTrees 0.994, ExtraTrees 0.993, Voting_Heterogeneous 0.993 | 0.992 | 14 | 0.30 | 0.06 | 0.020 |
| distribution_shift | auc | Voting_Heterogeneous 0.692, ExtraTrees 0.691, Voting_RF_ExtraTrees 0.686 | 0.683 | 6 | 0.44 | 0.12 | 0.035 |
| identical_100 | logloss | Dummy 0.694, NeuralForest_Differentiable 0.694, Stacking_RF_XGB 0.694 | 0.695 | 5 | 0.27 | 0.09 | 0.027 |
| intermediate_45 | auc | CatBoost 0.909, Stacking_Boosting 0.908, Voting_RF_CatBoost 0.907 | 0.901 | 12 | 0.33 | 0.08 | 0.025 |
| label_noise_20pct | auc | Voting_Heterogeneous 0.701, CatBoost 0.696, NeuralForest_Differentiable 0.696 | 0.696 | 5 | 0.35 | 0.14 | 0.032 |
| large_n | auc | CatBoost 0.779, Stacking_Boosting 0.777, Voting_RF_CatBoost 0.776 | 0.772 | 9 | 1.38 | 0.49 | 0.057 |
| missing_15pct | auc | Stacking_Boosting 0.780, Voting_Heterogeneous 0.779, CatBoost 0.779 | 0.773 | 8 | 0.40 | 0.14 | 0.032 |
| noise_50 | auc | GradientBoosting 0.786, AdaBoost 0.782, Voting_Boosting 0.781 | 0.773 | 19 | 0.73 | 0.29 | 0.054 |
| nonlinear_xor | f1 | Voting_RF_CatBoost 0.997, GradientBoosting 0.997, CatBoost 0.997 | 0.983 | 15 | 0.42 | 0.14 | 0.036 |
| overlap_80 | auc | Voting_Heterogeneous 0.675, AdaBoost 0.671, CatBoost 0.670 | 0.666 | 6 | 0.30 | 0.09 | 0.026 |
| presence_bg_1to1 | boyce | NeuralForest_SubSample50_Informed 0.995, Logistic 0.995, Voting_Heterogeneous 0.993 | 0.993 | 4 | 0.23 | 0.06 | 0.045 |
| presence_bg_1to5 | boyce | NeuralForest_PUInformed 0.990, NeuralForest_Differentiable 0.988, NeuralForest_PUSafe 0.988 | 0.986 | 7 | 0.61 | 0.23 | 0.052 |
| pseudoabsence_hard | boyce | Logistic 0.990, NeuralForest_PUStableTeacher 0.970, NeuralForest_PUSafe 0.968 | 0.721 | 8 | 0.23 | 0.06 | 0.042 |
| rare_10pct | pr_auc | Voting_Heterogeneous 0.521, Logistic 0.508, ExtraTrees 0.505 | 0.502 | 4 | 0.40 | 0.14 | 0.035 |
| rare_1pct | pr_auc | NeuralForest_Adaptive 0.183, NeuralForest_Calibrated 0.183, Voting_Heterogeneous 0.178 | 0.183 | 1 | 1.27 | 0.63 | 0.030 |
| rare_3pct | pr_auc | Voting_Heterogeneous 0.290, Logistic 0.284, ExtraTrees 0.272 | 0.272 | 4 | 0.73 | 0.29 | 0.044 |
| similar_97 | brier | Dummy 0.250, Stacking_RF_XGB 0.251, Stacking_Boosting 0.251 | 0.252 | 6 | 0.27 | 0.09 | 0.027 |
| small_n | balanced_accuracy | Voting_Heterogeneous 0.661, NeuralForest_UndersampledPlain 0.659, AdaBoost 0.657 | 0.650 | 15 | 0.13 | 0.04 | 0.019 |

## Interpretation

The validation set selects algorithms, model complexity and mixtures
without consulting independent test outcomes. Non-differentiable
metrics use a differentiable proxy in soft trees. Average precision
uses a smooth listwise ranking surrogate, while AUC and Boyce use
a pairwise logistic surrogate. Test performance always uses the
original metric, not its surrogate.

NeuralForest_Adaptive and NeuralForest_Calibrated include ridge-regularized linear and constant
experts. Its performance must be interpreted as a hybrid ensemble,
not attributed exclusively to learned tree divisions.

Historical seed-matched v0.2 comparisons are in
paired_v02_comparison.csv. Means over seeds 401 and 503 appear in
new_seed_primary_mean and new_seed_rank in summary.csv.

Presence-background training negatives are not confirmed human
labels. The independent test population has true synthetic labels,
which changes the interpretation of reported metrics.

The identical-distribution negative control should give chance
discrimination; a constant predictor often minimizes log loss.
These synthetic datasets do not establish universal superiority.
For 15 metric-specific comparisons and voting/stacking ensembles,
consult METRICS_REPORT.md. The extra metrics are evaluated only
on held-out test sets and were not used to select candidates.

## Result files

- all_runs.csv: all metrics, configurations and timed stages
- balanced under-sampling uses equal minority/majority per tree
  without replacement. Each tree uses a training-only class count.
- summary.csv: mean, standard deviation, ranking and phase timings
- paired_v02_comparison.csv: paired original-seed differences
- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison
- paired_v041_comparison.csv: six matched seeds against NeuralForest v0.4.1
- paired_v05_comparison.csv: eight matched seeds against v0.5
- METRICS_REPORT.md: rankings across fifteen independent test metrics
- external_validation.csv: out-of-distribution shifted cohort scores
- EXTERNAL_VALIDATION.md: shifted-cohort method comparisons
- PU_STABILITY.md: cross-fitted uncertainty-gating decisions
- pu_stability_choices.csv: every selected sampling strategy
- pu_safe_choices.csv: fidelity-gated candidate decisions
- paired_v07_comparison.csv: same-seed reference against v0.7
- all_metric_ranks.csv, metric_winners.csv: all metric rankings
- ensemble_comparison.csv: voted and stacked ensemble comparison
- paired_multi_metrics.csv: paired differences and 95% intervals
- average_metric_ranks.csv: rank summaries across scenario families
- sampling_comparison.csv: forest-by-forest subsampling comparison
- COMPUTE_TIME.md: selection, final fit and prediction time by algorithm
- manifest.json: provenance, seeds, failures and environment
- benchmark/README.md: full experimental protocol and limitations
