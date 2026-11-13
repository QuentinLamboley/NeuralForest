# Vampire benchmark: audited NeuralForest

All times are measured wall-clock values on GitHub-hosted CPU
runners with mathematical libraries constrained to one thread.
Selection time includes internal hyperparameter and expert search.
Final fit time measures the separately trained final estimator.
Prediction time includes evaluation inference on the held-out set.
The three stages are also available independently in summary.csv.

The experiment covers 19 scenarios, eight seeds and 25 methods.
Successful runs: 3800 / 3800.
Seeds 251 and 347 are new confirmation seeds absent from earlier v0.4.1 runs.

## Overall ranking for each scenario

| Scenario | Metric | Top three (eight-seed mean) | NeuralForest adaptatif | Rank | Selection (s) | Final fit (s) | Prediction (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| correlated_30 | auc | GradientBoosting 0.796, CatBoost 0.792, NeuralForest_Adaptive 0.790 | 0.790 | 3 | 0.64 | 0.23 | 0.024 |
| distinct_0 | auc | ExtraTrees 0.993, AdaBoost 0.993, CatBoost 0.992 | 0.992 | 5 | 0.24 | 0.05 | 0.008 |
| distribution_shift | auc | ExtraTrees 0.692, CatBoost 0.687, NeuralForest_Adaptive 0.686 | 0.686 | 3 | 0.48 | 0.12 | 0.016 |
| identical_100 | logloss | Dummy 0.694, SVM_RBF 0.697, NeuralForest_Calibrated 0.697 | 0.697 | 5 | 0.23 | 0.08 | 0.012 |
| intermediate_45 | auc | CatBoost 0.906, GradientBoosting 0.903, AdaBoost 0.901 | 0.900 | 5 | 0.27 | 0.07 | 0.010 |
| label_noise_20pct | auc | NeuralForest_Adaptive 0.697, NeuralForest_Calibrated 0.697, NeuralForest_Differentiable 0.697 | 0.697 | 1 | 0.41 | 0.14 | 0.018 |
| large_n | auc | CatBoost 0.778, XGBoost 0.774, GradientBoosting 0.773 | 0.769 | 7 | 1.59 | 0.49 | 0.029 |
| missing_15pct | auc | CatBoost 0.777, NeuralForest_Differentiable 0.773, NeuralForest_Adaptive 0.773 | 0.773 | 3 | 0.46 | 0.14 | 0.016 |
| noise_50 | auc | GradientBoosting 0.786, AdaBoost 0.782, CatBoost 0.779 | 0.771 | 15 | 0.66 | 0.26 | 0.022 |
| nonlinear_xor | f1 | GradientBoosting 0.997, CatBoost 0.996, LightGBM 0.994 | 0.982 | 8 | 0.48 | 0.15 | 0.018 |
| overlap_80 | auc | CatBoost 0.669, AdaBoost 0.666, GradientBoosting 0.663 | 0.661 | 5 | 0.27 | 0.08 | 0.011 |
| presence_bg_1to1 | boyce | NeuralForest_SubSample50_Informed 0.995, Logistic 0.994, NeuralForest_Adaptive 0.994 | 0.994 | 3 | 0.22 | 0.05 | 0.018 |
| presence_bg_1to5 | boyce | NeuralForest_UndersampledInformed 0.989, NeuralForest_Adaptive 0.988, NeuralForest_Calibrated 0.988 | 0.988 | 2 | 0.55 | 0.18 | 0.021 |
| pseudoabsence_hard | boyce | Logistic 0.989, AdaBoost 0.724, NeuralForest_Adaptive 0.673 | 0.673 | 3 | 0.24 | 0.05 | 0.016 |
| rare_10pct | pr_auc | Logistic 0.510, ExtraTrees 0.509, NeuralForest_Adaptive 0.508 | 0.508 | 3 | 0.45 | 0.14 | 0.017 |
| rare_1pct | pr_auc | Logistic 0.177, NeuralForest_Adaptive 0.176, NeuralForest_Calibrated 0.176 | 0.176 | 2 | 1.32 | 0.62 | 0.015 |
| rare_3pct | pr_auc | Logistic 0.287, NeuralForest_Adaptive 0.278, NeuralForest_Calibrated 0.278 | 0.278 | 2 | 0.77 | 0.29 | 0.021 |
| similar_97 | brier | Dummy 0.250, AdaBoost 0.252, SVM_RBF 0.252 | 0.253 | 6 | 0.23 | 0.08 | 0.011 |
| small_n | balanced_accuracy | AdaBoost 0.663, NeuralForest_UndersampledInformed 0.662, NeuralForest_UndersampledPlain 0.662 | 0.657 | 7 | 0.15 | 0.03 | 0.010 |

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
paired_v02_comparison.csv. Means over seeds 251 and 347 appear in
new_seed_primary_mean and new_seed_rank in summary.csv.

Presence-background training negatives are not confirmed human
labels. The independent test population has true synthetic labels,
which changes the interpretation of reported metrics.

The identical-distribution negative control should give chance
discrimination; a constant predictor often minimizes log loss.
These synthetic datasets do not establish universal superiority.

## Result files

- all_runs.csv: all metrics, configurations and timed stages
- balanced under-sampling uses equal minority/majority per tree
  without replacement. Each tree uses a training-only class count.
- summary.csv: mean, standard deviation, ranking and phase timings
- paired_v02_comparison.csv: paired original-seed differences
- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison
- paired_v041_comparison.csv: six matched seeds against NeuralForest v0.4.1
- sampling_comparison.csv: forest-by-forest subsampling comparison
- COMPUTE_TIME.md: selection, final fit and prediction time by algorithm
- manifest.json: provenance, seeds, failures and environment
- benchmark/README.md: full experimental protocol and limitations
