# Vampire benchmark: audited NeuralForest

All times are measured wall-clock values on GitHub-hosted CPU
runners with mathematical libraries constrained to one thread.
Selection time includes internal hyperparameter and expert search.
Final fit time measures the separately trained final estimator.
Prediction time includes evaluation inference on the held-out set.
The three stages are also available independently in summary.csv.

The experiment covers 19 scenarios, six seeds and 17 methods.
Successful runs: 1938 / 1938.
Seeds 131 and 197 were not used during the v0.3 experiment.

## Overall ranking for each scenario

| Scenario | Metric | Top three (six-seed mean) | NeuralForest calibrated | Rank | Selection (s) | Final fit (s) | Prediction (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| correlated_30 | auc | GradientBoosting 0.798, CatBoost 0.793, XGBoost 0.790 | 0.785 | 6 | 0.81 | 0.24 | 0.026 |
| distinct_0 | auc | ExtraTrees 0.993, AdaBoost 0.992, CatBoost 0.992 | 0.992 | 6 | 0.44 | 0.06 | 0.011 |
| distribution_shift | auc | ExtraTrees 0.692, CatBoost 0.687, NeuralForest_Adaptive 0.686 | 0.686 | 3 | 0.61 | 0.12 | 0.017 |
| identical_100 | logloss | Dummy 0.694, SVM_RBF 0.697, NeuralForest_Differentiable 0.698 | 0.698 | 4 | 0.34 | 0.09 | 0.015 |
| intermediate_45 | auc | CatBoost 0.907, GradientBoosting 0.904, AdaBoost 0.902 | 0.900 | 4 | 0.48 | 0.08 | 0.013 |
| label_noise_20pct | auc | ExtraTrees 0.698, NeuralForest_Adaptive 0.697, NeuralForest_Calibrated 0.697 | 0.697 | 2 | 0.72 | 0.19 | 0.028 |
| large_n | auc | CatBoost 0.779, GradientBoosting 0.775, XGBoost 0.774 | 0.771 | 5 | 2.06 | 0.37 | 0.024 |
| missing_15pct | auc | CatBoost 0.780, GradientBoosting 0.775, NeuralForest_CART 0.774 | 0.773 | 5 | 0.81 | 0.20 | 0.027 |
| noise_50 | auc | GradientBoosting 0.786, AdaBoost 0.781, CatBoost 0.778 | 0.771 | 9 | 0.98 | 0.29 | 0.036 |
| nonlinear_xor | f1 | GradientBoosting 0.997, CatBoost 0.997, HistGradientBoosting 0.994 | 0.984 | 8 | 0.53 | 0.14 | 0.020 |
| overlap_80 | auc | CatBoost 0.670, AdaBoost 0.667, GradientBoosting 0.665 | 0.663 | 5 | 0.44 | 0.09 | 0.014 |
| presence_bg_1to1 | boyce | Logistic 0.995, ExtraTrees 0.992, NeuralForest_Differentiable 0.991 | 0.990 | 5 | 0.29 | 0.07 | 0.027 |
| presence_bg_1to5 | boyce | NeuralForest_Adaptive 0.985, NeuralForest_Calibrated 0.985, NeuralForest_Differentiable 0.973 | 0.985 | 1 | 0.84 | 0.25 | 0.029 |
| pseudoabsence_hard | boyce | Logistic 0.988, NeuralForest_Adaptive 0.804, NeuralForest_Calibrated 0.804 | 0.804 | 2 | 0.33 | 0.07 | 0.025 |
| rare_10pct | pr_auc | ExtraTrees 0.520, Logistic 0.511, NeuralForest_Adaptive 0.508 | 0.508 | 3 | 0.48 | 0.10 | 0.014 |
| rare_1pct | pr_auc | Logistic 0.196, NeuralForest_Adaptive 0.186, NeuralForest_Calibrated 0.186 | 0.186 | 2 | 3.53 | 0.85 | 0.026 |
| rare_3pct | pr_auc | Logistic 0.290, NeuralForest_Adaptive 0.267, NeuralForest_Calibrated 0.267 | 0.267 | 2 | 1.14 | 0.22 | 0.018 |
| similar_97 | brier | Dummy 0.250, AdaBoost 0.252, SVM_RBF 0.252 | 0.252 | 4 | 0.35 | 0.09 | 0.014 |
| small_n | balanced_accuracy | AdaBoost 0.662, Logistic 0.660, NeuralForest_Differentiable 0.659 | 0.659 | 4 | 0.12 | 0.02 | 0.008 |

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
paired_v02_comparison.csv. Means over seeds 131 and 197 appear in
new_seed_primary_mean and new_seed_rank in summary.csv.

Presence-background training negatives are not confirmed human
labels. The independent test population has true synthetic labels,
which changes the interpretation of reported metrics.

The identical-distribution negative control should give chance
discrimination; a constant predictor often minimizes log loss.
These synthetic datasets do not establish universal superiority.

## Result files

- all_runs.csv: all metrics, configurations and timed stages
- summary.csv: mean, standard deviation, ranking and phase timings
- paired_v02_comparison.csv: paired original-seed differences
- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison
- manifest.json: provenance, seeds, failures and environment
- benchmark/README.md: full experimental protocol and limitations
