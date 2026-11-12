# Vampire benchmark: audited NeuralForest

All times are measured wall-clock values on GitHub-hosted CPU
runners with mathematical libraries constrained to one thread.
Selection time includes internal hyperparameter and expert search.
Final fit time measures the separately trained final estimator.
Prediction time includes evaluation inference on the held-out set.
The three stages are also available independently in summary.csv.

The experiment covers 19 scenarios, four seeds and 16 methods.
Successful runs: 1216 / 1216.
Seeds 71 and 97 were not used in the initial v0.2 benchmark.

## Overall ranking for each scenario

| Scenario | Metric | Top three (four-seed mean) | NeuralForest adaptive | Rank | Selection (s) | Final fit (s) | Prediction (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| correlated_30 | auc | GradientBoosting 0.793, CatBoost 0.792, AdaBoost 0.789 | 0.785 | 6 | 0.68 | 0.20 | 0.023 |
| distinct_0 | auc | ExtraTrees 0.994, CatBoost 0.993, AdaBoost 0.993 | 0.992 | 4 | 0.29 | 0.05 | 0.009 |
| distribution_shift | auc | ExtraTrees 0.692, CatBoost 0.687, NeuralForest_Adaptive 0.686 | 0.686 | 3 | 0.55 | 0.10 | 0.015 |
| identical_100 | logloss | Dummy 0.694, NeuralForest_Adaptive 0.695, NeuralForest_Differentiable 0.695 | 0.695 | 2 | 0.21 | 0.08 | 0.014 |
| intermediate_45 | auc | CatBoost 0.908, NeuralForest_Adaptive 0.905, GradientBoosting 0.904 | 0.905 | 2 | 0.33 | 0.07 | 0.011 |
| label_noise_20pct | auc | NeuralForest_Adaptive 0.694, NeuralForest_Differentiable 0.694, ExtraTrees 0.693 | 0.694 | 1 | 0.57 | 0.16 | 0.023 |
| large_n | auc | CatBoost 0.777, GradientBoosting 0.773, XGBoost 0.773 | 0.770 | 4 | 1.53 | 0.35 | 0.020 |
| missing_15pct | auc | CatBoost 0.779, GradientBoosting 0.774, NeuralForest_Adaptive 0.773 | 0.773 | 3 | 0.68 | 0.17 | 0.022 |
| noise_50 | auc | GradientBoosting 0.782, AdaBoost 0.780, CatBoost 0.775 | 0.769 | 8 | 0.91 | 0.24 | 0.030 |
| nonlinear_xor | f1 | CatBoost 0.998, GradientBoosting 0.997, HistGradientBoosting 0.995 | 0.988 | 7 | 0.48 | 0.12 | 0.019 |
| overlap_80 | auc | AdaBoost 0.678, CatBoost 0.674, Logistic 0.673 | 0.668 | 5 | 0.30 | 0.08 | 0.012 |
| presence_bg_1to1 | boyce | NeuralForest_Adaptive 0.996, Logistic 0.994, ExtraTrees 0.993 | 0.996 | 1 | 0.32 | 0.06 | 0.034 |
| presence_bg_1to5 | boyce | MLP 0.968, NeuralForest_CART 0.965, NeuralForest_Differentiable 0.962 | 0.955 | 5 | 0.84 | 0.23 | 0.025 |
| pseudoabsence_hard | boyce | Logistic 0.989, AdaBoost 0.809, NeuralForest_Adaptive 0.792 | 0.792 | 3 | 0.34 | 0.06 | 0.022 |
| rare_10pct | pr_auc | ExtraTrees 0.532, NeuralForest_Adaptive 0.527, Logistic 0.524 | 0.527 | 2 | 0.43 | 0.10 | 0.013 |
| rare_1pct | pr_auc | Logistic 0.116, AdaBoost 0.111, NeuralForest_Adaptive 0.110 | 0.110 | 3 | 2.83 | 0.74 | 0.021 |
| rare_3pct | pr_auc | Logistic 0.264, ExtraTrees 0.228, NeuralForest_Adaptive 0.222 | 0.222 | 3 | 0.85 | 0.21 | 0.015 |
| similar_97 | brier | Dummy 0.250, Logistic 0.251, NeuralForest_Adaptive 0.252 | 0.252 | 3 | 0.22 | 0.08 | 0.013 |
| small_n | balanced_accuracy | Logistic 0.669, NeuralForest_Adaptive 0.667, AdaBoost 0.665 | 0.667 | 2 | 0.09 | 0.02 | 0.006 |

## Interpretation

The validation set selects algorithms, model complexity and mixtures
without consulting independent test outcomes. Non-differentiable
metrics use a differentiable proxy in soft trees. Average precision
uses a smooth listwise ranking surrogate, while AUC and Boyce use
a pairwise logistic surrogate. Test performance always uses the
original metric, not its surrogate.

NeuralForest_Adaptive includes ridge-regularized linear and constant
experts. Its performance must be interpreted as a hybrid ensemble,
not attributed exclusively to learned tree divisions.

Historical seed-matched v0.2 comparisons are in
paired_v02_comparison.csv. Means over seeds 71 and 97 appear in
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
- manifest.json: provenance, seeds, failures and environment
- benchmark/README.md: full experimental protocol and limitations
