# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.014 | 0.002 | 0.0008 | 0.017 |
| Logistic | 0.034 | 0.005 | 0.0012 | 0.040 |
| MLP | 0.127 | 0.058 | 0.0017 | 0.187 |
| NeuralForest_UndersampledPlain | 0.091 | 0.103 | 0.0139 | 0.209 |
| XGBoost | 0.164 | 0.057 | 0.0035 | 0.224 |
| LightGBM | 0.172 | 0.079 | 0.0159 | 0.267 |
| SVM_RBF | 0.182 | 0.109 | 0.0316 | 0.323 |
| NeuralForest_SubSample50_Informed | 0.204 | 0.107 | 0.0135 | 0.327 |
| NeuralForest_UndersampledInformed | 0.311 | 0.103 | 0.0135 | 0.430 |
| ranger | 0.259 | 0.140 | 0.0380 | 0.437 |
| HistGradientBoosting | 0.297 | 0.139 | 0.0129 | 0.449 |
| NeuralForest_CART | 0.330 | 0.177 | 0.0144 | 0.526 |
| ExtraTrees | 0.344 | 0.160 | 0.0264 | 0.530 |
| NeuralForest_Differentiable | 0.425 | 0.177 | 0.0156 | 0.619 |
| NeuralForest_Calibrated | 0.476 | 0.179 | 0.0158 | 0.673 |
| NeuralForest_Adaptive | 0.480 | 0.178 | 0.0158 | 0.676 |
| CatBoost | 0.524 | 0.217 | 0.0020 | 0.743 |
| RF_Bootstrap50 | 0.494 | 0.260 | 0.0200 | 0.774 |
| AdaBoost | 0.535 | 0.309 | 0.0177 | 0.861 |
| GradientBoosting | 0.590 | 0.327 | 0.0029 | 0.920 |
| Voting_Boosting | 0.640 | 0.316 | 0.0178 | 0.974 |
| Voting_Heterogeneous | 0.641 | 0.316 | 0.0233 | 0.980 |
| RF_Subsample50 | 0.686 | 0.362 | 0.0299 | 1.077 |
| RandomForest | 0.679 | 0.384 | 0.0220 | 1.085 |
| Voting_RF_ExtraTrees | 0.762 | 0.418 | 0.0376 | 1.218 |
| Voting_RF_CatBoost | 0.892 | 0.477 | 0.0186 | 1.387 |
| RF_BalancedUndersampled | 0.917 | 0.502 | 0.0230 | 1.442 |
| RF_BalancedUndersampledPrior | 0.917 | 0.503 | 0.0231 | 1.443 |
| RF_Subsample80 | 0.932 | 0.526 | 0.0318 | 1.490 |
| Stacking_Boosting | 0.904 | 0.920 | 0.0103 | 1.834 |
| Stacking_RF_XGB | 1.056 | 1.186 | 0.0206 | 2.263 |

## Comparisons by scenario

A full per-scenario table of the standard RF, 50% and 80% row
subsampling, class-balanced under-sampling, ranger and informed
NeuralForest variants is in sampling_comparison.csv. Both the
primary objective gap and the total-time ratio versus RandomForest
are stored. Positive primary_advantage_vs_classic_rf is favorable.

All 5,890 per-seed score and timing records appear in all_runs.csv;
the cross-seed means and standard deviations are in summary.csv.
The training budget and the initial predictive objective are
not perfectly matched across algorithm families.
