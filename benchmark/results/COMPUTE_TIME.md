# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.016 | 0.003 | 0.0018 | 0.021 |
| Logistic | 19 | 0.039 | 0.005 | 0.0027 | 0.047 |
| MLP | 19 | 0.137 | 0.060 | 0.0036 | 0.201 |
| NeuralForest_UndersampledPlain | 19 | 0.092 | 0.103 | 0.0295 | 0.226 |
| XGBoost | 19 | 0.187 | 0.065 | 0.0073 | 0.260 |
| LightGBM | 19 | 0.190 | 0.087 | 0.0366 | 0.314 |
| NeuralForest_SubSample50_Informed | 19 | 0.199 | 0.101 | 0.0294 | 0.332 |
| SVM_RBF | 19 | 0.176 | 0.100 | 0.0632 | 0.339 |
| NeuralForest_PUNeural | 3 | 0.186 | 0.210 | 0.0209 | 0.417 |
| NeuralForest_UndersampledInformed | 19 | 0.318 | 0.105 | 0.0296 | 0.454 |
| ranger | 19 | 0.272 | 0.149 | 0.0847 | 0.505 |
| HistGradientBoosting | 19 | 0.325 | 0.155 | 0.0279 | 0.508 |
| NeuralForest_CART | 19 | 0.322 | 0.171 | 0.0310 | 0.526 |
| ExtraTrees | 19 | 0.379 | 0.175 | 0.0582 | 0.613 |
| NeuralForest_Differentiable | 19 | 0.425 | 0.169 | 0.0342 | 0.630 |
| NeuralForest_PUInformed | 3 | 0.276 | 0.328 | 0.0573 | 0.662 |
| NeuralForest_Calibrated | 19 | 0.462 | 0.171 | 0.0345 | 0.669 |
| NeuralForest_Adaptive | 19 | 0.472 | 0.170 | 0.0350 | 0.679 |
| CatBoost | 19 | 0.595 | 0.241 | 0.0041 | 0.840 |
| RF_Bootstrap50 | 19 | 0.533 | 0.276 | 0.0446 | 0.854 |
| GradientBoosting | 19 | 0.612 | 0.338 | 0.0063 | 0.957 |
| AdaBoost | 19 | 0.584 | 0.334 | 0.0412 | 0.960 |
| Voting_Boosting | 19 | 0.717 | 0.356 | 0.0403 | 1.114 |
| Voting_Heterogeneous | 19 | 0.720 | 0.351 | 0.0509 | 1.122 |
| RandomForest | 19 | 0.714 | 0.398 | 0.0485 | 1.161 |
| RF_Subsample50 | 19 | 0.745 | 0.386 | 0.0672 | 1.198 |
| Voting_RF_ExtraTrees | 19 | 0.816 | 0.441 | 0.0828 | 1.340 |
| NeuralForest_PUStable | 3 | 0.946 | 0.512 | 0.0445 | 1.503 |
| Voting_RF_CatBoost | 19 | 0.966 | 0.512 | 0.0408 | 1.518 |
| RF_BalancedUndersampled | 19 | 0.992 | 0.539 | 0.0508 | 1.582 |
| RF_BalancedUndersampledPrior | 19 | 0.993 | 0.539 | 0.0509 | 1.583 |
| RF_Subsample80 | 19 | 0.988 | 0.549 | 0.0713 | 1.609 |
| Stacking_Boosting | 19 | 1.026 | 1.044 | 0.0233 | 2.094 |
| Stacking_RF_XGB | 19 | 1.143 | 1.272 | 0.0453 | 2.461 |

## Comparisons by scenario

A full per-scenario table of the standard RF, 50% and 80% row
subsampling, class-balanced under-sampling, ranger and informed
NeuralForest variants is in sampling_comparison.csv. Both the
primary objective gap and the total-time ratio versus RandomForest
are stored. Positive primary_advantage_vs_classic_rf is favorable.

All 5,980 per-seed score and timing records appear in all_runs.csv;
the cross-seed means and standard deviations are in summary.csv.
Prediction time covers BOTH independent true-label test cohorts.
PU methods are evaluated on three presence/background scenarios
only. Their mean times are not directly comparable with the
19-scenario averages of other methods.
The training budget and the initial predictive objective are
not perfectly matched across algorithm families.
