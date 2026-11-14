# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.014 | 0.002 | 0.0016 | 0.018 |
| Logistic | 19 | 0.034 | 0.004 | 0.0023 | 0.041 |
| MLP | 19 | 0.118 | 0.051 | 0.0031 | 0.172 |
| NeuralForest_UndersampledPlain | 19 | 0.081 | 0.091 | 0.0257 | 0.200 |
| XGBoost | 19 | 0.158 | 0.055 | 0.0063 | 0.220 |
| LightGBM | 19 | 0.161 | 0.073 | 0.0317 | 0.266 |
| SVM_RBF | 19 | 0.147 | 0.084 | 0.0546 | 0.286 |
| NeuralForest_SubSample50_Informed | 19 | 0.174 | 0.089 | 0.0257 | 0.290 |
| NeuralForest_UndersampledInformed | 19 | 0.279 | 0.093 | 0.0259 | 0.399 |
| NeuralForest_PUNeural | 3 | 0.185 | 0.210 | 0.0218 | 0.417 |
| ranger | 19 | 0.226 | 0.120 | 0.0739 | 0.420 |
| HistGradientBoosting | 19 | 0.277 | 0.132 | 0.0246 | 0.434 |
| NeuralForest_CART | 19 | 0.283 | 0.151 | 0.0272 | 0.463 |
| ExtraTrees | 19 | 0.328 | 0.151 | 0.0519 | 0.531 |
| NeuralForest_Differentiable | 19 | 0.372 | 0.149 | 0.0299 | 0.554 |
| NeuralForest_Calibrated | 19 | 0.402 | 0.149 | 0.0300 | 0.583 |
| NeuralForest_Adaptive | 19 | 0.413 | 0.150 | 0.0305 | 0.596 |
| NeuralForest_PUInformed | 3 | 0.278 | 0.326 | 0.0567 | 0.661 |
| CatBoost | 19 | 0.505 | 0.203 | 0.0037 | 0.712 |
| RF_Bootstrap50 | 19 | 0.462 | 0.238 | 0.0395 | 0.739 |
| GradientBoosting | 19 | 0.523 | 0.287 | 0.0056 | 0.816 |
| AdaBoost | 19 | 0.503 | 0.286 | 0.0361 | 0.826 |
| NeuralForest_PUSafe | 3 | 0.654 | 0.210 | 0.0185 | 0.882 |
| Voting_Boosting | 19 | 0.607 | 0.301 | 0.0352 | 0.944 |
| Voting_Heterogeneous | 19 | 0.614 | 0.299 | 0.0458 | 0.959 |
| RandomForest | 19 | 0.617 | 0.344 | 0.0432 | 1.004 |
| RF_Subsample50 | 19 | 0.643 | 0.333 | 0.0595 | 1.035 |
| Voting_RF_ExtraTrees | 19 | 0.706 | 0.380 | 0.0738 | 1.159 |
| Voting_RF_CatBoost | 19 | 0.824 | 0.438 | 0.0363 | 1.298 |
| RF_Subsample80 | 19 | 0.854 | 0.473 | 0.0634 | 1.390 |
| RF_BalancedUndersampled | 19 | 0.873 | 0.477 | 0.0455 | 1.395 |
| RF_BalancedUndersampledPrior | 19 | 0.877 | 0.475 | 0.0452 | 1.397 |
| NeuralForest_PUStable | 3 | 0.951 | 0.470 | 0.0381 | 1.460 |
| NeuralForest_PUStableTeacher | 3 | 0.951 | 0.470 | 0.0432 | 1.465 |
| Stacking_Boosting | 19 | 0.860 | 0.876 | 0.0205 | 1.757 |
| Stacking_RF_XGB | 19 | 0.982 | 1.091 | 0.0402 | 2.114 |

## Comparisons by scenario

A full per-scenario table of the standard RF, 50% and 80% row
subsampling, class-balanced under-sampling, ranger and informed
NeuralForest variants is in sampling_comparison.csv. Both the
primary objective gap and the total-time ratio versus RandomForest
are stored. Positive primary_advantage_vs_classic_rf is favorable.

All 6,040 per-seed score and timing records appear in all_runs.csv;
the cross-seed means and standard deviations are in summary.csv.
Prediction time covers BOTH independent true-label test cohorts.
PU methods are evaluated on three presence/background scenarios
only. Their mean times are not directly comparable with the
19-scenario averages of other methods.
The training budget and the initial predictive objective are
not perfectly matched across algorithm families.
