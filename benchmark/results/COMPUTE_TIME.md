# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.018 | 0.003 | 0.0020 | 0.023 |
| Logistic | 19 | 0.044 | 0.006 | 0.0029 | 0.052 |
| MLP | 19 | 0.153 | 0.068 | 0.0038 | 0.225 |
| NeuralForest_UndersampledPlain | 19 | 0.101 | 0.114 | 0.0303 | 0.247 |
| XGBoost | 19 | 0.204 | 0.071 | 0.0073 | 0.282 |
| LightGBM | 19 | 0.205 | 0.095 | 0.0354 | 0.335 |
| NeuralForest_SubSample50_Informed | 19 | 0.209 | 0.108 | 0.0302 | 0.349 |
| SVM_RBF | 19 | 0.192 | 0.110 | 0.0687 | 0.371 |
| NeuralForest_PUNeural | 3 | 0.185 | 0.211 | 0.0214 | 0.417 |
| NeuralForest_UndersampledInformed | 19 | 0.342 | 0.115 | 0.0305 | 0.490 |
| ranger | 19 | 0.277 | 0.151 | 0.0859 | 0.515 |
| NeuralForest_CART | 19 | 0.339 | 0.180 | 0.0319 | 0.553 |
| HistGradientBoosting | 19 | 0.366 | 0.175 | 0.0288 | 0.570 |
| NeuralForest_PUInformed | 3 | 0.277 | 0.326 | 0.0565 | 0.659 |
| ExtraTrees | 19 | 0.411 | 0.187 | 0.0625 | 0.660 |
| NeuralForest_Differentiable | 19 | 0.449 | 0.179 | 0.0353 | 0.665 |
| NeuralForest_Calibrated | 19 | 0.488 | 0.180 | 0.0356 | 0.706 |
| NeuralForest_Adaptive | 19 | 0.496 | 0.180 | 0.0360 | 0.715 |
| NeuralForest_PUSafe | 3 | 0.649 | 0.209 | 0.0179 | 0.876 |
| RF_Bootstrap50 | 19 | 0.573 | 0.292 | 0.0484 | 0.913 |
| CatBoost | 19 | 0.654 | 0.270 | 0.0042 | 0.928 |
| GradientBoosting | 19 | 0.651 | 0.353 | 0.0065 | 1.011 |
| AdaBoost | 19 | 0.632 | 0.356 | 0.0480 | 1.036 |
| Voting_Boosting | 19 | 0.784 | 0.389 | 0.0389 | 1.212 |
| Voting_Heterogeneous | 19 | 0.784 | 0.382 | 0.0541 | 1.219 |
| RandomForest | 19 | 0.755 | 0.414 | 0.0523 | 1.221 |
| RF_Subsample50 | 19 | 0.794 | 0.405 | 0.0748 | 1.274 |
| Voting_RF_ExtraTrees | 19 | 0.869 | 0.462 | 0.0888 | 1.420 |
| NeuralForest_PUStable | 3 | 0.946 | 0.469 | 0.0377 | 1.453 |
| NeuralForest_PUStableTeacher | 3 | 0.946 | 0.469 | 0.0425 | 1.458 |
| Voting_RF_CatBoost | 19 | 1.038 | 0.547 | 0.0437 | 1.629 |
| RF_Subsample80 | 19 | 1.039 | 0.569 | 0.0789 | 1.687 |
| RF_BalancedUndersampled | 19 | 1.089 | 0.587 | 0.0545 | 1.731 |
| RF_BalancedUndersampledPrior | 19 | 1.090 | 0.588 | 0.0545 | 1.732 |
| Stacking_Boosting | 19 | 1.108 | 1.126 | 0.0230 | 2.258 |
| Stacking_RF_XGB | 19 | 1.222 | 1.348 | 0.0485 | 2.618 |

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
