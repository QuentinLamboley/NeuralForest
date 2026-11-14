# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.017 | 0.003 | 0.0020 | 0.022 |
| Logistic | 19 | 0.042 | 0.006 | 0.0030 | 0.051 |
| NeuralForest_UndersampledPlain | 19 | 0.096 | 0.107 | 0.0307 | 0.235 |
| MLP | 19 | 0.160 | 0.074 | 0.0040 | 0.238 |
| XGBoost | 19 | 0.208 | 0.072 | 0.0078 | 0.288 |
| LightGBM | 19 | 0.202 | 0.096 | 0.0367 | 0.335 |
| NeuralForest_SubSample50_Informed | 19 | 0.219 | 0.111 | 0.0306 | 0.363 |
| SVM_RBF | 19 | 0.207 | 0.123 | 0.0724 | 0.402 |
| NeuralForest_PUNeural | 3 | 0.185 | 0.211 | 0.0216 | 0.417 |
| NeuralForest_UndersampledInformed | 19 | 0.334 | 0.108 | 0.0307 | 0.475 |
| ranger | 19 | 0.296 | 0.162 | 0.0904 | 0.548 |
| HistGradientBoosting | 19 | 0.359 | 0.171 | 0.0290 | 0.559 |
| NeuralForest_CART | 19 | 0.349 | 0.185 | 0.0324 | 0.569 |
| ExtraTrees | 19 | 0.407 | 0.187 | 0.0606 | 0.654 |
| NeuralForest_PUInformed | 3 | 0.278 | 0.327 | 0.0562 | 0.660 |
| NeuralForest_Differentiable | 19 | 0.468 | 0.184 | 0.0360 | 0.690 |
| NeuralForest_Calibrated | 19 | 0.511 | 0.185 | 0.0363 | 0.735 |
| NeuralForest_Adaptive | 19 | 0.521 | 0.185 | 0.0368 | 0.745 |
| NeuralForest_PUSafe | 3 | 0.655 | 0.212 | 0.0182 | 0.886 |
| RF_Bootstrap50 | 19 | 0.569 | 0.295 | 0.0467 | 0.912 |
| CatBoost | 19 | 0.653 | 0.270 | 0.0045 | 0.928 |
| AdaBoost | 19 | 0.618 | 0.352 | 0.0456 | 1.016 |
| GradientBoosting | 19 | 0.655 | 0.362 | 0.0068 | 1.024 |
| Voting_Boosting | 19 | 0.782 | 0.389 | 0.0407 | 1.211 |
| Voting_Heterogeneous | 19 | 0.778 | 0.380 | 0.0531 | 1.212 |
| RandomForest | 19 | 0.766 | 0.427 | 0.0508 | 1.243 |
| RF_Subsample50 | 19 | 0.794 | 0.412 | 0.0736 | 1.279 |
| Voting_RF_ExtraTrees | 19 | 0.876 | 0.472 | 0.0861 | 1.434 |
| NeuralForest_PUStable | 3 | 0.958 | 0.475 | 0.0378 | 1.471 |
| NeuralForest_PUStableTeacher | 3 | 0.958 | 0.475 | 0.0428 | 1.476 |
| Voting_RF_CatBoost | 19 | 1.042 | 0.555 | 0.0426 | 1.641 |
| RF_BalancedUndersampled | 19 | 1.057 | 0.570 | 0.0532 | 1.680 |
| RF_BalancedUndersampledPrior | 19 | 1.058 | 0.571 | 0.0532 | 1.681 |
| RF_Subsample80 | 19 | 1.056 | 0.587 | 0.0778 | 1.721 |
| Stacking_Boosting | 19 | 1.106 | 1.131 | 0.0242 | 2.261 |
| Stacking_RF_XGB | 19 | 1.232 | 1.372 | 0.0477 | 2.651 |

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
