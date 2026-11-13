# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.013 | 0.002 | 0.0015 | 0.016 |
| Logistic | 19 | 0.031 | 0.004 | 0.0022 | 0.037 |
| MLP | 19 | 0.109 | 0.049 | 0.0030 | 0.161 |
| NeuralForest_UndersampledPlain | 19 | 0.088 | 0.100 | 0.0272 | 0.217 |
| XGBoost | 19 | 0.169 | 0.060 | 0.0062 | 0.235 |
| LightGBM | 19 | 0.174 | 0.083 | 0.0309 | 0.288 |
| SVM_RBF | 19 | 0.151 | 0.088 | 0.0557 | 0.295 |
| NeuralForest_SubSample50_Informed | 19 | 0.185 | 0.096 | 0.0270 | 0.310 |
| NeuralForest_PUNeural | 3 | 0.144 | 0.163 | 0.0186 | 0.326 |
| NeuralForest_UndersampledInformed | 19 | 0.299 | 0.101 | 0.0273 | 0.430 |
| ranger | 19 | 0.234 | 0.126 | 0.0712 | 0.432 |
| HistGradientBoosting | 19 | 0.283 | 0.137 | 0.0263 | 0.446 |
| NeuralForest_CART | 19 | 0.302 | 0.163 | 0.0287 | 0.495 |
| ExtraTrees | 19 | 0.321 | 0.152 | 0.0492 | 0.522 |
| NeuralForest_PUInformed | 3 | 0.224 | 0.267 | 0.0498 | 0.541 |
| NeuralForest_Differentiable | 19 | 0.389 | 0.161 | 0.0314 | 0.583 |
| NeuralForest_Calibrated | 19 | 0.419 | 0.163 | 0.0316 | 0.615 |
| NeuralForest_Adaptive | 19 | 0.429 | 0.162 | 0.0319 | 0.624 |
| RF_Bootstrap50 | 19 | 0.461 | 0.244 | 0.0364 | 0.741 |
| AdaBoost | 19 | 0.500 | 0.295 | 0.0300 | 0.825 |
| CatBoost | 19 | 0.584 | 0.239 | 0.0037 | 0.826 |
| GradientBoosting | 19 | 0.545 | 0.302 | 0.0054 | 0.852 |
| RandomForest | 19 | 0.629 | 0.358 | 0.0402 | 1.027 |
| RF_Subsample50 | 19 | 0.639 | 0.339 | 0.0535 | 1.031 |
| Voting_Heterogeneous | 19 | 0.680 | 0.333 | 0.0433 | 1.056 |
| Voting_Boosting | 19 | 0.690 | 0.356 | 0.0344 | 1.081 |
| Voting_RF_ExtraTrees | 19 | 0.710 | 0.391 | 0.0693 | 1.170 |
| RF_BalancedUndersampled | 19 | 0.868 | 0.482 | 0.0422 | 1.392 |
| RF_BalancedUndersampledPrior | 19 | 0.870 | 0.482 | 0.0424 | 1.394 |
| RF_Subsample80 | 19 | 0.865 | 0.490 | 0.0573 | 1.412 |
| Voting_RF_CatBoost | 19 | 0.898 | 0.486 | 0.0341 | 1.418 |
| Stacking_Boosting | 19 | 1.007 | 1.018 | 0.0198 | 2.044 |
| Stacking_RF_XGB | 19 | 1.008 | 1.132 | 0.0377 | 2.177 |

## Comparisons by scenario

A full per-scenario table of the standard RF, 50% and 80% row
subsampling, class-balanced under-sampling, ranger and informed
NeuralForest variants is in sampling_comparison.csv. Both the
primary objective gap and the total-time ratio versus RandomForest
are stored. Positive primary_advantage_vs_classic_rf is favorable.

All 5,950 per-seed score and timing records appear in all_runs.csv;
the cross-seed means and standard deviations are in summary.csv.
Prediction time covers BOTH independent true-label test cohorts.
PU methods are evaluated on three presence/background scenarios
only. Their mean times are not directly comparable with the
19-scenario averages of other methods.
The training budget and the initial predictive objective are
not perfectly matched across algorithm families.
