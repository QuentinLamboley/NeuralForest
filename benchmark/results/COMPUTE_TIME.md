# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.015 | 0.003 | 0.0017 | 0.019 |
| Logistic | 19 | 0.035 | 0.005 | 0.0024 | 0.043 |
| MLP | 19 | 0.129 | 0.060 | 0.0032 | 0.192 |
| NeuralForest_UndersampledPlain | 19 | 0.095 | 0.108 | 0.0275 | 0.232 |
| NeuralForest_PUNeural | 3 | 0.106 | 0.121 | 0.0130 | 0.240 |
| XGBoost | 19 | 0.181 | 0.063 | 0.0063 | 0.250 |
| LightGBM | 19 | 0.189 | 0.090 | 0.0317 | 0.311 |
| NeuralForest_SubSample50_Informed | 19 | 0.203 | 0.106 | 0.0275 | 0.339 |
| SVM_RBF | 19 | 0.182 | 0.108 | 0.0630 | 0.353 |
| NeuralForest_PUInformed | 3 | 0.171 | 0.199 | 0.0349 | 0.405 |
| NeuralForest_UndersampledInformed | 19 | 0.323 | 0.109 | 0.0276 | 0.461 |
| ranger | 19 | 0.264 | 0.146 | 0.0757 | 0.486 |
| HistGradientBoosting | 19 | 0.314 | 0.150 | 0.0257 | 0.489 |
| NeuralForest_CART | 19 | 0.331 | 0.179 | 0.0291 | 0.541 |
| NeuralForest_PUSafe | 3 | 0.412 | 0.121 | 0.0111 | 0.544 |
| ExtraTrees | 19 | 0.352 | 0.164 | 0.0533 | 0.570 |
| NeuralForest_Differentiable | 19 | 0.435 | 0.177 | 0.0319 | 0.646 |
| NeuralForest_Calibrated | 19 | 0.467 | 0.179 | 0.0322 | 0.680 |
| NeuralForest_Adaptive | 19 | 0.478 | 0.178 | 0.0325 | 0.691 |
| RF_Bootstrap50 | 19 | 0.502 | 0.265 | 0.0405 | 0.807 |
| CatBoost | 19 | 0.593 | 0.247 | 0.0037 | 0.844 |
| NeuralForest_PUStable | 3 | 0.569 | 0.272 | 0.0229 | 0.864 |
| NeuralForest_PUStableTeacher | 3 | 0.569 | 0.272 | 0.0276 | 0.869 |
| AdaBoost | 19 | 0.554 | 0.324 | 0.0367 | 0.914 |
| GradientBoosting | 19 | 0.612 | 0.338 | 0.0057 | 0.956 |
| Voting_Heterogeneous | 19 | 0.696 | 0.345 | 0.0466 | 1.087 |
| Voting_Boosting | 19 | 0.721 | 0.355 | 0.0350 | 1.111 |
| RandomForest | 19 | 0.686 | 0.387 | 0.0444 | 1.117 |
| RF_Subsample50 | 19 | 0.695 | 0.367 | 0.0612 | 1.124 |
| Voting_RF_ExtraTrees | 19 | 0.774 | 0.424 | 0.0761 | 1.275 |
| Voting_RF_CatBoost | 19 | 0.947 | 0.508 | 0.0371 | 1.492 |
| RF_BalancedUndersampled | 19 | 0.938 | 0.516 | 0.0465 | 1.501 |
| RF_BalancedUndersampledPrior | 19 | 0.942 | 0.516 | 0.0467 | 1.505 |
| RF_Subsample80 | 19 | 0.943 | 0.531 | 0.0651 | 1.539 |
| Stacking_Boosting | 19 | 1.024 | 1.035 | 0.0205 | 2.080 |
| Stacking_RF_XGB | 19 | 1.088 | 1.222 | 0.0409 | 2.351 |

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
