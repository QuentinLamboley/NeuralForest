# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.014 | 0.003 | 0.0016 | 0.018 |
| Logistic | 19 | 0.034 | 0.005 | 0.0024 | 0.041 |
| MLP | 19 | 0.125 | 0.056 | 0.0033 | 0.184 |
| NeuralForest_UndersampledPlain | 19 | 0.092 | 0.104 | 0.0280 | 0.226 |
| XGBoost | 19 | 0.179 | 0.063 | 0.0065 | 0.248 |
| NeuralForest_PUNeural | 3 | 0.111 | 0.126 | 0.0142 | 0.251 |
| LightGBM | 19 | 0.183 | 0.086 | 0.0323 | 0.301 |
| SVM_RBF | 19 | 0.166 | 0.097 | 0.0599 | 0.323 |
| NeuralForest_SubSample50_Informed | 19 | 0.203 | 0.105 | 0.0279 | 0.337 |
| NeuralForest_PUInformed | 3 | 0.179 | 0.207 | 0.0375 | 0.424 |
| NeuralForest_UndersampledInformed | 19 | 0.317 | 0.104 | 0.0278 | 0.451 |
| ranger | 19 | 0.253 | 0.138 | 0.0749 | 0.466 |
| HistGradientBoosting | 19 | 0.305 | 0.146 | 0.0269 | 0.478 |
| NeuralForest_CART | 19 | 0.328 | 0.175 | 0.0292 | 0.534 |
| NeuralForest_PUSafe | 3 | 0.441 | 0.128 | 0.0122 | 0.581 |
| ExtraTrees | 19 | 0.362 | 0.169 | 0.0516 | 0.582 |
| NeuralForest_Differentiable | 19 | 0.422 | 0.177 | 0.0322 | 0.633 |
| NeuralForest_Calibrated | 19 | 0.454 | 0.177 | 0.0326 | 0.666 |
| NeuralForest_Adaptive | 19 | 0.466 | 0.175 | 0.0328 | 0.676 |
| RF_Bootstrap50 | 19 | 0.506 | 0.271 | 0.0386 | 0.816 |
| CatBoost | 19 | 0.581 | 0.234 | 0.0039 | 0.819 |
| NeuralForest_PUStable | 3 | 0.599 | 0.284 | 0.0249 | 0.908 |
| AdaBoost | 19 | 0.556 | 0.323 | 0.0335 | 0.912 |
| NeuralForest_PUStableTeacher | 3 | 0.599 | 0.284 | 0.0311 | 0.914 |
| GradientBoosting | 19 | 0.594 | 0.328 | 0.0057 | 0.928 |
| Voting_Heterogeneous | 19 | 0.692 | 0.342 | 0.0456 | 1.079 |
| Voting_Boosting | 19 | 0.702 | 0.347 | 0.0360 | 1.086 |
| RF_Subsample50 | 19 | 0.701 | 0.373 | 0.0573 | 1.132 |
| RandomForest | 19 | 0.694 | 0.395 | 0.0423 | 1.132 |
| Voting_RF_ExtraTrees | 19 | 0.789 | 0.433 | 0.0734 | 1.295 |
| Voting_RF_CatBoost | 19 | 0.948 | 0.508 | 0.0363 | 1.492 |
| RF_BalancedUndersampled | 19 | 0.958 | 0.514 | 0.0444 | 1.516 |
| RF_BalancedUndersampledPrior | 19 | 0.961 | 0.515 | 0.0446 | 1.521 |
| RF_Subsample80 | 19 | 0.948 | 0.541 | 0.0616 | 1.550 |
| Stacking_Boosting | 19 | 1.003 | 1.033 | 0.0207 | 2.057 |
| Stacking_RF_XGB | 19 | 1.096 | 1.236 | 0.0395 | 2.372 |

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
