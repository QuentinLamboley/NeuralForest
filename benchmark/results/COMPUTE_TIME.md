# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.017 | 0.003 | 0.0020 | 0.022 |
| Logistic | 19 | 0.042 | 0.006 | 0.0029 | 0.051 |
| MLP | 19 | 0.151 | 0.067 | 0.0038 | 0.221 |
| NeuralForest_UndersampledPlain | 19 | 0.101 | 0.114 | 0.0312 | 0.248 |
| XGBoost | 19 | 0.205 | 0.072 | 0.0075 | 0.284 |
| LightGBM | 19 | 0.208 | 0.096 | 0.0372 | 0.341 |
| NeuralForest_SubSample50_Informed | 19 | 0.215 | 0.111 | 0.0311 | 0.359 |
| SVM_RBF | 19 | 0.191 | 0.109 | 0.0687 | 0.369 |
| NeuralForest_PUNeural | 3 | 0.186 | 0.209 | 0.0210 | 0.417 |
| NeuralForest_UndersampledInformed | 19 | 0.346 | 0.116 | 0.0314 | 0.495 |
| ranger | 19 | 0.290 | 0.160 | 0.0879 | 0.537 |
| HistGradientBoosting | 19 | 0.361 | 0.173 | 0.0296 | 0.564 |
| NeuralForest_CART | 19 | 0.348 | 0.187 | 0.0330 | 0.570 |
| NeuralForest_PUInformed | 3 | 0.278 | 0.329 | 0.0554 | 0.663 |
| ExtraTrees | 19 | 0.414 | 0.191 | 0.0621 | 0.667 |
| NeuralForest_Differentiable | 19 | 0.457 | 0.184 | 0.0364 | 0.680 |
| NeuralForest_Calibrated | 19 | 0.497 | 0.186 | 0.0367 | 0.722 |
| NeuralForest_Adaptive | 19 | 0.505 | 0.186 | 0.0370 | 0.730 |
| NeuralForest_PUSafe | 3 | 0.656 | 0.210 | 0.0178 | 0.884 |
| CatBoost | 19 | 0.654 | 0.269 | 0.0042 | 0.928 |
| RF_Bootstrap50 | 19 | 0.582 | 0.301 | 0.0481 | 0.930 |
| GradientBoosting | 19 | 0.668 | 0.367 | 0.0066 | 1.042 |
| AdaBoost | 19 | 0.635 | 0.362 | 0.0458 | 1.043 |
| Voting_Boosting | 19 | 0.792 | 0.394 | 0.0410 | 1.228 |
| Voting_Heterogeneous | 19 | 0.791 | 0.384 | 0.0543 | 1.229 |
| RandomForest | 19 | 0.774 | 0.430 | 0.0523 | 1.256 |
| RF_Subsample50 | 19 | 0.808 | 0.417 | 0.0732 | 1.299 |
| Voting_RF_ExtraTrees | 19 | 0.887 | 0.477 | 0.0884 | 1.452 |
| NeuralForest_PUStable | 3 | 0.948 | 0.470 | 0.0389 | 1.457 |
| Voting_RF_CatBoost | 19 | 1.055 | 0.561 | 0.0436 | 1.660 |
| RF_BalancedUndersampled | 19 | 1.083 | 0.587 | 0.0545 | 1.725 |
| RF_BalancedUndersampledPrior | 19 | 1.085 | 0.588 | 0.0545 | 1.727 |
| RF_Subsample80 | 19 | 1.067 | 0.590 | 0.0775 | 1.735 |
| Stacking_Boosting | 19 | 1.124 | 1.150 | 0.0238 | 2.298 |
| Stacking_RF_XGB | 19 | 1.244 | 1.381 | 0.0484 | 2.673 |

## Comparisons by scenario

A full per-scenario table of the standard RF, 50% and 80% row
subsampling, class-balanced under-sampling, ranger and informed
NeuralForest variants is in sampling_comparison.csv. Both the
primary objective gap and the total-time ratio versus RandomForest
are stored. Positive primary_advantage_vs_classic_rf is favorable.

All 6,010 per-seed score and timing records appear in all_runs.csv;
the cross-seed means and standard deviations are in summary.csv.
Prediction time covers BOTH independent true-label test cohorts.
PU methods are evaluated on three presence/background scenarios
only. Their mean times are not directly comparable with the
19-scenario averages of other methods.
The training budget and the initial predictive objective are
not perfectly matched across algorithm families.
