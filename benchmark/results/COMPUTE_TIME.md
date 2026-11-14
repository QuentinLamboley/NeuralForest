# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and ten seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Scenarios | Selection (s) | Final fit (s) | Prediction both cohorts (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 19 | 0.013 | 0.002 | 0.0015 | 0.017 |
| Logistic | 19 | 0.031 | 0.004 | 0.0023 | 0.037 |
| MLP | 19 | 0.113 | 0.052 | 0.0030 | 0.168 |
| NeuralForest_UndersampledPlain | 19 | 0.087 | 0.098 | 0.0268 | 0.213 |
| XGBoost | 19 | 0.172 | 0.060 | 0.0063 | 0.238 |
| NeuralForest_PUNeural | 3 | 0.108 | 0.123 | 0.0137 | 0.245 |
| LightGBM | 19 | 0.174 | 0.084 | 0.0324 | 0.290 |
| SVM_RBF | 19 | 0.161 | 0.095 | 0.0590 | 0.315 |
| NeuralForest_SubSample50_Informed | 19 | 0.191 | 0.098 | 0.0268 | 0.318 |
| NeuralForest_PUInformed | 3 | 0.174 | 0.204 | 0.0372 | 0.416 |
| NeuralForest_UndersampledInformed | 19 | 0.298 | 0.099 | 0.0269 | 0.426 |
| HistGradientBoosting | 19 | 0.281 | 0.135 | 0.0253 | 0.441 |
| ranger | 19 | 0.252 | 0.138 | 0.0731 | 0.463 |
| NeuralForest_CART | 19 | 0.309 | 0.167 | 0.0282 | 0.507 |
| ExtraTrees | 19 | 0.329 | 0.157 | 0.0503 | 0.536 |
| NeuralForest_PUSafe | 3 | 0.425 | 0.122 | 0.0120 | 0.559 |
| NeuralForest_Differentiable | 19 | 0.402 | 0.166 | 0.0308 | 0.600 |
| NeuralForest_Calibrated | 19 | 0.432 | 0.167 | 0.0310 | 0.631 |
| NeuralForest_Adaptive | 19 | 0.439 | 0.167 | 0.0315 | 0.639 |
| RF_Bootstrap50 | 19 | 0.469 | 0.251 | 0.0371 | 0.757 |
| CatBoost | 19 | 0.535 | 0.222 | 0.0037 | 0.760 |
| AdaBoost | 19 | 0.514 | 0.304 | 0.0315 | 0.849 |
| NeuralForest_PUStable | 3 | 0.581 | 0.275 | 0.0241 | 0.880 |
| NeuralForest_PUStableTeacher | 3 | 0.581 | 0.275 | 0.0280 | 0.884 |
| GradientBoosting | 19 | 0.571 | 0.318 | 0.0056 | 0.894 |
| Voting_Heterogeneous | 19 | 0.648 | 0.321 | 0.0441 | 1.013 |
| Voting_Boosting | 19 | 0.671 | 0.336 | 0.0358 | 1.043 |
| RF_Subsample50 | 19 | 0.650 | 0.347 | 0.0560 | 1.053 |
| RandomForest | 19 | 0.647 | 0.370 | 0.0408 | 1.057 |
| Voting_RF_ExtraTrees | 19 | 0.729 | 0.404 | 0.0707 | 1.203 |
| RF_BalancedUndersampled | 19 | 0.867 | 0.479 | 0.0429 | 1.389 |
| Voting_RF_CatBoost | 19 | 0.883 | 0.471 | 0.0345 | 1.389 |
| RF_BalancedUndersampledPrior | 19 | 0.868 | 0.479 | 0.0431 | 1.391 |
| RF_Subsample80 | 19 | 0.888 | 0.505 | 0.0597 | 1.453 |
| Stacking_Boosting | 19 | 0.931 | 0.948 | 0.0206 | 1.900 |
| Stacking_RF_XGB | 19 | 1.028 | 1.159 | 0.0384 | 2.225 |

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
