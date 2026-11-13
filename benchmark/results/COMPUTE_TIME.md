# Computation time: NeuralForest and forest baselines

Observed wall-clock seconds averaged across 19 scenarios and eight seeds.
Numerical libraries were restricted to one CPU thread. Different runners
may have distinct performance. Times do not include data generation
or dependency installation. Internal search budgets differ by estimator.

| Algorithm | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.015 | 0.002 | 0.0009 | 0.018 |
| Logistic | 0.034 | 0.005 | 0.0013 | 0.040 |
| MLP | 0.125 | 0.056 | 0.0018 | 0.184 |
| NeuralForest_UndersampledPlain | 0.097 | 0.108 | 0.0151 | 0.220 |
| XGBoost | 0.179 | 0.063 | 0.0038 | 0.246 |
| LightGBM | 0.184 | 0.086 | 0.0168 | 0.286 |
| SVM_RBF | 0.173 | 0.101 | 0.0309 | 0.305 |
| NeuralForest_SubSample50_Informed | 0.200 | 0.103 | 0.0141 | 0.317 |
| ranger | 0.250 | 0.140 | 0.0390 | 0.429 |
| NeuralForest_UndersampledInformed | 0.323 | 0.109 | 0.0142 | 0.446 |
| HistGradientBoosting | 0.306 | 0.148 | 0.0137 | 0.468 |
| NeuralForest_CART | 0.327 | 0.170 | 0.0152 | 0.512 |
| ExtraTrees | 0.349 | 0.164 | 0.0274 | 0.540 |
| NeuralForest_Differentiable | 0.444 | 0.171 | 0.0165 | 0.631 |
| NeuralForest_Calibrated | 0.501 | 0.172 | 0.0167 | 0.688 |
| NeuralForest_Adaptive | 0.508 | 0.171 | 0.0167 | 0.696 |
| RF_Bootstrap50 | 0.497 | 0.261 | 0.0204 | 0.778 |
| CatBoost | 0.589 | 0.256 | 0.0022 | 0.848 |
| AdaBoost | 0.546 | 0.320 | 0.0182 | 0.884 |
| GradientBoosting | 0.588 | 0.324 | 0.0031 | 0.915 |
| RandomForest | 0.676 | 0.380 | 0.0224 | 1.078 |
| RF_Subsample50 | 0.693 | 0.364 | 0.0304 | 1.087 |
| RF_Subsample80 | 0.932 | 0.521 | 0.0324 | 1.486 |
| RF_BalancedUndersampledPrior | 0.950 | 0.522 | 0.0233 | 1.495 |
| RF_BalancedUndersampled | 0.950 | 0.524 | 0.0235 | 1.497 |

## Comparisons by scenario

A full per-scenario table of the standard RF, 50% and 80% row
subsampling, class-balanced under-sampling, ranger and informed
NeuralForest variants is in sampling_comparison.csv. Both the
primary objective gap and the total-time ratio versus RandomForest
are stored. Positive primary_advantage_vs_classic_rf is favorable.

All 3,800 per-seed score and timing records appear in all_runs.csv;
the cross-seed means and standard deviations are in summary.csv.
The training budget and the initial predictive objective are
not perfectly matched across algorithm families.
