# Computation time, NeuralForest v0.4.1

Observed wall-clock durations, in seconds, from 19 scenarios, six independent generator seeds and 17 reference models/variants, for 1,938 successful fits. The numerical libraries were restricted to one thread. Runners can have different underlying hardware, so the values are workload- and infrastructure-specific. Data generation and dependency installation are excluded.

Selection includes hyperparameter search and neural/expert/calibration selection. Final fit is measured separately. Prediction is the independent-test population inference time. Each row below is the unweighted arithmetic mean over scenarios, not a universal speed ranking.

| Algorithm | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.018 | 0.003 | 0.001 | 0.022 |
| Logistic | 0.042 | 0.006 | 0.002 | 0.050 |
| MLP | 0.156 | 0.071 | 0.002 | 0.229 |
| XGBoost | 0.207 | 0.072 | 0.004 | 0.284 |
| LightGBM | 0.209 | 0.095 | 0.019 | 0.323 |
| SVM_RBF | 0.196 | 0.114 | 0.035 | 0.345 |
| ranger | 0.288 | 0.164 | 0.045 | 0.497 |
| NeuralForest_CART | 0.353 | 0.188 | 0.017 | 0.558 |
| HistGradientBoosting | 0.367 | 0.176 | 0.015 | 0.558 |
| ExtraTrees | 0.418 | 0.194 | 0.032 | 0.643 |
| CatBoost | 0.665 | 0.269 | 0.002 | 0.936 |
| NeuralForest_Differentiable | 0.745 | 0.187 | 0.020 | 0.952 |
| NeuralForest_Calibrated | 0.806 | 0.187 | 0.021 | 1.013 |
| NeuralForest_Adaptive | 0.811 | 0.187 | 0.021 | 1.018 |
| AdaBoost | 0.641 | 0.360 | 0.023 | 1.024 |
| GradientBoosting | 0.681 | 0.380 | 0.004 | 1.064 |
| RandomForest | 0.781 | 0.434 | 0.027 | 1.242 |

## Example scenario-specific comparisons

| Scenario | Model | Primary score | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rare_1pct | NeuralForest_Adaptive | 0.186 | 3.538 | 0.855 | 0.025 | 4.418 |
| rare_1pct | NeuralForest_Calibrated | 0.186 | 3.534 | 0.854 | 0.026 | 4.414 |
| rare_1pct | NeuralForest_Differentiable | 0.166 | 3.421 | 0.847 | 0.025 | 4.293 |
| rare_1pct | NeuralForest_CART | 0.157 | 1.518 | 0.848 | 0.019 | 2.385 |
| rare_1pct | Logistic | 0.196 | 0.054 | 0.009 | 0.002 | 0.065 |
| rare_1pct | CatBoost | 0.151 | 0.621 | 0.319 | 0.002 | 0.943 |
| rare_1pct | XGBoost | 0.146 | 0.215 | 0.101 | 0.005 | 0.322 |
| rare_1pct | ranger | 0.161 | 0.772 | 0.482 | 0.050 | 1.304 |
| rare_3pct | NeuralForest_Adaptive | 0.267 | 1.133 | 0.221 | 0.019 | 1.373 |
| rare_3pct | NeuralForest_Calibrated | 0.267 | 1.136 | 0.223 | 0.018 | 1.377 |
| rare_3pct | NeuralForest_Differentiable | 0.249 | 1.104 | 0.242 | 0.018 | 1.364 |
| rare_3pct | NeuralForest_CART | 0.235 | 0.402 | 0.222 | 0.014 | 0.638 |
| rare_3pct | Logistic | 0.290 | 0.032 | 0.005 | 0.001 | 0.038 |
| rare_3pct | CatBoost | 0.238 | 0.367 | 0.175 | 0.002 | 0.544 |
| rare_3pct | XGBoost | 0.205 | 0.121 | 0.056 | 0.005 | 0.182 |
| rare_3pct | ranger | 0.218 | 0.243 | 0.137 | 0.036 | 0.415 |
| large_n | NeuralForest_Adaptive | 0.771 | 2.051 | 0.367 | 0.024 | 2.442 |
| large_n | NeuralForest_Calibrated | 0.771 | 2.060 | 0.370 | 0.024 | 2.454 |
| large_n | NeuralForest_Differentiable | 0.771 | 2.037 | 0.369 | 0.024 | 2.430 |
| large_n | NeuralForest_CART | 0.770 | 0.661 | 0.366 | 0.020 | 1.047 |
| large_n | Logistic | 0.743 | 0.032 | 0.005 | 0.001 | 0.038 |
| large_n | CatBoost | 0.779 | 0.383 | 0.117 | 0.002 | 0.501 |
| large_n | XGBoost | 0.774 | 0.159 | 0.046 | 0.004 | 0.209 |
| large_n | ranger | 0.765 | 0.720 | 0.451 | 0.073 | 1.245 |
| presence_bg_1to5 | NeuralForest_Adaptive | 0.985 | 0.878 | 0.254 | 0.029 | 1.161 |
| presence_bg_1to5 | NeuralForest_Calibrated | 0.985 | 0.842 | 0.254 | 0.029 | 1.125 |
| presence_bg_1to5 | NeuralForest_Differentiable | 0.973 | 0.786 | 0.252 | 0.029 | 1.066 |
| presence_bg_1to5 | NeuralForest_CART | 0.973 | 0.468 | 0.252 | 0.026 | 0.746 |
| presence_bg_1to5 | Logistic | 0.963 | 0.041 | 0.006 | 0.001 | 0.048 |
| presence_bg_1to5 | CatBoost | 0.950 | 0.519 | 0.181 | 0.002 | 0.703 |
| presence_bg_1to5 | XGBoost | 0.907 | 0.185 | 0.089 | 0.005 | 0.280 |
| presence_bg_1to5 | ranger | 0.955 | 0.339 | 0.196 | 0.062 | 0.598 |
| pseudoabsence_hard | NeuralForest_Adaptive | 0.804 | 0.368 | 0.070 | 0.025 | 0.463 |
| pseudoabsence_hard | NeuralForest_Calibrated | 0.804 | 0.330 | 0.070 | 0.025 | 0.425 |
| pseudoabsence_hard | NeuralForest_Differentiable | 0.682 | 0.288 | 0.069 | 0.024 | 0.382 |
| pseudoabsence_hard | NeuralForest_CART | 0.473 | 0.146 | 0.070 | 0.020 | 0.237 |
| pseudoabsence_hard | Logistic | 0.988 | 0.037 | 0.005 | 0.001 | 0.043 |
| pseudoabsence_hard | CatBoost | 0.254 | 0.493 | 0.167 | 0.002 | 0.662 |
| pseudoabsence_hard | XGBoost | 0.164 | 0.156 | 0.046 | 0.004 | 0.207 |
| pseudoabsence_hard | ranger | 0.224 | 0.126 | 0.057 | 0.050 | 0.233 |
| identical_100 | NeuralForest_Adaptive | 0.698 | 0.293 | 0.092 | 0.016 | 0.401 |
| identical_100 | NeuralForest_Calibrated | 0.698 | 0.345 | 0.093 | 0.015 | 0.444 |
| identical_100 | NeuralForest_Differentiable | 0.698 | 0.248 | 0.091 | 0.015 | 0.354 |
| identical_100 | NeuralForest_CART | 0.706 | 0.187 | 0.092 | 0.012 | 0.291 |
| identical_100 | Logistic | 0.701 | 0.044 | 0.005 | 0.001 | 0.051 |
| identical_100 | CatBoost | 0.718 | 0.502 | 0.139 | 0.003 | 0.643 |
| identical_100 | XGBoost | 0.759 | 0.184 | 0.051 | 0.003 | 0.238 |
| identical_100 | ranger | 0.719 | 0.181 | 0.086 | 0.033 | 0.301 |
| nonlinear_xor | NeuralForest_Adaptive | 0.984 | 0.564 | 0.142 | 0.019 | 0.725 |
| nonlinear_xor | NeuralForest_Calibrated | 0.984 | 0.531 | 0.142 | 0.020 | 0.692 |
| nonlinear_xor | NeuralForest_Differentiable | 0.984 | 0.487 | 0.143 | 0.019 | 0.649 |
| nonlinear_xor | NeuralForest_CART | 0.983 | 0.283 | 0.143 | 0.017 | 0.444 |
| nonlinear_xor | Logistic | 0.504 | 0.044 | 0.005 | 0.002 | 0.051 |
| nonlinear_xor | CatBoost | 0.997 | 0.504 | 0.144 | 0.002 | 0.651 |
| nonlinear_xor | XGBoost | 0.993 | 0.163 | 0.065 | 0.005 | 0.233 |
| nonlinear_xor | ranger | 0.989 | 0.234 | 0.115 | 0.048 | 0.397 |

## Reading these values

The additional internal model selection in the NeuralForest variants can dominate their training time. It must be compared with each competitor's own selection budget rather than only its final fit. The benchmark intentionally uses bounded but not compute-matched tuning spaces. An algorithm being faster on these samples does not establish general dominance.

The complete data with per-seed times, evaluated metrics and internal configurations are in `all_runs.csv`; cross-seed standard deviations and per-scenario timings are in `summary.csv`. The controlled 2-versus-3-round neural ablation and its per-case costs are in `audit/rounds_report.md` and `audit/rounds_results.csv`.
