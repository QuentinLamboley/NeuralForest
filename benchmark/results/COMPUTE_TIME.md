# Computational time comparison

All values are measured wall-clock seconds on GitHub-hosted CPU runners with numerical libraries restricted to one thread. Each model was evaluated in all 19 scenarios with four random seeds. Values are unweighted averages across scenarios, so they are workload-specific, not general performance claims.

Selection includes hyperparameter search, neural guidance, differentiable-tree optimization and hybrid expert choice when applicable. Final fit is measured separately; prediction is performed on the external test set. Data generation and dependency installation are excluded.

## Mean time across scenarios

| Algorithm | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.014 | 0.002 | 0.001 | 0.018 |
| Logistic | 0.033 | 0.005 | 0.001 | 0.039 |
| MLP | 0.126 | 0.060 | 0.002 | 0.188 |
| XGBoost | 0.185 | 0.064 | 0.004 | 0.252 |
| LightGBM | 0.180 | 0.081 | 0.016 | 0.277 |
| SVM_RBF | 0.158 | 0.091 | 0.030 | 0.279 |
| ranger | 0.237 | 0.137 | 0.040 | 0.414 |
| HistGradientBoosting | 0.311 | 0.150 | 0.013 | 0.474 |
| NeuralForest_CART | 0.309 | 0.165 | 0.015 | 0.489 |
| ExtraTrees | 0.351 | 0.166 | 0.027 | 0.544 |
| NeuralForest_Differentiable | 0.616 | 0.163 | 0.019 | 0.799 |
| NeuralForest_Adaptive | 0.654 | 0.164 | 0.018 | 0.837 |
| AdaBoost | 0.535 | 0.312 | 0.018 | 0.865 |
| CatBoost | 0.600 | 0.263 | 0.002 | 0.866 |
| GradientBoosting | 0.572 | 0.325 | 0.003 | 0.900 |
| RandomForest | 0.671 | 0.378 | 0.022 | 1.071 |

## Representative scenarios

The table below shows selection, fitting, prediction time and the scenario's primary held-out score for selected algorithms. For the complete 19 x 16 table, inspect summary.csv.

| Scenario | Algorithm | Primary score | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rare_3pct | NeuralForest_Adaptive | 0.222 | 0.849 | 0.210 | 0.015 | 1.075 |
| rare_3pct | NeuralForest_Differentiable | 0.207 | 0.829 | 0.214 | 0.043 | 1.087 |
| rare_3pct | NeuralForest_CART | 0.204 | 0.381 | 0.207 | 0.012 | 0.601 |
| rare_3pct | ranger | 0.195 | 0.181 | 0.101 | 0.030 | 0.313 |
| rare_3pct | CatBoost | 0.212 | 0.315 | 0.149 | 0.002 | 0.465 |
| rare_3pct | XGBoost | 0.170 | 0.094 | 0.045 | 0.003 | 0.143 |
| rare_3pct | Logistic | 0.264 | 0.022 | 0.003 | 0.001 | 0.027 |
| rare_3pct | ExtraTrees | 0.228 | 0.244 | 0.124 | 0.022 | 0.389 |
| rare_1pct | NeuralForest_Adaptive | 0.110 | 2.827 | 0.739 | 0.021 | 3.586 |
| rare_1pct | NeuralForest_Differentiable | 0.095 | 2.773 | 0.733 | 0.021 | 3.527 |
| rare_1pct | NeuralForest_CART | 0.097 | 1.297 | 0.737 | 0.017 | 2.051 |
| rare_1pct | ranger | 0.108 | 0.581 | 0.363 | 0.040 | 0.984 |
| rare_1pct | CatBoost | 0.091 | 0.567 | 0.207 | 0.002 | 0.776 |
| rare_1pct | XGBoost | 0.090 | 0.172 | 0.082 | 0.004 | 0.259 |
| rare_1pct | Logistic | 0.116 | 0.037 | 0.007 | 0.001 | 0.045 |
| rare_1pct | ExtraTrees | 0.088 | 0.634 | 0.350 | 0.026 | 1.010 |
| large_n | NeuralForest_Adaptive | 0.770 | 1.532 | 0.349 | 0.020 | 1.901 |
| large_n | NeuralForest_Differentiable | 0.770 | 1.560 | 0.345 | 0.020 | 1.925 |
| large_n | NeuralForest_CART | 0.769 | 0.619 | 0.346 | 0.018 | 0.983 |
| large_n | ranger | 0.765 | 0.547 | 0.328 | 0.059 | 0.935 |
| large_n | CatBoost | 0.777 | 0.322 | 0.097 | 0.001 | 0.420 |
| large_n | XGBoost | 0.773 | 0.123 | 0.036 | 0.003 | 0.162 |
| large_n | Logistic | 0.743 | 0.023 | 0.003 | 0.001 | 0.027 |
| large_n | ExtraTrees | 0.761 | 0.417 | 0.197 | 0.038 | 0.652 |
| presence_bg_1to1 | NeuralForest_Adaptive | 0.996 | 0.318 | 0.062 | 0.034 | 0.414 |
| presence_bg_1to1 | NeuralForest_Differentiable | 0.990 | 0.262 | 0.059 | 0.024 | 0.347 |
| presence_bg_1to1 | NeuralForest_CART | 0.987 | 0.131 | 0.060 | 0.020 | 0.212 |
| presence_bg_1to1 | ranger | 0.991 | 0.123 | 0.055 | 0.057 | 0.234 |
| presence_bg_1to1 | CatBoost | 0.993 | 0.507 | 0.233 | 0.002 | 0.742 |
| presence_bg_1to1 | XGBoost | 0.984 | 0.157 | 0.045 | 0.004 | 0.206 |
| presence_bg_1to1 | Logistic | 0.994 | 0.043 | 0.005 | 0.002 | 0.049 |
| presence_bg_1to1 | ExtraTrees | 0.993 | 0.360 | 0.159 | 0.037 | 0.556 |
| label_noise_20pct | NeuralForest_Adaptive | 0.694 | 0.573 | 0.162 | 0.023 | 0.758 |
| label_noise_20pct | NeuralForest_Differentiable | 0.694 | 0.518 | 0.162 | 0.022 | 0.703 |
| label_noise_20pct | NeuralForest_CART | 0.687 | 0.304 | 0.163 | 0.019 | 0.485 |
| label_noise_20pct | ranger | 0.677 | 0.269 | 0.154 | 0.052 | 0.475 |
| label_noise_20pct | CatBoost | 0.690 | 0.403 | 0.114 | 0.002 | 0.519 |
| label_noise_20pct | XGBoost | 0.676 | 0.153 | 0.043 | 0.003 | 0.199 |
| label_noise_20pct | Logistic | 0.691 | 0.029 | 0.004 | 0.001 | 0.034 |
| label_noise_20pct | ExtraTrees | 0.693 | 0.353 | 0.155 | 0.028 | 0.535 |

## Interpretation

NeuralForest_Adaptive can improve some predictive scores, but its internal selection often takes longer than inexpensive linear or boosting alternatives. On the 1-percent rare-class condition its mean total runtime is approximately 3.59 s, compared with 0.045 s for logistic regression. On the large-sample condition it takes approximately 1.90 s, compared with 0.42 s for CatBoost and 0.16 s for XGBoost. These numbers apply only to the current workloads and GitHub runners.

The full CSV preserves per-seed timing, mean, standard deviation, every evaluation metric and the chosen configuration. Never equate lower wall-clock time with superior calibrated predictive performance.
