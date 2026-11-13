# NeuralForest v0.4.1 to v0.5 paired comparison

These observations pair 19 simulator cases, six original seeds
and 17 common model variants. Both evaluations use untouched
independently generated test samples; higher improvement is better.
Changes in software version and CPU runner can affect timings.

| Scenario | Model | Mean objective improvement |
| --- | --- | ---: |
| correlated_30 | NeuralForest_Adaptive | +0.00399 |
| correlated_30 | NeuralForest_CART | +0.00000 |
| correlated_30 | NeuralForest_Calibrated | +0.00399 |
| correlated_30 | NeuralForest_Differentiable | +0.00387 |
| distinct_0 | NeuralForest_Adaptive | +0.00036 |
| distinct_0 | NeuralForest_CART | +0.00000 |
| distinct_0 | NeuralForest_Calibrated | +0.00036 |
| distinct_0 | NeuralForest_Differentiable | +0.00036 |
| distribution_shift | NeuralForest_Adaptive | -0.00078 |
| distribution_shift | NeuralForest_CART | +0.00000 |
| distribution_shift | NeuralForest_Calibrated | -0.00078 |
| distribution_shift | NeuralForest_Differentiable | -0.00017 |
| identical_100 | NeuralForest_Adaptive | +0.00269 |
| identical_100 | NeuralForest_CART | +0.00000 |
| identical_100 | NeuralForest_Calibrated | +0.00224 |
| identical_100 | NeuralForest_Differentiable | +0.00385 |
| intermediate_45 | NeuralForest_Adaptive | -0.00103 |
| intermediate_45 | NeuralForest_CART | +0.00000 |
| intermediate_45 | NeuralForest_Calibrated | -0.00103 |
| intermediate_45 | NeuralForest_Differentiable | -0.00088 |
| label_noise_20pct | NeuralForest_Adaptive | +0.00062 |
| label_noise_20pct | NeuralForest_CART | +0.00000 |
| label_noise_20pct | NeuralForest_Calibrated | +0.00062 |
| label_noise_20pct | NeuralForest_Differentiable | -0.00034 |
| large_n | NeuralForest_Adaptive | +0.00010 |
| large_n | NeuralForest_CART | +0.00000 |
| large_n | NeuralForest_Calibrated | +0.00010 |
| large_n | NeuralForest_Differentiable | +0.00036 |
| missing_15pct | NeuralForest_Adaptive | -0.00086 |
| missing_15pct | NeuralForest_CART | +0.00000 |
| missing_15pct | NeuralForest_Calibrated | -0.00086 |
| missing_15pct | NeuralForest_Differentiable | -0.00121 |
| noise_50 | NeuralForest_Adaptive | -0.00095 |
| noise_50 | NeuralForest_CART | +0.00000 |
| noise_50 | NeuralForest_Calibrated | -0.00095 |
| noise_50 | NeuralForest_Differentiable | -0.00261 |
| nonlinear_xor | NeuralForest_Adaptive | +0.00018 |
| nonlinear_xor | NeuralForest_CART | +0.00000 |
| nonlinear_xor | NeuralForest_Calibrated | +0.00018 |
| nonlinear_xor | NeuralForest_Differentiable | +0.00018 |
| overlap_80 | NeuralForest_Adaptive | +0.00175 |
| overlap_80 | NeuralForest_CART | +0.00000 |
| overlap_80 | NeuralForest_Calibrated | +0.00175 |
| overlap_80 | NeuralForest_Differentiable | +0.00205 |
| presence_bg_1to1 | NeuralForest_Adaptive | +0.00148 |
| presence_bg_1to1 | NeuralForest_CART | +0.00000 |
| presence_bg_1to1 | NeuralForest_Calibrated | +0.00148 |
| presence_bg_1to1 | NeuralForest_Differentiable | -0.00133 |
| presence_bg_1to5 | NeuralForest_Adaptive | +0.00252 |
| presence_bg_1to5 | NeuralForest_CART | +0.00000 |
| presence_bg_1to5 | NeuralForest_Calibrated | +0.00252 |
| presence_bg_1to5 | NeuralForest_Differentiable | +0.01444 |
| pseudoabsence_hard | NeuralForest_Adaptive | +0.02854 |
| pseudoabsence_hard | NeuralForest_CART | +0.00000 |
| pseudoabsence_hard | NeuralForest_Calibrated | +0.02854 |
| pseudoabsence_hard | NeuralForest_Differentiable | +0.11497 |
| rare_10pct | NeuralForest_Adaptive | -0.00071 |
| rare_10pct | NeuralForest_CART | +0.00000 |
| rare_10pct | NeuralForest_Calibrated | -0.00071 |
| rare_10pct | NeuralForest_Differentiable | +0.00191 |
| rare_1pct | NeuralForest_Adaptive | +0.01393 |
| rare_1pct | NeuralForest_CART | +0.00000 |
| rare_1pct | NeuralForest_Calibrated | +0.01393 |
| rare_1pct | NeuralForest_Differentiable | -0.01133 |
| rare_3pct | NeuralForest_Adaptive | -0.00909 |
| rare_3pct | NeuralForest_CART | +0.00000 |
| rare_3pct | NeuralForest_Calibrated | -0.00909 |
| rare_3pct | NeuralForest_Differentiable | -0.00454 |
| similar_97 | NeuralForest_Adaptive | +0.00102 |
| similar_97 | NeuralForest_CART | +0.00000 |
| similar_97 | NeuralForest_Calibrated | +0.00093 |
| similar_97 | NeuralForest_Differentiable | +0.00080 |
| small_n | NeuralForest_Adaptive | +0.00126 |
| small_n | NeuralForest_CART | +0.00000 |
| small_n | NeuralForest_Calibrated | +0.00126 |
| small_n | NeuralForest_Differentiable | -0.00221 |

## Caveats

This is a matched-seed engineering evaluation, not an independent
confirmatory trial: the old scenario families were already inspected.
New seeds 251 and 347 are reported separately in summary.csv.
Version 0.5 adds seven new under-sampling and forest baseline
variants without historical counterparts. These do not appear
in the matched comparison but do appear in sampling_comparison.csv.
A positive comparison for a hybrid model cannot be attributed
to one component without an explicit ablation.

Times include algorithm-specific search and model fitting.
The full per-run paired metric and timing records are in
paired_v03_comparison.csv.
