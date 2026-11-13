# NeuralForest v0.4.1 to v0.5 paired comparison

These observations pair 19 simulator cases, six original seeds
and 17 common model variants. Both evaluations use untouched
independently generated test samples; higher improvement is better.
Changes in software version and CPU runner can affect timings.

| Scenario | Model | Mean objective improvement |
| --- | --- | ---: |
| correlated_30 | NeuralForest_Adaptive | +0.00273 |
| correlated_30 | NeuralForest_CART | +0.00000 |
| correlated_30 | NeuralForest_Calibrated | +0.00273 |
| correlated_30 | NeuralForest_Differentiable | +0.00168 |
| distinct_0 | NeuralForest_Adaptive | +0.00031 |
| distinct_0 | NeuralForest_CART | +0.00000 |
| distinct_0 | NeuralForest_Calibrated | +0.00031 |
| distinct_0 | NeuralForest_Differentiable | +0.00031 |
| distribution_shift | NeuralForest_Adaptive | +0.00083 |
| distribution_shift | NeuralForest_CART | +0.00000 |
| distribution_shift | NeuralForest_Calibrated | +0.00083 |
| distribution_shift | NeuralForest_Differentiable | +0.00167 |
| identical_100 | NeuralForest_Adaptive | +0.00092 |
| identical_100 | NeuralForest_CART | +0.00000 |
| identical_100 | NeuralForest_Calibrated | +0.00029 |
| identical_100 | NeuralForest_Differentiable | +0.00033 |
| intermediate_45 | NeuralForest_Adaptive | -0.00068 |
| intermediate_45 | NeuralForest_CART | +0.00000 |
| intermediate_45 | NeuralForest_Calibrated | -0.00068 |
| intermediate_45 | NeuralForest_Differentiable | -0.00160 |
| label_noise_20pct | NeuralForest_Adaptive | +0.00092 |
| label_noise_20pct | NeuralForest_CART | +0.00000 |
| label_noise_20pct | NeuralForest_Calibrated | +0.00092 |
| label_noise_20pct | NeuralForest_Differentiable | +0.00103 |
| large_n | NeuralForest_Adaptive | -0.00030 |
| large_n | NeuralForest_CART | +0.00000 |
| large_n | NeuralForest_Calibrated | -0.00030 |
| large_n | NeuralForest_Differentiable | +0.00014 |
| missing_15pct | NeuralForest_Adaptive | +0.00076 |
| missing_15pct | NeuralForest_CART | +0.00000 |
| missing_15pct | NeuralForest_Calibrated | +0.00076 |
| missing_15pct | NeuralForest_Differentiable | +0.00117 |
| noise_50 | NeuralForest_Adaptive | -0.00229 |
| noise_50 | NeuralForest_CART | +0.00000 |
| noise_50 | NeuralForest_Calibrated | -0.00229 |
| noise_50 | NeuralForest_Differentiable | -0.00320 |
| nonlinear_xor | NeuralForest_Adaptive | +0.00027 |
| nonlinear_xor | NeuralForest_CART | +0.00000 |
| nonlinear_xor | NeuralForest_Calibrated | +0.00027 |
| nonlinear_xor | NeuralForest_Differentiable | +0.00027 |
| overlap_80 | NeuralForest_Adaptive | -0.00081 |
| overlap_80 | NeuralForest_CART | +0.00000 |
| overlap_80 | NeuralForest_Calibrated | -0.00081 |
| overlap_80 | NeuralForest_Differentiable | -0.00004 |
| presence_bg_1to1 | NeuralForest_Adaptive | +0.00381 |
| presence_bg_1to1 | NeuralForest_CART | +0.00000 |
| presence_bg_1to1 | NeuralForest_Calibrated | +0.00381 |
| presence_bg_1to1 | NeuralForest_Differentiable | -0.00040 |
| presence_bg_1to5 | NeuralForest_Adaptive | +0.00156 |
| presence_bg_1to5 | NeuralForest_CART | +0.00000 |
| presence_bg_1to5 | NeuralForest_Calibrated | +0.00156 |
| presence_bg_1to5 | NeuralForest_Differentiable | +0.00190 |
| pseudoabsence_hard | NeuralForest_Adaptive | +0.04269 |
| pseudoabsence_hard | NeuralForest_CART | +0.00000 |
| pseudoabsence_hard | NeuralForest_Calibrated | +0.04269 |
| pseudoabsence_hard | NeuralForest_Differentiable | +0.15071 |
| rare_10pct | NeuralForest_Adaptive | -0.00152 |
| rare_10pct | NeuralForest_CART | +0.00000 |
| rare_10pct | NeuralForest_Calibrated | -0.00152 |
| rare_10pct | NeuralForest_Differentiable | -0.00806 |
| rare_1pct | NeuralForest_Adaptive | +0.00229 |
| rare_1pct | NeuralForest_CART | +0.00000 |
| rare_1pct | NeuralForest_Calibrated | +0.00229 |
| rare_1pct | NeuralForest_Differentiable | -0.01192 |
| rare_3pct | NeuralForest_Adaptive | +0.00458 |
| rare_3pct | NeuralForest_CART | +0.00000 |
| rare_3pct | NeuralForest_Calibrated | +0.00458 |
| rare_3pct | NeuralForest_Differentiable | +0.00141 |
| similar_97 | NeuralForest_Adaptive | +0.00011 |
| similar_97 | NeuralForest_CART | +0.00000 |
| similar_97 | NeuralForest_Calibrated | -0.00016 |
| similar_97 | NeuralForest_Differentiable | -0.00005 |
| small_n | NeuralForest_Adaptive | -0.00108 |
| small_n | NeuralForest_CART | +0.00000 |
| small_n | NeuralForest_Calibrated | -0.00108 |
| small_n | NeuralForest_Differentiable | -0.00096 |

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
