# NeuralForest v0.3 to v0.4 paired comparison

These observations pair 19 simulator cases, four original seeds
and 16 common model variants. Both evaluations use untouched
independently generated test samples; higher improvement is better.
Changes in software version and CPU runner can affect timings.

| Scenario | Model | Mean objective improvement |
| --- | --- | ---: |
| correlated_30 | NeuralForest_Adaptive | +0.00259 |
| correlated_30 | NeuralForest_CART | +0.00000 |
| correlated_30 | NeuralForest_Differentiable | +0.00335 |
| distinct_0 | NeuralForest_Adaptive | +0.00002 |
| distinct_0 | NeuralForest_CART | +0.00000 |
| distinct_0 | NeuralForest_Differentiable | +0.00002 |
| distribution_shift | NeuralForest_Adaptive | +0.00016 |
| distribution_shift | NeuralForest_CART | +0.00000 |
| distribution_shift | NeuralForest_Differentiable | +0.00211 |
| identical_100 | NeuralForest_Adaptive | -0.00110 |
| identical_100 | NeuralForest_CART | +0.00000 |
| identical_100 | NeuralForest_Differentiable | -0.00114 |
| intermediate_45 | NeuralForest_Adaptive | +0.00007 |
| intermediate_45 | NeuralForest_CART | +0.00000 |
| intermediate_45 | NeuralForest_Differentiable | -0.00054 |
| label_noise_20pct | NeuralForest_Adaptive | +0.00102 |
| label_noise_20pct | NeuralForest_CART | +0.00000 |
| label_noise_20pct | NeuralForest_Differentiable | +0.00075 |
| large_n | NeuralForest_Adaptive | -0.00018 |
| large_n | NeuralForest_CART | +0.00000 |
| large_n | NeuralForest_Differentiable | -0.00018 |
| missing_15pct | NeuralForest_Adaptive | +0.00003 |
| missing_15pct | NeuralForest_CART | +0.00000 |
| missing_15pct | NeuralForest_Differentiable | +0.00003 |
| noise_50 | NeuralForest_Adaptive | -0.00300 |
| noise_50 | NeuralForest_CART | +0.00000 |
| noise_50 | NeuralForest_Differentiable | -0.00340 |
| nonlinear_xor | NeuralForest_Adaptive | +0.00040 |
| nonlinear_xor | NeuralForest_CART | +0.00000 |
| nonlinear_xor | NeuralForest_Differentiable | +0.00040 |
| overlap_80 | NeuralForest_Adaptive | -0.00074 |
| overlap_80 | NeuralForest_CART | +0.00000 |
| overlap_80 | NeuralForest_Differentiable | +0.00098 |
| presence_bg_1to1 | NeuralForest_Adaptive | -0.00038 |
| presence_bg_1to1 | NeuralForest_CART | +0.00000 |
| presence_bg_1to1 | NeuralForest_Differentiable | +0.00071 |
| presence_bg_1to5 | NeuralForest_Adaptive | +0.03159 |
| presence_bg_1to5 | NeuralForest_CART | +0.00000 |
| presence_bg_1to5 | NeuralForest_Differentiable | +0.00613 |
| pseudoabsence_hard | NeuralForest_Adaptive | +0.03972 |
| pseudoabsence_hard | NeuralForest_CART | +0.00000 |
| pseudoabsence_hard | NeuralForest_Differentiable | +0.16529 |
| rare_10pct | NeuralForest_Adaptive | -0.00854 |
| rare_10pct | NeuralForest_CART | +0.00000 |
| rare_10pct | NeuralForest_Differentiable | -0.01614 |
| rare_1pct | NeuralForest_Adaptive | -0.00547 |
| rare_1pct | NeuralForest_CART | +0.00000 |
| rare_1pct | NeuralForest_Differentiable | -0.01635 |
| rare_3pct | NeuralForest_Adaptive | +0.00561 |
| rare_3pct | NeuralForest_CART | +0.00000 |
| rare_3pct | NeuralForest_Differentiable | +0.00018 |
| similar_97 | NeuralForest_Adaptive | -0.00110 |
| similar_97 | NeuralForest_CART | +0.00000 |
| similar_97 | NeuralForest_Differentiable | -0.00110 |
| small_n | NeuralForest_Adaptive | -0.00140 |
| small_n | NeuralForest_CART | +0.00000 |
| small_n | NeuralForest_Differentiable | +0.00129 |

## Caveats

This is a matched-seed engineering evaluation, not an independent
confirmatory trial: the old scenario families were already inspected.
New seeds 131 and 197 are reported separately in summary.csv.
Version 0.4 adds a calibration-enabled model that has no
matching v0.3 counterpart and therefore does not appear here.
A positive comparison for a hybrid model cannot be attributed
to one component without an explicit ablation.

Times include algorithm-specific search and model fitting.
The full per-run paired metric and timing records are in
paired_v03_comparison.csv.
