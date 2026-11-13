# NeuralForest v0.3 to v0.4 paired comparison

These observations pair 19 simulator cases, four original seeds
and 16 common model variants. Both evaluations use untouched
independently generated test samples; higher improvement is better.
Changes in software version and CPU runner can affect timings.

| Scenario | Model | Mean objective improvement |
| --- | --- | ---: |
| correlated_30 | NeuralForest_Adaptive | +0.00402 |
| correlated_30 | NeuralForest_CART | +0.00000 |
| correlated_30 | NeuralForest_Differentiable | +0.00618 |
| distinct_0 | NeuralForest_Adaptive | -0.00002 |
| distinct_0 | NeuralForest_CART | +0.00000 |
| distinct_0 | NeuralForest_Differentiable | -0.00002 |
| distribution_shift | NeuralForest_Adaptive | -0.00183 |
| distribution_shift | NeuralForest_CART | +0.00000 |
| distribution_shift | NeuralForest_Differentiable | -0.00025 |
| identical_100 | NeuralForest_Adaptive | +0.00064 |
| identical_100 | NeuralForest_CART | +0.00000 |
| identical_100 | NeuralForest_Differentiable | +0.00066 |
| intermediate_45 | NeuralForest_Adaptive | -0.00056 |
| intermediate_45 | NeuralForest_CART | +0.00000 |
| intermediate_45 | NeuralForest_Differentiable | +0.00018 |
| label_noise_20pct | NeuralForest_Adaptive | -0.00051 |
| label_noise_20pct | NeuralForest_CART | +0.00000 |
| label_noise_20pct | NeuralForest_Differentiable | -0.00227 |
| large_n | NeuralForest_Adaptive | +0.00012 |
| large_n | NeuralForest_CART | +0.00000 |
| large_n | NeuralForest_Differentiable | +0.00012 |
| missing_15pct | NeuralForest_Adaptive | -0.00009 |
| missing_15pct | NeuralForest_CART | +0.00000 |
| missing_15pct | NeuralForest_Differentiable | -0.00033 |
| noise_50 | NeuralForest_Adaptive | -0.00074 |
| noise_50 | NeuralForest_CART | +0.00000 |
| noise_50 | NeuralForest_Differentiable | -0.00210 |
| nonlinear_xor | NeuralForest_Adaptive | +0.00027 |
| nonlinear_xor | NeuralForest_CART | +0.00000 |
| nonlinear_xor | NeuralForest_Differentiable | +0.00027 |
| overlap_80 | NeuralForest_Adaptive | +0.00345 |
| overlap_80 | NeuralForest_CART | +0.00000 |
| overlap_80 | NeuralForest_Differentiable | +0.00405 |
| presence_bg_1to1 | NeuralForest_Adaptive | -0.00242 |
| presence_bg_1to1 | NeuralForest_CART | +0.00000 |
| presence_bg_1to1 | NeuralForest_Differentiable | -0.00089 |
| presence_bg_1to5 | NeuralForest_Adaptive | +0.03281 |
| presence_bg_1to5 | NeuralForest_CART | +0.00000 |
| presence_bg_1to5 | NeuralForest_Differentiable | +0.02442 |
| pseudoabsence_hard | NeuralForest_Adaptive | +0.06071 |
| pseudoabsence_hard | NeuralForest_CART | +0.00000 |
| pseudoabsence_hard | NeuralForest_Differentiable | +0.15828 |
| rare_10pct | NeuralForest_Adaptive | -0.00148 |
| rare_10pct | NeuralForest_CART | +0.00000 |
| rare_10pct | NeuralForest_Differentiable | +0.00122 |
| rare_1pct | NeuralForest_Adaptive | +0.01087 |
| rare_1pct | NeuralForest_CART | +0.00000 |
| rare_1pct | NeuralForest_Differentiable | -0.01150 |
| rare_3pct | NeuralForest_Adaptive | +0.00422 |
| rare_3pct | NeuralForest_CART | +0.00000 |
| rare_3pct | NeuralForest_Differentiable | +0.00009 |
| similar_97 | NeuralForest_Adaptive | +0.00012 |
| similar_97 | NeuralForest_CART | +0.00000 |
| similar_97 | NeuralForest_Differentiable | -0.00049 |
| small_n | NeuralForest_Adaptive | -0.00082 |
| small_n | NeuralForest_CART | +0.00000 |
| small_n | NeuralForest_Differentiable | -0.00265 |

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
