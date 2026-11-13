# NeuralForest v0.5 to v0.6 paired comparison

These observations pair 19 simulator cases, eight original seeds
and 25 common model variants. Both evaluations use untouched
independently generated test samples; higher improvement is better.
Changes in software version and CPU runner can affect timings.

| Scenario | Model | Mean objective improvement |
| --- | --- | ---: |
| correlated_30 | NeuralForest_Adaptive | +0.00100 |
| correlated_30 | NeuralForest_CART | +0.00000 |
| correlated_30 | NeuralForest_Calibrated | +0.00100 |
| correlated_30 | NeuralForest_Differentiable | +0.00189 |
| correlated_30 | NeuralForest_SubSample50_Informed | +0.00000 |
| correlated_30 | NeuralForest_UndersampledInformed | +0.00000 |
| correlated_30 | NeuralForest_UndersampledPlain | +0.00000 |
| distinct_0 | NeuralForest_Adaptive | -0.00002 |
| distinct_0 | NeuralForest_CART | +0.00000 |
| distinct_0 | NeuralForest_Calibrated | -0.00002 |
| distinct_0 | NeuralForest_Differentiable | -0.00011 |
| distinct_0 | NeuralForest_SubSample50_Informed | +0.00000 |
| distinct_0 | NeuralForest_UndersampledInformed | +0.00000 |
| distinct_0 | NeuralForest_UndersampledPlain | +0.00000 |
| distribution_shift | NeuralForest_Adaptive | -0.00128 |
| distribution_shift | NeuralForest_CART | +0.00000 |
| distribution_shift | NeuralForest_Calibrated | -0.00128 |
| distribution_shift | NeuralForest_Differentiable | -0.00123 |
| distribution_shift | NeuralForest_SubSample50_Informed | +0.00000 |
| distribution_shift | NeuralForest_UndersampledInformed | +0.00000 |
| distribution_shift | NeuralForest_UndersampledPlain | +0.00000 |
| identical_100 | NeuralForest_Adaptive | +0.00237 |
| identical_100 | NeuralForest_CART | +0.00000 |
| identical_100 | NeuralForest_Calibrated | +0.00180 |
| identical_100 | NeuralForest_Differentiable | +0.00376 |
| identical_100 | NeuralForest_SubSample50_Informed | +0.00000 |
| identical_100 | NeuralForest_UndersampledInformed | +0.00000 |
| identical_100 | NeuralForest_UndersampledPlain | +0.00000 |
| intermediate_45 | NeuralForest_Adaptive | +0.00013 |
| intermediate_45 | NeuralForest_CART | +0.00000 |
| intermediate_45 | NeuralForest_Calibrated | +0.00013 |
| intermediate_45 | NeuralForest_Differentiable | +0.00012 |
| intermediate_45 | NeuralForest_SubSample50_Informed | +0.00000 |
| intermediate_45 | NeuralForest_UndersampledInformed | +0.00000 |
| intermediate_45 | NeuralForest_UndersampledPlain | +0.00000 |
| label_noise_20pct | NeuralForest_Adaptive | -0.00022 |
| label_noise_20pct | NeuralForest_CART | +0.00000 |
| label_noise_20pct | NeuralForest_Calibrated | -0.00022 |
| label_noise_20pct | NeuralForest_Differentiable | -0.00038 |
| label_noise_20pct | NeuralForest_SubSample50_Informed | +0.00000 |
| label_noise_20pct | NeuralForest_UndersampledInformed | +0.00000 |
| label_noise_20pct | NeuralForest_UndersampledPlain | +0.00000 |
| large_n | NeuralForest_Adaptive | +0.00054 |
| large_n | NeuralForest_CART | +0.00000 |
| large_n | NeuralForest_Calibrated | +0.00054 |
| large_n | NeuralForest_Differentiable | -0.00011 |
| large_n | NeuralForest_SubSample50_Informed | +0.00000 |
| large_n | NeuralForest_UndersampledInformed | +0.00000 |
| large_n | NeuralForest_UndersampledPlain | +0.00000 |
| missing_15pct | NeuralForest_Adaptive | -0.00153 |
| missing_15pct | NeuralForest_CART | +0.00000 |
| missing_15pct | NeuralForest_Calibrated | -0.00153 |
| missing_15pct | NeuralForest_Differentiable | -0.00191 |
| missing_15pct | NeuralForest_SubSample50_Informed | +0.00000 |
| missing_15pct | NeuralForest_UndersampledInformed | +0.00000 |
| missing_15pct | NeuralForest_UndersampledPlain | +0.00000 |
| noise_50 | NeuralForest_Adaptive | +0.00103 |
| noise_50 | NeuralForest_CART | +0.00000 |
| noise_50 | NeuralForest_Calibrated | +0.00103 |
| noise_50 | NeuralForest_Differentiable | +0.00046 |
| noise_50 | NeuralForest_SubSample50_Informed | +0.00000 |
| noise_50 | NeuralForest_UndersampledInformed | +0.00000 |
| noise_50 | NeuralForest_UndersampledPlain | +0.00000 |
| nonlinear_xor | NeuralForest_Adaptive | -0.00126 |
| nonlinear_xor | NeuralForest_CART | +0.00000 |
| nonlinear_xor | NeuralForest_Calibrated | -0.00126 |
| nonlinear_xor | NeuralForest_Differentiable | -0.00126 |
| nonlinear_xor | NeuralForest_SubSample50_Informed | +0.00000 |
| nonlinear_xor | NeuralForest_UndersampledInformed | +0.00000 |
| nonlinear_xor | NeuralForest_UndersampledPlain | +0.00000 |
| overlap_80 | NeuralForest_Adaptive | +0.00295 |
| overlap_80 | NeuralForest_CART | +0.00000 |
| overlap_80 | NeuralForest_Calibrated | +0.00295 |
| overlap_80 | NeuralForest_Differentiable | +0.00200 |
| overlap_80 | NeuralForest_SubSample50_Informed | +0.00000 |
| overlap_80 | NeuralForest_UndersampledInformed | +0.00000 |
| overlap_80 | NeuralForest_UndersampledPlain | +0.00000 |
| presence_bg_1to1 | NeuralForest_Adaptive | -0.00149 |
| presence_bg_1to1 | NeuralForest_CART | +0.00000 |
| presence_bg_1to1 | NeuralForest_Calibrated | -0.00149 |
| presence_bg_1to1 | NeuralForest_Differentiable | -0.00070 |
| presence_bg_1to1 | NeuralForest_SubSample50_Informed | +0.00000 |
| presence_bg_1to1 | NeuralForest_UndersampledInformed | +0.00000 |
| presence_bg_1to1 | NeuralForest_UndersampledPlain | +0.00000 |
| presence_bg_1to5 | NeuralForest_Adaptive | -0.00069 |
| presence_bg_1to5 | NeuralForest_CART | +0.00000 |
| presence_bg_1to5 | NeuralForest_Calibrated | -0.00069 |
| presence_bg_1to5 | NeuralForest_Differentiable | +0.00978 |
| presence_bg_1to5 | NeuralForest_SubSample50_Informed | +0.00000 |
| presence_bg_1to5 | NeuralForest_UndersampledInformed | +0.00000 |
| presence_bg_1to5 | NeuralForest_UndersampledPlain | +0.00000 |
| pseudoabsence_hard | NeuralForest_Adaptive | +0.02023 |
| pseudoabsence_hard | NeuralForest_CART | +0.00000 |
| pseudoabsence_hard | NeuralForest_Calibrated | +0.02023 |
| pseudoabsence_hard | NeuralForest_Differentiable | -0.00444 |
| pseudoabsence_hard | NeuralForest_SubSample50_Informed | +0.00000 |
| pseudoabsence_hard | NeuralForest_UndersampledInformed | +0.00000 |
| pseudoabsence_hard | NeuralForest_UndersampledPlain | +0.00000 |
| rare_10pct | NeuralForest_Adaptive | +0.00019 |
| rare_10pct | NeuralForest_CART | +0.00000 |
| rare_10pct | NeuralForest_Calibrated | +0.00019 |
| rare_10pct | NeuralForest_Differentiable | +0.00482 |
| rare_10pct | NeuralForest_SubSample50_Informed | +0.00000 |
| rare_10pct | NeuralForest_UndersampledInformed | +0.00000 |
| rare_10pct | NeuralForest_UndersampledPlain | +0.00000 |
| rare_1pct | NeuralForest_Adaptive | +0.01041 |
| rare_1pct | NeuralForest_CART | +0.00000 |
| rare_1pct | NeuralForest_Calibrated | +0.01041 |
| rare_1pct | NeuralForest_Differentiable | -0.00769 |
| rare_1pct | NeuralForest_SubSample50_Informed | +0.00000 |
| rare_1pct | NeuralForest_UndersampledInformed | +0.00000 |
| rare_1pct | NeuralForest_UndersampledPlain | +0.00000 |
| rare_3pct | NeuralForest_Adaptive | -0.01026 |
| rare_3pct | NeuralForest_CART | +0.00000 |
| rare_3pct | NeuralForest_Calibrated | -0.01026 |
| rare_3pct | NeuralForest_Differentiable | -0.00751 |
| rare_3pct | NeuralForest_SubSample50_Informed | +0.00000 |
| rare_3pct | NeuralForest_UndersampledInformed | +0.00000 |
| rare_3pct | NeuralForest_UndersampledPlain | +0.00000 |
| similar_97 | NeuralForest_Adaptive | +0.00085 |
| similar_97 | NeuralForest_CART | +0.00000 |
| similar_97 | NeuralForest_Calibrated | +0.00094 |
| similar_97 | NeuralForest_Differentiable | +0.00077 |
| similar_97 | NeuralForest_SubSample50_Informed | +0.00000 |
| similar_97 | NeuralForest_UndersampledInformed | +0.00000 |
| similar_97 | NeuralForest_UndersampledPlain | +0.00000 |
| small_n | NeuralForest_Adaptive | +0.00183 |
| small_n | NeuralForest_CART | +0.00000 |
| small_n | NeuralForest_Calibrated | +0.00183 |
| small_n | NeuralForest_Differentiable | -0.00153 |
| small_n | NeuralForest_SubSample50_Informed | +0.00000 |
| small_n | NeuralForest_UndersampledInformed | +0.00000 |
| small_n | NeuralForest_UndersampledPlain | +0.00000 |

## Caveats

This is a matched-seed engineering evaluation, not an independent
confirmatory trial: the old scenario families were already inspected.
New seeds 401 and 503 are reported separately in summary.csv.
Version 0.6 adds six voted/stacked ensemble baselines
without historical counterparts. These appear in
ensemble_comparison.csv and METRICS_REPORT.md.
A positive comparison for a hybrid model cannot be attributed
to one component without an explicit ablation.

Times include algorithm-specific search and model fitting.
The full per-run paired metric and timing records are in
paired_v03_comparison.csv.
