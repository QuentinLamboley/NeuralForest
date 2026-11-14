# Independent real-record v0.7 versus v0.8 paired audit

Three public datasets x three seeds x 33 previously available
methods produce 297 matched SCAR comparisons. No test outcome
labels were used for model selection. The SAR mechanism and the
new stable-guidance model are excluded from historical pairs
because they have no old counterpart.

| Previously available method | Mean AUC change |
| --- | ---: |
| NeuralForest_Adaptive | +0.000000 |
| NeuralForest_CART | +0.000000 |
| NeuralForest_Calibrated | +0.000000 |
| NeuralForest_Differentiable | +0.000000 |
| NeuralForest_PUInformed | +0.000000 |
| NeuralForest_PUNeural | +0.000000 |
| NeuralForest_SubSample50_Informed | +0.000000 |
| NeuralForest_UndersampledInformed | +0.000000 |
| NeuralForest_UndersampledPlain | +0.000000 |

Timings may vary between GitHub-hosted machines.
Neither repeatedly inspected SCAR datasets nor in-simulator
transfer cohorts constitute independent field validation.
Detailed paired metrics/timings: paired_v07_scar.csv.
