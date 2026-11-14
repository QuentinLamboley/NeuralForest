# Independent real-record v0.8 versus v0.9 paired audit

Three public datasets x three seeds x two regimes x 34 available
methods produce 612 matched comparisons. No test outcome
labels were used for model selection. The
new fidelity-constrained method is excluded from historical pairs
because it has no old counterpart.

| Previously available method | Mean AUC change |
| --- | ---: |
| NeuralForest_Adaptive | +0.000000 |
| NeuralForest_CART | +0.000000 |
| NeuralForest_Calibrated | +0.000000 |
| NeuralForest_Differentiable | +0.000000 |
| NeuralForest_PUInformed | +0.000000 |
| NeuralForest_PUNeural | +0.000000 |
| NeuralForest_PUStable | +0.000000 |
| NeuralForest_SubSample50_Informed | +0.000000 |
| NeuralForest_UndersampledInformed | +0.000000 |
| NeuralForest_UndersampledPlain | +0.000000 |

Timings may vary between GitHub-hosted machines.
Neither repeatedly inspected SCAR datasets nor in-simulator
transfer cohorts constitute independent field validation.
Detailed paired metrics/timings: paired_v08_scar_sar.csv.
