# NeuralForest v0.3 engineering audit

## Scope

Inspected native tree training, differentiable routing, sampling weights,
encoding, metric selection, internal holdouts and benchmark timing.
The package is an experimental research implementation. Passing software
tests does not establish universal validity, correctness of every score
under every statistical design, or superiority over other methods.

## Findings and changes

| Issue | Previous behavior | Updated behavior | Verification |
| --- | --- | --- | --- |
| Duplicate observation weighting | Bootstrap probabilities and split impurity both used the same weight | Native trees default to single-weighted bootstrap; double mode remains explicitly selectable | Binary single/double training tests |
| Predictor preparation leakage | Numeric medians and factor encodings were estimated before the holdout split | Internal training data define the encoding; holdout data use fixed training statistics | Validation-covariate perturbation test |
| Round selection after refit | The differentiable blend could be selected using predictions from a native forest already refit on validation observations | All rounds, soft-tree blending and linear experts are selected before any full-data refit | Independent holdout protocol and code order |
| Average precision mismatch | R used trapezoidal PR AUC, Python used noninterpolated average precision | R uses the noninterpolated AP definition | Analytic 4-point AP test |
| Rare-event soft-tree objective | AP selection used an AUC pairwise proxy | Soft trees now support a differentiable smooth-AP ranking objective | Finite-difference gradient and rare-event training tests |
| Class weighting at high imbalance | Auto exponent reached 0.75 and was applied twice | Default single weighting and lower auto exponents | Full holdout benchmark |
| Limited model selection | Validation selected only CART and soft-tree variants | Optional ridge linear and constant experts, with validation-controlled mixing | Expert selection and probability tests |
| Compute accounting | Model selection and final fit were aggregated | Selection, final fit and prediction wall times are reported separately for every model | Benchmark output schema |
| Missingness benchmark | No explicit missing-covariate stress test | New missing-data scenario with train-only median imputation for all reference models | Python simulator protocol tests and full benchmark |
| Insufficient sensitivity analysis | Two random seeds | Four seeds, including two new confirmation seeds | Per-seed CSV and separate new-seed rankings |

## Differentiable objective map

- Regression: mean squared error.
- Binary log loss and Brier: differentiable objectives.
- AUC and Boyce: pairwise logistic ranking proxy.
- Average precision: listwise soft precision-at-rank proxy.
- F1 and several discrete criteria: focal loss proxy.
- Selection on internal validation uses the actual requested metric.
- External test performance always reports actual metrics, not proxy losses.

## Remaining limitations

There is no algorithm that can dominate every competing estimator over
every possible data-generating distribution without assumptions.
The No Free Lunch principle makes such a claim untenable.

Native CART trees and the differentiable oblique trees are distinct
ensembles. Gradient updates change the latter's splits directly.
Selection may set their final mixture weight to zero, which is a
valid complexity reduction rather than an optimization failure.

The linear expert is a hybrid extension. When it is selected, any gain
cannot be attributed solely to neural-guided trees.
The chosen model does not provide calibrated uncertainty intervals.
Pure presence-background experiments still involve unknown labels;
background observations are not verified absences.
Statistical significance requires more independent datasets and seeds
than a small simulator benchmark.

Measured training time includes internal validation search. It is
hardware-dependent and can be influenced by runtime caching, memory,
input encoding and task size. Internal metric selection is not a
substitute for an external test or nested resampling.

## Reproduction

The R package checks run on Linux, Windows and macOS.
The benchmark imports the original generator at
QuentinLamboley/td-machinelearning revision
48c2b28ca7874cbef1f7dde15de962c6e56c5027.

See benchmark/README.md for exact scenarios and comparison budgets.
See benchmark/results for measured scores and timings. The complete
version 0.2 results are preserved in audit/ for paired comparisons.
