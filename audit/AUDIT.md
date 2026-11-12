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

## Observed v0.3 benchmark

The audited benchmark evaluated 16 methods across 19 scenarios and four seeds: 1,216 successful fits. The adaptive hybrid ranks first on two scenarios, the 20-percent label-noise AUC experiment and the 1:1 presence-background Boyce experiment. Logistic regression, CatBoost, ExtraTrees and other methods remain better elsewhere.

For 3 percent positives, the previous differentiable NeuralForest achieved 0.132 average precision on seeds 17 and 43. The new differentiable variant achieved 0.207 across four seeds; the adaptive hybrid reached 0.222. Logistic regression remained best at 0.264. The adaptive model's average selection, final fitting and prediction times were 0.849, 0.211 and 0.015 s. Logistic regression required 0.022, 0.003 and 0.001 s.

For the presence-background 1:5 experiment, the old two-seed differentiable Boyce score of 0.995 did not persist after the audit: audited differentiable score 0.962 and adaptive score 0.955 across four seeds. This regression is retained in all score tables. The results cannot identify which individual correction caused the difference without additional targeted ablations.

Two new seeds, 71 and 97, are reported separately in benchmark/results/summary.csv. This helps detect improvements that only reproduce on the previously inspected samples, but two confirmation seeds are not a substitute for an external benchmark suite.

All algorithms' model-selection time, refit time, prediction time and total are published in benchmark/results/COMPUTE_TIME.md and summary.csv. The full per-run record is in all_runs.csv.

## Version 0.4: further engineering review

The previous experiment exposed two shortcomings that were not addressed by
the v0.3 modifications:

* Neural guidance was warm-started after recomputing predictor means and
  standard deviations without transporting the first-layer weights. The
  function represented by the network could change before its first new
  optimization update. Version 0.4 transports first-layer weights and
  biases so the original unsaturated function is preserved. Adam moments
  are reset after the coordinate transformation. A dedicated numerical
  invariance test checks the predictions before and after transport.
* The linear experts were restricted to relatively strong ridge penalties.
  The candidate set now extends into weak regularization, while remaining
  subject to inner-validation choice. Weak regularization is not always
  preferable, especially when correlated predictors are numerous.

Binary proper-scoring metrics can now select an out-of-bag Platt calibration
on the inner-validation fold. The calibrator is fitted on inner-training OOB
probabilities. If the final native forest is refitted, the calibrator is
relearned on the refitted forest's OOB predictions. This avoids fitting the
calibration coefficients to the same validation labels that decide whether
calibration is applied. The new stage is separately timed.

Version 0.4 adds an explicit calibration-enabled model to the comparison.
A six-seed trial on the 19 original scenario families includes the four
v0.3 seeds and two additional confirmation seeds. The 19 x 17 x 6
design requires 1,938 completed model evaluations before publishing
a complete report. The results are not expected to favor the new version
in every condition, and every observed regression must be retained.

### Limitations that cannot be eliminated by software fixes

The data-generating distribution is unknown, so no classifier can guarantee
uniform dominance on all test distributions. These synthetic examples are
not a substitute for externally gathered real datasets.
Metric-specific model selection on a small inner validation sample can
overfit, particularly for rare-event average precision and Boyce.
Out-of-bag calibration is approximate because observations are used in
other trees of the ensemble and model selection remains data-dependent.
A 0.5 decision threshold does not optimize all thresholded metrics.
The current benchmark uses bounded and unequal hyperparameter budgets,
and the recorded wall-clock times are runner-specific.

### Additional defect detected by third-round ablation

The initial three-round ablation failed because the warm-start
compatibility check applied ncol() to layer lists, rather than
to each layer's numeric weight matrix. This code path is not reached
by a benchmark configured for only two neural rounds. Version 0.4.1
corrects the compatibility check and adds an end-to-end three-round
regression test. The corrected ablation must complete before claiming
that the warm-start implementation has been empirically evaluated.
