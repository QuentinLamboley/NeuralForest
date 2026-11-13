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

## v0.4.1 test and benchmark disposition

The native package passes R CMD check on Linux, macOS and Windows
for the v0.4.1 commit that repairs the third-round warm-start
compatibility check. The primary benchmark was rerun on the same
19 scenarios, 17 reference variants and six seeds, yielding 1,938
successful fits. The generator revision is unchanged, making
the score comparisons to the earlier v0.4 release controlled,
although runner hardware and elapsed times can vary.

A separate third-round ablation on independent seeds 227 and 311
completed 76 fits. A third guided round improved six scenarios,
degraded two and tied eleven by their primary metrics. Across all
38 paired training tasks, the average additional total time was
approximately 0.153 seconds. Internal validation selected the
third round in 15/38 three-round fits. The full ablation is
reported in audit/rounds_report.md.

The principal remaining architectural weaknesses are
model-selection variance with extremely few positive cases,
soft-tree optimisation cost relative to boosted-tree baselines,
heavy dependence on the initial feature representation, and
the absence of distribution-free generalization guarantees.
The reported mean runtimes are not hardware-normalized
performance guarantees. The laboratory's separate real-world
validation datasets should be evaluated before any deployment.

## Version 0.5: informed versus class-balanced and classic forests

The v0.4.1 benchmark did not include directly comparable non-neural
undersampled random forests or a native class-balanced undersampling
strategy. The native v0.5 implementation now draws equal-sized class
strata without replacement, scaled by sample_fraction relative to the
minority class size. This changes both the statistical sampling problem
and the effective training sample size. It is not equivalent to tuning
the same ordinary random forest, so each configuration is reported
as a separate algorithm.

The native engine exports sampled_class_counts for every tree.
Regression tests assert equal per-tree class counts, verify no
replacement and preserve ordinary unbalanced bootstrap behavior.
An explicit informed/uninformed balanced pair separates learned OOB
observation-priority guidance from the subsampling itself. A separate
ordinary 50% without-replacement informed forest is included.

The previous implementation trained separate class-0 and class-1 soft
forests for binary targets, although a single sigmoid probability
determines both class probabilities. Version 0.5 trains exactly one
soft forest for the positive label, returns its complementary class
probability and validates the normalisation. This halves the number
of trained binary soft ensembles. Prediction and model selection are
not guaranteed to match the earlier OVR implementation.

New external baselines include a plain sklearn RandomForest,
partial bootstrap (50%), independent 50% and 80% subsampling
without replacement, imbalanced-learn balanced random forests,
and a separately identified training-prior-corrected balanced
forest. The prior correction is a deterministic Bayes odds update
using the **training** class prevalence and cannot use test labels.
It depends on the prior-shift assumption, which can fail under
presence-background contamination and distribution shift.

The full 19 x 8 x 25 evaluation reports 3,800 model runs on six
previous seeds and two new confirmation seeds, including selection,
final training, and inference time separately. The version v0.4.1
raw measurements are frozen under audit/baseline-v041-all_runs.csv.
All score changes are shown including declines.

Further limitations remain: no finite training set can guarantee
universal superiority, model selection can overfit very rare positive
classes, balanced sampling can worsen calibration, and benchmarks
on one synthetic generator do not establish performance on ecology,
veterinary clinical data, or other real-world applications.

## Version 0.5 measured findings and statistical uncertainty

The completed benchmark has 3,800 successful evaluations from 25 models,
19 scenarios and eight simulator seeds. All three R CMD check platforms
passed. The informed class-balanced native forest improved the selected
primary metric over its otherwise matched uninformed version in 14 of
19 scenario averages, and worsened the remaining five. The average
wall-clock total was 0.4462 s for the informed balanced forest and
0.2195 s for its plain counterpart.

These numerical wins must not be described as established superiority.
Paired two-sided 95% Student t confidence intervals for the mean
eight-seed score differences include zero in all 19 individual
scenarios. Example intervals (favorable-direction informed minus
plain): presence/background 1:5 Boyce +0.00406, 95% CI
[-0.00260, +0.01072]; pseudoabsence-hard Boyce +0.03620,
95% CI [-0.02517, +0.09757]; rare 1% average precision
+0.00808, 95% CI [-0.00396, +0.02011]; noise_50 AUC -0.00197,
95% CI [-0.00435, +0.00041]. These intervals are approximate and
do not account for reusing the original simulator scenario families.
No multiple-comparison adjustment could make a non-significant
unadjusted interval significant.

In the 1:5 presence-background scenario, the raw balanced random
forest obtains mean Boyce 0.98743 and its training-prior-corrected
version obtains 0.91644. This is not necessarily evidence that
the classifier changed its ranking. The current implementation of
Boyce discretizes the numerical suitability scale into equal-width
bins, so monotone transformations of probabilities can change its
statistic. The ROC AUC of the two versions is nearly identical.
Do not treat equal-width-bin Boyce as a purely rank-invariant metric;
test alternate binning definitions before reporting ecological model
selection conclusions.

The informed 50% ordinary subsample, the class-balanced informed
forest and the hybrid adaptive NeuralForest remain distinct model
families. Neither their differences in samples per tree nor their
different hyperparameter search budgets establish a fair
same-compute causal comparison. The paired informed-versus-plain
balanced comparison is the most controlled estimate of network
guidance, but would still benefit from many more seeds and real
datasets.

## Important failure on a new presence-background seed

The independent seed 251 of the pseudoabsence_hard experiment reveals
a clear failure that the six-seed averages concealed. The adaptive
NeuralForest selects a native forest using an inner holdout Boyce
score of 0.90436, but the independent true-label test scores are
Boyce -0.25112 and AUC 0.48903. The selected soft-tree weight is
zero, as is the selected linear-expert weight. A standard Logistic
regression on exactly the same simulated data scores test Boyce
0.98968 and AUC 0.71145, despite an inner pseudo-label Boyce
of only 0.27373.

This is evidence of an objective-domain mismatch. The inner
validation treats sampled background as negatives, although it
contains unknown positives. The external synthetic test is
evaluated against verified original simulator labels, so the
internal selection metric is not estimating the same target.
No tree architecture or parameter search can guarantee that the
pseudo-label validation optimum generalizes to the true labels.
A future genuine positive-unlabelled analysis would need
specified prevalence assumptions, independent presence-only
evaluation or verified negatives, and sensitivity analysis.
The unfavourable result is retained in all_runs.csv. It must not
be fixed post hoc by selecting parameters from the test set.

The eight-seed mean of Boyce for the adaptive forest on this
scenario is 0.67309, versus 0.98887 for Logistic regression;
the extreme negative seed contributes to that reduction.

## Version 0.6: independent early-stopping and ensemble comparator audit

Earlier gradient-trained soft trees consumed the outer model-selection
validation labels for early stopping and then reused those labels to
choose their mixture weight and expert combination. This entangles
early stopping and outer selection, increasing winner's-curse bias.
Version 0.6 holds out a separate stratified 15% subset from
the internal training fold exclusively for soft-tree early stopping.
Only the remainder of the internal training partition trains
differentiable splits. The original outer validation remains reserved
for model selection. A regression test changes only outer validation
labels and checks identical learned soft-tree probabilities.

Previous timing accounting added calibration_selection_seconds
to soft_selection_seconds, but the soft stopwatch was started before
the calibration stage. Moving the soft timer after calibration
eliminates this double counting. Per-stage costs still exclude
data generation and dependency/environment installation.

Six independent scikit-learn ensemble baselines are included:
four soft-voting models combining random forests, extra trees,
boosting and/or regularized logistic regression, and two
three-fold out-of-fold stacking models. Each stacking base
estimator performs its own missing-data imputation inside
the fold rather than on all of the meta-training observations.
The evaluation protocol makes no claim of equal hyperparameter
budgets, and stacking entails intrinsically higher training cost.

A ten-seed, 31-method, 19-scenario experiment is designed to produce
5,890 runs. Eight seeds match v0.5 exactly and two are additional
confirmation seeds. The v0.5 3,800-run raw file is frozen for
a matched before/after comparison. Every method is scored
on 15 reported metrics (AUC, AP, F1, MCC, balanced accuracy,
precision, recall, specificity, accuracy, Brier, logloss,
standard equal-width Boyce, rank-based Boyce, ECE, precision@10%,
and recall@10%); accuracy makes sixteen total including the
thresholded accuracy metric.

Important limitations: training and selection on pseudo-absence
labels still cannot guarantee performance against true outcomes.
The soft-trees training subfold reduces effective fitting sample
size and may decrease predictive scores. Rank-based Boyce is
a new sensitivity diagnostic and must not be passed off as
the continuous Boyce index used in ecological modelling.
Statistical intervals from repeated seeds do not generalize
automatically to independent scientific datasets, and
multiple comparisons require caution.
