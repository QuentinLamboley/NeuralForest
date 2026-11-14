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

### Additional baseline audit: ranger imputation and compute-time scope

The R-based ranger baseline formerly computed median imputation
from all of its training data before the inner validation split.
This is a modest but avoidable transductive use of the selector
holdout's predictors. The selection-stage medians now depend only
on the inner training fold. The final ranger refit computes
medians on the complete training partition and applies them
to the independent test features. Data split and imputation
preparation are included in the measured selection stage,
and final-fit preprocessing is included in final-fit time.

The result aggregator refuses to certify or publish results
unless *all* 5,890 model executions succeeded, including every
new voting and stacking ensemble and all ranger fits. The complete
error history remains in CI logs for any failed execution.

## Final v0.6 measured disposition: 13 November 2026

The completed benchmark reused the pinned vampire simulator and evaluated
31 model types across 19 scenarios and 10 independent seeds. Exactly
5,890/5,890 executions succeeded. All three OS-specific R CMD checks
passed. The complete measured data are published under
benchmark/results and retained as the version 0.6 reproducibility
artifact. The original aggregation workflow completed all five
benchmark groups successfully but lacked SciPy in its separate
aggregation job. A dedicated publication workflow downloaded the
original five group archives, installed SciPy, verified every model
success and published the final reports. No new independent
simulation was substituted for the failed aggregation step.

The strongest mean primary-metric ranks over the 19 scenario
families were Voting_Heterogeneous at 4.1053, followed by
NeuralForest_Calibrated at 7.7895 and
NeuralForest_Adaptive at 7.8421. Voting_Heterogeneous has
seven sole first-place primary ranks; NeuralForest_Adaptive
has one sole first-place rank (rare_1pct), tied with
NeuralForest_Calibrated by score. A separate informed
half-subsampled NeuralForest led the presence_bg_1to1
scenario and the standalone differentiable NeuralForest led
presence_bg_1to5. These are rankings from selected metrics
that vary by scenario, not proof of a best model in general.

The evaluation also records fifteen diagnostic metrics plus
accuracy without changing the tuning target on the test set.
Voting_Heterogeneous achieved the best cross-scenario average
rank for ROC AUC, average precision, balanced accuracy, MCC,
precision, specificity, Brier, logloss and top-decile retrieval.
Voting_RF_CatBoost performed best on F1 by average rank,
GradientBoosting on recall, while NeuralForest_Calibrated
led the two Boyce rank summaries. All metrics and per-scenario
winners are in METRICS_REPORT.md and all_metric_ranks.csv.

A rare-positive advantage remains conditional. At 1% positive
prevalence NeuralForest_Adaptive achieved AP 0.1834 versus
Voting_Heterogeneous 0.1780 and Logistic 0.1732; however its
fixed-threshold F1 was only 0.0398 versus 0.0522 (voting)
and 0.0887 (logistic). This demonstrates that ranking and
decision threshold are separate optimization problems.
The matched eight-seed AP improvement over v0.5 was +0.0104,
but variability was large (standard deviation ~0.0195).
At 3% positive prevalence the adaptive model's AP was
0.2722 versus Voting_Heterogeneous 0.2900.
At 10% prevalence it was 0.5017 versus 0.5209.
At 15% missingness, Stacking_Boosting reached AUC 0.7800,
NeuralForest_Adaptive 0.7733.

Mean end-to-end wall-clock time across these short simulator
scenarios was 0.6761 seconds for NeuralForest_Adaptive
(selection 0.480, final refit 0.178, prediction 0.016),
0.9801 for Voting_Heterogeneous,
1.0845 for plain RandomForest, 1.8338 for
Stacking_Boosting and 2.2633 for Stacking_RF_XGB.
Search spaces, implementation languages, candidate counts
and runner hardware are not compute-matched, so absolute
time differences are descriptive only.

NeuralForest remains substantially behind Logistic regression
on pseudoabsence_hard Boyce: 0.7211 versus 0.9898. The
opposite direction of the internal pseudo-label and
independent true-label objectives remains a fundamental
scientific limitation, not a software defect that can
be eliminated with another search. Secondary metric
rankings, t intervals and individual seeds have been
inspected repeatedly during these development cycles;
future confirmatory claims require truly unused data,
and comparisons should adjust for repeated testing.
Neither a finite synthetic benchmark nor a passing software
test proves the absence of errors or universal dominance.

## Version 0.7: native nnPU uncertainty and transport test

The major remaining domain mismatch is that observed
presences are verified positives whereas background
cases are NOT verified negatives. Optimizing a standard
positive-versus-pseudo-negative holdout score can
prefer an incorrect model even when an independent
true-label evaluation suggests another choice.

The new C++17 module `src/pu.cpp` implements a
one-hidden-layer tanh MLP trained with Adam on the
non-negative positive-unlabeled risk estimator
(Kiryo et al., 2017; NeurIPS). It contains native
forward and backward differentiation and an
independent training-only early stopping partition.
Three prespecified positive-class-prior assumptions
(.12, .25, .40) represent sensitivity, not
post-hoc choices from the test set.

Two PU-specific variants separate (1) a standalone
native PU neural ensemble from (2) a native random
forest whose bootstrap draw weights discount
unlabeled cases with high PU-positive scores.
An outer PU-labeled holdout chooses a fixed
neural/forest blending coefficient through a
worst-case sensitivity risk, without seeing
independent true labels. This is NOT a universally
valid correction: representative positive and
unlabeled sampling are needed, whereas
`pseudoabsence_hard` deliberately violates
representative background sampling. The class
prior is not identifiable from arbitrary P/U
data without further assumptions.

Version 0.7 preserves the original 31 methods,
19 scenarios, ten seeds and 15 detailed
diagnostic metrics. It adds two PU-specific
methods only to the 3 presence scenarios,
for 5,950 total model fits when all succeed.
Its second independent synthetic test cohort
changes the class-feature distribution by a
prespecified similarity shift, and its true
labels are never used in fitting or tuning.

This is a **simulation-based transfer test**,
not external field/clinical/ecological validation.
The simulator and scenario designs have been
repeatedly inspected during method development;
their use as confirmatory evidence is limited.
Real data and genuinely fresh datasets are needed
before clinical or ecological deployment, and
we cannot promise universal best-in-class
performance or that there are zero defects.

Methods basis: Kiryo R, Niu G, du Plessis MC,
Sugiyama M. Positive-Unlabeled Learning with
Non-Negative Risk Estimator. NeurIPS 2017;
and Bekker J, Davis J. Learning from positive
and unlabeled data: a survey. Machine Learning
2020.

## Version 0.8 preregistered engineering corrections

**Failure A: in-sample neural guidance.** v0.7 estimated
PU positivity for training cases using a neural ensemble
fitted to those very same cases, which can amplify
overfitting through sampling. v0.8 cross-fits the native
C++ nnPU prior ensemble over two disjoint partitions.
Only other-fold predictions are passed to the C++ sampler.
If a split would have fewer than about 25 positives
available before the inner stop split, no guidance is
allowed. The original PU neural and informed variants
are preserved as negative and positive reference controls.

**Failure B: overconfident fixed bootstrap.** v0.7 used
the fixed multiplier max(0.15, 1 - 0.8*posterior) for
unlabeled cases. v0.8 uses six neural predictions
(two random initializations for each of three declared
class priors), retains all confirmed positive weights
at one, and discounts only strongly positive cases when
the six predictions agree. C++ normalizes the sum of
unlabeled weights back to their original count. It then
shrinks every deviation from one analytically until
ESS/n_unlabeled >= 0.96. Two prespecified intensities
are compared to a common uniform-forest baseline using
the same native forest seed and 130 trees.

**Failure C: validation regret from guided forests.**
v0.8 compares neural-only fallback, plain unweighted
forest, both guided strengths, and fixed 50% teacher
mixtures using only a disjoint outer PU holdout.
Selection uses the maximum non-negative PU risk over
predeclared priors on each of two stratified validation
halves. A candidate forest must improve upon the teacher
by more than 0.004 on BOTH halves, otherwise the
neural-only teacher is retained. This gate only
constrains PU surrogate risk under assumed representative
sampling; it cannot certify generalization or guard
against wrong PU priors or SAR/SNAR label selection.
No independent test labels are available to the gate.

**New falsification tests:** In addition to the identical
19 scenarios and original ten seeds, report the full
cross-fitted selection policy on the three PU-background
scenarios, including what proportion of cases selects the
teacher fallback, and compare 16 scoring metrics on both
the standard and shifted simulator cohort. Real public
data (Digits, Wine, Wisconsin Breast Cancer) are evaluated
under both uniform SCAR and feature-dependent SAR simulated
positive confirmation, on three seeds. Both evaluation
partitions retain original verified labels but these
are never accessed by model selection. These are
real records with artificial labels, NOT an independently
collected clinical, ecological or regional cohort.

**Computational fairness:** All 31 previous baselines and
two v0.7 PU variants are rerun unchanged, with separate
selection, refit and two-cohort inference time. PU-stable
timings include fitting all candidate forests, C++ sampling
gates, two-fold teacher cross-fitting and full-data
refit. The extra model is scoped to PU-background
scenarios, so its mean runtime must not be directly
compared with 19-scenario baselines. Mathematical libraries
use one CPU thread in the workflow. Shared hardware,
budgets and model complexities are not perfectly matched.

Scientific basis: Kiryo et al., "Positive-Unlabeled Learning
with Non-Negative Risk Estimator", NeurIPS 2017; Kumar
& Lambert, "Positive Unlabeled Learning Selected Not At
Random (PULSNAR)", PeerJ Computer Science 2024.
Both sources emphasize that assumed class priors and
positive-labeling mechanisms matter. v0.8 does NOT
implement PULSNAR propensity correction. Subjecting the
method to a deliberately feature-selected SAR labeling
regime is a stress test, not a proof of robustness.

## v0.8 observed failure and v0.9 exploratory countermeasure

The completed v0.8 experiment included 5,980 successful
synthetic evaluations, 612 public-record SCAR/SAR evaluations
and passing Linux/Windows/macOS checks. On hard selected
background, the stability-guided forest achieved average
true-label Boyce 0.4512 while the original neural-only nnPU
achieved 0.9670; on shifted synthetic test, 0.4093 vs 0.9218.
The supposedly conservative PU validation gate selected a
forest in all ten seeds of this scenario. Its empirical
nnPU surrogate objective improved but the independent
true-label ranking deteriorated. This is a clear refutation
of the idea that two-fold PU risk gating alone guarantees
stable generalization under selected background.

Version 0.9 adds a separate `nf_pu_safe_forest` with a trust
region around the exact original native nnPU teacher.
The original teacher is trained with the SAME prior grid,
number of restarts (one), encoding, random seed and
refit scheme as v0.7 `nf_pu_forest(mode="neural")`.
A regression test checks exact fallback equality.
Forests still use the previously audited native
C++ OOF neural guidance and ESS-regularized sampler.
A candidate is evaluated using the exact probability-clipped
output it will deploy. It must improve worst-prior PU risk
by >0.004 separately on both selection folds AND
preserve Spearman rank agreement >=0.995 in both.
At inference the final predictor is restricted to at
most 0.02 absolute probability distance from teacher.
These thresholds are exploratory engineering tolerances,
not universally optimal constants.

The old v0.8 5,980-row main simulator data and 612-row
public-record SCAR/SAR data are frozen by blob SHA in audit.
Comparisons are paired across the same seeds and datasets,
but the new method was designed *after examining* those
outcomes. A future independent geographical, temporal
or clinical site remains necessary to support an external
predictive claim. This strategy prevents unbounded
probability deviation but cannot guarantee preserving
a ranking metric among nearly tied observations.
