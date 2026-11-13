## NeuralForest 0.7: positive–unlabeled uncertainty, native C++ training

Version 0.7 introduces two **experimental** methods for the
positive/background problem: `NeuralForest_PUNeural` and
`NeuralForest_PUInformed`. The neural component is a 1-hidden-layer
tanh MLP written in C++17/Rcpp and optimized by full-batch Adam with
the non-negative PU logistic risk. A grid of **prespecified**
class-prior assumptions (.12, .25, .40) is averaged as a sensitivity
ensemble; no true label from the independent tests is consulted.
The informed forest discounts unlabeled examples assigned a large
positive probability, while retaining confirmed positives. A
**distinct outer PU validation partition** selects the neural/forest
mix by worst-case nnPU risk over assumed priors.

Usage for **confirmed positives and unlabeled/background**, not
verified negatives:

```r
library(NeuralForest)
model <- nf_pu_forest(x, y, mode="informed",
                      priors=c(.12,.25,.40))
prob <- predict(model,newdata,type="prob")
```

**Critical assumption:** PU estimators generally require representative
observed positives, representative unlabeled sampling, and an
appropriate prior. Environmentally chosen **hard pseudo-absences**
violate the representativeness assumption. Good performance cannot
be guaranteed under this violation. Avoid interpreting nnPU predictions
as calibrated true-disease probabilities without external evidence.

The original v0.6 comparison remains unchanged:
31 methods × 19 cases × 10 seeds = 5,890 fits. The two native PU
experiments are added **only** for the three presence/background
cases (60 further fits), giving a planned 5,950 complete model fits.

Every fitted model predicts both the original independent test
population and an additional independent **shifted synthetic**
population, with multiple classification/ranking/calibration metrics
and selection/fitting/inference times. The latter is an external-style
transport stress test, **not independent real-world validation**.

See [the benchmark report](benchmark/results/RESULTS.md),
[the transfer evaluation](benchmark/results/EXTERNAL_VALIDATION.md)
and [the full audit](audit/AUDIT.md).

---

# NeuralForest

**Neural-guided random forests for R** : a research-stage package with a native
C++17 tree engine and an adaptive neural sampler.

NeuralForest combines weighted, randomized CART trees and a small multilayer
perceptron (MLP). The MLP learns **which observations are difficult to predict**
from out-of-bag (OOB) errors and adapts sampling probabilities for subsequent
forests. Its learned input-layer saliencies also guide feature proposals.

Version 0.6 separates soft-tree early stopping from model-selection holdouts
and fixes a timing accounting error in calibration selection. The expanded
reproducible benchmark compares 31 methods, including soft-voting and
cross-validated stacking ensembles, across 19 scenario families, ten
seeds and 15 independent-test metrics. Newly introduced ranking-based
Boyce is provided only as a monotone-transform sensitivity diagnostic,
not a replacement for the ecological continuous Boyce index. See
[benchmark/results/METRICS_REPORT.md](benchmark/results/METRICS_REPORT.md).
Genuine universal algorithm dominance cannot be established on finite
benchmark samples.

Version 0.5 adds genuine class-balanced under-sampling without replacement,
per-tree class-draw diagnostics, and a single sigmoid soft-tree model for
binary classification rather than two redundant one-vs-rest models.
Comparative benchmarks explicitly separate plain, under-sampled and
neural-informed forests; no advantage is assumed in advance.

Version 0.4.1 corrects a latent warm-start dimension check, discovered by
training three guided neural rounds. Regression tests now exercise the
third round rather than only checking network weight transport in isolation.

Version 0.4 stabilizes neural warm starts when input scaling changes between
rounds, broadens ridge-regularization candidates, and optionally learns binary
probability calibration on out-of-bag predictions. The validation fold selects
whether calibration is used; calibration is disabled unless explicitly
requested. Improvements to rank-based or binary decision metrics are not
guaranteed.

Version 0.3 adds a single-weighted sampling option, training-fold-only
preprocessing, a smooth average-precision surrogate for rare classes, and
an optional validation-selected regularized linear expert. The expert is
explicitly identified as a hybrid alternative and can be disabled.

With `differentiable=TRUE`, an auxiliary soft decision-tree ensemble uses
sigmoid routing and analytic reverse-mode gradients to optimize each split
projection, threshold and terminal value. A validation-selected mixture combines
the differentiable trees with the native CART ensemble. Non-differentiable
selection metrics use stated surrogate objectives, not exact metric gradients.

Claims of universal superiority over ranger or boosting methods are not made.
Independent benchmark results are published separately from the package checks.

## Install

Install R (4.3+) and a C++17 toolchain, then:

    install.packages("remotes")
    remotes::install_github("QuentinLamboley/NeuralForest")

Or install locally after cloning:

    install.packages("Rcpp")
    Rcpp::compileAttributes(".")
    install.packages(".", repos = NULL, type = "source")

## Quick start

    library(NeuralForest)
    data(iris)
    set.seed(42)
    id <- sample.int(nrow(iris), 120)
    fit <- neural_forest(
      iris[id, 1:4], iris$Species[id],
      control = nf_control(
        n_trees = 250,
        neural_rounds = 3,
        neural_hidden = c(48, 24),
        n_threads = 4,
        max_depth = 12,
        min_leaf = 2,
        neural_guidance = 0.6,
        differentiable = TRUE,
        soft_n_trees = 6,
        soft_depth = 3,
        soft_loss = "auto"
      ),
      metric = "macro_auc",
      seed = 42
    )
    print(fit)
    predict(fit, iris[-id,1:4])
    predict(fit, iris[-id,1:4], type = "prob")
    nf_importance(fit)
    nf_diagnostics(fit)
    nf_uncertainty(fit,iris[-id,1:4],measure="entropy")
    nf_permutation_importance(fit,iris[-id,1:4],iris$Species[-id],
                              repeats=3)
    nf_metrics("classification")
    fit$timings  # preprocessing, forest selection, soft trees, experts, refit

Regression:

    fit <- neural_forest(
      mtcars[, setdiff(names(mtcars), "mpg")],
      mtcars$mpg,
      task = "regression",
      control = nf_control(n_trees=100, neural_rounds=2,
                           min_leaf=2, max_depth=6),
      metric = "rmse", validation = 0.25
    )
    predict(fit, mtcars[,setdiff(names(mtcars),"mpg")])

Hyperparameter search with **nested holdouts** inside each outer fold:

    search <- list(n_trees=c(100L,300L), mtry=c(2L,4L),
                   min_leaf=c(2L,5L),
                   neural_guidance=c(0,0.4,0.8),
                   feature_guidance=c(0,0.3))
    tuning <- nf_tune(iris[,1:4], iris$Species, search=search,
                      trials=8, folds=3, metric="macro_auc")
    tuning$results
    tuning$best_control

An indicative comparison with ranger on a common outer holdout:

    nf_benchmark(iris[,1:4],iris$Species)

ranger is optional and installed separately for that comparison.

## Architecture

1. **Data adapter**: factor coding, numeric medians for missing values,
   training-only preprocessing, consistent prediction columns.
2. **Target profile**: empirical class frequencies, entropy, target quantiles,
   skewness, kurtosis, zero fraction and input missingness.
3. **C++17 training**: bootstrap or weighted sampling without replacement;
   feature subsampling; exact or binned split candidates; weighted CART Gini,
   entropy or variance reduction; leaf smoothing; OOB probability estimates;
   optional OpenMP parallelism.
4. **Differentiable tree learner**: weighted oblique splits route observations
   through sigmoid gates; leaf logits and split coefficients are optimized by
   analytic backpropagation with Adam. Separate internal validation selects
   the contribution of these trees to the final ensemble.
5. **Neural difficulty learner**: configurable ReLU MLP with sigmoid output,
   soft OOB-ranked error targets, mini-batch Adam, weight decay and early
   stopping. It adapts observation and variable proposal weights. Guidance
   can be disabled to compare a conventional forest baseline.
6. **Round selection**: disjoint validation set used to choose the best
   forest round by a user-selected metric; early stopping.
7. **Model analysis**: native impurity importance, holdout permutation
   importance, uncalibrated ensemble disagreement, per-tree predictions,
   feature distribution profiles, selection history and OOB coverage.
8. **Tuning and comparisons**: random/grid search with outer CV, optional
   group-aware splitting, ranger benchmark.

### Training flow

    training data -> tree forest -> OOB predictions -> difficulty labels
           ^                                          |
           |                                          v
       weighted sampling <- neural proposal <- MLP on training rows

The separate holdout is evaluated after each round. It is never used as
a neural training target. Selecting among rounds on a holdout nevertheless
means its reported score is a **selection score**, not an untouched test score.

### Common controls

| Control | Function |
| --- | --- |
| n_trees | Native CART trees per round |
| mtry | Proposed features at each split |
| min_leaf, min_split | Minimum observations in leaves / nodes |
| max_depth, max_nodes | Complexity and memory limits |
| max_candidates | Candidate split count per feature (0 exhausts all) |
| split_search | Exact splits or histogram-based approximate splits |
| histogram_bins | Number of bins for approximate splitting |
| extra_random | Random offsets for candidate thresholds |
| split_rule | Gini / entropy classification, variance regression |
| sample_fraction, replace | Sampling size and bootstrap policy |
| min_gain, laplace | Split penalty and leaf probability smoothing |
| neural_rounds, neural_hidden | Meta-learning iterations and layer widths |
| neural_epochs, neural_batch | Neural minibatch training |
| neural_lr, neural_weight_decay | Optimizer configuration |
| neural_guidance | Strength of adaptive example sampling |
| feature_guidance | Strength of learned feature proposals |
| neural_temperature, max_weight_ratio | Weight smoothness / caps |
| class_balance | None, balanced or auto |
| adaptive | Target-aware default values |
| n_threads | OpenMP threads (falls back to serial if unavailable) |
| sampling_weight_mode | Single prevents duplicating bootstrap weighting in node impurity |
| tree_sampling | Standard or class-balanced undersampling without replacement |
| linear_expert, linear_penalties | Optional ridge-regularized linear alternative |
| calibration, calibration_penalties | Optional out-of-bag Platt calibration of binary probabilities |
| calibration_min_gain | Required holdout improvement before calibration is selected |
| expert_blend, expert_min_gain | Validation-selected mixture strength and improvement guard |
| differentiable, soft_n_trees, soft_depth | Train gradient-optimized soft splits |
| soft_loss | Metric-specific differentiable surrogate or explicit choice |
| soft_blend | Holdout-selected contribution of differentiable trees |
| soft_epochs, soft_batch, soft_lr | Gradient optimization controls |
| soft_temperature, soft_clip, soft_patience | Routing and optimization stability |
| patience, neural_patience | Early stopping |
| verbose | Round-level progress |

Use the help system and nf_metrics() to inspect metric identifiers and
optimization directions. Some statistics (e.g. Boyce, AUC or MCC) are
undefined on degenerate folds. Boyce is intended for appropriate
presence-background problems and should not be interpreted as a generic
diagnostic metric. MASE depends on the observation order.

## Methodological precautions

- The engine is experimental. Speed, accuracy and calibration are NOT
  guaranteed to improve on ranger. Tune against a true held-out benchmark.
- The native C++ implementation is designed for numeric encoded inputs.
  Factor design matrices may expand high-cardinality inputs.
- The current regression output is the **forest mean**. Per-tree predictions
  are accessible, but this is not a formal quantile regression forest.
- Impurity importance is known to favor some variable types; test permutation
  importance on a separate holdout for scientific conclusions.
- Spatial, temporal and group-dependent data require appropriate splits.
  Use the groups argument or your own external evaluation.
- Neural warm starts now transport first-layer weights to each new
  standardized input space, preserving the initial prediction function.
- A held-out selector can accept or reject OOB Platt probability calibration;
  it is not a substitute for external calibration assessment.
- Linear experts are selected using an internal holdout. This adds model
  selection capacity; it is not a guarantee of improved test accuracy.
- Linear and constant experts, when enabled, alter the model family;
  results must be presented separately from a pure forest.
- Neural sampling is predictive importance-weighted resampling.
  Soft-tree split parameters are differentiated directly, but AUC, Boyce,
  F1 and related discrete metrics use ranking, smooth average-precision,
  or focal surrogates; held-out selection uses the requested metric.
- Multiclass AUC uses one-vs-rest averaging; ECE uses confidence calibration.
- Workload depends strongly on n, p, node size, tree count and neural rounds.
  CPU memory and compute usage can be substantial.

## Validation

The repository ships testthat tests and a GitHub Actions R CMD check
workflow. The simulator-based benchmark is specified in
[benchmark/README.md](benchmark/README.md), with measured scores and
timings published under [benchmark/results](benchmark/results). The
comparison evaluates XGBoost, LightGBM, CatBoost, ranger and other
reference methods across changes in class overlap, prevalence, feature
structure and presence-background sampling.

## Origins and license

Developed from an experimental neural-guided forest concept used in
equine Lyme and Ixodes ricinus research, reimplemented as a standalone
R package with a native tree backend.

CC0-1.0 (see LICENSE). Original research-grade software, provided as-is.

## Audited benchmark and runtime

The full 19-scenario, four-seed benchmark is available in
[benchmark/results/RESULTS.md](benchmark/results/RESULTS.md).
Separate parameter-selection, final-fit and prediction timings for
all 16 algorithms are tabulated in
[benchmark/results/COMPUTE_TIME.md](benchmark/results/COMPUTE_TIME.md).
A record of identified defects, changes and remaining limitations
is maintained in [audit/AUDIT.md](audit/AUDIT.md).

## Third-round ablation and reproducible compute costs

The v0.4.1 third-round warm-start defect is covered by an
end-to-end regression test. The independent two-versus-three
neural-round experiment reports all 76 fits in
[audit/rounds_results.csv](audit/rounds_results.csv) and
[audit/rounds_report.md](audit/rounds_report.md).
The six-seed comparison between all 17 candidate methods is
maintained in [benchmark/results](benchmark/results), including
[COMPUTE_TIME.md](benchmark/results/COMPUTE_TIME.md).

## Balanced subsampling example

A genuinely class-balanced, undersampled forest can be trained without
any neural guidance:

```r
baseline <- neural_forest(x, y, task="classification", metric="pr_auc",
  control=nf_control(n_trees=200, neural=FALSE,
    tree_sampling="balanced_undersample", sample_fraction=0.8))
baseline$native$sampled_class_counts
```

To examine whether neural guidance helps beyond the same sampling design:

```r
guided <- neural_forest(x, y, task="classification", metric="pr_auc",
  control=nf_control(n_trees=200, neural=TRUE,
    neural_rounds=3, tree_sampling="balanced_undersample",
    sample_fraction=0.8))
```

Balanced undersampling can discard many majority-class examples and can
change probability calibration; use independent test data and compare
Brier and log loss as well as AUC, average precision and Boyce. The
benchmarks also include unbalanced RandomForest, 50% and 80%
subsampling without replacement, a partial-row bootstrap RF, and
imbalanced-learn balanced forests with and without prior correction.

## Why model-selection holdouts matter (v0.6)

Previously the auxiliary soft trees used the same holdout both to
early-stop their gradient updates and to choose their ensemble
mixture with the native forest. Version 0.6 creates a disjoint
training-only soft early-stopping split; the outer holdout
is used strictly to select the round, soft mixture, calibration
and expert. This removes one concrete source of optimistic
validation selection. Tests verify that changing outer holdout
labels cannot change trained soft-tree parameters.

The main comparison reports 31 models, including classic and
subsampled random forests plus independent soft-voting and
three-fold stacking ensembles. Base learners inside stacking
fit their missing-value imputers inside each cross-validation
fold. Their fixed tuning budgets and computation are recorded.
Model selection still uses each scenario's predefined target
metric and never consults independent test labels.

Analysis includes ROC AUC, average precision, class threshold
scores, proper probability scores, expected calibration error,
top-decile precision/recall, and both ordinary and rank-transformed
Boyce. The separate metrics are diagnostics for the *same fitted
models* and are not retrospectively used to tune them.
