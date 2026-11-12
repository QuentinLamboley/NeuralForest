# NeuralForest

**Neural-guided random forests for R** : a research-stage package with a native
C++17 tree engine and an adaptive neural sampler.

NeuralForest combines weighted, randomized CART trees and a small multilayer
perceptron (MLP). The MLP learns **which observations are difficult to predict**
from out-of-bag (OOB) errors and adapts sampling probabilities for subsequent
forests. Its learned input-layer saliencies also guide feature proposals.

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
