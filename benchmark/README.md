# Vampire benchmark

The experiment uses the original simulator from
QuentinLamboley/td-machinelearning, revision
48c2b28ca7874cbef1f7dde15de962c6e56c5027.
The generator functions are imported, not reimplemented.

## Reproduction

Install R 4.3 or later, a C++17 toolchain, Rcpp, jsonlite,
ranger and Python 3.12 with numpy, scipy, pandas, scikit-learn,
xgboost, lightgbm and catboost.

Checkout the pinned td-machinelearning revision into the
td-source subdirectory. Install NeuralForest using
R CMD INSTALL . and execute:

    python benchmark/run.py --group core
    python benchmark/run.py --group imbalance
    python benchmark/run.py --group structure
    python benchmark/run.py --group presence
    python benchmark/run.py --group stress
    python benchmark/summarize.py
    python audit/pair_v03.py

GitHub Actions performs the same steps and records
summary.csv, all_runs.csv, RESULTS.md and manifest.json.
Runs use six independent seeds: 17, 43, 71, 97, 131 and 197.
The first four match the v0.3 experiment. Seeds 131 and 197
are new confirmation seeds, included in the dedicated summary
columns for model ranking and every primary metric.

## Experimental scenarios

The original generator produces ten independent
class-conditional truncated-normal features. Its
parameters_with_proximity routine smoothly increases
similarity between humans and vampires.

Core: similarity 0, 45, 80, 97 and 100 percent.
Imbalance: 10 percent vampires, 3 percent vampires,
small samples and large samples.
Structure: 50 noise features, 30 correlated predictors,
nonlinear XOR outcome and changed test distribution.
Stress: one percent vampires, 20 percent noisy class labels,
and 15 percent missing values in each predictor.
Missing data are median-imputed from each training partition
without using its corresponding validation/test partition.
Presence only: observed vampires versus randomly drawn
unlabeled background at ratios 1:1 and 1:5, or
background points selected to be environmentally
similar to the observed presences. True background
labels are not used at selection time.

At 100 percent similarity, the class-conditional feature
distributions are identical, providing a chance-level
negative control. Test samples are generated with a
different random seed and never used for training.

## Reference methods

DummyClassifier, LogisticRegression, RandomForest,
ExtraTrees, GradientBoosting, HistGradientBoosting,
AdaBoost, XGBoost, LightGBM, CatBoost, RBF SVM,
multilayer perceptron, ranger, NeuralForest without
differentiable trees, NeuralForest with gradient-trained
oblique soft trees, NeuralForest with an adaptive ridge expert,
and NeuralForest with both adaptive expert and optional
out-of-bag Platt calibration. A direct comparison isolates
the additional calibration stage.

All models use the same training and independent
test sets per scenario and seed. Python methods select
between one or two fixed hyperparameter candidates
on a 20 percent training-only stratified holdout,
then refit to the entire training set. Ranger
uses a 20 percent training-only holdout.
NeuralForest selects training rounds and the
differentiable mixture coefficient on a training-only
20 percent validation sample, then refits its native
ensemble. Gradient trees train without external
test observations.

This is a bounded tuning budget, not an exhaustive
or perfectly compute-matched search. Models have
different training costs and search spaces.

## Optimization and metrics

The selected metric is AUC, average precision,
balanced accuracy, F1, Brier, log loss or Boyce
depending on the scenario. NeuralForest maps
non-differentiable metrics to pairwise ranking
or focal losses. Such proxies are distinct from
the mathematical gradient of the metric itself.
The actual metric is used for validation selection.

All models are scored on twelve test metrics:
ROC AUC, average precision, accuracy, balanced
accuracy, precision, recall, specificity, F1,
Matthews correlation, Brier, log loss and Boyce.
The default classification threshold is fixed
at 0.5. Interpret prevalence-sensitive metrics
alongside the sample's positive rate.

Reported wall-clock training time includes
hyperparameter selection and final fitting.
Prediction time is measured separately. CPU
thread counts are fixed at one for reproducibility.
Timing excludes dependencies and data generation.

Presence-background training negatives may contain
unobserved vampires and must not be interpreted
as confirmed humans. The independent test sample
does retain verified synthetic labels.

The code and data are synthetic experiments.
Performance on these scenarios does not establish
universal superiority on other datasets.
Results include failed runs and undefined metrics.

## Historical comparisons

The original v0.2 and v0.3 benchmark results are frozen under audit/.
The original v0.3 experiments are paired to the new results over the
same four seeds and the same simulator cases. The new comparison
reports objective improvements and changes in compute time.
The Platt calibrator is learned on inner-training OOB predictions
and activated only for proper scoring metrics when the internal
validation score improves. A refitted native forest receives
a freshly fitted OOB calibrator using training labels only.
Warm-start neural networks are transported into each new scaling
space, preserving their initial predictions. No test-set labels
are consulted in any training or parameter-selection step.
The updated report contains a seed-matched comparison
for original seeds 17 and 43 and independent confirmation
means for newly introduced seeds 71 and 97.
Changes to hyperparameter budgets and algorithm composition
are recorded in the configurations of each run.

## Version 0.5 forest sampling audit

The full evaluation compares 25 distinct methods, including
standard scikit-learn RandomForest and ranger implementations;
RandomForest with 50% bootstrap rows; 50% and 80% row
subsamples without replacement; class-balanced under-sampling
without replacement; and an otherwise identical balanced forest
with training-prevalence correction. Class-prior correction is
determined from the training fold only, not from test labels.

Native NeuralForest adds three separate controls:
a neural-informed 50% undersampled forest with no replacement,
a class-balanced undersampled forest without neural guidance,
and the same class-balanced design with a three-round
observation-priority learner. Their timing differences and
predictive scores isolate the cost and potential benefit of
neural guidance. The normal and hybrid NeuralForest variants
are still reported separately.

Eight independent generator seeds are evaluated:
17, 43, 71, 97, 131, 197, 251 and 347. The final two seeds
were not included in the previous six-seed experiment.
All 3,800 individual model fits (19 x 8 x 25) and their
selection, final-fitting, and prediction times are recorded.

The baseline version 0.4.1 measurements are archived under
audit/baseline-v041-all_runs.csv before the new benchmark.
audit/pair_v041.py compares all 1,938 exact matches on
the six previous seeds. New algorithms have no historical
counterpart, so they are not falsely included in the matched
comparison. Summary files are automatically regenerated from
per-seed measurements and are not manually transcribed.

Balanced under-sampling changes the class prevalence within
each tree, and training-prior correction only holds under
ordinary prior-shift assumptions. This does not solve
presence/background label contamination or distribution shift.
