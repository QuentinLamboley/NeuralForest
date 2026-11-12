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
    python benchmark/summarize.py

GitHub Actions performs the same steps and records
summary.csv, all_runs.csv, RESULTS.md and manifest.json.
Runs use two independent seeds, 17 and 43.

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
differentiable trees and NeuralForest with
gradient-trained oblique soft trees.

All models use the same training and independent
test sets per scenario and seed. Python methods select
between one or two fixed hyperparameter candidates
on a 25 percent training-only stratified holdout,
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
