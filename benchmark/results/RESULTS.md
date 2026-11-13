# Vampire benchmark: audited NeuralForest

All times are measured wall-clock values on GitHub-hosted CPU
runners with mathematical libraries constrained to one thread.
Selection time includes internal hyperparameter and expert search.
Final fit time measures the separately trained final estimator.
Prediction time includes evaluation inference on the held-out set.
The three stages are also available independently in summary.csv.

The experiment covers 19 scenarios, eight seeds and 25 methods.
Successful runs: 3800 / 3800.
Seeds 251 and 347 are new confirmation seeds absent from earlier v0.4.1 runs.

## Overall ranking for each scenario

| Scenario | Metric | Top three (eight-seed mean) | NeuralForest adaptatif | Rank | Selection (s) | Final fit (s) | Prediction (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| correlated_30 | auc | GradientBoosting 0.796, CatBoost 0.792, NeuralForest_Adaptive 0.790 | 0.790 | 3 | 0.64 | 0.23 | 0.024 |
| distinct_0 | auc | ExtraTrees 0.993, AdaBoost 0.993, CatBoost 0.992 | 0.992 | 5 | 0.24 | 0.05 | 0.008 |
| distribution_shift | auc | ExtraTrees 0.692, CatBoost 0.687, NeuralForest_Adaptive 0.686 | 0.686 | 3 | 0.48 | 0.12 | 0.016 |
| identical_100 | logloss | Dummy 0.694, SVM_RBF 0.697, NeuralForest_Calibrated 0.697 | 0.697 | 5 | 0.23 | 0.08 | 0.012 |
| intermediate_45 | auc | CatBoost 0.906, GradientBoosting 0.903, AdaBoost 0.901 | 0.900 | 5 | 0.27 | 0.07 | 0.010 |
| label_noise_20pct | auc | NeuralForest_Adaptive 0.697, NeuralForest_Calibrated 0.697, NeuralForest_Differentiable 0.697 | 0.697 | 1 | 0.41 | 0.14 | 0.018 |
| large_n | auc | CatBoost 0.778, XGBoost 0.774, GradientBoosting 0.773 | 0.769 | 7 | 1.59 | 0.49 | 0.029 |
| missing_15pct | auc | CatBoost 0.777, NeuralForest_Differentiable 0.773, NeuralForest_Adaptive 0.773 | 0.773 | 3 | 0.46 | 0.14 | 0.016 |
| noise_50 | auc | GradientBoosting 0.786, AdaBoost 0.782, CatBoost 0.779 | 0.771 | 15 | 0.66 | 0.26 | 0.022 |
| nonlinear_xor | f1 | GradientBoosting 0.997, CatBoost 0.996, LightGBM 0.994 | 0.982 | 8 | 0.48 | 0.15 | 0.018 |
| overlap_80 | auc | CatBoost 0.669, AdaBoost 0.666, GradientBoosting 0.663 | 0.661 | 5 | 0.27 | 0.08 | 0.011 |
| presence_bg_1to1 | boyce | NeuralForest_SubSample50_Informed 0.995, Logistic 0.994, NeuralForest_Adaptive 0.994 | 0.994 | 3 | 0.22 | 0.05 | 0.018 |
| presence_bg_1to5 | boyce | NeuralForest_UndersampledInformed 0.989, NeuralForest_Adaptive 0.988, NeuralForest_Calibrated 0.988 | 0.988 | 2 | 0.55 | 0.18 | 0.021 |
| pseudoabsence_hard | boyce | Logistic 0.989, AdaBoost 0.724, NeuralForest_Adaptive 0.673 | 0.673 | 3 | 0.24 | 0.05 | 0.016 |
| rare_10pct | pr_auc | Logistic 0.510, ExtraTrees 0.509, NeuralForest_Adaptive 0.508 | 0.508 | 3 | 0.45 | 0.14 | 0.017 |
| rare_1pct | pr_auc | Logistic 0.177, NeuralForest_Adaptive 0.176, NeuralForest_Calibrated 0.176 | 0.176 | 2 | 1.32 | 0.62 | 0.015 |
| rare_3pct | pr_auc | Logistic 0.287, NeuralForest_Adaptive 0.278, NeuralForest_Calibrated 0.278 | 0.278 | 2 | 0.77 | 0.29 | 0.021 |
| similar_97 | brier | Dummy 0.250, AdaBoost 0.252, SVM_RBF 0.252 | 0.253 | 6 | 0.23 | 0.08 | 0.011 |
| small_n | balanced_accuracy | AdaBoost 0.663, NeuralForest_UndersampledInformed 0.662, NeuralForest_UndersampledPlain 0.662 | 0.657 | 7 | 0.15 | 0.03 | 0.010 |

## Interpretation

The validation set selects algorithms, model complexity and mixtures
without consulting independent test outcomes. Non-differentiable
metrics use a differentiable proxy in soft trees. Average precision
uses a smooth listwise ranking surrogate, while AUC and Boyce use
a pairwise logistic surrogate. Test performance always uses the
original metric, not its surrogate.

NeuralForest_Adaptive and NeuralForest_Calibrated include ridge-regularized linear and constant
experts. Its performance must be interpreted as a hybrid ensemble,
not attributed exclusively to learned tree divisions.

Historical seed-matched v0.2 comparisons are in
paired_v02_comparison.csv. Means over seeds 251 and 347 appear in
new_seed_primary_mean and new_seed_rank in summary.csv.

Presence-background training negatives are not confirmed human
labels. The independent test population has true synthetic labels,
which changes the interpretation of reported metrics.

The identical-distribution negative control should give chance
discrimination; a constant predictor often minimizes log loss.
These synthetic datasets do not establish universal superiority.

## Result files

- all_runs.csv: all metrics, configurations and timed stages
- balanced under-sampling uses equal minority/majority per tree
  without replacement. Each tree uses a training-only class count.
- summary.csv: mean, standard deviation, ranking and phase timings
- paired_v02_comparison.csv: paired original-seed differences
- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison
- paired_v041_comparison.csv: six matched seeds against NeuralForest v0.4.1
- sampling_comparison.csv: forest-by-forest subsampling comparison
- COMPUTE_TIME.md: selection, final fit and prediction time by algorithm
- manifest.json: provenance, seeds, failures and environment
- benchmark/README.md: full experimental protocol and limitations

## Statistical qualification of the sampling comparison

Across 19 scenario means, the native balanced under-sampling model
with neural guidance improves the primary metric in 14 scenarios
and worsens it in five. However, **none of the 19 approximate
paired 95% confidence intervals (eight seeds) excludes zero**.
The median per-scenario training-plus-prediction time ratio is
approximately 2.00; mean total time is 0.4462 s with guidance
versus 0.2195 s without it.

These observations suggest opportunities for better informed sampling,
but they do not demonstrate statistically reliable superiority.
The 1:5 presence-background Boyce advantage is 0.00406 with an
approximate 95% CI from -0.00260 to +0.01072. The hard pseudoabsence
advantage is 0.03620 with an interval from -0.02517 to +0.09757.

Boyce values here are based on equal-width bins. They can change under
monotone calibration transformations that leave AUC unchanged.
Interpret the particularly large raw-versus-prior-corrected balanced
random-forest Boyce differences with care. The complete scores,
classifiers' configurations and measured CPU times remain in
all_runs.csv, summary.csv and sampling_comparison.csv.

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
