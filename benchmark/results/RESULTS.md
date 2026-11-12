# Vampire benchmark results

All results come from real model fits to
independently generated training and test data. Two seeds were
evaluated for each scenario. Runtime is measured on GitHub-hosted
CPUs and includes internal selection and model fitting.

## Model rankings by scenario

| Scenario | Selection metric | Three leading models | NeuralForest differential (score, rank) | Training + prediction (s) |
| --- | --- | --- | --- | --- |
| correlated_30 | auc | GradientBoosting: 0.803, CatBoost: 0.802, XGBoost: 0.798 | 0.788, #7 | 0.84 |
| distinct_0 | auc | ExtraTrees: 0.994, CatBoost: 0.993, RandomForest: 0.993 | 0.992, #4 | 0.41 |
| distribution_shift | auc | ExtraTrees: 0.698, Logistic: 0.693, CatBoost: 0.693 | 0.684, #7 | 0.54 |
| identical_100 | logloss | Dummy: 0.693, Logistic: 0.695, SVM_RBF: 0.695 | 0.706, #7 | 0.35 |
| intermediate_45 | auc | CatBoost: 0.914, HistGradientBoosting: 0.909, XGBoost: 0.909 | 0.908, #6 | 0.48 |
| large_n | auc | CatBoost: 0.785, GradientBoosting: 0.780, XGBoost: 0.780 | 0.774, #4 | 3.33 |
| noise_50 | auc | AdaBoost: 0.783, CatBoost: 0.781, GradientBoosting: 0.780 | 0.772, #8 | 1.04 |
| nonlinear_xor | f1 | CatBoost: 0.997, GradientBoosting: 0.997, HistGradientBoosting: 0.993 | 0.988, #8 | 0.52 |
| overlap_80 | auc | CatBoost: 0.687, AdaBoost: 0.686, Logistic: 0.684 | 0.675, #6 | 0.47 |
| presence_bg_1to1 | boyce | Logistic: 0.997, ExtraTrees: 0.995, RandomForest: 0.995 | 0.986, #9 | 0.28 |
| presence_bg_1to5 | boyce | NeuralForest_Differentiable: 0.995, NeuralForest_CART: 0.991, CatBoost: 0.975 | 0.995, #1 | 0.82 |
| pseudoabsence_hard | boyce | Logistic: 0.993, MLP: 0.947, NeuralForest_Differentiable: 0.914 | 0.914, #3 | 0.30 |
| rare_10pct | pr_auc | Logistic: 0.524, ExtraTrees: 0.523, CatBoost: 0.515 | 0.474, #7 | 0.55 |
| rare_3pct | pr_auc | Logistic: 0.279, ExtraTrees: 0.225, CatBoost: 0.208 | 0.132, #12 | 1.15 |
| similar_97 | brier | Logistic: 0.249, Dummy: 0.250, AdaBoost: 0.251 | 0.254, #7 | 0.35 |
| small_n | balanced_accuracy | Logistic: 0.671, RandomForest: 0.662, NeuralForest_CART: 0.661 | 0.661, #3 | 0.13 |

## Reading the results

The full score matrix (12 metrics, training time and prediction time)
is in all_runs.csv. summary.csv has mean, standard deviation and rank
for every combination of scenario and algorithm.

Non-differentiable model-selection metrics (AUC, average precision,
Boyce and F1) have proxy gradient losses in NeuralForest. The
requested metric is nevertheless used for independent internal
selection and external testing. A proxy loss is not its exact gradient.

For presence-background designs, training negatives are unlabeled
background observations, potentially containing vampires.
True class labels are only used in the independent test population.
The labels in the presence-background training task differ from
true human-vampire labels and require separate interpretation.

The 100% similarity condition has indistinguishable class-specific
covariate distributions and should yield chance discrimination.
A lack of advantage over simpler methods should be reported as such.

Failures remain in the CSV and manifest rather than being excluded.
All time comparisons are specific to this hardware and workload.
See benchmark/README.md for caveats and exact configuration.
