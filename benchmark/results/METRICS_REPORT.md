# Performance by metric: complete ensemble comparison

Model selection and hyperparameter search used each scenario's
predefined primary objective, **not** these test-set diagnostics.
The independent outcomes for 19 scenarios, 10 seeds and all models
are in all_runs.csv. Rankings are provided separately for all
15 metrics, including proper scores, discrimination,
fixed-threshold classification, top-decile retrieval, and Boyce.

## Metric-level comparisons

The following values average **ranks**, not incompatible numeric
scores, across scenarios. Lower average rank is more favorable;
first-place counts are descriptive. These are not significance
tests and do not establish universal algorithm dominance.

| Metric | Best mean-rank method | Average rank |
| --- | --- | ---: |
| roc_auc | Voting_Heterogeneous | 3.37 |
| pr_auc | Voting_Heterogeneous | 2.05 |
| balanced_accuracy | Voting_Heterogeneous | 6.11 |
| f1 | Voting_RF_CatBoost | 8.53 |
| mcc | Voting_Heterogeneous | 4.47 |
| precision | Voting_Heterogeneous | 3.63 |
| recall | GradientBoosting | 8.68 |
| specificity | Voting_Heterogeneous | 6.63 |
| brier | Voting_Heterogeneous | 5.79 |
| logloss | Voting_Heterogeneous | 6.74 |
| boyce | NeuralForest_Calibrated | 6.63 |
| boyce_rank | NeuralForest_Calibrated | 8.26 |
| ece | Dummy | 3.26 |
| precision_at_10pct | Voting_Heterogeneous | 2.32 |
| recall_at_10pct | Voting_Heterogeneous | 2.32 |

## NeuralForest versus voted and stacked models

The table below compares representative families on every
scenario's predefined metric, including actual mean score
and the three measured compute stages. It does not imply the
same hyperparameter budget across models.

| Scenario | Primary metric | NF adaptive | Best ensemble | Ensemble score | Classical RF |
| --- | --- | ---: | --- | ---: | ---: |
| correlated_30 | auc | 0.7905 | Voting_Heterogeneous | 0.7961 | 0.7850 |
| distinct_0 | auc | 0.9924 | Voting_RF_ExtraTrees | 0.9935 | 0.9924 |
| distribution_shift | auc | 0.6834 | Voting_Heterogeneous | 0.6920 | 0.6754 |
| identical_100 | logloss | 0.6949 | Stacking_RF_XGB | 0.6941 | 0.7157 |
| intermediate_45 | auc | 0.9012 | Stacking_Boosting | 0.9076 | 0.8987 |
| label_noise_20pct | auc | 0.6955 | Voting_Heterogeneous | 0.7009 | 0.6907 |
| large_n | auc | 0.7719 | Stacking_Boosting | 0.7771 | 0.7658 |
| missing_15pct | auc | 0.7733 | Stacking_Boosting | 0.7800 | 0.7686 |
| noise_50 | auc | 0.7725 | Voting_Boosting | 0.7812 | 0.7751 |
| nonlinear_xor | f1 | 0.9829 | Voting_RF_CatBoost | 0.9969 | 0.9853 |
| overlap_80 | auc | 0.6659 | Voting_Heterogeneous | 0.6754 | 0.6529 |
| presence_bg_1to1 | boyce | 0.9926 | Voting_Heterogeneous | 0.9934 | 0.9883 |
| presence_bg_1to5 | boyce | 0.9863 | Voting_Heterogeneous | 0.9735 | 0.9664 |
| pseudoabsence_hard | boyce | 0.7211 | Voting_Heterogeneous | 0.7449 | 0.1966 |
| rare_10pct | pr_auc | 0.5017 | Voting_Heterogeneous | 0.5209 | 0.4697 |
| rare_1pct | pr_auc | 0.1834 | Voting_Heterogeneous | 0.1780 | 0.1420 |
| rare_3pct | pr_auc | 0.2722 | Voting_Heterogeneous | 0.2900 | 0.2431 |
| similar_97 | brier | 0.2515 | Stacking_RF_XGB | 0.2506 | 0.2598 |
| small_n | balanced_accuracy | 0.6498 | Voting_Heterogeneous | 0.6612 | 0.6499 |

## Uncertainty and interpretation

paired_multi_metrics.csv contains all matched-seed differences
between NeuralForest Adaptive and reference algorithms, with
exploratory t-based 95% intervals. Positive improvement_mean
favors NeuralForest. With thousands of model-metric-scenario
comparisons, unadjusted intervals do not control multiplicity.
Extreme class rarity can make AP and top-decile retrieval volatile.

Equal-width-bin Boyce is sensitive to monotone score rescaling.
boyce_rank first transforms all predictions into their ranks and
is invariant to strictly monotone calibration without ties.
It is a robustness diagnostic, not a standard replacement for
the published continuous Boyce index. The numeric scale and
ecological meaning of a habitat suitability score still matter.

ECE is a descriptive calibration diagnostic and is not a proper
scoring rule. Brier and log-loss are separately reported.
Threshold metrics use the same fixed 0.5 cutoff for every
algorithm and may unfairly penalize deliberately class-balanced
or presence/background models. The top-decile retrieval metrics
are threshold-independent with respect to probability scale.

The simulated background labels are not verified negatives.
Optimizing a Boyce value on pseudo-labels can disagree with
the independent true-label target. No model is guaranteed to
be optimal for every possible data-generating distribution.

Files: all_metric_ranks.csv, metric_winners.csv,
ensemble_comparison.csv, paired_multi_metrics.csv,
average_metric_ranks.csv and primary_rank_summary.csv.
