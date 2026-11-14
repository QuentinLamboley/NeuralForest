# PU guided forest: paired teacher attribution and transport audit

Evaluated commit: be249ede97aa03b7a388f008eb0faac32303db12.
Domain: synthetic. Both evaluation cohorts are test-only.
No test outcomes were used to fit, select or calibrate any model.

**Crucial control:** PUStableTeacher is the exact fitted two-restart
neural component of PUStable. It shares training and search time,
so its total time is NOT an independent neural-only compute cost.
Incremental-tree comparisons use identical networks and encoders.
The cheaper PUNeural reference has only one neural restart.

## Matching teacher, tree contribution, and transport

| Group | Test metric | Comparison | Mean gain | Losses / seeds | Shifted gain |
| --- | --- | --- | ---: | ---: | ---: |
| presence_bg_1to1 | boyce | combined_effect | +0.0011 | 3/10 | -0.0052 |
| presence_bg_1to1 | pr_auc | combined_effect | +0.0034 | 3/10 | +0.0016 |
| presence_bg_1to1 | roc_auc | combined_effect | +0.0036 | 1/10 | +0.0027 |
| presence_bg_1to1 | boyce | extra_teacher_restarts | +0.0011 | 3/10 | -0.0052 |
| presence_bg_1to1 | pr_auc | extra_teacher_restarts | +0.0034 | 3/10 | +0.0016 |
| presence_bg_1to1 | roc_auc | extra_teacher_restarts | +0.0036 | 1/10 | +0.0027 |
| presence_bg_1to1 | boyce | incremental_tree_effect | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to1 | pr_auc | incremental_tree_effect | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to1 | roc_auc | incremental_tree_effect | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to1 | boyce | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to1 | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to1 | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to5 | boyce | combined_effect | -0.0021 | 7/10 | -0.0006 |
| presence_bg_1to5 | pr_auc | combined_effect | +0.0040 | 2/10 | +0.0028 |
| presence_bg_1to5 | roc_auc | combined_effect | +0.0017 | 2/10 | +0.0014 |
| presence_bg_1to5 | boyce | extra_teacher_restarts | -0.0021 | 7/10 | -0.0006 |
| presence_bg_1to5 | pr_auc | extra_teacher_restarts | +0.0040 | 2/10 | +0.0028 |
| presence_bg_1to5 | roc_auc | extra_teacher_restarts | +0.0017 | 2/10 | +0.0014 |
| presence_bg_1to5 | boyce | incremental_tree_effect | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to5 | pr_auc | incremental_tree_effect | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to5 | roc_auc | incremental_tree_effect | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to5 | boyce | rank_guard_vs_legacy | +0.0009 | 0/10 | +0.0015 |
| presence_bg_1to5 | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |
| presence_bg_1to5 | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |
| pseudoabsence_hard | boyce | combined_effect | -0.2099 | 7/10 | -0.1856 |
| pseudoabsence_hard | pr_auc | combined_effect | -0.0437 | 6/10 | -0.0255 |
| pseudoabsence_hard | roc_auc | combined_effect | -0.0459 | 6/10 | -0.0287 |
| pseudoabsence_hard | boyce | extra_teacher_restarts | +0.0026 | 5/10 | -0.0158 |
| pseudoabsence_hard | pr_auc | extra_teacher_restarts | +0.0152 | 2/10 | +0.0036 |
| pseudoabsence_hard | roc_auc | extra_teacher_restarts | +0.0087 | 2/10 | +0.0016 |
| pseudoabsence_hard | boyce | incremental_tree_effect | -0.2125 | 4/10 | -0.1698 |
| pseudoabsence_hard | pr_auc | incremental_tree_effect | -0.0589 | 4/10 | -0.0290 |
| pseudoabsence_hard | roc_auc | incremental_tree_effect | -0.0546 | 4/10 | -0.0303 |
| pseudoabsence_hard | boyce | rank_guard_vs_legacy | +0.0014 | 0/10 | -0.0018 |
| pseudoabsence_hard | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |
| pseudoabsence_hard | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/10 | +0.0000 |

## Core comparators (not selected by these results)

| Group | Method | AUC | AP | Shifted AUC | Seconds, both cohorts |
| --- | --- | ---: | ---: | ---: | ---: |
| presence_bg_1to1 | RandomForest | 0.7971 | 0.5991 | 0.7109 | 0.829 |
| presence_bg_1to1 | RF_BalancedUndersampled | 0.7832 | 0.5704 | 0.6992 | 1.425 |
| presence_bg_1to1 | RF_Subsample50 | 0.8010 | 0.6084 | 0.7140 | 1.011 |
| presence_bg_1to1 | XGBoost | 0.7838 | 0.5758 | 0.6981 | 0.232 |
| presence_bg_1to1 | Voting_Heterogeneous | 0.8127 | 0.6528 | 0.7260 | 0.984 |
| presence_bg_1to1 | Stacking_Boosting | 0.8051 | 0.6142 | 0.7163 | 1.620 |
| presence_bg_1to1 | NeuralForest_PUNeural | 0.7843 | 0.6215 | 0.7038 | 0.261 |
| presence_bg_1to1 | NeuralForest_PUStableTeacher | 0.7879 | 0.6249 | 0.7065 | 0.825 |
| presence_bg_1to1 | NeuralForest_PUStable | 0.7879 | 0.6249 | 0.7065 | 0.820 |
| presence_bg_1to1 | NeuralForest_PUSafe | 0.7843 | 0.6215 | 0.7038 | 0.525 |
| presence_bg_1to5 | RandomForest | 0.7738 | 0.5940 | 0.6949 | 1.427 |
| presence_bg_1to5 | RF_BalancedUndersampled | 0.8049 | 0.6028 | 0.7155 | 1.468 |
| presence_bg_1to5 | RF_Subsample50 | 0.7774 | 0.6040 | 0.6985 | 1.455 |
| presence_bg_1to5 | XGBoost | 0.7867 | 0.5747 | 0.7037 | 0.277 |
| presence_bg_1to5 | Voting_Heterogeneous | 0.8106 | 0.6527 | 0.7276 | 1.159 |
| presence_bg_1to5 | Stacking_Boosting | 0.8068 | 0.6259 | 0.7207 | 1.898 |
| presence_bg_1to5 | NeuralForest_PUNeural | 0.7933 | 0.6364 | 0.7121 | 0.761 |
| presence_bg_1to5 | NeuralForest_PUStableTeacher | 0.7950 | 0.6404 | 0.7135 | 2.645 |
| presence_bg_1to5 | NeuralForest_PUStable | 0.7950 | 0.6404 | 0.7135 | 2.639 |
| presence_bg_1to5 | NeuralForest_PUSafe | 0.7933 | 0.6364 | 0.7121 | 1.625 |
| pseudoabsence_hard | RandomForest | 0.5390 | 0.3209 | 0.5259 | 0.876 |
| pseudoabsence_hard | RF_BalancedUndersampled | 0.5376 | 0.3112 | 0.5257 | 1.529 |
| pseudoabsence_hard | RF_Subsample50 | 0.5451 | 0.3261 | 0.5293 | 1.047 |
| pseudoabsence_hard | XGBoost | 0.5146 | 0.2914 | 0.5100 | 0.226 |
| pseudoabsence_hard | Voting_Heterogeneous | 0.6094 | 0.4323 | 0.5681 | 0.981 |
| pseudoabsence_hard | Stacking_Boosting | 0.5188 | 0.3007 | 0.5135 | 1.614 |
| pseudoabsence_hard | NeuralForest_PUNeural | 0.6931 | 0.4944 | 0.6186 | 0.228 |
| pseudoabsence_hard | NeuralForest_PUStableTeacher | 0.7018 | 0.5096 | 0.6202 | 0.906 |
| pseudoabsence_hard | NeuralForest_PUStable | 0.6472 | 0.4507 | 0.5899 | 0.901 |
| pseudoabsence_hard | NeuralForest_PUSafe | 0.6931 | 0.4944 | 0.6186 | 0.477 |

## Important limits

Matched-forest gains can be positive or negative, especially under
unrepresentative background selection and feature-dependent
positive confirmation. APUUC/SCAR assumptions must not be inferred
from these results. The synthetic shifted cohort is generated by
the same simulator, and the real-data partitions come from the
same original datasets; neither is an external clinical site.
Only 3 seeds per real regime and 10 synthetic seeds were used.
Intervals across reused simulation designs are exploratory.
Outcome-based gate tuning on these test results is prohibited.
A separate reserved verified-label audit must be used to make
prospective selection decisions.

For precise scores and phase-level timings see the paired CSV
and the full benchmark all_runs.csv, as well as
pu_baseline_controls.csv.
