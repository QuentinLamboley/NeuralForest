# PU guided forest: paired teacher attribution and transport audit

Evaluated commit: 4623083d07a7485a1b232c9e571f0d39cda3a35f.
Domain: real. Both evaluation cohorts are test-only.
No test outcomes were used to fit, select or calibrate any model.

**Crucial control:** PUStableTeacher is the exact fitted two-restart
neural component of PUStable. It shares training and search time,
so its total time is NOT an independent neural-only compute cost.
Incremental-tree comparisons use identical networks and encoders.
The cheaper PUNeural reference has only one neural restart.

## Matching teacher, tree contribution, and transport

| Group | Test metric | Comparison | Mean gain | Losses / seeds | Shifted gain |
| --- | --- | --- | ---: | ---: | ---: |
| Digits_Eight/SAR | boyce | combined_effect | -0.0000 | 1/3 | -0.1286 |
| Digits_Eight/SAR | pr_auc | combined_effect | +0.0111 | 0/3 | +0.0103 |
| Digits_Eight/SAR | roc_auc | combined_effect | +0.0023 | 1/3 | +0.0006 |
| Digits_Eight/SAR | boyce | extra_teacher_restarts | -0.0000 | 1/3 | -0.1286 |
| Digits_Eight/SAR | pr_auc | extra_teacher_restarts | +0.0111 | 0/3 | +0.0103 |
| Digits_Eight/SAR | roc_auc | extra_teacher_restarts | +0.0023 | 1/3 | +0.0006 |
| Digits_Eight/SAR | boyce | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SAR | pr_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SAR | roc_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SAR | boyce | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SAR | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SAR | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SCAR | boyce | combined_effect | -0.3435 | 2/3 | +0.1594 |
| Digits_Eight/SCAR | pr_auc | combined_effect | +0.0075 | 1/3 | +0.0102 |
| Digits_Eight/SCAR | roc_auc | combined_effect | +0.0042 | 0/3 | +0.0012 |
| Digits_Eight/SCAR | boyce | extra_teacher_restarts | -0.3435 | 2/3 | +0.1594 |
| Digits_Eight/SCAR | pr_auc | extra_teacher_restarts | +0.0075 | 1/3 | +0.0102 |
| Digits_Eight/SCAR | roc_auc | extra_teacher_restarts | +0.0042 | 0/3 | +0.0012 |
| Digits_Eight/SCAR | boyce | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SCAR | pr_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SCAR | roc_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SCAR | boyce | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SCAR | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Digits_Eight/SCAR | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SAR | boyce | combined_effect | +nan | 0/0 | +nan |
| Wine_Class0/SAR | pr_auc | combined_effect | +0.0043 | 0/3 | +0.0053 |
| Wine_Class0/SAR | roc_auc | combined_effect | +0.0023 | 0/3 | +0.0035 |
| Wine_Class0/SAR | boyce | extra_teacher_restarts | +nan | 0/0 | +nan |
| Wine_Class0/SAR | pr_auc | extra_teacher_restarts | +0.0043 | 0/3 | +0.0053 |
| Wine_Class0/SAR | roc_auc | extra_teacher_restarts | +0.0023 | 0/3 | +0.0035 |
| Wine_Class0/SAR | boyce | incremental_tree_effect | +nan | 0/0 | +0.0000 |
| Wine_Class0/SAR | pr_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SAR | roc_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SAR | boyce | rank_guard_vs_legacy | +0.0000 | 0/1 | +nan |
| Wine_Class0/SAR | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SAR | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SCAR | boyce | combined_effect | +nan | 0/0 | +0.0000 |
| Wine_Class0/SCAR | pr_auc | combined_effect | +0.0021 | 0/3 | +0.0096 |
| Wine_Class0/SCAR | roc_auc | combined_effect | +0.0012 | 0/3 | +0.0069 |
| Wine_Class0/SCAR | boyce | extra_teacher_restarts | +nan | 0/0 | +0.0000 |
| Wine_Class0/SCAR | pr_auc | extra_teacher_restarts | +0.0021 | 0/3 | +0.0096 |
| Wine_Class0/SCAR | roc_auc | extra_teacher_restarts | +0.0012 | 0/3 | +0.0069 |
| Wine_Class0/SCAR | boyce | incremental_tree_effect | +nan | 0/0 | +0.0000 |
| Wine_Class0/SCAR | pr_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SCAR | roc_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SCAR | boyce | rank_guard_vs_legacy | +0.0000 | 0/1 | +0.0000 |
| Wine_Class0/SCAR | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wine_Class0/SCAR | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SAR | boyce | combined_effect | -0.0543 | 1/3 | +0.0200 |
| Wisconsin_Breast_Cancer/SAR | pr_auc | combined_effect | -0.0003 | 1/3 | +0.0002 |
| Wisconsin_Breast_Cancer/SAR | roc_auc | combined_effect | +0.0000 | 1/3 | +0.0001 |
| Wisconsin_Breast_Cancer/SAR | boyce | extra_teacher_restarts | -0.0543 | 1/3 | +0.0200 |
| Wisconsin_Breast_Cancer/SAR | pr_auc | extra_teacher_restarts | -0.0003 | 1/3 | +0.0002 |
| Wisconsin_Breast_Cancer/SAR | roc_auc | extra_teacher_restarts | +0.0000 | 1/3 | +0.0001 |
| Wisconsin_Breast_Cancer/SAR | boyce | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SAR | pr_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SAR | roc_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SAR | boyce | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SAR | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SAR | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SCAR | boyce | combined_effect | -0.1661 | 2/3 | +0.0654 |
| Wisconsin_Breast_Cancer/SCAR | pr_auc | combined_effect | +0.0004 | 1/3 | -0.0000 |
| Wisconsin_Breast_Cancer/SCAR | roc_auc | combined_effect | +0.0001 | 1/3 | -0.0000 |
| Wisconsin_Breast_Cancer/SCAR | boyce | extra_teacher_restarts | -0.1661 | 2/3 | +0.0654 |
| Wisconsin_Breast_Cancer/SCAR | pr_auc | extra_teacher_restarts | +0.0004 | 1/3 | -0.0000 |
| Wisconsin_Breast_Cancer/SCAR | roc_auc | extra_teacher_restarts | +0.0001 | 1/3 | -0.0000 |
| Wisconsin_Breast_Cancer/SCAR | boyce | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SCAR | pr_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SCAR | roc_auc | incremental_tree_effect | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SCAR | boyce | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SCAR | pr_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |
| Wisconsin_Breast_Cancer/SCAR | roc_auc | rank_guard_vs_legacy | +0.0000 | 0/3 | +0.0000 |

## Core comparators (not selected by these results)

| Group | Method | AUC | AP | Shifted AUC | Seconds, both cohorts |
| --- | --- | ---: | ---: | ---: | ---: |
| Digits_Eight/SAR | RandomForest | 0.9689 | 0.8466 | 0.9811 | 0.482 |
| Digits_Eight/SAR | RF_BalancedUndersampled | 0.9583 | 0.8457 | 0.9822 | 0.556 |
| Digits_Eight/SAR | RF_Subsample50 | 0.9653 | 0.8460 | 0.9833 | 0.569 |
| Digits_Eight/SAR | XGBoost | 0.9457 | 0.8281 | 0.9747 | 0.119 |
| Digits_Eight/SAR | Voting_Heterogeneous | 0.9678 | 0.8507 | 0.9850 | 0.556 |
| Digits_Eight/SAR | Stacking_Boosting | 0.9602 | 0.8621 | 0.9797 | 0.960 |
| Digits_Eight/SAR | NeuralForest_PUNeural | 0.9495 | 0.8084 | 0.9771 | 0.799 |
| Digits_Eight/SAR | NeuralForest_PUStableTeacher | 0.9518 | 0.8195 | 0.9778 | 2.531 |
| Digits_Eight/SAR | NeuralForest_PUStable | 0.9518 | 0.8195 | 0.9778 | 2.532 |
| Digits_Eight/SAR | NeuralForest_PUSafe | 0.9495 | 0.8084 | 0.9771 | 1.410 |
| Digits_Eight/SCAR | RandomForest | 0.9903 | 0.9326 | 0.9938 | 0.488 |
| Digits_Eight/SCAR | RF_BalancedUndersampled | 0.9831 | 0.9278 | 0.9906 | 0.580 |
| Digits_Eight/SCAR | RF_Subsample50 | 0.9907 | 0.9342 | 0.9940 | 0.608 |
| Digits_Eight/SCAR | XGBoost | 0.9891 | 0.9381 | 0.9922 | 0.122 |
| Digits_Eight/SCAR | Voting_Heterogeneous | 0.9917 | 0.9395 | 0.9925 | 0.534 |
| Digits_Eight/SCAR | Stacking_Boosting | 0.9924 | 0.9558 | 0.9944 | 0.957 |
| Digits_Eight/SCAR | NeuralForest_PUNeural | 0.9692 | 0.8831 | 0.9877 | 0.839 |
| Digits_Eight/SCAR | NeuralForest_PUStableTeacher | 0.9733 | 0.8906 | 0.9889 | 2.632 |
| Digits_Eight/SCAR | NeuralForest_PUStable | 0.9733 | 0.8906 | 0.9889 | 2.634 |
| Digits_Eight/SCAR | NeuralForest_PUSafe | 0.9692 | 0.8831 | 0.9877 | 1.439 |
| Wine_Class0/SAR | RandomForest | 0.9896 | 0.9815 | 0.9954 | 0.279 |
| Wine_Class0/SAR | RF_BalancedUndersampled | 0.9965 | 0.9930 | 0.9994 | 0.460 |
| Wine_Class0/SAR | RF_Subsample50 | 0.9931 | 0.9877 | 0.9942 | 0.382 |
| Wine_Class0/SAR | XGBoost | 0.9641 | 0.9544 | 0.9919 | 0.053 |
| Wine_Class0/SAR | Voting_Heterogeneous | 0.9954 | 0.9913 | 0.9988 | 0.294 |
| Wine_Class0/SAR | Stacking_Boosting | 0.9873 | 0.9767 | 0.9896 | 0.377 |
| Wine_Class0/SAR | NeuralForest_PUNeural | 0.9977 | 0.9957 | 0.9931 | 0.036 |
| Wine_Class0/SAR | NeuralForest_PUStableTeacher | 1.0000 | 1.0000 | 0.9965 | 0.067 |
| Wine_Class0/SAR | NeuralForest_PUStable | 1.0000 | 1.0000 | 0.9965 | 0.066 |
| Wine_Class0/SAR | NeuralForest_PUSafe | 0.9977 | 0.9957 | 0.9931 | 0.033 |
| Wine_Class0/SCAR | RandomForest | 0.9884 | 0.9665 | 0.9942 | 0.288 |
| Wine_Class0/SCAR | RF_BalancedUndersampled | 0.9948 | 0.9872 | 1.0000 | 0.464 |
| Wine_Class0/SCAR | RF_Subsample50 | 0.9838 | 0.9625 | 0.9878 | 0.389 |
| Wine_Class0/SCAR | XGBoost | 0.9815 | 0.9321 | 0.9873 | 0.061 |
| Wine_Class0/SCAR | Voting_Heterogeneous | 0.9942 | 0.9891 | 0.9965 | 0.301 |
| Wine_Class0/SCAR | Stacking_Boosting | 0.9850 | 0.9570 | 0.9861 | 0.410 |
| Wine_Class0/SCAR | NeuralForest_PUNeural | 0.9977 | 0.9957 | 0.9873 | 0.040 |
| Wine_Class0/SCAR | NeuralForest_PUStableTeacher | 0.9988 | 0.9979 | 0.9942 | 0.067 |
| Wine_Class0/SCAR | NeuralForest_PUStable | 0.9988 | 0.9979 | 0.9942 | 0.066 |
| Wine_Class0/SCAR | NeuralForest_PUSafe | 0.9977 | 0.9957 | 0.9873 | 0.036 |
| Wisconsin_Breast_Cancer/SAR | RandomForest | 0.9839 | 0.9820 | 0.9905 | 0.370 |
| Wisconsin_Breast_Cancer/SAR | RF_BalancedUndersampled | 0.9870 | 0.9846 | 0.9919 | 0.546 |
| Wisconsin_Breast_Cancer/SAR | RF_Subsample50 | 0.9885 | 0.9854 | 0.9916 | 0.457 |
| Wisconsin_Breast_Cancer/SAR | XGBoost | 0.9876 | 0.9869 | 0.9901 | 0.172 |
| Wisconsin_Breast_Cancer/SAR | Voting_Heterogeneous | 0.9912 | 0.9885 | 0.9966 | 1.038 |
| Wisconsin_Breast_Cancer/SAR | Stacking_Boosting | 0.9865 | 0.9852 | 0.9946 | 2.115 |
| Wisconsin_Breast_Cancer/SAR | NeuralForest_PUNeural | 0.9922 | 0.9909 | 0.9976 | 0.190 |
| Wisconsin_Breast_Cancer/SAR | NeuralForest_PUStableTeacher | 0.9922 | 0.9907 | 0.9977 | 0.583 |
| Wisconsin_Breast_Cancer/SAR | NeuralForest_PUStable | 0.9922 | 0.9907 | 0.9977 | 0.582 |
| Wisconsin_Breast_Cancer/SAR | NeuralForest_PUSafe | 0.9922 | 0.9909 | 0.9976 | 0.339 |
| Wisconsin_Breast_Cancer/SCAR | RandomForest | 0.9876 | 0.9864 | 0.9955 | 0.395 |
| Wisconsin_Breast_Cancer/SCAR | RF_BalancedUndersampled | 0.9877 | 0.9853 | 0.9943 | 0.559 |
| Wisconsin_Breast_Cancer/SCAR | RF_Subsample50 | 0.9890 | 0.9861 | 0.9942 | 0.459 |
| Wisconsin_Breast_Cancer/SCAR | XGBoost | 0.9853 | 0.9828 | 0.9921 | 0.231 |
| Wisconsin_Breast_Cancer/SCAR | Voting_Heterogeneous | 0.9908 | 0.9892 | 0.9967 | 1.433 |
| Wisconsin_Breast_Cancer/SCAR | Stacking_Boosting | 0.9886 | 0.9870 | 0.9908 | 2.108 |
| Wisconsin_Breast_Cancer/SCAR | NeuralForest_PUNeural | 0.9909 | 0.9908 | 0.9978 | 0.187 |
| Wisconsin_Breast_Cancer/SCAR | NeuralForest_PUStableTeacher | 0.9910 | 0.9912 | 0.9978 | 0.603 |
| Wisconsin_Breast_Cancer/SCAR | NeuralForest_PUStable | 0.9910 | 0.9912 | 0.9978 | 0.602 |
| Wisconsin_Breast_Cancer/SCAR | NeuralForest_PUSafe | 0.9909 | 0.9908 | 0.9978 | 0.351 |

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
