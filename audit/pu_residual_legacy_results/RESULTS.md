# PU residual gradient: matched comparison on the pinned synthetic benchmark

This predeclared add-on runs the new C++17 learner and its exact
matched neural teacher on the same 3 presence/background scenarios,
10 seeds, true-label test samples and shifted cohort as the v0.9.2
reference benchmark. Comparators reuse recorded frozen v0.9.2
outcomes, not retrained estimates. Test labels never select models.
The simulator has been inspected repeatedly and is NOT independent
confirmatory evidence. All 60 prediction records reflect 30 new
actual model fits, since the teacher is a shared-fit ablation.

| Scenario | Original nnPU AUC | Stable matched teacher AUC | New teacher AUC | Residual AUC | Incremental AUC | Boyce gain | Shifted AUC gain | Selected rounds/10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| presence_bg_1to1 | 0.7843 | 0.7879 | 0.7879 | 0.7882 | +0.0003 | +0.0010 | +0.0002 | 2/10 |
| presence_bg_1to5 | 0.7933 | 0.7950 | 0.7950 | 0.7950 | +0.0000 | +0.0000 | +0.0000 | 0/10 |
| pseudoabsence_hard | 0.6931 | 0.7018 | 0.7018 | 0.7006 | -0.0012 | -0.0027 | -0.0006 | 8/10 |

## Controls and compute time

| Scenario | Method | AUC | AP | Boyce | Shifted AUC | Total seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| presence_bg_1to1 | CatBoost | 0.8076 | 0.6184 | 0.9909 | 0.7189 | 0.732 |
| presence_bg_1to1 | NeuralForest_PUNeural | 0.7843 | 0.6215 | 0.9865 | 0.7038 | 0.261 |
| presence_bg_1to1 | NeuralForest_PUResidual | 0.7882 | 0.6256 | 0.9885 | 0.7067 | 0.646 |
| presence_bg_1to1 | NeuralForest_PUResidualTeacher | 0.7879 | 0.6249 | 0.9875 | 0.7065 | 0.647 |
| presence_bg_1to1 | NeuralForest_PUSafe | 0.7843 | 0.6215 | 0.9865 | 0.7038 | 0.525 |
| presence_bg_1to1 | NeuralForest_PUStable | 0.7879 | 0.6249 | 0.9875 | 0.7065 | 0.820 |
| presence_bg_1to1 | NeuralForest_PUStableTeacher | 0.7879 | 0.6249 | 0.9875 | 0.7065 | 0.825 |
| presence_bg_1to1 | RF_BalancedUndersampled | 0.7832 | 0.5704 | 0.9897 | 0.6992 | 1.425 |
| presence_bg_1to1 | RF_Subsample50 | 0.8010 | 0.6084 | 0.9911 | 0.7140 | 1.011 |
| presence_bg_1to1 | RandomForest | 0.7971 | 0.5991 | 0.9883 | 0.7109 | 0.829 |
| presence_bg_1to1 | Stacking_Boosting | 0.8051 | 0.6142 | 0.9725 | 0.7163 | 1.620 |
| presence_bg_1to1 | Voting_Heterogeneous | 0.8127 | 0.6528 | 0.9934 | 0.7260 | 0.984 |
| presence_bg_1to1 | XGBoost | 0.7838 | 0.5758 | 0.9732 | 0.6981 | 0.232 |
| presence_bg_1to5 | CatBoost | 0.8082 | 0.6323 | 0.9606 | 0.7221 | 0.745 |
| presence_bg_1to5 | NeuralForest_PUNeural | 0.7933 | 0.6364 | 0.9872 | 0.7121 | 0.761 |
| presence_bg_1to5 | NeuralForest_PUResidual | 0.7950 | 0.6404 | 0.9851 | 0.7135 | 1.864 |
| presence_bg_1to5 | NeuralForest_PUResidualTeacher | 0.7950 | 0.6404 | 0.9851 | 0.7135 | 1.867 |
| presence_bg_1to5 | NeuralForest_PUSafe | 0.7933 | 0.6364 | 0.9881 | 0.7121 | 1.625 |
| presence_bg_1to5 | NeuralForest_PUStable | 0.7950 | 0.6404 | 0.9851 | 0.7135 | 2.639 |
| presence_bg_1to5 | NeuralForest_PUStableTeacher | 0.7950 | 0.6404 | 0.9851 | 0.7135 | 2.645 |
| presence_bg_1to5 | RF_BalancedUndersampled | 0.8049 | 0.6028 | 0.9871 | 0.7155 | 1.468 |
| presence_bg_1to5 | RF_Subsample50 | 0.7774 | 0.6040 | 0.9707 | 0.6985 | 1.455 |
| presence_bg_1to5 | RandomForest | 0.7738 | 0.5940 | 0.9664 | 0.6949 | 1.427 |
| presence_bg_1to5 | Stacking_Boosting | 0.8068 | 0.6259 | 0.9110 | 0.7207 | 1.898 |
| presence_bg_1to5 | Voting_Heterogeneous | 0.8106 | 0.6527 | 0.9735 | 0.7276 | 1.159 |
| presence_bg_1to5 | XGBoost | 0.7867 | 0.5747 | 0.9206 | 0.7037 | 0.277 |
| pseudoabsence_hard | CatBoost | 0.5204 | 0.3002 | 0.1529 | 0.5157 | 0.685 |
| pseudoabsence_hard | NeuralForest_PUNeural | 0.6931 | 0.4944 | 0.9670 | 0.6186 | 0.228 |
| pseudoabsence_hard | NeuralForest_PUResidual | 0.7006 | 0.5086 | 0.9669 | 0.6196 | 0.720 |
| pseudoabsence_hard | NeuralForest_PUResidualTeacher | 0.7018 | 0.5096 | 0.9697 | 0.6202 | 0.720 |
| pseudoabsence_hard | NeuralForest_PUSafe | 0.6931 | 0.4944 | 0.9684 | 0.6186 | 0.477 |
| pseudoabsence_hard | NeuralForest_PUStable | 0.6472 | 0.4507 | 0.7572 | 0.5899 | 0.901 |
| pseudoabsence_hard | NeuralForest_PUStableTeacher | 0.7018 | 0.5096 | 0.9697 | 0.6202 | 0.906 |
| pseudoabsence_hard | RF_BalancedUndersampled | 0.5376 | 0.3112 | 0.0665 | 0.5257 | 1.529 |
| pseudoabsence_hard | RF_Subsample50 | 0.5451 | 0.3261 | 0.2553 | 0.5293 | 1.047 |
| pseudoabsence_hard | RandomForest | 0.5390 | 0.3209 | 0.1966 | 0.5259 | 0.876 |
| pseudoabsence_hard | Stacking_Boosting | 0.5188 | 0.3007 | -0.1136 | 0.5135 | 1.614 |
| pseudoabsence_hard | Voting_Heterogeneous | 0.6094 | 0.4323 | 0.7449 | 0.5681 | 0.981 |
| pseudoabsence_hard | XGBoost | 0.5146 | 0.2914 | 0.1044 | 0.5100 | 0.226 |

## Methodological limitations

The PU-risk gate is only a heuristic under SCAR and predeclared
class priors; SAR, SNAR or hard-selected backgrounds can produce
strongly misleading decisions. The new method does not claim
universally superior discrimination. The same fixed simulator
seeds have been used in earlier iterations; the experiment is
an engineering audit, not an untouched external validation.
Runtime measurements are not directly comparable across jobs
or different GitHub CPU runner vintages. Shared-fit teacher
training time must not be added to the residual learner total.

Full rows: all_runs.csv, comparison.csv, paired_effect.csv.
