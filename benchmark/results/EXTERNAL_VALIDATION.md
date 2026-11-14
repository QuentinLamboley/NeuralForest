# Independent shifted-cohort validation: synthetic transport

Every fitted model also predicted a second **independent** cohort
from the same pinned synthetic generator with a prespecified
proximity change (usually 12 percentage points). Training,
threshold selection, prior assumptions and hyperparameter search
used neither this cohort nor its true labels. This is NOT
validation on an independently collected real-world dataset.

The main analysis includes 6040 successful prediction records and
records all 16 metrics on both independent test populations.

| Scenario | Metric | Best shifted-cohort model | Shifted score | NeuralForest Adaptive |
| --- | --- | --- | ---: | ---: |
| correlated_30 | auc | Voting_Heterogeneous | 0.7046 | 0.6976 |
| distinct_0 | auc | ExtraTrees | 0.9804 | 0.9747 |
| distribution_shift | auc | Voting_Heterogeneous | 0.5882 | 0.5842 |
| identical_100 | logloss | Dummy | 0.6936 | 0.6971 |
| intermediate_45 | auc | Voting_Heterogeneous | 0.8398 | 0.8337 |
| label_noise_20pct | auc | Voting_Heterogeneous | 0.6649 | 0.6616 |
| large_n | auc | CatBoost | 0.6652 | 0.6617 |
| missing_15pct | auc | Voting_Heterogeneous | 0.6844 | 0.6790 |
| noise_50 | auc | CatBoost | 0.6940 | 0.6873 |
| nonlinear_xor | f1 | GradientBoosting | 0.9967 | 0.9833 |
| overlap_80 | auc | Voting_Heterogeneous | 0.5756 | 0.5711 |
| presence_bg_1to1 | boyce | ExtraTrees | 0.9882 | 0.9869 |
| presence_bg_1to5 | boyce | NeuralForest_UndersampledInformed | 0.9852 | 0.9655 |
| pseudoabsence_hard | boyce | Logistic | 0.9522 | 0.5869 |
| rare_10pct | pr_auc | Voting_Heterogeneous | 0.3414 | 0.3269 |
| rare_1pct | pr_auc | ExtraTrees | 0.1170 | 0.1071 |
| rare_3pct | pr_auc | Voting_Heterogeneous | 0.1768 | 0.1588 |
| similar_97 | brier | Logistic | 0.2485 | 0.2503 |
| small_n | balanced_accuracy | CatBoost | 0.5982 | 0.5965 |

## Positive-unlabeled variants

These methods are run only in the three presence/background
scenarios; they must not be credited with winning the 19-scenario
overall rank or compared by overall mean compute time.

| Scenario | Model | Main test metric | Shifted-cohort metric | Mean total time (s) |
| --- | --- | ---: | ---: | ---: |
| presence_bg_1to1 | Logistic | 0.9948 | 0.9836 | 0.053 |
| presence_bg_1to1 | NeuralForest_Adaptive | 0.9926 | 0.9869 | 0.343 |
| presence_bg_1to1 | NeuralForest_PUInformed | 0.9865 | 0.9721 | 0.406 |
| presence_bg_1to1 | NeuralForest_PUNeural | 0.9865 | 0.9721 | 0.261 |
| presence_bg_1to1 | NeuralForest_PUSafe | 0.9865 | 0.9721 | 0.525 |
| presence_bg_1to1 | NeuralForest_PUStable | 0.9875 | 0.9669 | 0.820 |
| presence_bg_1to1 | NeuralForest_PUStableTeacher | 0.9875 | 0.9669 | 0.825 |
| presence_bg_1to1 | Voting_Heterogeneous | 0.9934 | 0.9826 | 0.984 |
| presence_bg_1to5 | Logistic | 0.9627 | 0.9345 | 0.057 |
| presence_bg_1to5 | NeuralForest_Adaptive | 0.9863 | 0.9655 | 0.897 |
| presence_bg_1to5 | NeuralForest_PUInformed | 0.9899 | 0.9809 | 1.202 |
| presence_bg_1to5 | NeuralForest_PUNeural | 0.9872 | 0.9792 | 0.761 |
| presence_bg_1to5 | NeuralForest_PUSafe | 0.9881 | 0.9808 | 1.625 |
| presence_bg_1to5 | NeuralForest_PUStable | 0.9851 | 0.9786 | 2.639 |
| presence_bg_1to5 | NeuralForest_PUStableTeacher | 0.9851 | 0.9786 | 2.645 |
| presence_bg_1to5 | Voting_Heterogeneous | 0.9735 | 0.9559 | 1.159 |
| pseudoabsence_hard | Logistic | 0.9898 | 0.9522 | 0.052 |
| pseudoabsence_hard | NeuralForest_Adaptive | 0.7211 | 0.5869 | 0.340 |
| pseudoabsence_hard | NeuralForest_PUInformed | 0.6202 | 0.5968 | 0.370 |
| pseudoabsence_hard | NeuralForest_PUNeural | 0.9670 | 0.9218 | 0.228 |
| pseudoabsence_hard | NeuralForest_PUSafe | 0.9684 | 0.9200 | 0.477 |
| pseudoabsence_hard | NeuralForest_PUStable | 0.7572 | 0.7362 | 0.901 |
| pseudoabsence_hard | NeuralForest_PUStableTeacher | 0.9697 | 0.9060 | 0.906 |
| pseudoabsence_hard | Voting_Heterogeneous | 0.7449 | 0.6469 | 0.981 |

The native nnPU objective assumes representative observed-positive
and unlabeled distributions and prespecified class-prior values.
The hard-background selection intentionally violates the
representative-unlabeled assumption. An apparent improvement on
this simulated population does not validate those assumptions.
Repeated inspection of these 19 scenario families has also
reduced their value as confirmatory external evidence.

Source measurements: all_runs.csv and external_validation.csv.
