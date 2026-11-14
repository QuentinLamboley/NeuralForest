# NNPU residual-tree pilot, only exploratory

The two public data tasks have been inspected in earlier work;
fresh random seeds do NOT make them externally independent.
SCAR or feature-dependent SAR masks true training positives.
No test label was used to train, tune or select a model.
The matched teacher shares the exact residual-model training.
Its training time should not be double-counted.

| Source | Regime | Residual AUC | Teacher AUC | Delta | Selected / runs |
| --- | --- | ---: | ---: | ---: | --- |
| Diabetes_Progression180 | SAR | 0.8246 | 0.8246 | +0.0000 | neural_fallback, neural_fallback |
| Diabetes_Progression180 | SCAR | 0.8322 | 0.8322 | -0.0000 | neural_fallback, residual_4_1.00 |
| Digits_Eight | SAR | 0.9620 | 0.9620 | +0.0000 | neural_fallback, neural_fallback |
| Digits_Eight | SCAR | 0.9806 | 0.9806 | +0.0000 | neural_fallback, neural_fallback |

## Baseline means and runtime

| Source | Regime | Method | AUC | AP | Shifted AUC | Total seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Diabetes_Progression180 | SAR | CatBoost | 0.8132 | 0.7040 | 0.7881 | 0.050 |
| Diabetes_Progression180 | SAR | NeuralForest_PUResidual | 0.8246 | 0.7238 | 0.7948 | 0.212 |
| Diabetes_Progression180 | SAR | NeuralForest_PUResidualTeacher | 0.8246 | 0.7238 | 0.7948 | 0.212 |
| Diabetes_Progression180 | SAR | RF_Subsample50 | 0.8110 | 0.7225 | 0.7695 | 0.192 |
| Diabetes_Progression180 | SAR | RandomForest | 0.8005 | 0.6909 | 0.7686 | 0.143 |
| Diabetes_Progression180 | SAR | XGBoost | 0.7937 | 0.6630 | 0.7728 | 0.023 |
| Diabetes_Progression180 | SCAR | CatBoost | 0.8220 | 0.7347 | 0.8192 | 0.080 |
| Diabetes_Progression180 | SCAR | NeuralForest_PUResidual | 0.8322 | 0.7489 | 0.8273 | 0.252 |
| Diabetes_Progression180 | SCAR | NeuralForest_PUResidualTeacher | 0.8322 | 0.7489 | 0.8265 | 0.251 |
| Diabetes_Progression180 | SCAR | RF_Subsample50 | 0.8291 | 0.7452 | 0.8176 | 0.196 |
| Diabetes_Progression180 | SCAR | RandomForest | 0.8124 | 0.7134 | 0.8231 | 0.150 |
| Diabetes_Progression180 | SCAR | XGBoost | 0.7923 | 0.6718 | 0.7878 | 0.025 |
| Digits_Eight | SAR | CatBoost | 0.9638 | 0.8650 | 0.9368 | 0.104 |
| Digits_Eight | SAR | NeuralForest_PUResidual | 0.9620 | 0.8554 | 0.9425 | 2.058 |
| Digits_Eight | SAR | NeuralForest_PUResidualTeacher | 0.9620 | 0.8554 | 0.9425 | 2.054 |
| Digits_Eight | SAR | RF_Subsample50 | 0.9723 | 0.8980 | 0.9542 | 0.261 |
| Digits_Eight | SAR | RandomForest | 0.9755 | 0.9006 | 0.9611 | 0.210 |
| Digits_Eight | SAR | XGBoost | 0.9420 | 0.8338 | 0.9220 | 0.053 |
| Digits_Eight | SCAR | CatBoost | 0.9862 | 0.9283 | 0.9914 | 0.114 |
| Digits_Eight | SCAR | NeuralForest_PUResidual | 0.9806 | 0.8995 | 0.9675 | 2.125 |
| Digits_Eight | SCAR | NeuralForest_PUResidualTeacher | 0.9806 | 0.8995 | 0.9675 | 2.122 |
| Digits_Eight | SCAR | RF_Subsample50 | 0.9842 | 0.8994 | 0.9863 | 0.264 |
| Digits_Eight | SCAR | RandomForest | 0.9888 | 0.9242 | 0.9891 | 0.211 |
| Digits_Eight | SCAR | XGBoost | 0.9794 | 0.8981 | 0.9872 | 0.054 |

## Caveats

The C++ gradient approximates nonnegative PU log-loss, assuming
representative positives/background and a predeclared prior grid.
SAR/SNAR and selected pseudo-absence populations violate this.
Even when the PU-risk selection gate accepts trees, external
discrimination and calibration may worsen. The pilot has only
two seeds per scenario and is NOT an efficacy demonstration.
Gradient-boosted trees are not necessarily superior to neural-only
or conventional RF/boosting; the purpose is to find model failures.
Full data and exact timing: all_runs.csv; paired_effect.csv.
