# Native PU residual-gradient forest: predeclared controlled challenge

Three synthetic mechanisms: XOR interactions, linear outcome, and
an independent-feature null; SCAR and feature-dependent SAR masks.
Five fixed seeds, 820 PU training examples, 320 verified-label
audit examples and two disjoint 420-example test cohorts.
Verified audit outcomes only gate already fitted models; they do
not train trees or networks, and independent test outcomes do
not select hyperparameters or model families.

| Mechanism | PU confirmation | Teacher AUC | Residual AUC | Verified-gate AUC | Forest selected | Verified gate selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| interaction_xor | SAR | 0.5103 | 0.5126 | 0.5121 | 4/5 | 1/5 |
| interaction_xor | SCAR | 0.5563 | 0.5563 | 0.5563 | 1/5 | 1/5 |
| linear | SAR | 0.7988 | 0.7988 | 0.7988 | 0/5 | 0/5 |
| linear | SCAR | 0.8167 | 0.8169 | 0.8167 | 1/5 | 0/5 |
| null | SAR | 0.4863 | 0.4864 | 0.4863 | 1/5 | 0/5 |
| null | SCAR | 0.4751 | 0.4751 | 0.4751 | 0/5 | 0/5 |

## Held-out differential performance and timing

| Mechanism | Regime | Method | AUC | AP | Brier | Shifted AUC | Total seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| interaction_xor | SAR | CatBoost | 0.8417 | 0.7448 | 0.1859 | 0.8943 | 0.053 |
| interaction_xor | SAR | NeuralForest_PUResidual | 0.5126 | 0.4445 | 0.2737 | 0.5282 | 0.639 |
| interaction_xor | SAR | NeuralForest_PUResidualTeacher | 0.5103 | 0.4425 | 0.2740 | 0.5228 | 0.639 |
| interaction_xor | SAR | NeuralForest_PUResidualVerified | 0.5121 | 0.4435 | 0.2735 | 0.5277 | 0.683 |
| interaction_xor | SAR | RF_Subsample50 | 0.7082 | 0.6501 | 0.2240 | 0.8504 | 0.256 |
| interaction_xor | SAR | RandomForest | 0.7144 | 0.6572 | 0.2218 | 0.8482 | 0.221 |
| interaction_xor | SAR | XGBoost | 0.7753 | 0.7005 | 0.2048 | 0.8843 | 0.028 |
| interaction_xor | SCAR | CatBoost | 0.8796 | 0.7813 | 0.1568 | 0.8961 | 0.165 |
| interaction_xor | SCAR | NeuralForest_PUResidual | 0.5563 | 0.4986 | 0.2504 | 0.5112 | 0.485 |
| interaction_xor | SCAR | NeuralForest_PUResidualTeacher | 0.5563 | 0.4987 | 0.2501 | 0.5104 | 0.485 |
| interaction_xor | SCAR | NeuralForest_PUResidualVerified | 0.5563 | 0.4986 | 0.2504 | 0.5112 | 0.525 |
| interaction_xor | SCAR | RF_Subsample50 | 0.8511 | 0.7452 | 0.1896 | 0.8553 | 0.259 |
| interaction_xor | SCAR | RandomForest | 0.8531 | 0.7416 | 0.1854 | 0.8572 | 0.229 |
| interaction_xor | SCAR | XGBoost | 0.8624 | 0.7557 | 0.1769 | 0.8888 | 0.028 |
| linear | SAR | CatBoost | 0.7839 | 0.5691 | 0.1655 | 0.7958 | 0.053 |
| linear | SAR | NeuralForest_PUResidual | 0.7988 | 0.5893 | 0.1668 | 0.7953 | 0.496 |
| linear | SAR | NeuralForest_PUResidualTeacher | 0.7988 | 0.5893 | 0.1668 | 0.7953 | 0.496 |
| linear | SAR | NeuralForest_PUResidualVerified | 0.7988 | 0.5893 | 0.1668 | 0.7953 | 0.538 |
| linear | SAR | RF_Subsample50 | 0.7659 | 0.5457 | 0.1692 | 0.7825 | 0.246 |
| linear | SAR | RandomForest | 0.7674 | 0.5429 | 0.1690 | 0.7831 | 0.207 |
| linear | SAR | XGBoost | 0.7685 | 0.5438 | 0.1723 | 0.7882 | 0.027 |
| linear | SCAR | CatBoost | 0.8038 | 0.6034 | 0.1617 | 0.7858 | 0.053 |
| linear | SCAR | NeuralForest_PUResidual | 0.8169 | 0.6238 | 0.1571 | 0.8130 | 0.539 |
| linear | SCAR | NeuralForest_PUResidualTeacher | 0.8167 | 0.6232 | 0.1572 | 0.8126 | 0.539 |
| linear | SCAR | NeuralForest_PUResidualVerified | 0.8167 | 0.6232 | 0.1572 | 0.8126 | 0.581 |
| linear | SCAR | RF_Subsample50 | 0.7787 | 0.5682 | 0.1650 | 0.7753 | 0.248 |
| linear | SCAR | RandomForest | 0.7769 | 0.5598 | 0.1651 | 0.7796 | 0.213 |
| linear | SCAR | XGBoost | 0.7856 | 0.5658 | 0.1672 | 0.7782 | 0.027 |
| null | SAR | CatBoost | 0.4950 | 0.2633 | 0.2073 | 0.4902 | 0.052 |
| null | SAR | NeuralForest_PUResidual | 0.4864 | 0.2645 | 0.2199 | 0.4672 | 0.521 |
| null | SAR | NeuralForest_PUResidualTeacher | 0.4863 | 0.2647 | 0.2200 | 0.4671 | 0.521 |
| null | SAR | NeuralForest_PUResidualVerified | 0.4863 | 0.2647 | 0.2200 | 0.4671 | 0.562 |
| null | SAR | RF_Subsample50 | 0.4809 | 0.2583 | 0.2124 | 0.5000 | 0.255 |
| null | SAR | RandomForest | 0.4771 | 0.2546 | 0.2130 | 0.5074 | 0.222 |
| null | SAR | XGBoost | 0.4880 | 0.2563 | 0.2159 | 0.5053 | 0.027 |
| null | SCAR | CatBoost | 0.4735 | 0.2527 | 0.2022 | 0.4932 | 0.052 |
| null | SCAR | NeuralForest_PUResidual | 0.4751 | 0.2516 | 0.1996 | 0.4882 | 0.421 |
| null | SCAR | NeuralForest_PUResidualTeacher | 0.4751 | 0.2516 | 0.1996 | 0.4882 | 0.421 |
| null | SCAR | NeuralForest_PUResidualVerified | 0.4751 | 0.2516 | 0.1996 | 0.4882 | 0.460 |
| null | SCAR | RF_Subsample50 | 0.4539 | 0.2499 | 0.2072 | 0.4950 | 0.266 |
| null | SCAR | RandomForest | 0.4495 | 0.2463 | 0.2084 | 0.4956 | 0.234 |
| null | SCAR | XGBoost | 0.4705 | 0.2566 | 0.2096 | 0.4829 | 0.027 |

## Interpretation restrictions

These are deliberately designed synthetic mechanisms, not an
untouched confirmatory experiment or a field/cohort evaluation.
The verified-label audit uses outcomes unavailable to a typical
presence-only workflow. Results from SCAR cannot justify use
under SAR/SNAR or targeted pseudo-absence sampling.
A selected forest can fail to generalize even when all empirical
PU risk checks pass. Keep the neural fallback and do not select
new hyperparameters using the held-out test observations.
Training times of verified, teacher and residual predictions
SHARE the same initial model fit and must not be added.
Per-seed data and paired effects: all_runs.csv, paired_effect.csv.
