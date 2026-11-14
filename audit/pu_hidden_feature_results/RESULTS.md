# Native nnPU hidden-feature-guided forest, paired ablation

Predeclared training-only comparisons: original numeric features
versus identical original features plus frozen C++ hidden nnPU
activations. The native neural teachers are tested for EXACT
agreement across raw and augmented fits for every seed.
A size-matched untrained random-tanh projection controls for
the mere addition of nonlinear feature dimensions.
Sole ablated factors: frozen learned vs random vs original
features and their constrained feature-sampling mass.
All other native tree settings
and the per-seed training P/U indicators are held fixed.

The candidate scores shown below were NOT selected on true labels.
They are exploratory research ablations, not deployable predictors.
Guarded versions use a distinct verified-outcome cohort to
choose candidate vs the matching neural teacher, on two halves.
An entirely separate test and shifted test remain untouched.
The supervised controls classify observed P/U indicators and
are not assumed to estimate true positive status under SAR.

| Mechanism | PU labels | Teacher AUC | Raw AUC | Random-tanh AUC | Learned-tanh AUC | Learned minus random | Learned minus raw | Verified learned AUC | Selections |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interaction_xor | SAR | 0.5074 | 0.6043 | 0.6349 | 0.6226 | -0.0124 | +0.0182 | 0.6226 | 10/10 |
| interaction_xor | SCAR | 0.5468 | 0.7591 | 0.7906 | 0.8053 | +0.0147 | +0.0462 | 0.8053 | 10/10 |
| linear | SAR | 0.7746 | 0.7635 | 0.7653 | 0.7644 | -0.0009 | +0.0009 | 0.7746 | 0/10 |
| linear | SCAR | 0.8138 | 0.8047 | 0.8063 | 0.8088 | +0.0025 | +0.0041 | 0.8138 | 0/10 |
| no_signal | SAR | 0.5029 | 0.5026 | 0.4993 | 0.5010 | +0.0017 | -0.0016 | 0.5023 | 1/10 |
| no_signal | SCAR | 0.4923 | 0.4928 | 0.4896 | 0.4874 | -0.0022 | -0.0054 | 0.4923 | 0/10 |

## Exploratory paired AUC intervals: learned against controls

Two-sided 95% Student-t intervals use synthetic seed variation only.
They are NOT corrected for multiple comparisons, nor externally
validated confidence intervals for unknown ecological settings.

| Mechanism | Regime | Cohort | Contrast | Mean AUC delta | 95% interval | Wins / seeds |
| --- | --- | --- | --- | ---: | --- | ---: |
| interaction_xor | SAR | shift | learned_hidden_vs_random_tanh | -0.0226 | [-0.0491, +0.0039] | 2/10 |
| interaction_xor | SAR | test | learned_hidden_vs_random_tanh | -0.0124 | [-0.0246, -0.0002] | 3/10 |
| interaction_xor | SAR | shift | learned_hidden_vs_raw | +0.0062 | [-0.0294, +0.0417] | 3/10 |
| interaction_xor | SAR | test | learned_hidden_vs_raw | +0.0182 | [+0.0016, +0.0349] | 7/10 |
| interaction_xor | SCAR | shift | learned_hidden_vs_random_tanh | +0.0220 | [+0.0005, +0.0434] | 7/10 |
| interaction_xor | SCAR | test | learned_hidden_vs_random_tanh | +0.0147 | [+0.0015, +0.0279] | 9/10 |
| interaction_xor | SCAR | shift | learned_hidden_vs_raw | +0.0358 | [+0.0148, +0.0569] | 9/10 |
| interaction_xor | SCAR | test | learned_hidden_vs_raw | +0.0462 | [+0.0259, +0.0666] | 10/10 |
| linear | SAR | shift | learned_hidden_vs_random_tanh | -0.0017 | [-0.0089, +0.0055] | 6/10 |
| linear | SAR | test | learned_hidden_vs_random_tanh | -0.0009 | [-0.0044, +0.0027] | 4/10 |
| linear | SAR | shift | learned_hidden_vs_raw | -0.0069 | [-0.0133, -0.0005] | 1/10 |
| linear | SAR | test | learned_hidden_vs_raw | +0.0009 | [-0.0029, +0.0048] | 4/10 |
| linear | SCAR | shift | learned_hidden_vs_random_tanh | +0.0001 | [-0.0050, +0.0052] | 6/10 |
| linear | SCAR | test | learned_hidden_vs_random_tanh | +0.0025 | [-0.0003, +0.0053] | 6/10 |
| linear | SCAR | shift | learned_hidden_vs_raw | +0.0039 | [-0.0015, +0.0092] | 6/10 |
| linear | SCAR | test | learned_hidden_vs_raw | +0.0041 | [+0.0016, +0.0066] | 9/10 |
| no_signal | SAR | shift | learned_hidden_vs_random_tanh | +0.0000 | [-0.0125, +0.0125] | 5/10 |
| no_signal | SAR | test | learned_hidden_vs_random_tanh | +0.0017 | [-0.0008, +0.0041] | 8/10 |
| no_signal | SAR | shift | learned_hidden_vs_raw | -0.0032 | [-0.0089, +0.0025] | 4/10 |
| no_signal | SAR | test | learned_hidden_vs_raw | -0.0016 | [-0.0038, +0.0006] | 3/10 |
| no_signal | SCAR | shift | learned_hidden_vs_random_tanh | +0.0016 | [-0.0107, +0.0139] | 4/10 |
| no_signal | SCAR | test | learned_hidden_vs_random_tanh | -0.0022 | [-0.0096, +0.0052] | 3/10 |
| no_signal | SCAR | shift | learned_hidden_vs_raw | -0.0013 | [-0.0111, +0.0085] | 4/10 |
| no_signal | SCAR | test | learned_hidden_vs_raw | -0.0054 | [-0.0131, +0.0022] | 3/10 |

## All controls, multi-metric tests and time

| Mechanism | Regime | Method | AUC | PR-AUC | Brier | Shift AUC | Train s | Predict s | Hidden importance share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interaction_xor | SAR | CatBoost | 0.8590 | 0.7526 | 0.1796 | 0.8893 | 0.094 | 0.001 | nan |
| interaction_xor | SAR | ExtraTrees | 0.6094 | 0.5408 | 0.2476 | 0.6862 | 0.105 | 0.018 | nan |
| interaction_xor | SAR | NeuralForest_PUHiddenLatentCandidate | 0.6226 | 0.5725 | 0.2329 | 0.7559 | 0.246 | 0.014 | 0.379 |
| interaction_xor | SAR | NeuralForest_PUHiddenLatentVerified | 0.6226 | 0.5725 | 0.2329 | 0.7559 | 0.279 | 0.014 | 0.379 |
| interaction_xor | SAR | NeuralForest_PUHiddenRandomCandidate | 0.6349 | 0.5877 | 0.2291 | 0.7785 | 0.244 | 0.012 | 0.336 |
| interaction_xor | SAR | NeuralForest_PUHiddenRandomVerified | 0.6349 | 0.5877 | 0.2291 | 0.7785 | 0.276 | 0.014 | 0.336 |
| interaction_xor | SAR | NeuralForest_PUHiddenRawCandidate | 0.6043 | 0.5527 | 0.2367 | 0.7497 | 0.219 | 0.012 | 0.000 |
| interaction_xor | SAR | NeuralForest_PUHiddenRawVerified | 0.6043 | 0.5527 | 0.2367 | 0.7497 | 0.255 | 0.012 | 0.000 |
| interaction_xor | SAR | NeuralForest_PUHiddenTeacher | 0.5074 | 0.4204 | 0.2632 | 0.5312 | 0.219 | 0.007 | nan |
| interaction_xor | SAR | RandomForest | 0.6627 | 0.6137 | 0.2387 | 0.7992 | 0.200 | 0.019 | nan |
| interaction_xor | SAR | XGBoost | 0.7964 | 0.7081 | 0.1963 | 0.8813 | 0.043 | 0.003 | nan |
| interaction_xor | SCAR | CatBoost | 0.8956 | 0.7840 | 0.1502 | 0.8872 | 0.101 | 0.001 | nan |
| interaction_xor | SCAR | ExtraTrees | 0.7566 | 0.6561 | 0.2425 | 0.7002 | 0.106 | 0.018 | nan |
| interaction_xor | SCAR | NeuralForest_PUHiddenLatentCandidate | 0.8053 | 0.7034 | 0.2267 | 0.7566 | 0.226 | 0.013 | 0.399 |
| interaction_xor | SCAR | NeuralForest_PUHiddenLatentVerified | 0.8053 | 0.7034 | 0.2267 | 0.7566 | 0.257 | 0.014 | 0.399 |
| interaction_xor | SCAR | NeuralForest_PUHiddenRandomCandidate | 0.7906 | 0.6881 | 0.2286 | 0.7347 | 0.226 | 0.014 | 0.368 |
| interaction_xor | SCAR | NeuralForest_PUHiddenRandomVerified | 0.7906 | 0.6881 | 0.2286 | 0.7347 | 0.255 | 0.012 | 0.368 |
| interaction_xor | SCAR | NeuralForest_PUHiddenRawCandidate | 0.7591 | 0.6531 | 0.2315 | 0.7208 | 0.204 | 0.011 | 0.000 |
| interaction_xor | SCAR | NeuralForest_PUHiddenRawVerified | 0.7591 | 0.6531 | 0.2315 | 0.7208 | 0.238 | 0.011 | 0.000 |
| interaction_xor | SCAR | NeuralForest_PUHiddenTeacher | 0.5468 | 0.4700 | 0.2409 | 0.5332 | 0.204 | 0.005 | nan |
| interaction_xor | SCAR | RandomForest | 0.8318 | 0.7124 | 0.2176 | 0.7673 | 0.203 | 0.019 | nan |
| interaction_xor | SCAR | XGBoost | 0.8858 | 0.7724 | 0.1683 | 0.8606 | 0.043 | 0.003 | nan |
| linear | SAR | CatBoost | 0.7448 | 0.5106 | 0.1635 | 0.8094 | 0.092 | 0.001 | nan |
| linear | SAR | ExtraTrees | 0.7807 | 0.5524 | 0.1677 | 0.8187 | 0.103 | 0.018 | nan |
| linear | SAR | NeuralForest_PUHiddenLatentCandidate | 0.7644 | 0.5323 | 0.1749 | 0.8037 | 0.241 | 0.014 | 0.592 |
| linear | SAR | NeuralForest_PUHiddenLatentVerified | 0.7746 | 0.5364 | 0.1743 | 0.7986 | 0.272 | 0.005 | nan |
| linear | SAR | NeuralForest_PUHiddenRandomCandidate | 0.7653 | 0.5322 | 0.1745 | 0.8054 | 0.238 | 0.012 | 0.391 |
| linear | SAR | NeuralForest_PUHiddenRandomVerified | 0.7761 | 0.5376 | 0.1733 | 0.8012 | 0.271 | 0.007 | 0.414 |
| linear | SAR | NeuralForest_PUHiddenRawCandidate | 0.7635 | 0.5310 | 0.1734 | 0.8105 | 0.218 | 0.011 | 0.000 |
| linear | SAR | NeuralForest_PUHiddenRawVerified | 0.7746 | 0.5364 | 0.1743 | 0.7986 | 0.253 | 0.007 | nan |
| linear | SAR | NeuralForest_PUHiddenTeacher | 0.7746 | 0.5364 | 0.1743 | 0.7986 | 0.218 | 0.005 | nan |
| linear | SAR | RandomForest | 0.7434 | 0.5098 | 0.1619 | 0.8016 | 0.192 | 0.018 | nan |
| linear | SAR | XGBoost | 0.7248 | 0.4847 | 0.1721 | 0.7855 | 0.040 | 0.003 | nan |
| linear | SCAR | CatBoost | 0.8014 | 0.5715 | 0.1520 | 0.8041 | 0.093 | 0.001 | nan |
| linear | SCAR | ExtraTrees | 0.8115 | 0.5981 | 0.1670 | 0.8211 | 0.105 | 0.018 | nan |
| linear | SCAR | NeuralForest_PUHiddenLatentCandidate | 0.8088 | 0.5983 | 0.1598 | 0.8147 | 0.248 | 0.014 | 0.540 |
| linear | SCAR | NeuralForest_PUHiddenLatentVerified | 0.8138 | 0.6076 | 0.1570 | 0.8238 | 0.280 | 0.005 | nan |
| linear | SCAR | NeuralForest_PUHiddenRandomCandidate | 0.8063 | 0.5981 | 0.1596 | 0.8146 | 0.247 | 0.012 | 0.382 |
| linear | SCAR | NeuralForest_PUHiddenRandomVerified | 0.8138 | 0.6076 | 0.1570 | 0.8238 | 0.281 | 0.006 | nan |
| linear | SCAR | NeuralForest_PUHiddenRawCandidate | 0.8047 | 0.5971 | 0.1594 | 0.8108 | 0.225 | 0.011 | 0.000 |
| linear | SCAR | NeuralForest_PUHiddenRawVerified | 0.8138 | 0.6076 | 0.1570 | 0.8238 | 0.261 | 0.007 | nan |
| linear | SCAR | NeuralForest_PUHiddenTeacher | 0.8138 | 0.6076 | 0.1570 | 0.8238 | 0.225 | 0.005 | nan |
| linear | SCAR | RandomForest | 0.7921 | 0.5676 | 0.1548 | 0.7934 | 0.198 | 0.019 | nan |
| linear | SCAR | XGBoost | 0.7720 | 0.5245 | 0.1595 | 0.7748 | 0.042 | 0.003 | nan |
| no_signal | SAR | CatBoost | 0.5041 | 0.2556 | 0.2037 | 0.5110 | 0.092 | 0.001 | nan |
| no_signal | SAR | ExtraTrees | 0.5010 | 0.2571 | 0.1949 | 0.5047 | 0.104 | 0.018 | nan |
| no_signal | SAR | NeuralForest_PUHiddenLatentCandidate | 0.5010 | 0.2595 | 0.2130 | 0.4940 | 0.234 | 0.013 | 0.407 |
| no_signal | SAR | NeuralForest_PUHiddenLatentVerified | 0.5023 | 0.2609 | 0.2208 | 0.4931 | 0.267 | 0.007 | 0.437 |
| no_signal | SAR | NeuralForest_PUHiddenRandomCandidate | 0.4993 | 0.2579 | 0.2134 | 0.4940 | 0.234 | 0.011 | 0.362 |
| no_signal | SAR | NeuralForest_PUHiddenRandomVerified | 0.5023 | 0.2613 | 0.2208 | 0.4929 | 0.266 | 0.007 | 0.348 |
| no_signal | SAR | NeuralForest_PUHiddenRawCandidate | 0.5026 | 0.2605 | 0.2126 | 0.4972 | 0.212 | 0.011 | 0.000 |
| no_signal | SAR | NeuralForest_PUHiddenRawVerified | 0.5025 | 0.2608 | 0.2207 | 0.4941 | 0.247 | 0.006 | 0.000 |
| no_signal | SAR | NeuralForest_PUHiddenTeacher | 0.5029 | 0.2622 | 0.2217 | 0.4938 | 0.212 | 0.007 | nan |
| no_signal | SAR | RandomForest | 0.5054 | 0.2544 | 0.2015 | 0.4956 | 0.195 | 0.018 | nan |
| no_signal | SAR | XGBoost | 0.5102 | 0.2613 | 0.2141 | 0.4963 | 0.041 | 0.003 | nan |
| no_signal | SCAR | CatBoost | 0.4906 | 0.2534 | 0.1979 | 0.5009 | 0.091 | 0.001 | nan |
| no_signal | SCAR | ExtraTrees | 0.4903 | 0.2494 | 0.1939 | 0.5084 | 0.103 | 0.018 | nan |
| no_signal | SCAR | NeuralForest_PUHiddenLatentCandidate | 0.4874 | 0.2474 | 0.2000 | 0.5046 | 0.211 | 0.013 | 0.418 |
| no_signal | SCAR | NeuralForest_PUHiddenLatentVerified | 0.4923 | 0.2499 | 0.2020 | 0.5081 | 0.242 | 0.006 | nan |
| no_signal | SCAR | NeuralForest_PUHiddenRandomCandidate | 0.4896 | 0.2480 | 0.2000 | 0.5030 | 0.211 | 0.013 | 0.400 |
| no_signal | SCAR | NeuralForest_PUHiddenRandomVerified | 0.4923 | 0.2499 | 0.2020 | 0.5081 | 0.239 | 0.005 | nan |
| no_signal | SCAR | NeuralForest_PUHiddenRawCandidate | 0.4928 | 0.2533 | 0.1996 | 0.5059 | 0.190 | 0.010 | 0.000 |
| no_signal | SCAR | NeuralForest_PUHiddenRawVerified | 0.4929 | 0.2496 | 0.2012 | 0.5106 | 0.223 | 0.006 | 0.000 |
| no_signal | SCAR | NeuralForest_PUHiddenTeacher | 0.4923 | 0.2499 | 0.2020 | 0.5081 | 0.190 | 0.005 | nan |
| no_signal | SCAR | RandomForest | 0.4944 | 0.2550 | 0.1964 | 0.5015 | 0.196 | 0.018 | nan |
| no_signal | SCAR | XGBoost | 0.4921 | 0.2576 | 0.2081 | 0.5065 | 0.041 | 0.003 | nan |

## Scope and limitations

This is a synthetic research simulation, not a real-world
independent clinical or ecological field validation.
XOR and null settings were inspected in earlier audits, so
this cannot be treated as an untouched confirmation test.
The neural hidden units are learned in-sample from PU labels;
only the subsequent evaluation/test cases are fully held out.
Under SAR/SNAR and selected background, P/U classification
may be a systematically biased surrogate.
Raw, untrained-random and learned-hidden candidates were
trained independently with identical teacher seeds, not
from a sharedfit in memory: their training costs are REAL and separate.
The teacher row is an ablation of the raw model's trained
networks: do not add its recorded training time to raw total.
The experiment must not be used to tune on the test outcomes.
See all_runs.csv, paired_effect.csv and paired_intervals.csv
for every per-seed outcome and unadjusted interval.
