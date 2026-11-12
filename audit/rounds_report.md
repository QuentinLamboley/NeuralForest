# Neural round count ablation

The 19 original scenarios were independently generated on seeds 227
and 311, which were not evaluated by the previous benchmark.
Two and three guided forest rounds use the same model structure
and the same training and test observations. The internal holdout
still selects the best round and expert without test labels.
An extra round may be rejected; more compute is not guaranteed
to improve generalization.

| Scenario | Metric | Two rounds | Three rounds | Three minus two, favorable direction | Extra total time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| correlated_30 | auc | 0.7817 | 0.7817 | +0.0000 | +0.121 |
| distinct_0 | auc | 0.9916 | 0.9924 | +0.0008 | +0.048 |
| distribution_shift | auc | 0.6893 | 0.6893 | +0.0000 | +0.142 |
| identical_100 | logloss | 0.6935 | 0.6935 | -0.0000 | +0.067 |
| intermediate_45 | auc | 0.8870 | 0.8870 | +0.0000 | +0.068 |
| label_noise_20pct | auc | 0.6996 | 0.7000 | +0.0004 | +0.191 |
| large_n | auc | 0.7705 | 0.7713 | +0.0009 | +0.332 |
| missing_15pct | auc | 0.7740 | 0.7740 | +0.0000 | +0.198 |
| noise_50 | auc | 0.7728 | 0.7692 | -0.0036 | +0.151 |
| nonlinear_xor | f1 | 0.9878 | 0.9878 | +0.0000 | +0.162 |
| overlap_80 | auc | 0.6544 | 0.6550 | +0.0006 | +0.075 |
| presence_bg_1to1 | boyce | 0.9930 | 0.9930 | +0.0000 | +0.090 |
| presence_bg_1to5 | boyce | 0.9704 | 0.9943 | +0.0240 | +0.133 |
| pseudoabsence_hard | boyce | 0.7152 | 0.7152 | +0.0000 | +0.095 |
| rare_10pct | pr_auc | 0.4230 | 0.4208 | -0.0021 | +0.144 |
| rare_1pct | pr_auc | 0.1688 | 0.1688 | +0.0000 | +0.642 |
| rare_3pct | pr_auc | 0.2363 | 0.2363 | +0.0000 | +0.153 |
| similar_97 | brier | 0.2502 | 0.2502 | -0.0000 | +0.072 |
| small_n | balanced_accuracy | 0.6448 | 0.6448 | +0.0000 | +0.020 |

Three rounds improved 6 scenarios, worsened 2,
and tied 11 at this precision on two new seeds.

## Interpretation

The core six-seed benchmark with 17 algorithms remains the primary
cross-method comparison. This targeted ablation assesses whether
the transported warm-start mechanism is exercised by an additional
neural round, and records its separate computational overhead.
The difference cannot be attributed exclusively to weight transport
because the number of training rounds also changes.
Do not infer general superiority from this synthetic two-seed ablation.

The raw score vector and selection, final-fit and prediction
durations for each case are available in rounds_results.csv.
