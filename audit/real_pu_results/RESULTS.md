# PU validation beyond the vampire simulator

Three real publicly bundled datasets are used: Wisconsin Diagnostic
Breast Cancer, Wine recognition, and handwritten Digits (digit eight
vs other digits). Each dataset is evaluated under SCAR (uniform
positive confirmation) and SAR (feature-dependent positive
confirmation) training-label mechanisms. All negative cases and
30% of positive training cases have unknown labels. Ground truth
from two reserved independent evaluation partitions is never
used by model selection. SAR confirmation propensities depend
only on a high-variance feature identified among training positives.

This is out-of-simulator validation on real records, but NOT an
independent hospital, veterinary, geographical or field cohort.
PU priors remain fixed at 0.12, 0.25 and 0.40; no test-label
estimate of the class prior is permitted. Three seeds, 35 methods,
two confirmation regimes and 16 metrics yield 630 evaluations.

| Dataset | Regime | Best AUC method | AUC | NF safe | NF stable | NF PU neural | NF PU informed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Digits_Eight | SAR | ExtraTrees | 0.9771 | 0.9495 | 0.9518 | 0.9495 | 0.9559 |
| Digits_Eight | SCAR | Voting_RF_ExtraTrees | 0.9955 | 0.9692 | 0.9771 | 0.9692 | 0.9740 |
| Wine_Class0 | SAR | NeuralForest_PUStable | 1.0000 | 0.9977 | 1.0000 | 0.9977 | 0.9965 |
| Wine_Class0 | SCAR | NeuralForest_PUStable | 0.9988 | 0.9977 | 0.9988 | 0.9977 | 0.9942 |
| Wisconsin_Breast_Cancer | SAR | NeuralForest_PUStable | 0.9922 | 0.9922 | 0.9922 | 0.9922 | 0.9913 |
| Wisconsin_Breast_Cancer | SCAR | NeuralForest_PUStable | 0.9929 | 0.9909 | 0.9929 | 0.9909 | 0.9892 |

## Stable PU vs predecessor: all metrics and computation

No test-based selection is performed. The stable method's
cross-fitting, candidate forests, teacher fitting and holdout
selection are counted under selection_seconds. Its refit and
inference costs are recorded separately. All model scores and
times appear in summary.csv and all_runs.csv.

| Dataset | Regime | Safe AP | Stable AP | Neural AP | Safe total (s) | Stable total (s) | Neural total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Digits_Eight | SAR | 0.8084 | 0.8195 | 0.8084 | 2.321 | 4.345 | 1.384 |
| Digits_Eight | SCAR | 0.8831 | 0.9043 | 0.8831 | 2.406 | 4.518 | 1.461 |
| Wine_Class0 | SAR | 0.9957 | 1.0000 | 0.9957 | 0.065 | 0.131 | 0.069 |
| Wine_Class0 | SCAR | 0.9957 | 0.9979 | 0.9957 | 0.071 | 0.135 | 0.074 |
| Wisconsin_Breast_Cancer | SAR | 0.9909 | 0.9907 | 0.9909 | 0.511 | 1.029 | 0.298 |
| Wisconsin_Breast_Cancer | SCAR | 0.9908 | 0.9919 | 0.9908 | 0.538 | 1.204 | 0.298 |

## Limits

The pseudo-unlabeled regime is simulated from known real labels.
Outcome labels from the independent test partitions do not train
any model. However, the number of original datasets and seeds is
small, and no independent clinical site is represented.
No universal superiority is established by these comparisons.
The method's label-selection assumptions are more plausible here
than in environmentally selected hard backgrounds but still need
domain-specific evaluation.

Full results: all_runs.csv, summary.csv and manifest.json.
