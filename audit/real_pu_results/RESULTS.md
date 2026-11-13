# PU validation beyond the vampire simulator

Three real, publicly bundled datasets are used: Wisconsin Diagnostic
Breast Cancer, Wine recognition, and handwritten Digits (digit eight
vs other digits). Labels are artificially hidden for most negative
and 30% of positive training cases. All methods train on positive
confirmation versus unlabeled status. Independent original labels
are used **only** for the two reserved evaluation partitions.

This is out-of-simulator validation on real records, but NOT an
independent hospital, veterinary, geographical or field cohort.
PU priors remain fixed at 0.12, 0.25 and 0.40 and are not inferred
from true test outcomes. Three split seeds, 33 methods and 16
metrics are used, for 297 complete evaluations.

| Dataset | Best ROC-AUC method | AUC | NF PU informed | NF adaptive |
| --- | --- | ---: | ---: | ---: |
| Digits_Eight | Voting_RF_ExtraTrees | 0.9955 | 0.9740 | 0.9918 |
| Wine_Class0 | NeuralForest_PUNeural | 0.9977 | 0.9942 | 0.9977 |
| Wisconsin_Breast_Cancer | ExtraTrees | 0.9919 | 0.9892 | 0.9889 |

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
