# Cross-fitted uncertainty-guided PU forests: stability audit

Three legacy PU-background scenarios are compared on ten seeds.
Only the 3 specialized PU methods see confirmed-positive versus
unlabeled labels. The 31 comparator methods remain unchanged.
The stability-gated model uses six native C++ nnPU networks
(two initializations at each of three fixed class-prior assumptions).
Its training-example guidance is based on disjoint 2-fold
cross-fitted teacher predictions, never in-sample network outputs.
C++ weighting preserves group mass and ensures unlabeled
effective sample fraction at least 0.96. A candidate forest
must beat neural-only worst-prior PU validation risk on both
separate holdout halves by more than 0.004; otherwise it falls
back to the neural-only predictor.

| Scenario | Legacy nnPU Boyce | PU informed Boyce | Stable Boyce | Stable shifted Boyce | Stable selections |
| --- | ---: | ---: | ---: | ---: | --- |
| presence_bg_1to1 | 0.9865 | 0.9865 | 0.9875 | 0.9669 | neural_fallback: 10 |
| presence_bg_1to5 | 0.9872 | 0.9899 | 0.9851 | 0.9786 | neural_fallback: 10 |
| pseudoabsence_hard | 0.9670 | 0.6202 | 0.4512 | 0.4093 | guided_0.25: 4, guided_0.25_blend50: 1, guided_0.55: 3, plain: 1, plain_blend50: 1 |

## What the gate can and cannot ensure

The gate protects ONLY the empirical, worst-case selected-PU
surrogate on two training holdout halves. It does not
guarantee non-inferiority on true outcome labels or on a
feature-selected unrepresentative background (SAR/SNAR).
The score can mis-rank candidate forests if PU priors or
labeling assumptions are misspecified. The benchmark's
independent shifted cohort also comes from the SAME simulator.
Its metrics are test-only; no threshold or model selection
is allowed to consult its labels.

The 19 synthetic scenarios have been inspected repeatedly.
Out-of-simulator public-data and SAR tests are required
before claiming improvement beyond this simulator.
The time to train *all* guidance candidates and all
cross-fitted C++ neural ensembles is INCLUDED under
selection_seconds; native final refit is separately timed.
All 31 established forest, boosting, neural, voting and
stacking comparators remain unchanged.

See pu_stability_choices.csv and all_runs.csv for all outcomes.
