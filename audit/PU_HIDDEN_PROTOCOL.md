# Native nnPU hidden-feature forest: experiment design

Status: v0.9.4 EXPERIMENTAL. Do not interpret scores from the old
simulators as confirmatory field, veterinary or medical validation.

## Research question

Can learned, frozen C++ nnPU neural representations improve a native
forest's *split decisions*, beyond using only the original clinical or
ecological predictors, while remaining robust to mislabeled negatives,
unrepresentative backgrounds and compute cost?

This is separate from:
- PUStable: reweight a native classification forest with neural estimates
- PUResidual: adjust the teacher's logits with nnPU gradient trees
- PUSafe: strictly monotone teacher-based score correction (cannot
  improve AUC/AP rankings).

## Frozen neural supervision and structural ablation

Both feature modes fit exactly the same deterministic two-restart nnPU
ensemble per fixed prior (0.12/0.25/0.40), with the same training-only
sample and fixed seed. Each version builds its own C++ native forest.
"raw" splits only on the original encoded variables; "hidden" splits
on those SAME variables plus a maximum of two tanh activations from
each neural network, selected by training-only absolute readout weight.

The total feature-sampling mass allotted to hidden units is fixed at
35%; 65% remains with the original variables. The network features are
bounded to [-1,1], C++ builds the trees, and forest outputs are
calibrated to the nnPU score distribution from training-only OOB
predictions, without using verified labels. A fixed 65% forest blend
is used, not tuned on test outcomes.

Because the hidden activations were trained on the same P/U training
labels, hidden features can encode overfitting. The disjoint verified
audit and untouched tests are therefore mandatory for any claims.
A raw+hidden improvement may also reflect high-capacity random
projections rather than meaningful biological relationships; an
additional random-feature ablation would be required to isolate the
biological value of learned representations.

## Partitioning and target-label leakage

Each scenario uses separately generated independent individuals:
- PU training cohort: 820; 70% of its true positive statuses become
  visible as 1, all other cases become UNLABELED 0.
- Verified audit: 320 real positive/negative outcomes; used ONLY for
  selecting each already fitted candidate versus the same teacher.
- Test cohort: 420 true outcomes, never used for model fitting or
  choosing coefficients, priors, network widths or forest depths.
- Shifted test cohort: 420 separately generated records with
  changed values of features x3/x4; not an external institution.

The controls include vanilla RandomForest and ExtraTrees and tuned-
only-by-prespecified-settings CatBoost/XGBoost trained against the same
observed P/U indicators. They are P/U CLASSIFIERS, not proven
estimators of true infection or disease status.

The candidate ablation is intentionally unguarded and MUST NOT be
used for clinical deployment. "verified" uses two prespecified audit
halves with no less than 15 true outcomes of each class per half and
accepts a candidate ONLY after strict AUC gain with AP and Brier
non-inferiority guards. The audit cannot serve as a final test.

## Metrics and attribution

For EVERY seed and scenario: report ROC AUC, PR AUC, Boyce (if
estimable), Brier, logloss, mean train time (including NN and OOB
alignment), prediction time on both cohorts, latent importance
fraction and raw/hidden/teacher paired differences. Report raw
candidate and hidden candidate independently of their audit
decisions so a fallback cannot hide structural failure.

No universal 'best in all circumstances' conclusion is possible:
realistic test distributions, clinical sites, covariate shifts,
SAR/SNAR labeling propensities, multicollinearity and rare events
are future independent research requirements. Even a PU risk
improvement can reduce genuine target-outcome accuracy.

## Reproducibility

The implementation is in R/pu_hidden_forest.R,
src/pu_latent.cpp, native registered Rcpp wrappers, and the
tests/testthat/test-pu-hidden-forest.R suite.
The exploratory pilot code lives in
benchmark/pu_hidden_pilot.py and
benchmark/pu_hidden_pilot_r.R.
CI: .github/workflows/pu-hidden-feature.yml.
Outputs: audit/pu_hidden_feature_results/RESULTS.md,
all_runs.csv, paired_effect.csv, manifest.json (when finished).
