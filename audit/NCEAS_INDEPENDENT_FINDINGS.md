# Native presence-only NeuralForest: independent NCEAS survey pilot

## Provenance and safeguards

The NCEAS dataset distributed in the CRAN R package \`disdat\`
is derived from Elith et al. (2020), DOI
10.17161/bi.vi15i2.13384 (use canonical publication identifier
10.17161/bi.v15i2.13384), and provides training presence-only
occurrences, available environmental backgrounds and genuinely
independently surveyed presence/absence evaluation outcomes.

The pilot used Australian Wet Tropics (AWT) data, **eight
bird species chosen solely by descending TRAINING presence count**:
awt01, awt03, awt04, awt06, awt08, awt15, awt17, awt18.
Independent survey outcomes did not affect species inclusion,
model training, model selection, hyperparameters or score
calibration. All algorithms shared a fixed common 2,500-point
region environmental background subsample; no background
location was described as verified unoccupied.

The R and C++ model versions tested are the v0.10.0
\`nf_po_forest\` neural, raw, learned-hidden and random-tanh
feature variants, plus maxnet LQ/LQPH, GLM, GAM, ranger RF,
standard RF, class-balanced-bootstrap RF, ExtraTrees, XGBoost
and CatBoost — 14 methods on 8 species = **112 fitted models**.

AUC and average precision were calculated against *independently
surveyed* presence/non-detection observations only AFTER every
model was trained; neither score is an occupancy calibration.

## All-method average on eight birds

| Model | Independent survey ROC-AUC | Average precision |
| --- | ---: | ---: |
| XGBoost | **0.6526** | **0.5560** |
| Bootstrap class-balanced weighted RF | 0.6518 | 0.5558 |
| CatBoost | 0.6512 | 0.5450 |
| GAM | 0.6413 | 0.5368 |
| MaxEnt/maxnet LQPH | 0.6348 | 0.5233 |
| NeuralForest PO random tanh | 0.6263 | 0.5318 |
| NeuralForest PO learned hidden tanh | 0.6238 | 0.5209 |
| NeuralForest PO neural only | 0.6225 | 0.5245 |

The class-balanced RF uses \`class_weight="balanced_subsample"\`,
which adjusts class weights on each bootstrap sample; it is
NOT itself a true class undersampling estimator. This distinction
must be retained in the manuscript.

Species-specific held-out ROC-AUC:

| Species | Native hidden | MaxEnt LQPH | CatBoost |
| --- | ---: | ---: | ---: |
| awt01 | 0.5348 | 0.5062 | 0.5607 |
| awt03 | 0.5081 | 0.4928 | 0.5543 |
| awt04 | 0.4527 | 0.4795 | 0.4665 |
| awt06 | 0.6396 | 0.6865 | 0.6906 |
| awt08 | 0.6773 | 0.7139 | 0.7415 |
| awt15 | 0.7977 | 0.7928 | 0.8018 |
| awt17 | 0.8305 | 0.8283 | 0.8134 |
| awt18 | 0.5497 | 0.5779 | 0.5805 |

The results DO NOT demonstrate NeuralForest's general superiority.
Some species favor it; the mean AUC is below XGBoost, balanced
RF and CatBoost. Learned tanh features did not outperform the
same-size untrained random feature control on this pilot.

## What is missing before manuscript submission

1. Evaluate all six NCEAS regions / all species meeting a
   predeclared training sample-size criterion, NOT only the eight
   most frequent AWT birds.
2. Respect the categorical predictor encoding, survey-specific
   absence/detection conventions and region-specific geographic
   information.
3. Compare the newer v0.10.1 objective-aligned gradient forest
   on the same training-only material, along with DeepMaxent and
   PO-specific RF weighting/downsampling.
4. Use equal-budget, NESTED spatial hyperparameter tuning.
   Independent survey testing is a strong external source, but
   survey locations are not automatically geographically disjoint
   from opportunistic training records.
5. Report primary contrasts, per-species paired differences,
   species-group/taxon/regional heterogeneity, correction for
   multiple testing and uncertainty with species/regions as
   independent units when justified.
6. These are already inspected public data, not an untouched
   future prospective ecological study. Reserve independent
   species/time/regions for confirmatory conclusions.

Source implementation:
\`benchmark/po_nceas_export.R\`,
\`benchmark/po_nceas_models.R\`,
\`benchmark/po_nceas_run.py\`,
\`.github/workflows/po-nceas-independent.yml\`.

Raw published data and true independent outcomes are NOT
redistributed here; install the source \`disdat\` R package and
cite the original Elith/Valavi data authors. Model-score
CSV and reports are stored in \`benchmark/po_nceas_results/\`.
