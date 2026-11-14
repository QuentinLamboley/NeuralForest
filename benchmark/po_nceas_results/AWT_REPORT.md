# NCEAS independently surveyed ecological test: AWT

**Pilot, not a publishable multi-region conclusion.**
Species were chosen using ONLY their training presence count,
not survey outcomes or performance. Published NCEAS disdat
presence-only training and environmental background were fitted
with 14 algorithm variants and assessed against genuinely
independent designed-survey presence/absence outcomes.
Those surveyed outcomes were read only AFTER all methods fit.

The survey true negatives are genuinely observed non-detections
under the survey protocol, not assigned to background. No
detection histories or full occupancy probability calibration
are inferred from them. This is separate-source validation,
not necessarily geographic transfer validation.

| Method | Mean independent AUC | Mean independent AP | Train s |
| --- | ---: | ---: | ---: |
| BalancedSubsample_RF | 0.6518 | 0.5558 | 0.716 |
| CatBoost | 0.6512 | 0.5450 | 0.208 |
| ExtraTrees | 0.6277 | 0.5258 | 0.268 |
| GAM_smooth | 0.6413 | 0.5368 | 2.232 |
| GLM_logistic | 0.6011 | 0.5235 | 0.010 |
| NeuralForest_PO_hidden | 0.6238 | 0.5209 | 0.501 |
| NeuralForest_PO_neural | 0.6225 | 0.5245 | 0.255 |
| NeuralForest_PO_random | 0.6263 | 0.5318 | 0.497 |
| NeuralForest_PO_raw | 0.6257 | 0.5282 | 0.459 |
| RandomForest | 0.6243 | 0.5285 | 0.585 |
| XGBoost | 0.6526 | 0.5560 | 0.088 |
| maxnet_lq | 0.6110 | 0.5273 | 2.010 |
| maxnet_lqph | 0.6348 | 0.5233 | 2.711 |
| ranger_RF | 0.6182 | 0.5364 | 0.247 |

## Species-by-species AUC, including unfavorable outcomes

| Species | Group | Survey presences | Survey negatives | NN IPP | Hidden forest | MaxEnt LQPH | CatBoost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| awt01 | bird | 92 | 248 | 0.5266 | 0.5348 | 0.5062 | 0.5607 |
| awt03 | bird | 129 | 211 | 0.4705 | 0.5081 | 0.4928 | 0.5543 |
| awt04 | bird | 158 | 182 | 0.4573 | 0.4527 | 0.4795 | 0.4665 |
| awt06 | bird | 187 | 153 | 0.6595 | 0.6396 | 0.6865 | 0.6906 |
| awt08 | bird | 137 | 203 | 0.6861 | 0.6773 | 0.7139 | 0.7415 |
| awt15 | bird | 214 | 126 | 0.8005 | 0.7977 | 0.7928 | 0.8018 |
| awt17 | bird | 103 | 237 | 0.8364 | 0.8305 | 0.8283 | 0.8134 |
| awt18 | bird | 137 | 203 | 0.5431 | 0.5497 | 0.5779 | 0.5805 |

## Interpretation limits

This is a small pilot from a well-known public benchmark that
has been studied many times. A truly confirmatory study
requires prespecified species/regions, matched hyperparameter
tuning inside spatial outer-training blocks, and comparison
against published DeepMaxent and strong PO-specific RF models.
Independent surveys are not a guarantee of no geographic
autocorrelation or perfect detection.
AUC and AP on surveys are genuine evaluation metrics but do not
calibrate the relative intensity scores into occupancy.
The common available background was subsampled to 2500 training
records for runtime; this is not a full 10k-background benchmark.
All per-species unfavorable results, runtime, sample sizes and
undefined AUCs are retained in the machine-readable CSV.

Raw disdat species predictors/survey labels are NOT redistributed
by this workflow. Use the published package with attribution.
