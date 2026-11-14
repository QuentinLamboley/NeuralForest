# NeuralForest presence-only: NCEAS six-region exploratory pilot

**These are separate-source surveyed species evaluations, not
a preregistered independent publication or proof of superiority.**

Training uses opportunistic presence records plus independently
sampled environmental availability, not invented true absences.
The training-only x/y geography of opportunistic presences and
background is used to hold out an inner band for selecting
native IPP-gradient stage count, when such support is sufficient.
A fallback to RANDOM inner holdout is explicitly flagged.

Categorical predictors documented in disdat (ontveg, vegsys,
soilfert, age, toxicats, calc) were one-hot coded with levels
derived exclusively from train PO/background sources. Coordinates
are never supplied as ecological predictors. Each region has a
predeclared minimum presence count and up to four species by
training PO frequency, not by independent survey results.

Surveyed presence/non-detection outcomes are used to assess
ROC-AUC and average precision ONLY after every model fitted
and all prediction vectors were frozen. A survey non-detection
is not proven ecological unoccupancy when detection is imperfect.

## Coverage (six regions including those with zero eligible species)

| Region | Species above PO minimum | Evaluated species | Models run |
| --- | ---: | ---: | ---: |
| AWT | 20 | 4 | 80 |
| CAN | 14 | 4 | 80 |
| NSW | 12 | 4 | 80 |
| NZ | 15 | 4 | 80 |
| SA | 11 | 4 | 80 |
| SWI | 28 | 4 | 80 |

## Macro-regional average on evaluable species

Each represented region receives one equal weight, regardless
of number of species. The species-weighted averages and all
raw species-level outcomes are in the accompanying CSV files.

| Model | Survey ROC-AUC | Average precision | Mean fit seconds |
| --- | ---: | ---: | ---: |
| BalancedSubsample_RF | 0.7314 | 0.4235 | 0.662 |
| CatBoost_weighted | 0.7266 | 0.4181 | 0.211 |
| ExtraTrees | 0.7221 | 0.4142 | 0.234 |
| CatBoost | 0.7216 | 0.4090 | 0.226 |
| GAM_smooth | 0.7203 | 0.3876 | 3.484 |
| XGBoost_weighted | 0.7189 | 0.4066 | 0.083 |
| maxnet_lqph | 0.7188 | 0.3975 | 7.221 |
| RandomForest | 0.7167 | 0.4100 | 0.554 |
| NeuralForest_PO_raw | 0.7165 | 0.4096 | 0.477 |
| NeuralForest_PO_random | 0.7149 | 0.4077 | 0.509 |
| NeuralForest_POGrad_random | 0.7128 | 0.4067 | 0.595 |
| NeuralForest_POGrad_raw | 0.7123 | 0.4061 | 0.584 |
| XGBoost | 0.7122 | 0.4000 | 0.082 |
| NeuralForest_POGrad_hidden | 0.7114 | 0.4058 | 0.595 |
| NeuralForest_POGrad_teacher | 0.7108 | 0.4051 | 0.281 |
| NeuralForest_PO_hidden | 0.7094 | 0.4039 | 0.507 |
| NeuralForest_PO_neural | 0.7053 | 0.4003 | 0.299 |
| ranger_RF | 0.7016 | 0.4000 | 0.304 |
| maxnet_lq | 0.7015 | 0.3952 | 5.320 |
| GLM_logistic | 0.6918 | 0.3838 | 0.017 |

## Primary candidate versus controls, paired on the SAME species

| Reference | Mean region AUC gain | Mean region AP gain | Species wins (AUC) |
| --- | ---: | ---: | ---: |
| NeuralForest_POGrad_teacher | +0.0005 | +0.0007 | 5/24 |
| NeuralForest_POGrad_raw | -0.0010 | -0.0002 | 4/24 |
| NeuralForest_POGrad_random | -0.0014 | -0.0008 | 2/24 |
| NeuralForest_PO_hidden | +0.0020 | +0.0019 | 12/24 |
| NeuralForest_PO_random | -0.0035 | -0.0018 | 11/24 |
| maxnet_lqph | -0.0074 | +0.0083 | 9/24 |
| GAM_smooth | -0.0017 | +0.0089 | 10/23 |
| ranger_RF | +0.0097 | +0.0058 | 13/24 |
| RandomForest | -0.0053 | -0.0042 | 9/24 |
| BalancedSubsample_RF | -0.0201 | -0.0176 | 5/24 |
| XGBoost | -0.0009 | +0.0058 | 11/24 |
| XGBoost_weighted | -0.0076 | -0.0008 | 10/24 |
| CatBoost | -0.0103 | -0.0032 | 8/24 |
| CatBoost_weighted | -0.0152 | -0.0123 | 5/24 |

## Per-region primary and comparator survey AUC

| Region | Native IPP-gradient hidden | Matched teacher | MaxEnt LQPH | CatBoost weighted |
| --- | ---: | ---: | ---: | ---: |
| AWT | 0.6855 | 0.6860 | 0.6928 | 0.7036 |
| CAN | 0.6913 | 0.6910 | 0.6966 | 0.6951 |
| NSW | 0.7104 | 0.7104 | 0.7139 | 0.7192 |
| NZ | 0.7468 | 0.7495 | 0.7614 | 0.7496 |
| SA | 0.6979 | 0.6919 | 0.7107 | 0.7330 |
| SWI | 0.7363 | 0.7360 | 0.7371 | 0.7590 |

## Numerical convergence and missing fit audit

Number of model optimizer failures: 1.
Failed model rows are retained with NA AUC/AP, explicit
fit_failure_reason and training time, not removed or set to zero.
REML-to-ML/BFGS is a deterministic numerical fallback on the
TRAINING fit only; its use is recorded in fit_optimizer.
See fit_failures.csv and all_species_models.csv.

## Methodological caveats and next studies

- Species were selected by TRAINING occurrence count, so this
  *pilot* overrepresents relatively well-observed species.
- Test survey outcomes come from separate collection protocols;
  locations can still overlap opportunistic occurrence regions.
- Species sharing regional survey sites are NOT independent
  observational units. Six regions are too few for robust
  confirmatory region-level significance tests.
- Native IPP-gradient stages are inner-selected, whereas
  competitor hyperparameters are predeclared and fixed, not
  matched nested tuning. Comparability of compute budgets is
  not established.
- Some categories are unseen at survey sites and are encoded
  as all-zero training-known one-hot levels; categorical
  transfer/extrapolation needs a dedicated sensitivity audit.
- Maximum four species per region and 2,500 available points
  are preliminary runtime budgets, not full 226-species data.
- DeepMaxent is not yet included: a validated implementation
  must be pinned and evaluated on the identical splits.
- Because the pilot's outcomes have now been inspected, any
  subsequent architecture change needs a NEW untouched
  dataset/region/time confirmation. Do not claim universal best.
- NeuralForest intensity outputs are NOT calibrated occupancy
  probabilities or pathogen risk.

All individual outcomes (including losses and undefined scores),
inner spatial/random fallback frequencies, timings, paired
contrasts and counts are available in the CSV artifacts.
