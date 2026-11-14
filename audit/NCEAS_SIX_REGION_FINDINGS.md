# Presence-only NeuralForest: six-region independently surveyed NCEAS audit

**Status: completed exploratory benchmark, 14 November 2026.**
PR #8 (`feat/nceas-six-region-ipp-gradient-validation`), merged as
`61b59a7d2ed3f5c53b34c7831a0e26b7f072a157`.
Successful six-region workflow: [run 43650340580](https://github.com/QuentinLamboley/NeuralForest/actions/runs/43650340580).
Successful R CMD check on Linux, Windows and macOS: run 43650340582.

## Scope and leakage precautions

The CRAN `disdat` NCEAS study supplies opportunistic **presence-only**
training records, **environmental availability** background and
separately surveyed presence/non-detection evaluation outcomes.
We used all six regions, with the **four species having most usable
training presences** per region (at least 70 each; selection never
examined surveyed outcome distributions, AUC or AP).

The 2,500 sampled environment-availability records per region are
not verified species absences. Survey observations are read only AFTER
all model predictions are frozen. Regional coordinate information
is never a biological predictor; a disjoint training-only geographic
band selects the native IPP gradient tree count. All 24 species
supported geographic inner bands (no random inner-validation
fallback). All twenty model prediction variants are present per
species. Categorical ecological predictors are encoded with levels
from training records alone; independent surveys do not determine
the encoding.

There are **24 species × 20 models = 480 prediction records** across
**456 independent model fits** (the exact gradient teacher shares the
raw native gradient fit). One GAM failed numerical convergence; its
diagnostics and missing AUC/AP are retained instead of being replaced
by GLM or dropping that species. Another GAM used the prespecified
ML+BFGS optimizer fallback, selected by TRAINING numerical failure
rather than test performance.

**Crucial:** these are independently collected survey **non-detections**
and presence detections, not perfect knowledge of true occupancy,
and the survey source does not guarantee geographic separation
from every opportunistic training point. AUC and AP do not calibrate
absolute occupancy or tick/pathogen risk.

## Prespecified primary metric: survey ROC-AUC

Regional macro means; with four species in every region these are
numerically identical to the pooled species means. Comparators were
fixed before independent survey metrics were examined and were NOT
given an equal-budget nested spatial hyperparameter search.

| Model variant | Mean ROC-AUC | Mean average precision |
| --- | ---: | ---: |
| **BalancedSubsample_RF (bootstrap class weights)** | **0.7314** | **0.4235** |
| CatBoost_weighted | 0.7266 | 0.4181 |
| ExtraTrees | 0.7221 | 0.4142 |
| CatBoost | 0.7216 | 0.4090 |
| GAM_smooth (23 of 24 species only) | 0.7202 | 0.3875 |
| XGBoost_weighted | 0.7189 | 0.4066 |
| MaxEnt/maxnet LQPH | 0.7188 | 0.3975 |
| Standard RandomForest | 0.7167 | 0.4100 |
| NeuralForest 0.10.0 raw forest | 0.7165 | 0.4096 |
| NeuralForest 0.10.0 random-tanh forest | 0.7149 | 0.4077 |
| NeuralForest 0.10.1 random-tanh IPP-gradient | 0.7128 | 0.4067 |
| NeuralForest 0.10.1 raw IPP-gradient | 0.7123 | 0.4061 |
| XGBoost unweighted | 0.7122 | 0.4000 |
| **NeuralForest 0.10.1 learned-hidden IPP-gradient** | **0.7114** | **0.4058** |
| Exact same-fit native IPP-gradient neural teacher | 0.7108 | 0.4051 |
| NeuralForest 0.10.0 learned-hidden forest | 0.7094 | 0.4039 |
| NeuralForest 0.10.0 neural IPP teacher | 0.7053 | 0.4003 |
| ranger classic RF | 0.7016 | 0.4000 |
| MaxEnt/maxnet LQ | 0.7015 | 0.3952 |
| Logistic GLM | 0.6918 | 0.3838 |

**Naming caution:** `BalancedSubsample_RF` applies
`class_weight="balanced_subsample"` within each bootstrap sample;
this is **not literal class undersampling**.

## Exact paired attribution (same 24 species)

For the proposed 0.10.1 learned-hidden IPP-gradient forest, the
average ROC-AUC difference compared with:

| Strictly matched comparator | Mean AUC difference | Wins / eligible species |
| --- | ---: | ---: |
| Exact same-fit neural-only teacher | **+0.0005** | 5/24 |
| Untrained equal-dimensional random-tanh IPP-gradient forest | **-0.0014** | 2/24 |
| Raw-feature IPP-gradient forest | **-0.0010** | 4/24 |
| MaxEnt LQPH | **-0.0074** | 9/24 |
| CatBoost weighted | **-0.0152** | 5/24 |
| Balanced-bootstrap-weighted RF | **-0.0201** | 5/24 |

The learned-hidden forest improved the exact teacher on only
5 species, degraded on 3, and tied on 16. Stage counts were:
**0: 16 species, 1: 3 species, 4: 5 species**. These stage counts
were selected by TRAINING-only geographically disjoint inner
validation. When zero stages were chosen, the model had exact
native IPP teacher predictions, not a claimed forest gain.

All six region-macro AUC differences versus balanced-bootstrap RF
were negative; five of six against weighted CatBoost were negative.
The learned representation did not beat its untrained random-control
forest on the macro mean. Thus **this experiment does not demonstrate
superiority, or even a reliably positive incremental neural-feature
contribution**.

## Per-region AUC (survey outcomes)

| NCEAS region | NeuralForest 0.10.1 hidden | Exact teacher | MaxEnt LQPH | CatBoost weighted |
| --- | ---: | ---: | ---: | ---: |
| AWT | 0.6855 | 0.6860 | 0.6928 | 0.7036 |
| CAN | 0.6913 | 0.6910 | 0.6966 | 0.6951 |
| NSW | 0.7104 | 0.7104 | 0.7139 | 0.7192 |
| NZ | 0.7468 | 0.7495 | 0.7614 | 0.7496 |
| SA | 0.6979 | 0.6919 | 0.7107 | 0.7330 |
| SWI | 0.7363 | 0.7360 | 0.7371 | 0.7590 |

## Numerical failures and reproducibility

The first six-region run 43649536955 stopped on a GAM REML
optimizer failure for `NZ/nz36`. The subsequent fixed policy
retains all species and attempts the same GAM formula using
a second numerical optimiser (ML+BFGS) **only if REML fails**
on training data; neither optimiser can see survey outcomes.
For NZ/nz36 both failed, so its GAM outcome is missing and
its error message and runtime are recorded in `fit_failures.csv`.
The final full six-region run 43650340580 completed successfully.
R CMD checks succeeded on Linux, Windows and macOS.

The model and per-species report CSVs, input support thresholds,
geographical inner-validation modes, per-model timings,
convergence outcomes, and all paired metric differences are
available via that workflow's `nceas-six-region-complete-audit`
artifact. `benchmark/po_nceas_multi_results/` is populated
by the main-branch workflow's results publishing step.

## Research and publication decisions

1. **Retain 0.10.1 as a documented experimental algorithm**;
   do NOT claim it is the best presence-only SDM.
2. Document the methodological novelty precisely (native C++ IPP
   objective-aligned gradient forests and frozen neural features)
   while recognizing prior DeepMaxent neural intensity modelling.
3. To improve scientific performance, examine identifiable
   **observation-effort correction**, target-group or survey effort
   surfaces, source-domain heterogeneity and a pooled multi-species
   representation. An opportunistic presence probability conflates
   habitat preference and observation effort.
4. Evaluate published **DeepMaxent** (pin its repository commit
   and implementation, no apples-to-oranges paper-score comparison),
   MaxEnt regularization sweeps and strong weighted/undersampled RF
   with matched **nested spatial tuning budgets**.
5. Freeze algorithms and evaluation plans BEFORE collecting new
   species/regions/time-domain independent evidence. The NCEAS
   benchmark was inspected during development; it is no longer
   untouched confirmatory evidence.
6. Preserve validation assumptions: independent surveyed
   non-detections are not proof of biological absence, and shared
   regional survey sites mean species cannot automatically be
   counted as independent experimental replicates.

**This is an honest research audit, not a finished publication claim.**
