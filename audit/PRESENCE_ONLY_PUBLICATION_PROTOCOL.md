# NeuralForest PO — publication-readiness and independent validation protocol

Status: 2026-11-14. PRELIMINARY RESEARCH; does not assert superiority.

## Scientific claim to test

> A native hybrid neural/forest estimator trained on the
> inhomogeneous Poisson point-process (IPP) presence-only objective
> recovers relative habitat suitability more accurately than
> appropriately tuned ecological baselines, especially in nonlinear
> interactions, under controlled observation effort and spatial
> transfer; this is an empirical hypothesis, NOT an assumption.

An unbiased claim requires simultaneous improvements in (a)
discrimination of withheld presences from accessible environment,
(b) recovery of known habitat suitability in virtual species,
(c) spatial transfer, and (d) robustness to observation bias.
Report calibration-like scores ONLY when true labels are known,
not by calling background locations verified negatives.

A model may be valuable if competitive rather than the overall winner,
for example better under strong nonlinear interaction *and* low
sample size without a time-cost explosion; report narrow claims
and explicitly show losses under linear or highly biased designs.

## What is frozen already

The existing v0.10.0 1,800-evaluation spatial audit showed:
- Simple linear suitability: standard GLM / MaxEnt stronger overall.
- Nonlinear interaction: CatBoost oracle correlation stronger.
- Learned hidden features often help over raw features, but equal-size
  untrained nonlinear projections are essential controls.
- Presence-background Boyce can reward models that do not most closely
  recover the oracle latent ecological intensity.
- Native scores are relative intensity, not occupancy probabilities.
No gains on any of these previously inspected virtual species can be
advertised as a prospective independent confirmation.

Version 0.10.1 adds a separate native C++17 IPP derivative:
log(mean_weighted_background(exp(log_intensity))) -
mean_presence(log_intensity). The native forest uses shallow regression
trees to approximate negative gradients of THIS objective. No
background sample is assigned true biological absence status.

## Locked development benchmarks, no tuning on their outer outcomes

Three virtual species: linear, nonlinear interaction, multimodal.
Two observation-effort settings: unbiased, feature/spatial biased.
Two background policies: uniform accessible landscape and
detection-effort matched target group. Spatial CV: 5 geographic
blocks; initial gradient experiment tests outer blocks 0,2,4 for
two predetermined new seeds. A *different* geographic training block
selects the count of IPP gradient stages; after that count is
chosen, the learner refits on the complete outer-training area.
No geographical overlap between inner-validated candidates and
their training subset, and none between any training and outer test.

The complete baseline suite for the initial development benchmark
includes MaxEnt/maxnet, GLM, GAM, ranger and sklearn RF, extra-trees,
XGBoost, CatBoost and a fixed committee. The nested gradient
comparison includes matched IPP neural teacher, native IPP gradient
trees on raw, random-tanh and learned-tanh coordinates, maxnet LQPH,
GAM, RF, XGBoost, CatBoost. The fixed hyperparameters are not tuned
using outer tests. As implemented, only the number of IPP stages
is inner-selected; baseline nested tuning of equal budget is
FUTURE work before confirmatory superiority claims.

All means, losses, uncertainty, runtimes and validation failures
must be reported, without filtering to the best ecological niche.
Distinguish cost shared by the exactly matched neural teacher
from incremental native tree fitting.

## Critical field-data validation still missing

1. **Genuine presences and accessible environment M.**
   Define spatial extent independently of observed presence density;
   document suitability grid, duplicate handling, spatial uncertainty,
   survey period, recording biases and environmental variable versions.
   Independent *accessible background* is crucial. Hard pseudonegatives
   outside the known-presence FAMD hull are NOT an unbiased quadrature
   sample for the IPP likelihood.

2. **Survey effort / detection.**
   Supply a target-group background/detection-effort layer that is
   independently documented. Compare uniformly sampled accessible
   environment, target-group bias-matched background, spatial thinning,
   and effort strata. If effort is unknowable, report partial
   identifiability and domain sensitivity rather than claiming the
   model has discovered true ecological suitability.

3. **Independent ecological evaluations.**
   Two or more separately sourced species/regions with prespecified
   holdout geography and genuine independent observations (preferably
   structured field surveys with detection histories). For a true
   occupied/unoccupied evaluation, require designed surveys and
   detection-aware methods; do NOT relabel opportunistic background
   as nonoccurrence.

4. **Competitor fairness.**
   MaxEnt/maxnet with prespecified feature classes and an inner-selected
   regularization multiplier, GAM with suitable smoothness penalty,
   GLM/regularized logistic, RF including balanced and undersampled
   variants, XGBoost/CatBoost/LightGBM, and a fixed or train-selected
   ensemble. Tune all models with equal nested spatial fold structure
   and disclosed evaluation/computation budgets. Include equal-size
   random-feature controls and teacher-only ablation.

5. **Predictor leakage / spatial autocorrelation.**
   Fit imputers, encoders, neural hidden units, bias surfaces,
   quadrature weights, tuning, and ensembling EXCLUSIVELY inside outer
   training blocks. Report block extents in geography and
   environmental space separately. Leave-one-environmental-block-out
   is not itself leave-one-geographic-block-out. Shared occurrence
   cells across repetitions cannot be treated as independent surveys.

6. **Evaluation, uncertainty and maps.**
   Primary: withheld presence-background AUC or site-selectivity,
   calibrated to the chosen accessible background. Secondary:
   correctly defined continuous Boyce, environmental transfer,
   spatial-temporal transfer, mean/SD of relative intensity maps,
   ecological rank stability and map extrapolation/mismatch.
   For synthetic species, include oracle truth correlation/log skill.
   Compare fit time, prediction time, memory and total tuning budget,
   not just fitted-model prediction time.
   Avoid classifying a predicted relative intensity as an absolute
   occupancy probability or disease risk.

7. **Statistical inference.**
   Prespecify a primary model contrast, primary outcome and test
   domains *before collecting untouched field validation*.
   Use independent species/region/date as independent units,
   not correlated folds or five pseudoabsence repetitions.
   Report confidence intervals, multiplicity sensitivity, effect
   sizes, failure rates and negative findings.

## Connection to the *Ixodes ricinus* study

The existing study repository
https://github.com/QuentinLamboley/Iricinus-suitability-map
and its independent benchmark
https://github.com/QuentinLamboley/Iricinus-suitability-map-benchmark
are READ-ONLY for this NeuralForest research branch. Their Stage-25
files are deliberately not hosted on GitHub, and their five FAMD B
blocks are environmentally distinct rather than spatially independent.

A valid experiment using the species data would require an
**additional accessible-environment quadrature sample** and
independently documented effort design rather than reusing
environmental hard pseudoabsences as if they were unbiased
quadrature points. The original training labels may still support
a separately labeled sensitivity benchmark. Preserve the original
paper's frozen data, validation folds and outputs unchanged.

## Publication path

Possible research contribution:
"Objective-aligned neural-guided native forests for presence-only
species distribution modelling: nested spatial evaluation under
observation bias" — provisional research question, not a title
to submit without sufficient evidence.

The manuscript should contain equations and derivatives for IPP
network and native tree gradients, explicit random feature negative
control, simulation generator and seeds, nested spatial block design,
calibration/normalization definitions, representative competitor
hyperparameter budgets, per-study transfer results and reproducible
C++/R tests. Repository metadata and numerical claims must reflect
the EXACT benchmark commit and its archived outputs.

There is no universal best learner (No-Free-Lunch considerations
apply to heterogeneous data-generating distributions). Strong,
reproducible performance in a defined ecological regime is a more
defensible research claim than "best in all circumstances".
