# v0.9.4 — Native neural-feature forest: ten-seed research audit

## Status

The native C++17 frozen-hidden-unit forest, its equal-capacity
untrained random-tanh feature control, and raw-feature native forest
control passed multi-platform R CMD check (Linux, macOS, Windows),
the attribution protocol checks, and the paired synthetic experiment.
All comparisons are exploratory, not external independent clinical
or ecological evidence. The method is opt-in; neither the original
nnPU teacher nor the existing safer PU models are replaced.

The final GitHub Actions PR validation run used commit
`f685950fab01b1ea885e9c9986dacedda38d8a39`.
The PR was squash-merged in main with commit
`29c6914ed7201c0d4125ce6909e40825255cab5c`.

## Frozen experiment

Three mechanisms (XOR interaction, linear, no signal), SCAR and
feature-dependent SAR positive-label visibility, ten fixed random
seeds (5701, 5717, 5741, 5779, 5801, 5813, 5839, 5879, 5903,
5939), and separate training, true-outcome verification audit,
original test, and shifted test cohorts.

For each of 60 combinations, seven NeuralForest predictions and four
ordinary P/U classifiers are recorded: **660 predictions from 420
independently fitted models**. The native nnPU teacher and the
verified-gated versions reuse the respective fitted component; their
training time must not be double-counted.

The untrained random control is the same size as the learned hidden
representation and has the same raw predictors, feature-sampling
mass, forest hyperparameters and fixed teacher predictions.
Different distributions of learned versus random activations and
the in-sample hidden-unit fitting are remaining limitations.
Class labels for test and shifted cohorts NEVER choose algorithms.

## XOR AUC comparison (10 seeds)

| Positive-label mechanism | Teacher | Raw forest | Random-tanh forest | Learned-tanh forest | CatBoost |
| --- | ---: | ---: | ---: | ---: | ---: |
| SCAR | 0.5468 | 0.7591 | 0.7906 | 0.8053 | **0.8956** |
| SAR | 0.5074 | 0.6043 | **0.6349** | 0.6226 | **0.8590** |

Learned minus random-tanh AUC, with exploratory, unadjusted two-sided
paired 95% Student-t intervals across the ten independent simulated
seeds:

| Mechanism and evaluation cohort | Mean AUC difference | 95% paired interval | Learned wins |
| --- | ---: | --- | ---: |
| SCAR, original test | **+0.0147** | [+0.0015, +0.0279] | 9/10 |
| SCAR, shifted test | **+0.0220** | [+0.0005, +0.0434] | 7/10 |
| SAR, original test | **-0.0124** | [-0.0246, -0.0002] | 3/10 |
| SAR, shifted test | **-0.0226** | [-0.0491, +0.0039] | 2/10 |

Learned minus raw forest AUC:

| Mechanism and evaluation cohort | Mean AUC difference | 95% paired interval |
| --- | ---: | --- |
| SCAR, original | +0.0462 | [+0.0259, +0.0666] |
| SCAR, shifted | +0.0358 | [+0.0148, +0.0569] |
| SAR, original | +0.0182 | [+0.0016, +0.0349] |
| SAR, shifted | +0.0062 | [-0.0294, +0.0417] |

These paired intervals reflect variation across simulator seeds only,
are unadjusted for the many comparisons, and cannot be used as
clinical/ecological confidence intervals. The initial three seeds
and overall generator design were inspected before the extended
run, so this is a robustness check, NOT untouched confirmation.

## Failure modes remain

On XOR, CatBoost and XGBoost often outperform the hidden-feature
forest. In SCAR XOR, the learned-hidden forest's average Brier
was 0.2267 versus 0.1502 for CatBoost; it is therefore not
universally well calibrated. On SAR XOR, the learned model
underperformed its untrained random control, particularly after
distribution shift. This is especially concerning for selected
presence/background studies.

On linear signals, the verified audit generally preserved the nnPU
teacher (0/10 learned candidates accepted in SCAR and SAR).
On the null mechanism, the audit returned the teacher in all
SCAR runs but accepted one learned candidate out of ten in SAR.
That case slightly degraded held-out AUC, illustrating why a
verified two-half gate is an empirical safeguard, NOT a guarantee.
The ungated `gate="candidate"` mode must remain research-only.

The learned-hidden forest is lightweight but is not the fastest
competitor on the reported runner. In SCAR XOR, average native
learned-hidden training took ~0.198 s versus ~0.082 s for
CatBoost with its prespecified settings. These times reflect the
specified small synthetic datasets and single-thread settings.

## Implications and next experimental hypothesis

1. Keep `nf_pu_feature_forest(mode="hidden")` opt-in and retain
   the nnPU and rank-safe PUSafe references.
2. Include a frozen untrained nonlinear-feature negative control
   in EVERY subsequent neural-guidance performance claim.
3. Study the SAR mechanism explicitly: feature-dependent
   positive confirmation can cause learned neural representations
   to model selection propensity instead of true state.
4. Evaluate verified-positive/verified-negative domain holdouts,
   genuine independently collected data and a third fully
   untouched external test cohort; do not use previous benchmark
   outcomes to tune or choose hyperparameters.
5. For deployment, assess AUC, average precision, calibration
   (Brier/logloss), Boyce, external shift and training/prediction
   time simultaneously, not AUC alone.

## Reproduction

Source: `src/pu_latent.cpp`, `R/pu_hidden_forest.R`,
`benchmark/pu_hidden_pilot.py`, `benchmark/pu_hidden_pilot_r.R`.

CI: `.github/workflows/pu-hidden-feature.yml`.

Raw reproducible outputs: `audit/pu_hidden_feature_results/`
(`RESULTS.md`, `all_runs.csv`, `paired_effect.csv`,
`paired_intervals.csv`, `manifest.json`), after the main-branch
benchmark publisher completes. The PR workflow already produced a
complete archived run artifact.
