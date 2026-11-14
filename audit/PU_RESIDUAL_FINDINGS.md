# NeuralForest v0.9.3 — experimental native PU residual trees

14 November 2026. All comparisons are exploratory; neither repeated
simulators nor previously inspected public datasets are confirmatory
external evidence. Performance from separate GitHub runners is not a
perfectly controlled compute-time comparison.

## Method

Native C++17 computes nonnegative-PU log-loss gradient targets
for shallow regression trees. The teacher is an ensemble of two
neural restarts for each prespecified class prior. Guidance uses
disjoint cross-fitted neural scores; logit corrections are bounded.
Any deployment requires held-out PU-risk improvements for EVERY
prior on BOTH validation halves and strong rank concordance.
Otherwise the exact two-restart neural teacher is used.

This DOES NOT guarantee improvements in real-outcome AUC, Boyce,
calibration or robustness to SAR/SNAR and ecological background
selection. The method is opt-in; previous algorithms are unchanged.

## Paired ten-seed presence/background simulator benchmark

| Scenario | Matched teacher AUC | Corrected AUC | AUC change | Boyce change | Shifted AUC change | Corrections selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| presence_bg_1to1 | 0.7879 | 0.7882 | +0.0003 | +0.0010 | +0.0002 | 2/10 |
| presence_bg_1to5 | 0.7950 | 0.7950 | 0.0000 | 0.0000 | 0.0000 | 0/10 |
| pseudoabsence_hard | 0.7018 | 0.7006 | -0.0012 | -0.0027 | -0.0006 | 8/10 |

The previous PUStable standard positive-vs-unlabeled forest lost
0.0546 AUC versus its exactly matched teacher on pseudoabsence_hard.
The bounded gradient residual reduces that failure sharply but
does not fully eliminate it. No universal superiority is established.

## Public data masking pilot

Two already-inspected public sources, two seeds and SCAR/SAR
produced eight comparisons: seven neural fallbacks and one
selected four-stage correction. Held-out AUC changes were
approximately zero. Training cost includes cross-fitting,
candidate search, final neural refit and shallow tree fitting.

## Designed interaction/noninteraction stress challenge

Five seeds per mechanism and confirmation regime were evaluated
with true synthetic outcomes. A distinct verified-label audit
cohort was reserved to choose between the residual learner and
its exact teacher; another two cohorts were test-only.

| Mechanism | Regime | nnPU teacher AUC | Residual AUC | CatBoost P/U AUC |
| --- | --- | ---: | ---: | ---: |
| XOR | SCAR | 0.5563 | 0.5563 | 0.8796 |
| XOR | SAR | 0.5103 | 0.5126 | 0.8417 |
| Linear | SCAR | 0.8167 | 0.8169 | 0.8038 |
| Linear | SAR | 0.7988 | 0.7988 | 0.7839 |
| Null | SCAR | 0.4751 | 0.4751 | 0.4735 |
| Null | SAR | 0.4863 | 0.4864 | 0.4950 |

The XOR failure is decisive for the intended model concept:
shallow residual corrections do not automatically repair a
weakly learned neural interaction. A genuinely complementary
neural-feature-guided forest is the next hypothesis to test,
preferably with a separately collected verified-label cohort.

## Audit conclusions

- Retain PUSafe / the neural teacher as conservative options for
  tasks without verified negative outcomes.
- Keep PUResidual as an optional research model, not as the new
  default or a clinical diagnostic predictor.
- Inspect matched-teacher deltas and fold-by-prior risk matrices,
  not just the aggregated overall benchmark score.
- Do not use held-out test outcomes to select algorithms, residual
  strengths, or class-prior assumptions.
- Real multi-center evaluation requires truly independently
  collected outcomes and a locked final test cohort.

Workflow source:
.github/workflows/pu-residual-pilot.yml
.github/workflows/pu-residual-legacy.yml
