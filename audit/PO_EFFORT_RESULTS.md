# v0.11.0 — independent effort-offset neural/forest ecological audit

**Research status:** mathematically implemented and verified, but **NOT
a broadly superior presence-only SDM and NOT independently confirmed
on real-species field datasets**.

Merged PR #9, commit `232e9102aa0a4087ad071660390f65bde2318f11`.
GitHub Actions original completed benchmark: [run
43652070678](https://github.com/QuentinLamboley/NeuralForest/actions/runs/43652070678).
R CMD check on Ubuntu/Linux, Windows and macOS:
[run 43652070709](https://github.com/QuentinLamboley/NeuralForest/actions/runs/43652070709)
(all three passed).

The public opt-in `nf_po_effort_forest()` can use the native C++17
point-process neural teacher (`method="neural"`) or its
objective-gradient forest (`method="gradient"`) for known independent
detection-effort intensity. The existing default `nf_po_forest()`
and `nf_po_gradient_forest()` are unchanged.

## Scientific implementation

The observed report intensity is
`lambda_observed(s) = q_effort(s) * lambda_habitat(s)`.
The exact IPP risk used by the native C++ backend is

`L(f) = log(sum_BG(w_area * q_effort * exp(f)) /
            sum_BG(w_area * q_effort)) - mean_P(f)`,

with true absences **never** invented. The omitted observed-presence
`mean(log q)` term is constant in `f`. The ecological habitat output
is `exp(f)` normalized with the **area weights only**; it does not
include q in predictions and is NOT calibrated occupancy, population
abundance, pathogen infection probability or prevalence.

A spatially constant q is cancelled exactly (including numerical
reproducibility on the three supported operating systems). The C++
negative gradient was checked against finite differences on an
independently written offset likelihood; area-only normalization,
invalid effort weights and disjoint validation effort all have tests.

Crucially, q must be externally estimated from an independently
documented survey-effort, recording-coverage or justified non-focal
target-group process. A value of q estimated from the very same
focal-species presences without separate information does not identify
the detection process; automatic bias correction cannot be claimed.
See `audit/PO_EFFORT_IDENTIFIABILITY.md` for methods and assumptions.

## A new spatial virtual-species challenge

Two newly constructed habitat processes (`ecotone`,
`interaction`), two observation regimes (uniform, strongly biased),
3 fixed simulation seeds and 3 nonoverlapping geographic outer
blocks. Inside each outer training region, a separate geographic
inner block selects native gradient stages.

**36 paired cases, 13 named model variants, 468 prediction records**.
Training never accesses the virtual species' true habitat intensity,
which is used ONLY to assess predictions after model fitting.
All runs passed, with no omitted unfavorable conditions.

Primary score: **Spearman correlation of estimated habitat relative
intensity with the virtual species' true ecological intensity**
on held-out available habitat. This metric is available in simulated
populations only, not ordinary opportunistic real-species records.

| Habitat | Recording effort | Neural IPP naive | Neural IPP known q | Gradient forest naive | Gradient forest known q | Gradient forest imperfect q | CatBoost effort-weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ecotone | biased | 0.8122 | 0.6723 | 0.8103 | 0.6829 | **0.8206** | **0.8429** |
| Ecotone | uniform | 0.7904 | 0.7904 | 0.7965 | 0.7965 | 0.7965 | **0.8815** |
| Interactions | biased | 0.2991 | 0.3506 | 0.3074 | 0.3842 | 0.4529 | **0.8760** |
| Interactions | uniform | 0.6775 | 0.6775 | 0.7045 | 0.7045 | 0.7045 | **0.9089** |

The known q offset helped the native gradient forest by **+0.0768
Spearman rho** in the biased interaction process (6 of 9 paired
spatial evaluations), but **harmed it by −0.1273** in the biased
ecotone (only 3 of 9 wins). When q was uniform, the corrected
and naive models returned identical scores. **In both biased
scenarios, the prespecified independent imperfect q proxy was
more favorable than the perfectly known q**, with no universal
interpretation beyond this simulated mechanism.

The ecological oracle expected log-intensity skill led to the same
qualitative caution: in biased interactions, naive gradient
skill −0.0343 improved to +0.0608 with known q, whereas in
biased ecotones it worsened from +0.2149 to +0.1758.

A strong weighted CatBoost used exactly the same independent
effort information as the native offset. It remained much better
than NeuralForest on biased interactions (0.8760 versus 0.3842)
and on the biased ecotone (0.8429 versus 0.6829). Thus **the
success of sample-effort correction is not unique to NeuralForest**.

## Why an exact offset can worsen held-out ecological correlations

The simulated presence generator selects unique grid cells
*without replacement* from habitat-times-effort weights (roughly
550 cells from a 3,025-cell grid). An ideal continuous-space
Poisson point process can have repeated events in a spatial cell;
deduplicating and saturating cells alters the observation model
when detection is high. Such differences matter when applying
the same IPP likelihood to presence-only records prefiltered to
one record per cell. Environmental covariate misspecification,
spatial transfer beyond training blocks, finite sampling, and
imperfect q can also contribute. This analysis has NOT isolated
the causal reason for each observed degradation; do not select
a favored explanation post hoc.

An important subsequent simulator sensitivity is explicitly to
compare independently generated event-process counts **with
replacement** and deduplicated grid-cell presence records, using
new fixed seeds and a generator frozen before results are seen.
Consider point-process/quadrature discretization and finite-cell
detection models instead of pretending the observed design is
an ideal continuous-time Poisson event process.

## Publication judgement

- Implementation and numerical loss equivalence: **checked**.
- Positive improvement under a specific biased-interaction virtual
  habitat: **observed**.
- Universal correction benefit, reliable better-than-random learned
  features, superiority to weighted CatBoost, real-world field
  external validation: **NOT established**.
- Independence and power: spatial folds overlap in their
  training data; three seeds and two artificially defined
  habitats are not 36 independent species or field studies.
- Prior art: observation-effort offsets and weighted background
  likelihoods are known point-process concepts. Their presence
  alone is NOT the innovation. Candidate publication novelty
  would require a robust native neural-guided objective-gradient
  forest that adds reproducible benefits in carefully defined,
  independently validated ecological regimes.

## Reproducibility

- `R/po_effort.R`, `man/nf_po_effort_forest.Rd`;
- `tests/testthat/test-po-effort.R` (finite-difference gradient,
  exact constant-effort equivalence, weighted habitat normalization,
  independent validation), `benchmark/test_po_effort_design.py`;
- simulator `benchmark/po_effort_design.py`;
- R/Python runners `benchmark/po_effort_models.R`,
  `benchmark/po_effort_run.py`;
- aggregation `benchmark/po_effort_summarize.py`;
- CI `.github/workflows/po-effort-challenge.yml`;
- raw full matrices, all paired effects and stage counts in the
  successful `po-effort-offset-full-audit` workflow artifact,
  and `benchmark/po_effort_results/` after the main publishing
  workflow succeeds.

No ecology article should claim that NeuralForest was the best in
all circumstances. The most defensible next path is explicit
effort identifiability, careful point-process observation design,
paired ablations, and new independently sampled ecological surveys.
