# Independent observer-effort offsets in presence-only point-process forests

**Experimental NeuralForest 0.11.0 branch. Not a validated clinical or
ecological risk model, and not a guaranteed accuracy improvement.**

## Identifiability in opportunistic presence-only data

A spatial presence record is sampled from *observed* intensity

`lambda_observed(s) = q(s) × lambda_habitat(s)`

where `q(s)` is detectability, reporting or observer effort,
and `lambda_habitat(s)` is the ecological habitat intensity of interest.
With presence-only events alone, `q` and `lambda_habitat` are not
separately identifiable without extra information. A model trained to
discriminate opportunistic presences from uniformly available space can
perfectly learn observer accessibility and still fail to represent
species habitat.

Let `f(s) = log(lambda_habitat(s))`. With independent positive
relative effort `q_j` measured at accessible environmental
quadrature cells `b_j`, and inverse background design/inclusion
weights `w_j`, the **observed-point IPP loss** is

`L(f) = log( sum_j w_j q_j exp(f(b_j)) / sum_j w_j q_j )
         - (1/n_presence) sum_i f(p_i)`.

The term `mean_P(log q(p_i))` is constant in f and is omitted from
parameter optimization. The denominator `sum_j w_j q_j` is also
constant in f; including it avoids arbitrary scale changes in the
monitored objective. The existing native C++17 IPP neural and
gradient-forest functions optimize this EXACT objective when the
quadrature sampling weights are replaced by `w_j × q_j` (common
multiplicative scaling cancels).

The **ecological relative habitat intensity** is predicted by
`exp(f(s))`, normalized with `w_j` **only**, not `w_j q_j`.
It is NOT occupancy probability, abundance, prevalence, confirmed
absence or a tick infection-risk probability. The known observer-effort
surface must NOT be multiplied back into the habitat prediction,
unless a distinct use case explicitly wants *future observed reports*
instead of ecological habitat.

## Data requirements and exclusion boundaries

- Both `q_j` and `w_j` must be positive, finite and correspond to
  identical accessible-background rows as the model predictors.
- `w` represents available-area quadrature inclusion probabilities,
  not detection itself. For truly uniform accessible background,
  equal `w` can be scientifically appropriate. A hard pseudoabsence
  distribution selected by environmental separation from known
  presences is not an unbiased available-area quadrature.
- `q` must come from INDEPENDENT observation-effort information:
  structured survey effort, recorded observer visits or a justified
  non-focal target-group reporting surface. If estimated from focal
  species presences without extra assumptions, this correction
  reuses the biological response and confounds detection and habitat.
- A separate inner geographic validation cohort for gradient-stage
  selection requires an independent `q` for that cohort. The outer
  survey locations and their true observations cannot be used to
  choose q, fit the network, tune parameters, or select stages.
- Zero effort makes occurrence impossible in a region; such areas
  cannot be used to infer habitat without extrapolation and new
  detectability assumptions. This first research interface requires
  strictly positive q rather than pretending it learned those areas.
- Poorly measured q or wrong accessible-area weights can worsen
  predictions. Excessively concentrated q reduces effective
  background support, measured by
  `ESS=(sum(w*q))^2 / sum((w*q)^2)`.

## Validation controls

1. **Unit-level analytic test**: The native C++ numerical
   point-process gradient with `w*q` must match the finite
   difference of the independently written known-effort IPP loss.
2. **Exact null test**: When q is spatially constant, the same
   seeds/architecture must recover identical native IPP network
   parameters and predictions to the uncorrected network.
3. **New independent simulator**: New virtual habitat mechanisms
   `ecotone` and `interaction`, detection effort
   `uniform`/ `biased`, a separate imperfect effort proxy,
   3 prespecified seeds × 3 strict outer geographic folds.
   Inner spatial blocks are disjoint from both teacher fitting
   and outer test. **Oracle ecological suitability** is
   evaluated only after all predictions are finalized.
4. **Confounding negative control**: Report *both* ecological oracle
   fidelity and opportunistic-presence/background AUC; the latter
   can improve just by predicting reporting effort.
5. **Fair sampling correction control**: Compare with the same
   training set and a fixed CatBoost whose negative background
   sample weights also receive independent effort, plus
   raw/random-feature native forest and naïve MaxEnt/GLM
   baselines. Do not attribute a generic bias correction's
   gain to the neural architecture without matched controls.
6. **Real-world validation**: NCEAS `disdat` survey data do not
   automatically provide independently measured species-specific
   q. Its current 24-species benchmark CANNOT be relabelled
   "effort corrected" by this code. Acquire independently verified
   target-group/survey effort surfaces and reserve new unseen
   study regions and structured survey observations before
   confirmatory claims.

## Publication interpretation

An independent effort offset is a known point-process modelling
idea, not a new discovery. Candidate algorithmic novelty is in
combining it with a *native C++ objective-aligned gradient forest*,
frozen learned/raw/random features, conservative inner geographic
stage selection, and reproduced cross-domain accuracy/robustness.
If improvements exist only for oracle-perfect effort in artificial
data, but not for realistic imperfect target-group effort or new
survey sites, the evidence supports at most a methodological
research note, not an algorithm superior to DeepMaxent, MaxEnt,
weighted RF or CatBoost.

Avoid tuning model hyperparameters after looking at the outer
oracle results and then calling a rerun "external confirmation."
Keep every negative outcome, runtime and uncertainty diagnostic.
