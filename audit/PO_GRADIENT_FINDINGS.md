# v0.10.1 — objective-aligned Poisson point-process native forest

## Audit and implementation status

Merged pull request #6, main commit
\`707caa86178637638f932974344cf225e16d9868\`.

The native C++17 function \`nf_po_cpp_gradient\` computes the exact
point-process risk derivative
\(L=\log(\sum_j w_j \exp f(b_j)/\sum_j w_j)-
\overline{f(p_i)}\) before prespecified gradient clipping.
A native regression forest fits negative gradient pseudo-responses
and adds them to the frozen relative log-intensity nnPU/IPP teacher.
This **does not treat environmental background as true absence**.

The \`nf_po_gradient_forest\` R API uses a disjoint inner training
holdout to select a count of 0-4 native boosted forest stages, with
an exact teacher-only fallback. The spatial benchmark supplies
inner blocks geographically distinct from both the stage-training
subset and the final untouched outer block. After stage choice,
the final learner is refitted on all outer training records.
Scores are background-normalized relative habitat intensity,
NEVER absolute occupancy probability.

## Nested-spatial virtual species audit

All native tests, numerical finite differences and Linux/macOS/Windows
R CMD checks succeeded on the experimental branch. The benchmark
workflow completed **648 prediction records over 72 matched
spatial cases** (three virtual species, unbiased/biased detection,
uniform/effort-matched environmental background, two new seeds and
three held-out outer blocks) for nine methods. This corresponds to
576 independent fits because the exactly matched teacher is a
component of the raw variant.

Mean oracle Spearman correlation with TRUE simulated relative habitat
suitability, across all effort/background settings:

| Method | Linear | Nonlinear interaction | Multimodal |
| --- | ---: | ---: | ---: |
| Native IPP neural teacher | 0.628 | 0.556 | **0.778** |
| IPP-gradient neural-feature native forest | 0.625 | 0.600 | 0.775 |
| IPP-gradient random-feature native forest | 0.618 | 0.602 | **0.779** |
| IPP-gradient raw native forest | 0.632 | 0.587 | 0.777 |
| MaxEnt/maxnet LQPH | 0.614 | 0.847 | 0.662 |
| GAM | **0.881** | 0.088 | 0.714 |
| CatBoost | 0.449 | **0.896** | 0.612 |

There is a +0.0433 improvement of the gradient-hidden forest
over the same IPP teacher in the nonlinear experiment,
but learned hidden features do NOT reliably outperform an
equally sized random-tanh projection. CatBoost and MaxEnt
remain stronger than the native hybrid for this virtual
nonlinear mechanism. On linear and multimodal landscapes,
the hybrid does not consistently improve upon its teacher.

No single metric, including Boyce, can replace oracle recovery
in this virtual-species study. Mean AUC_pb discrimination against
background is not true occupied-versus-absent sensitivity.
Hyperparameters for the competing models were fixed without
equal-budget nested search; independent outer results have already
been inspected, so any later revisions need NEW untouched domains.

## Publication implications

- **Validated engineering**: Native C++ derivative implementation,
  finite-difference tests, additive forest fitting, nested spatial
  stage selection and same-teacher controls.
- **Not validated**: Universal superiority, true occupancy inference,
  calibration to prevalence, independence from detection-effort bias,
  superiority over DeepMaxent and multi-region survey transfer.
- **Next robust evidence**: NCEAS disdat independently surveyed
  presence/absence outcomes across diverse species/regions; use
  training-only backgrounds, external target-group effort controls,
  equal-budget nested spatial tuning and prespecified primary
  contrasts before claiming superiority.
- **Prior-art caution**: DeepMaxent already published a neural
  normalized Poisson/maximum-entropy approach in 2026. A
  single-species neural IPP objective is NOT by itself a new method.
  The potential technical novelty is native gradient-regression
  forests sharing this objective and their validated ecological
  robustness and computational cost.

Reproduce: \`.github/workflows/po-gradient-spatial.yml\`.
Outputs: \`benchmark/po_gradient_results/\` (workflow artifact and
later main-branch records). This document is not itself a
confirmatory journal result.
