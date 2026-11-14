\name{nf_po_effort_forest}
\alias{nf_po_effort_forest}
\title{Presence-only native point-process habitat model with an independent observation-effort offset}
\usage{
nf_po_effort_forest(
  presences, background, effort_background,
  availability_weights = NULL,
  method = c("gradient", "neural"),
  feature_mode = c("hidden", "raw", "random"),
  validation_presences = NULL, validation_background = NULL,
  validation_effort_background = NULL,
  validation_availability_weights = NULL,
  seed = 42L, hidden = 12L, epochs = 75L,
  max_stages = 4L, trees_per_stage = 5L,
  max_depth = 3L, min_leaf = 8L,
  hidden_features = 8L, latent_mass = 0.35,
  step_size = 0.12, min_gain = 0.002
)
}
\arguments{
\item{presences}{Observed species presence predictors: not a complete occupancy inventory.}
\item{background}{Available-environment predictors (not known absence sites) with the same predictor columns.}
\item{effort_background}{Positive externally measured relative detection/reporting effort at each TRAINING background row. It MUST NOT be estimated using the held-out evaluation labels or tuned against them.}
\item{availability_weights}{Positive inverse-inclusion weights for the sampled accessible-area quadrature. If background is uniformly sampled across accessible space, these may be omitted. These weights differ from detection effort.}
\item{method}{Either a native neural Poisson point-process teacher or a gradient-trained native C++ forest augmenting it.}
\item{feature_mode}{For gradient trees, frozen trained neural features, raw environmental features, or equally sized random untrained tanh projections.}
\item{validation_presences}{Disjoint inner training-only observed presence predictors for gradient stage selection.}
\item{validation_background}{Disjoint inner availability predictors for gradient stage selection.}
\item{validation_effort_background}{Measured effort for disjoint inner background quadrature (required with explicit inner validation).}
\item{validation_availability_weights}{Area inclusion weights for inner validation available-background sampling.}
\item{seed}{Reproducible random seed.}
\item{hidden}{Native C++ network hidden width.}
\item{epochs}{Native C++ network optimization epochs.}
\item{max_stages}{Maximum native IPP gradient correction stages.}
\item{trees_per_stage}{Number of regression trees per stage.}
\item{max_depth}{Maximum gradient-tree depth.}
\item{min_leaf}{Minimum terminal leaf size.}
\item{hidden_features}{Additional raw/random/neural split-feature count.}
\item{latent_mass}{Split-sampling mass for augmented features.}
\item{step_size}{Fixed gradient-forest stage step size.}
\item{min_gain}{Minimum held-out IPP-risk reduction required for a stage.}
}
\description{
This experimental interface treats observed opportunistic presence
intensity as species relative habitat intensity multiplied by a
separately known detection-effort surface. It uses the existing
native C++17 inhomogeneous Poisson-point-process objective with
availability quadrature weights multiplied by observed effort.
The fixed habitat-intensity predictions and their normalization
do NOT include detection effort.
}
\details{
With known background effort \eqn{q(s)} and positive accessible-area
quadrature weights \eqn{w(s)}, the IPP loss is proportional to
\eqn{\log\{\sum_{s\in BG}w(s)q(s)\exp(f(s))/\sum_{s\in BG}w(s)q(s)\}
 - \mathrm{mean}_{P}f(s)}.
The omitted mean log-effort over observed presences is constant
with respect to habitat score \eqn{f}. Multiplying the original
native point-process quadrature weights by independently supplied
effort is therefore an EXACT reparameterization of the desired
observation-intensity objective, not a neural heuristic.

The returned relative habitat intensity is normalized to weighted
mean one over the original accessible background, using
\code{availability_weights} ALONE, NOT the effort-weighted observation
distribution. This still does not identify absolute occupancy,
abundance, prevalence or pathogen risk. A known effort surface
must be scientifically justified independently of the focal species;
an effort surface inferred from the focal species occurrence labels
would confound species habitat preference with preferential reporting.
Incomplete, zero, biased or spatially nonstationary detection-effort
measurements can invalidate the correction.

For method=\code{"gradient"}, use separately reserved geographic
inner validation backgrounds and their effort when possible;
internal random validation is NOT spatially independent.
}
\value{An object inheriting \code{nf_po_forest} or
\code{nf_po_gradient_forest} with \code{effort_design} metadata and
\code{predict(...,type="relative")} scores.}
\seealso{\code{nf_po_forest}, \code{nf_po_gradient_forest}}
