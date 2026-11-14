% Fidelity constrained neural-informed PU model
\name{nf_pu_safe_forest}
\alias{nf_pu_safe_forest}
\title{Fidelity-constrained PU forest with exact native neural fallback}
\usage{
nf_pu_safe_forest(
  x, y, priors = c(0.12, 0.25, 0.40), seed = 42L,
  hidden = 20L, epochs = 60L, n_trees = 130L,
  min_gain = 0.004, strengths = c(0.25, 0.55),
  min_ess = 0.96, uncertainty_limit = 0.35,
  fidelity_cap = 0.02, min_rank_correlation = 0.995
)
}
\arguments{
\item{x}{Training predictors, as matrix or data frame.}
\item{y}{Binary indicators: 1 confirmed positive, 0 unlabeled.}
\item{priors}{Prespecified prior sensitivity values.}
\item{seed}{Seed for reproducible fitting.}
\item{hidden}{Hidden layer size of C++ nnPU network.}
\item{epochs}{Maximum native nnPU optimizer epochs.}
\item{n_trees}{Native C++ tree count for candidate forests.}
\item{min_gain}{Minimum PU holdout improvement in BOTH halves.}
\item{strengths}{Candidate uncertainty guidance strengths.}
\item{min_ess}{Minimum unlabeled effective sample size fraction.}
\item{uncertainty_limit}{Maximum tolerated spread in teacher probabilities before guidance weights vanish.}
\item{fidelity_cap}{Maximum absolute deviation from legacy nnPU teacher predictions, imposed on validation and all future predictions.}
\item{min_rank_correlation}{Minimum Spearman rank concordance separately on both PU validation halves.}
}
\description{
Extends \code{nf_pu_stable_forest} with a strict trust region around
the same C++ positive-unlabeled neural-only model from v0.7.
When neither a plain forest nor a guided forest passes the
rank-concordance and worst-prior risk selection criteria, the
result is exactly the legacy nnPU standalone prediction.
}
\details{
All tree sampling, disagreement gating, and nnPU neural optimization
are implemented in C++17. The final fidelity bound is also imposed
at prediction time. That limits absolute probability drift but
DOES NOT guarantee non-inferiority in AUC, Boyce, average precision
or calibration; the correct class prior and the positive-label
confirmation mechanism are unknown in many ecological applications.
Comparisons on previously inspected synthetic cases are exploratory
rather than independent confirmatory evidence.
}
\value{An object of class \code{nf_pu_forest} with compatible \code{predict}.}
