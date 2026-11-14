% NeuralForest stable PU learning, version 0.8
\name{nf_pu_stable_forest}
\alias{nf_pu_stable_forest}
\title{Stability-constrained, cross-fitted positive-unlabeled neural-guided forest}
\usage{
nf_pu_stable_forest(
  x, y, priors = c(0.12, 0.25, 0.40), seed = 42L,
  hidden = 20L, epochs = 60L, n_trees = 130L,
  min_gain = 0.004, strengths = c(0.25, 0.55),
  min_ess = 0.96, uncertainty_limit = 0.35,
  teacher_restarts = 2L, fidelity_cap = NULL,
  min_rank_correlation = 0, rank_lock = FALSE
)
}
\arguments{
\item{x}{Training features (matrix or data frame).}
\item{y}{Binary PU indicator: 1 is a confirmed positive; 0 is unlabeled, not necessarily negative.}
\item{priors}{A prespecified set of plausible positive-class prior values, never derived from test labels.}
\item{seed}{Reproducible seed.}
\item{hidden}{C++ neural hidden layer width.}
\item{epochs}{Number of native C++ network epochs.}
\item{n_trees}{Native C++ tree count per candidate forest.}
\item{min_gain}{Minimum PU risk improvement on both holdout halves required to replace neural fallback.}
\item{strengths}{Prespecified strengths of uncertainty-constrained tree resampling.}
\item{min_ess}{Lower bound on effective sample size as a proportion of unlabeled group size.}
\item{uncertainty_limit}{Maximum disagreement across neural networks before suppressing guided weights.}
\item{teacher_restarts}{One or two independent restarts per assumed prior.}
\item{fidelity_cap}{Optional absolute probability deviation cap.}
\item{min_rank_correlation}{Minimum Spearman concordance on both halves.}
\item{rank_lock}{Whether to replace the forest predictions by a globally monotone, native C++ forest-distilled map of the teacher score. Requires fidelity_cap. This can preserve but not improve rank-based discrimination.}
}
\description{
Combines native C++ positive-unlabeled neural ensembles, two-fold
out-of-fold bootstrap weights and a native C++ stability-constrained
sampling gate. A disjoint PU holdout chooses between a teacher-only
fallback, a uniform forest, and bounded guided forests, with optional
50 percent teacher blends. The selected candidate must improve the
PU risk for EVERY prespecified prior on BOTH holdout halves.
}
\details{
No unverified background labels or independent test outcomes are used
for training or selection. Neural networks are fitted on separate folds
to create training-case importance weights. Class mass is conserved and
the native sampler enforces a lower effective-sample-size limit.
When training or holdout class support is too small, conservative
fallback may disable guided forest fitting. PU risk and class-prior
assumptions can be misspecified, especially with environmentally
selected background or selected-not-at-random positive confirmation.
This is an experimental validation heuristic, NOT a guarantee of
correct predictions or universal superiority.
}
\value{An object of class \code{nf_pu_forest} with
\code{selection_history}, \code{stability_diagnostics},
\code{selected_candidate}, \code{selected_strength} and a
\code{predict(..., type = "prob")} method.}
