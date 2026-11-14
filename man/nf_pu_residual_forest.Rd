\name{nf_pu_residual_forest}
\alias{nf_pu_residual_forest}
\title{Neural-guided positive-unlabeled residual regression trees}
\usage{
nf_pu_residual_forest(
 x, y, priors = c(0.12, 0.25, 0.40), seed = 42L,
 hidden = 20L, epochs = 60L, rounds = 4L,
 trees_per_round = 2L, max_depth = 2L, learning_rate = 0.12,
 max_logit_change = 0.4, min_gain = 0.002,
 min_rank_correlation = 0.99, multipliers = c(0.25, 0.5, 1)
)
}
\arguments{
\item{x}{Data frame or matrix of model predictors.}
\item{y}{0/1 indicators: one means confirmed positive, zero means unlabeled, not confirmed negative.}
\item{priors}{Prespecified plausible class proportions for PU-risk sensitivity analysis.}
\item{seed}{Random seed for neural networks and native residual trees.}
\item{hidden}{Width of each C++17 PU neural network.}
\item{epochs}{Neural training epoch limit.}
\item{rounds}{Number of shallow residual regression-tree stages (1-20).}
\item{trees_per_round}{Number of bagged native C++ trees at each additive stage.}
\item{max_depth}{Maximum depth of each residual tree (1-4).}
\item{learning_rate}{Logit step size at each stage.}
\item{max_logit_change}{Bound on total logit correction relative to the neural teacher.}
\item{min_gain}{Strict positive PU risk gain on BOTH validation halves for EVERY prior.}
\item{min_rank_correlation}{Lower bound on Spearman rank correlation with the neural teacher on each validation half.}
\item{multipliers}{Predeclared attenuation factors for the accumulated residual correction.}
}
\description{
An experimental alternative to using a positive-versus-unlabeled
classification forest. Computes negative gradients of nonnegative PU
log-loss in native C++17, fits bounded shallow regression trees to those
gradients, and adds the predictions to the logits of a fixed
two-restart-per-prior native nnPU teacher. Training-case neural guidance
uses disjoint two-fold cross-fitted predictions. The teacher-only
fallback is selected whenever the correction does not improve PU risk
for every prior on both training-only holdout halves.
}
\details{
Positive-unlabeled assumptions can be incorrect under selected-at-random
(SAR), selected-not-at-random (SNAR), geographically selected backgrounds,
or misspecified class prevalences. Reducing empirical PU risk does NOT
guarantee improvement on verified true outcomes, AUC, Boyce or calibration.
This method is intended for controlled experimental comparisons, including
the exact same-network baseline returned by
\code{nf_pu_matched_teacher()}. The verified-label gate can provide
additional assessment only when a genuinely independent cohort with
confirmed positives and confirmed negatives was reserved beforehand.
}
\value{An \code{nf_pu_forest} model with \code{residual_trees},
\code{selected_rounds}, \code{selection_history}, phase-level timings,
and the standard \code{predict(..., type = "prob")} method.}
