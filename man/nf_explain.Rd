\name{nf_permutation_importance}
\alias{nf_permutation_importance}
\alias{nf_uncertainty}
\title{Independent permutation importance and ensemble disagreement}
\description{Estimate metric deterioration on shuffled evaluation features, or inspect uncalibrated tree prediction dispersion.}
\usage{
nf_permutation_importance(object, x, y, metric=object$metric,
  repeats=5L, seed=42L, groups=NULL)
nf_uncertainty(object, newdata,
  measure=c("sd","entropy","margin","vote"))
}
\arguments{
\item{object}{Fitted neural_forest object.}
\item{x}{Independent evaluation feature data.}
\item{y}{Corresponding evaluation outcomes.}
\item{metric}{Metric to assess.}
\item{repeats}{Number of permutations.}
\item{seed}{Random seed.}
\item{groups}{Optional IDs restricting permutations to within groups.}
\item{newdata}{Feature data for predictions.}
\item{measure}{Tree-level standard deviation, normalized predictive entropy, inverse class margin, or vote disagreement.}
}
\value{nf_permutation_importance returns a data frame; nf_uncertainty returns a vector or matrix.}
\details{All uncertainty values are diagnostic proxies and are not calibrated confidence intervals. For scientific interpretability, use external evaluation data.}
