\name{nf_tune}
\alias{nf_tune}
\title{Cross-validated neural forest hyperparameter search}
\description{Random or grid-based search across specified nf_control parameters with outer validation folds.}
\usage{
nf_tune(x, y, search=list(n_trees=c(100L,300L,500L),
  mtry=c(2L,4L,8L),min_leaf=c(1L,3L,8L),
  neural_guidance=c(0,0.3,0.6,0.9),
  feature_guidance=c(0,0.25,0.5),
  neural_rounds=c(1L,3L,5L)), trials=20L, folds=5L,
  metric=NULL, task="auto", control=nf_control(),
  seed=42L, groups=NULL, grid=FALSE)
}
\arguments{
\item{x}{Data frame/matrix of predictors.}
\item{y}{Response vector.}
\item{search}{Named list of parameter candidate vectors.}
\item{trials}{Random search trial count.}
\item{folds}{Number of outer folds.}
\item{metric}{Metric to optimize.}
\item{task}{Auto, regression, or classification.}
\item{control}{Base nf_control configuration.}
\item{seed}{Seed.}
\item{groups}{Optional group IDs for grouped folds.}
\item{grid}{Evaluate Cartesian grid if TRUE.}
}
\value{List containing results and best_control.}
