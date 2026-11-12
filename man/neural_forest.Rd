\name{neural_forest}
\alias{neural_forest}
\alias{predict.neural_forest}
\alias{print.neural_forest}
\alias{summary.neural_forest}
\title{Adaptive neural-guided random forest}
\description{Fits a native C++17 random forest with optional neural sampling adaptation using out-of-bag errors.}
\usage{
neural_forest(x, y, task="auto", control=nf_control(), metric=NULL,
  validation=0.2, seed=42L, groups=NULL, refit=FALSE)
\method{predict}{neural_forest}(object, newdata,
  type=c("response","prob","class","trees"), ...)
}
\arguments{
\item{x}{Data frame or numeric matrix of predictors.}
\item{y}{Factor/character class labels or numeric target.}
\item{task}{One of auto, classification, regression.}
\item{control}{Object returned by nf_control.}
\item{metric}{Selection metric; see nf_metrics.}
\item{validation}{Fraction reserved for round selection.}
\item{seed}{Pseudo-random seed.}
\item{groups}{Optional group IDs for grouped splitting.}
\item{refit}{Train a final ensemble on all data after selecting the round.}
\item{object}{Fitted neural_forest.}
\item{newdata}{Data frame or matrix with training predictor names and order.}
\item{type}{Predicted response, probability matrix, class, or raw per-tree predictions.}
\item{...}{Unused additional arguments.}
}
\value{A fitted neural_forest object; predict returns classes, numeric predictions, or arrays.}
\details{The observation-priority network adjusts sample weights and feature proposals. When differentiable=TRUE, an additional ensemble of soft oblique trees learns split projections and thresholds by backpropagation with Adam. This ensemble uses a differentiable surrogate for non-differentiable metrics and its mixing coefficient is selected on internal validation data. The validation score is used for model selection and is not an unbiased test-set estimate.}
\examples{
fit <- neural_forest(iris[,1:4], iris$Species, metric="accuracy",
   control=nf_control(n_trees=15,neural=FALSE), seed=42)
predict(fit,iris[1:5,1:4],type="prob")
}
