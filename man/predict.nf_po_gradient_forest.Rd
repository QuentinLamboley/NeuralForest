\name{predict.nf_po_gradient_forest}
\alias{predict.nf_po_gradient_forest}
\title{Predict relative presence-only intensity with native point-process gradient trees}
\usage{
\method{predict}{nf_po_gradient_forest}(
 object,newdata,type=c("relative","log_relative","suitability"),...
)
}
\arguments{
\item{object}{A fitted \code{nf_po_gradient_forest}.}
\item{newdata}{Predictors with the same columns as training.}
\item{type}{Relative intensity, its logarithm, or a bounded monotone suitability transform. NOT occupancy probability.}
\item{...}{Unused.}
}
\description{Predicts a fixed background-normalised log-intensity
with zero to several trained C++ regression-forest corrections.}
\value{Numeric relative habitat-intensity scores.}
