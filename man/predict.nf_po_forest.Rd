\name{predict.nf_po_forest}
\alias{predict.nf_po_forest}
\title{Predict relative habitat intensity from a native point-process forest}
\usage{
\method{predict}{nf_po_forest}(
 object, newdata,
 type = c("relative", "log_relative", "suitability"), ...
)
}
\arguments{
\item{object}{A fitted \code{nf_po_forest}.}
\item{newdata}{Predictor rows with the same named columns as training.}
\item{type}{\code{relative} gives background-normalised positive relative intensity, \code{log_relative} its logarithm, and \code{suitability} a monotonic bounded transformation of relative intensity. The bounded transformation is NOT occupancy probability.}
\item{...}{Unused.}
}
\description{
Evaluate a frozen native presence-only neural or neural-feature forest
without re-estimating the background normalisation on test data.
}
\value{A numeric vector of relative scores.}
