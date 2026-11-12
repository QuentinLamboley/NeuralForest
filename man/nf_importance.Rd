\name{nf_importance}
\alias{nf_importance}
\alias{nf_diagnostics}
\title{Variable importance and diagnostics}
\description{Summarize weighted CART impurity improvements and training diagnostics.}
\usage{
nf_importance(object, normalize=TRUE)
nf_diagnostics(object)
}
\arguments{
\item{object}{Fitted neural_forest.}
\item{normalize}{Scale importance values to sum to one.}
}
\value{nf_importance returns a ranked named vector; nf_diagnostics returns a list.}
\details{Impurity-based importance can be biased; prefer independent permutation studies for inference.}
