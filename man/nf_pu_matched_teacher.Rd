\name{nf_pu_matched_teacher}
\alias{nf_pu_matched_teacher}
\title{Extract the exact trained neural ensemble from a PU forest}
\usage{
nf_pu_matched_teacher(object)
}
\arguments{
\item{object}{A fitted \code{nf_pu_forest}, such as PUStable or PUSafe.}
}
\description{
Returns an otherwise identical fitted PU model with the native forest
and monotone calibration disabled. For PUStable this retains the same
two C++ neural restarts per assumed prior and the same encoder. It
provides the only valid control to isolate the incremental contribution
of tree guidance instead of confounding it with additional neural fits.
}
\value{A model of class \code{nf_pu_forest}.}
