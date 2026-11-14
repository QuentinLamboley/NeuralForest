\name{nf_po_metrics}
\alias{nf_po_metrics}
\title{Evaluate predictions at presence sites against environmental availability}
\usage{
nf_po_metrics(presence_scores, background_scores)
}
\arguments{
\item{presence_scores}{Scores for held-out presence localities.}
\item{background_scores}{Scores for held-out background availability localities, not confirmed absences.}
}
\description{
Compute background-discrimination AUC, a binned observed-over-expected
Boyce index and presence/background relative intensity means. Both
metrics depend on the chosen accessible/background distribution.
They do not establish occupancy accuracy or absolute predictive risk.
}
\value{Named numeric metrics: \code{auc_pb}, \code{boyce},
\code{mean_presence}, \code{mean_background},
\code{presence_background_ratio}.}
