\name{nf_metric}
\alias{nf_metric}
\alias{nf_metrics}
\title{Metrics for classification and regression}
\description{Evaluate probabilities and responses using a registry of metrics, with directions used during optimization.}
\usage{
nf_metrics(task=c("all","classification","regression"))
nf_metric(actual, predicted, metric, task="auto", threshold=0.5,
  beta=1, top_k=3L, quantile=0.5, huber_delta=1, bins=20L)
}
\arguments{
\item{task}{all, classification, regression, or auto for nf_metric.}
\item{actual}{Vector of observed responses.}
\item{predicted}{Numeric regression prediction or classification probability vector/matrix.}
\item{metric}{String metric identifier from nf_metrics.}
\item{threshold}{Binary class threshold.}
\item{beta}{F-beta coefficient.}
\item{top_k}{Number of classes for top-k accuracy.}
\item{quantile}{Pinball quantile.}
\item{huber_delta}{Huber loss transition point.}
\item{bins}{Calibration or Boyce bins.}
}
\value{nf_metrics returns a data frame; nf_metric returns a named numeric scalar.}
\details{Some metrics are undefined for folds with a single class. For presence-background problems, Boyce has specialized ecological interpretation. MASE assumes ordered observations.}
