\name{nf_benchmark}
\alias{nf_benchmark}
\title{Time NeuralForest and optionally ranger}
\description{Run an indicative train/predict timing comparison on a shared outer holdout.}
\usage{
nf_benchmark(x,y,control=nf_control(n_trees=200L,
  neural_rounds=2L),seed=42L,validation=.2)
}
\arguments{
\item{x}{Data frame or matrix of predictors.}
\item{y}{Response.}
\item{control}{Neural forest settings.}
\item{seed}{Seed.}
\item{validation}{Outer holdout fraction.}
}
\value{Data frame with model name, training seconds, prediction seconds, score.}
\details{This is not a controlled experimental comparison. Ranger and NeuralForest have different defaults and preprocessing.}
