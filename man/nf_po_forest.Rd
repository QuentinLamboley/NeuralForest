\name{nf_po_forest}
\alias{nf_po_forest}
\title{Experimental native presence-only point-process neural forest}
\usage{
nf_po_forest(
 presences, background, background_weights = NULL,
 mode = c("hidden","raw","random","neural"),
 seed = 42L, hidden = 16L, epochs = 90L,
 learning_rate = 0.01, decay = 0.0005,
 n_trees = 130L, max_depth = 5L, min_leaf = 5L,
 hidden_features = 8L, latent_mass = 0.35,
 forest_fraction = 0.65
)
}
\arguments{
\item{presences}{Observed-presence predictors (data frame or matrix).}
\item{background}{Environmental-availability predictors with the same columns. Background points are NOT verified absences.}
\item{background_weights}{Positive design inclusion weights for background availability. Passing these does NOT automatically correct spatial observation-effort bias.}
\item{mode}{Native C++ IPP neural network alone (\code{neural}); native raw forest (\code{raw}); learned hidden-unit augmented forest (\code{hidden}); same-size untrained random-tanh augmented forest (\code{random}).}
\item{seed}{Prespecified seed.}
\item{hidden}{Number of point-process network hidden units.}
\item{epochs}{Full-batch Adam optimisation iterations.}
\item{learning_rate}{Adam step size.}
\item{decay}{L2 regularization penalty.}
\item{n_trees}{Number of native CART trees for forest variants.}
\item{max_depth}{Maximum native tree depth.}
\item{min_leaf}{Minimum terminal-leaf size.}
\item{hidden_features}{Number of learned or random tanh features.}
\item{latent_mass}{Share of candidate-split feature-sampling weight allocated to learned/random features; original predictors retain more than half.}
\item{forest_fraction}{Fixed blend weight of aligned native forest log-ratio scores and neural point-process scores.}
}
\description{
Trains a native C++17 neural inhomogeneous point-process scorer
that maximises mean presence score relative to a background
log-mean-exponential partition function. It does not require
an assumed population prevalence or true negative outcomes.
}
\details{
The method compares observed presences with sampled environmental
availability, which may include occupied sites. The learned output is
a relative habitat intensity / suitability score, NOT occupancy
probability, species prevalence, incidence or a calibrated disease
risk. The standardised output has weighted mean relative intensity
one over the training background distribution. This normalisation
cannot identify absolute ecological abundance.

The native forest is fitted to the sampled presence-vs-background
contrast (not to known absence outcomes). Its OOB scores are
mapped to the teacher's score scale without looking at held-out
test locations or labels. Forest feature projections are frozen.
Sampling-effort bias, accessibility area, repeated presences, spatial
autocorrelation, transferability, and target-group background must
be addressed by study design and spatial blocked validation. This
experimental estimator has not established universal superiority
over MaxEnt/maxnet or other SDM algorithms.
}
\value{A fitted object of class \code{nf_po_forest}.}
\seealso{\code{nf_po_metrics}}
