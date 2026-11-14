\name{nf_po_gradient_forest}
\alias{nf_po_gradient_forest}
\title{Native gradient forest improving a point-process neural habitat-intensity scorer}
\usage{
nf_po_gradient_forest(
 presences, background, validation_presences = NULL,
 validation_background = NULL, background_weights = NULL,
 validation_background_weights = NULL,
 feature_mode = c("hidden", "raw", "random"),
 seed = 42L, hidden = 12L, epochs = 75L,
 max_stages = 4L, trees_per_stage = 5L,
 max_depth = 3L, min_leaf = 8L,
 hidden_features = 8L, latent_mass = 0.35,
 step_size = 0.12, min_gain = 0.002
)
}
\arguments{
\item{presences}{Observed species-presence predictors, not all occupied cells.}
\item{background}{Sampled available environment, NOT verified absence observations.}
\item{validation_presences}{Optional completely disjoint, ideally spatially held-out observed presences reserved solely for choosing the number of stages.}
\item{validation_background}{Corresponding disjoint accessible environmental background.}
\item{background_weights}{Positive training-background inclusion weights. Do not mistake this for automatic observation bias correction.}
\item{validation_background_weights}{Positive disjoint validation-background inclusion weights.}
\item{feature_mode}{Learned frozen hidden features, raw original features, or an equal-sized untrained random-tanh negative control.}
\item{seed}{Reproducible seed.}
\item{hidden}{Width of the base C++ neural point-process teacher.}
\item{epochs}{Base neural training iterations.}
\item{max_stages}{Maximum additive native regression-forest stages.}
\item{trees_per_stage}{Native regression trees per stage.}
\item{max_depth}{CART maximum depth.}
\item{min_leaf}{Minimum number of examples per leaf.}
\item{hidden_features}{Number of learned or random feature coordinates.}
\item{latent_mass}{Total split-sampling mass allocated to extra features; original predictors retain majority.}
\item{step_size}{Fixed positive additive log-intensity step factor.}
\item{min_gain}{Minimum improvement in held-out IPP risk required for each sequential stage.}
}
\description{
Fits a native C++17 inhomogeneous Poisson point-process neural scorer
and additional native regression forests on negative gradients of the
SAME point-process loss, rather than on artificial presence/background
binary classification labels. Stage selection is constrained to a
disjoint training-only presence/background validation cohort, with
exact teacher-only fallback when no forest stage is accepted.
}
\details{
If no explicit validation records are supplied, stratified random
training-only holdouts are used, but such splits are not sufficient to
establish transferability under geographic autocorrelation. For a
publishable study, supply disjoint spatial inner validation,
independent outer spatial folds and a final untouched ecological
transfer set. The class prior and true absences are unknown.
The forest uses a relative intensity score, NOT occurrence probability,
true-negative discrimination, prevalence or tick burden.
No universal improvement is guaranteed, especially when background
selection does not represent environmental availability or sampling
effort differs from the target group.
}
\value{A fitted object of class \code{nf_po_gradient_forest}, with
\code{selected_stages}, \code{validation}, \code{training} and a
\code{predict(...,type="relative")} method.}
