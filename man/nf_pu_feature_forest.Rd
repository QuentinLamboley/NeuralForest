\name{nf_pu_feature_forest}
\alias{nf_pu_feature_forest}
\title{Experimental native PU forest using frozen nnPU hidden representations}
\usage{
nf_pu_feature_forest(
 x,y,mode=c("hidden","raw","random"),gate=c("pu","verified","candidate"),
 audit_x=NULL,audit_y=NULL,priors=c(.12,.25,.40),
 seed=42L,hidden=20L,epochs=60L,
 n_trees=130L,max_depth=5L,min_leaf=5L,
 per_network=2L,latent_mass=.35,forest_fraction=.65,
 min_pu_gain=.003,min_rank_correlation=.99,
 verified_min_gain=.005
)
}
\arguments{
\item{x}{Training data frame or matrix with original predictor variables.}
\item{y}{Observed positive (1) versus unlabeled (0) indicators; zeros are NOT verified negatives.}
\item{mode}{Frozen trained hidden-feature forest, raw-only forest, or equal-size untrained random tanh features as a negative control for dimensionality.}
\item{gate}{Choose training-only surrogate PU holdout, a disjoint verified-label audit, or a completely UNGATED candidate for exploratory ablation only. Never deploy a candidate without valid assessment.}
\item{audit_x}{Disjoint predictors for a verified-outcome audit, required for gate="verified".}
\item{audit_y}{Disjoint TRUE positive/negative outcomes, never PU labels, required for gate="verified".}
\item{priors}{Prespecified class-prior sensitivity grid, fixed without examining outcomes.}
\item{seed}{Deterministic training and native forest seed.}
\item{hidden}{Width of the native C++ neural teacher hidden layers.}
\item{epochs}{Maximum nnPU neural training epochs.}
\item{n_trees}{Number of native CART trees; at least 25.}
\item{max_depth}{Maximum native tree depth, up to 12.}
\item{min_leaf}{Minimum terminal node size.}
\item{per_network}{Hidden units selected by absolute output weight from each frozen nnPU network.}
\item{latent_mass}{Bounded proportion of feature-sampling weight assigned to latent features; raw features retain a majority.}
\item{forest_fraction}{Fixed blend fraction of the forest after label-free OOB-to-neural score alignment.}
\item{min_pu_gain}{Minimum strict improvement of PU-risk in every prior/half under the surrogate gate.}
\item{min_rank_correlation}{Minimum teacher/forest rank concordance on both PU holdout halves.}
\item{verified_min_gain}{Minimum strict AUC gain on both verified audit halves.}
}
\description{
An opt-in experimental method using strongest native C++ nnPU
hidden-neuron activations as additional split coordinates for a
native C++ random forest. Mode="raw" fits the SAME forest method,
prevalence assumptions and nnPU teacher but does not append latent
features. Hidden feature sampling mass is bounded to avoid overwhelming
original clinical or ecological evidence. The forest classifies
observed P/U indicators; the OOB-only monotone alignment matches its
score scale to the frozen nnPU teacher without using outcomes.
}
\details{
The learned hidden representation is frozen after the teacher has been
fitted on the respective training subset. Holdout/test outcome labels
are never used for hidden representation learning. The hidden units
extracted for fitting trees are in-sample with respect to training
labels, so latent-feature overfitting remains possible and MUST be
assessed with held-out outcomes.

The PU gate relies on nonnegative PU risk, which is not a true-outcome
risk under feature-dependent confirmation (SAR/SNAR), spatially
selected pseudoabsences, or class-prior misspecification. By default
the method returns the exact matched neural teacher when no strict
two-half benefit is shown. The verified gate needs truly independent
positives and negatives and a separately preserved final test cohort.
There is no distribution-free guarantee of superiority or safety.
}
\value{
A fitted \code{nf_pu_forest} with native forest components (when
selected), feature-importance diagnostics, model-selection history,
per-stage timings, and standard \code{predict(...,type="prob")}
interface. Use \code{nf_pu_matched_teacher} to extract the SAME
networks with the forest removed.
}
