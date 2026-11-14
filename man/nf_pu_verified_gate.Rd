\name{nf_pu_verified_gate}
\alias{nf_pu_verified_gate}
\title{Verified-outcome two-half decision gate for a PU-guided forest}
\usage{
nf_pu_verified_gate(
 object, audit_x, audit_y, objective = c("auc", "pr_auc"),
 min_gain = 0.002, max_ap_loss = 0.01,
 max_brier_increase = 0.01, min_group_half = 15L,
 seed = 42L
)
}
\arguments{
\item{object}{A previously fitted \code{nf_pu_forest}.}
\item{audit_x}{A separate, reserved cohort of predictors.}
\item{audit_y}{True verified outcomes (0 or 1), NOT PU labels.}
\item{objective}{A prespecified AUC or average-precision target.}
\item{min_gain}{Required strict improvement in each audit half.}
\item{max_ap_loss}{Maximum allowed loss of average precision on each half.}
\item{max_brier_increase}{Maximum allowed Brier-score increase on each half.}
\item{min_group_half}{Minimum number of verified positives and negatives in every half.}
\item{seed}{Prespecified random seed for stratifying the audit halves.}
}
\description{
Compares the deployed forest with its EXACT matched neural teacher
using two disjoint subsets of an independently reserved, verified-label
audit cohort. Selects the forest only if it improves the objective in
both halves while meeting AP and Brier constraints; otherwise uses
the neural teacher. Does not retrain either predictor.
}
\details{
The verified cohort must not have been used by the PU network, tree
fitting, hyperparameter choice, class-prior estimation, or previous
model iteration. This cannot be verified automatically from features
alone. A final independent test cohort must remain untouched, and
the verified audit must never be used repeatedly to tune the method.
The procedure is an empirical safeguard, NOT a distribution-free
performance guarantee. With inadequate verified-class support the
method falls back to its matched teacher. Additional verified outcomes
are required to improve beyond the intrinsic PU surrogate limitations.
}
\value{A fitted \code{nf_pu_forest} with \code{verified_gate} diagnostics.}
