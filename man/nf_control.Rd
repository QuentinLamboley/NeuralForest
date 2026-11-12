\name{nf_control}
\alias{nf_control}
\title{Configure the NeuralForest ensemble and neural learner}
\description{Specifies native tree structure, sampling, neural architecture, and compute settings.}
\usage{
nf_control(n_trees=300L, mtry=NULL, min_leaf=NULL, min_split=NULL,
  max_depth=NULL, max_nodes=100000L, max_candidates=128L,
  sample_fraction=0.8, replace=TRUE, split_rule="auto",
  split_search="auto", histogram_bins=128L, extra_random=FALSE, min_gain=0, laplace=1, n_threads=1L,
  neural=TRUE, neural_rounds=3L, neural_hidden=NULL,
  neural_epochs=35L, neural_batch=256L, neural_lr=0.002,
  neural_weight_decay=1e-4, neural_patience=6L,
  neural_guidance=0.6, feature_guidance=0.25,
  neural_temperature=0.8, max_weight_ratio=6,
  class_balance="auto", adaptive=TRUE, patience=2L,
  sampling_weight_mode="single", linear_expert="none",
  linear_penalties=c(0.03,0.3,3), expert_blend=c(0.25,0.5,0.75,1),
  expert_min_gain=0.001, differentiable=FALSE, soft_n_trees=6L, soft_depth=3L,
  soft_epochs=35L, soft_batch=128L, soft_lr=0.015,
  soft_temperature=0.8, soft_loss="auto", soft_blend="auto",
  soft_weight_decay=1e-4, soft_patience=7L, soft_clip=4,
  verbose=FALSE)
}
\arguments{
\item{n_trees}{Number of trees per round.}
\item{mtry}{Candidate features per split; NULL chooses an adaptive value.}
\item{min_leaf}{Minimum observations in terminal nodes.}
\item{min_split}{Minimum observations to attempt splitting.}
\item{max_depth}{Maximum depth per tree.}
\item{max_nodes}{Maximum number of nodes per tree.}
\item{max_candidates}{Approximate evaluated split positions per variable; 0 means all.}
\item{sample_fraction}{Bootstrap draw fraction of training observations.}
\item{replace}{Sample with replacement.}
\item{split_rule}{Auto, gini, entropy, or variance.}
\item{split_search}{Auto, exact sorted thresholds, or histogram bins.}
\item{histogram_bins}{Histogram resolution (minimum 8).}
\item{extra_random}{Randomly offset candidate thresholds.}
\item{min_gain}{Minimum impurity improvement.}
\item{laplace}{Additive probability smoothing at leaves.}
\item{n_threads}{Requested OpenMP threads.}
\item{neural}{Enable guided resampling.}
\item{neural_rounds}{Number of alternating neural/forest rounds.}
\item{neural_hidden}{Vector of hidden layer widths.}
\item{neural_epochs}{Maximum epochs per MLP fit.}
\item{neural_batch}{Minibatch size.}
\item{neural_lr}{Adam learning rate.}
\item{neural_weight_decay}{Decoupled weight decay.}
\item{neural_patience}{Neural early stopping patience.}
\item{neural_guidance}{Sample-guidance weight between 0 and 1.}
\item{feature_guidance}{Feature-proposal guidance weight between 0 and 1.}
\item{neural_temperature}{Controls sharpness of sample priority.}
\item{max_weight_ratio}{Maximum multiplicative priority weighting.}
\item{class_balance}{Auto, balanced, or none.}
\item{adaptive}{Choose unset defaults using training target profile.}
\item{patience}{Outer early stopping patience.}
\item{sampling_weight_mode}{Single uses sample importance only in the draw, double also weights node splitting and leaves. Double preserves the legacy behavior.}
\item{linear_expert}{None or auto. Auto selects a ridge-regularized linear or constant expert on an internal holdout.}
\item{linear_penalties}{Candidate positive ridge penalties.}
\item{expert_blend}{Candidate linear expert mixture coefficients.}
\item{expert_min_gain}{Minimum validation improvement required to replace the current model with an expert.}
\item{differentiable}{Train soft oblique trees by gradient descent and validate a mixture with the native forest.}
\item{soft_n_trees}{Differentiable trees in the secondary ensemble.}
\item{soft_depth}{Depth of each differentiable tree, from 1 to 7.}
\item{soft_epochs}{Maximum gradient training epochs.}
\item{soft_batch}{Minibatch size.}
\item{soft_lr}{Adam learning rate.}
\item{soft_temperature}{Sigmoid routing temperature.}
\item{soft_loss}{Auto, logloss, brier, focal, auc ranking surrogate, smooth ap surrogate, or mse.}
\item{soft_blend}{Auto validation selection or numeric mixture coefficients.}
\item{soft_weight_decay}{AdamW weight decay.}
\item{soft_patience}{Early stopping by validation surrogate loss.}
\item{soft_clip}{Gradient norm clipping threshold.}
\item{verbose}{Print training progress.}
}
\value{An nf_control list.}
