\name{nf_control}
\alias{nf_control}
\title{Configure the NeuralForest ensemble and neural learner}
\description{Specifies native tree structure, sampling, neural architecture, and compute settings.}
\usage{
nf_control(n_trees=300L, mtry=NULL, min_leaf=NULL, min_split=NULL,
  max_depth=NULL, max_nodes=100000L, max_candidates=128L,
  sample_fraction=0.8, replace=TRUE, split_rule="auto",
  extra_random=FALSE, min_gain=0, laplace=1, n_threads=1L,
  neural=TRUE, neural_rounds=3L, neural_hidden=NULL,
  neural_epochs=35L, neural_batch=256L, neural_lr=0.002,
  neural_weight_decay=1e-4, neural_patience=6L,
  neural_guidance=0.6, feature_guidance=0.25,
  neural_temperature=0.8, max_weight_ratio=6,
  class_balance="auto", adaptive=TRUE, patience=2L,
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
\item{verbose}{Print training progress.}
}
\value{An nf_control list.}
