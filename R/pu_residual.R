# Experimental additive PU residual trees, not a PU supervised classifier.
# Raw U labels are never treated as verified negative outcomes.
# The neural teacher and the residual model are compared on an untouched
# training-only holdout, for every prespecified prior and both halves.
# In SAR/SNAR contexts even this surrogate may select a harmful model.

nf_pu_residual_cfg <- function(n,p,trees_per_round,max_depth,min_leaf) {
  list(n_trees=as.integer(trees_per_round),
    mtry=as.integer(max(1L,ceiling(sqrt(p)))),
    min_leaf=as.integer(min_leaf),
    min_split=as.integer(2L*min_leaf),
    max_depth=as.integer(max_depth),
    max_nodes=as.integer(2^(max_depth+1L)-1L),
    max_candidates=96L,sample_fraction=.8,
    replace=TRUE,split_rule=0L,
    histogram_mode=1L,histogram_bins=64L,
    extra_random=FALSE,min_gain=1e-10,
    laplace=0,n_threads=1L,
    split_weighted=FALSE,sampling_mode=0L)
}

nf_pu_residual_offset <- function(trees,X,rate,rounds=length(trees)) {
  out <- numeric(nrow(X))
  if(rounds<1L)return(out)
  for(i in seq_len(rounds))
    out <- out + rate*as.numeric(nf_cpp_predict(
      trees[[i]],X,1L,FALSE)[,1L])
  out
}

nf_pu_residual_train <- function(X,y,oof_prior_probability,priors,
        rounds,trees_per_round,max_depth,min_leaf,rate,cap,seed) {
  if(nrow(X)!=length(y) || length(oof_prior_probability)!=length(y))
    stop("Residual inputs must have matching rows")
  cfg <- nf_pu_residual_cfg(nrow(X),ncol(X),trees_per_round,
                             max_depth,min_leaf)
  list_of_trees <- vector("list",rounds)
  offset <- numeric(nrow(X))
  active <- matrix(NA_real_,rounds,length(priors))
  for(j in seq_len(rounds)) {
    predicted <- nf_pu_cpp_logit_adjust(
      oof_prior_probability,offset,1,cap)
    gradient <- nf_pu_cpp_residual_targets(
      predicted,as.numeric(y),as.numeric(priors),3)
    target <- as.numeric(gradient$target)
    active[j,] <- as.numeric(gradient$active_negative)
    fit <- nf_cpp_fit(X,target,rep(1,nrow(X)),
      rep(1,ncol(X)),cfg,0L,1L,
      as.integer(seed+j*271L))
    list_of_trees[[j]] <- fit$trees
    offset <- offset+rate*as.numeric(nf_cpp_predict(
      fit$trees,X,1L,FALSE)[,1L])
  }
  list(trees=list_of_trees,active_prior_fraction=colMeans(active))
}

#' Positive-unlabeled residual trees guided by a fixed neural teacher
#'
#' The alternative to positive-versus-unlabeled RF classification is to
#' construct shallow *regression* trees that approximate negative gradients
#' of the nonnegative PU risk at the current neural score. All gradients
#' and logit corrections are computed in C++17. A strict empirical
#' holdout gate evaluates every declared class prior in each of two
#' disjoint validation halves; it also requires high teacher rank
#' concordance. A failed gate deploys the original two-restart neural
#' ensemble exactly. This does not make PU risk valid under SAR/SNAR.
#'
#' @param x Training predictors, matrix or data frame.
#' @param y Observed positive (1) and unlabeled (0) indicators.
#' @param priors Predeclared target prevalence sensitivity values.
#' @param seed Reproducible seed.
#' @param hidden Native C++ PU neural layer width.
#' @param epochs Neural training epochs.
#' @param rounds Additive residual tree stages.
#' @param trees_per_round Native C++ shallow trees per boosting stage.
#' @param max_depth Native CART depth for each residual tree.
#' @param learning_rate Small cumulative logit update factor.
#' @param max_logit_change Bound on the total logit correction.
#' @param min_gain Minimum PU-risk improvement for every prior and fold.
#' @param min_rank_correlation Minimum per-half rank concordance with teacher.
#' @param multipliers Predeclared residual correction strengths.
#' @return Fitted \code{nf_pu_forest}, with selected correction and
#'   matched neural-only predictor available through
#'   \code{nf_pu_matched_teacher}.
#' @export
nf_pu_residual_forest <- function(x,y,priors=c(.12,.25,.40),
   seed=42L,hidden=20L,epochs=60L,rounds=4L,
   trees_per_round=2L,max_depth=2L,learning_rate=.12,
   max_logit_change=.4,min_gain=.002,min_rank_correlation=.99,
   multipliers=c(.25,.5,1)) {
  if(!is.data.frame(x) && !is.matrix(x))stop("Invalid predictors")
  yy <- as.character(y)
  if(length(y)!=nrow(x) || anyNA(yy) || any(!yy %in% c("0","1")))
    stop("PU labels must match x and contain only 0/1 indicators")
  yy <- as.integer(yy)
  if(min(tabulate(yy+1L,nbins=2L))<30L)
    stop("Residual PU method needs at least 30 cases in each PU group")
  if(any(!is.finite(priors)) || !length(priors) ||
     any(priors<=0|priors>=1) ||
     length(rounds)!=1L || !is.finite(rounds) ||
     rounds<1L||rounds>20L||rounds!=as.integer(rounds)||
     length(trees_per_round)!=1L || !is.finite(trees_per_round)||
     trees_per_round<1L||trees_per_round>20L||
     trees_per_round!=as.integer(trees_per_round)||
     length(max_depth)!=1L || !is.finite(max_depth)||
     max_depth<1L||max_depth>4L||max_depth!=as.integer(max_depth)||
     length(learning_rate)!=1L||!is.finite(learning_rate)||
     learning_rate<=0||learning_rate>.5 ||
     length(max_logit_change)!=1L||!is.finite(max_logit_change)||
     max_logit_change<=0||max_logit_change>2 ||
     length(min_gain)!=1L||!is.finite(min_gain)||min_gain<0||
     length(min_rank_correlation)!=1L||!is.finite(min_rank_correlation)||
     min_rank_correlation<0||min_rank_correlation>1||
     !length(multipliers)||any(!is.finite(multipliers))||
     any(multipliers<=0|multipliers>1)||
     length(hidden)!=1L||hidden<1||hidden>256||
     length(epochs)!=1L||epochs<1||epochs>1500)
    stop("Invalid residual PU configuration")
  now <- function()proc.time()[["elapsed"]]
  initial <- now()
  split <- nf_split_indices(yy,"classification",.2,seed)
  enc <- nf_encoding(x[split$train,,drop=FALSE])
  Xtr <- enc$X
  Xv <- nf_encoding(x[split$validation,,drop=FALSE],
                          fit=enc$encoding)$X
  yt <- yy[split$train]
  yv <- yy[split$validation]
  nn <- nf_pu_stable_ensemble(Xtr,yt,priors,hidden,epochs,
                             seed,restarts=2L)
  teacher_val <- nf_pu_ensemble_predict(nn,Xv)
  enough <- min(tabulate(yv+1L,nbins=2L))>=6L
  if(enough) {
    h <- nf_split_indices(yv,"classification",.49,seed+811L)
    folds <- list(h$train,h$validation)
  } else folds <- list(seq_along(yv))
  base_risk <- nf_pu_stable_risk_grid(yv,teacher_val,priors,folds)
  best <- list(rounds=0L,multiplier=0,
               worst_risk=max(base_risk),rank=1)
  history <- data.frame(rounds=0L,multiplier=0,
        worst_pu_risk=max(base_risk),min_prior_half_gain=0,
        rank_min=1,eligible=TRUE)
  oof <- if(enough) nf_pu_stable_oof(
    Xtr,yt,priors,hidden,epochs,seed+30011L,restarts=2L) else NULL
  train_ess <- NULL
  min_leaf <- max(8L,as.integer(ceiling(.025*nrow(Xtr))))
  if(!is.null(oof)) {
    # Disjoint cross-fitting: no training row is scored by a network
    # that was trained on that row's PU label.
    train_ess <- nf_pu_residual_train(
      Xtr,yt,rowMeans(oof),priors,as.integer(rounds),
      as.integer(trees_per_round),as.integer(max_depth),min_leaf,
      learning_rate,max_logit_change,seed+4709L)
    val_offset <- numeric(nrow(Xv))
    for(j in seq_len(rounds)) {
      val_offset <- val_offset + learning_rate*as.numeric(
          nf_cpp_predict(train_ess$trees[[j]],Xv,1L,FALSE)[,1L])
      for(frac in multipliers) {
        pv <- nf_pu_cpp_logit_adjust(
          teacher_val,val_offset,frac,max_logit_change)
        grid <- nf_pu_stable_risk_grid(yv,pv,priors,folds)
        gains <- base_risk-grid
        rank_min <- min(vapply(folds,function(ix)
          suppressWarnings(stats::cor(
            teacher_val[ix],pv[ix],method="spearman")),
          numeric(1)))
        eligible <- enough && all(is.finite(grid)) &&
            all(gains>min_gain) && is.finite(rank_min) &&
            rank_min>=min_rank_correlation
        history <- rbind(history,data.frame(
          rounds=j,multiplier=frac,worst_pu_risk=max(grid),
          min_prior_half_gain=min(gains),rank_min=rank_min,
          eligible=eligible))
        if(eligible && max(grid)<best$worst_risk-1e-9) {
          best <- list(rounds=as.integer(j),
               multiplier=as.numeric(frac),worst_risk=max(grid),
               rank=rank_min)
        }
      }
    }
  }
  history$selected <- history$rounds==best$rounds &
                       history$multiplier==best$multiplier
  select_seconds <- now()-initial
  final_start <- now()
  full <- nf_encoding(x) # fit all imputation statistics on training only
  final_teacher <- nf_pu_stable_ensemble(
    full$X,yy,priors,hidden,epochs,seed+989L,restarts=2L)
  final_residual <- NULL
  if(best$rounds>0L) {
    final_oof <- nf_pu_stable_oof(full$X,yy,priors,hidden,epochs,
                               seed+14027L,restarts=2L)
    if(!is.null(final_oof)) {
      final_residual <- nf_pu_residual_train(
        full$X,yy,rowMeans(final_oof),priors,best$rounds,
        as.integer(trees_per_round),as.integer(max_depth),
        max(8L,as.integer(ceiling(.025*nrow(full$X)))),
        learning_rate,max_logit_change,seed+4709L)$trees
    }
  }
  final_seconds <- now()-final_start
  if(best$rounds>0L && is.null(final_residual)) {
    # Rare-sample refit failure: exact neural fallback, never half-fit.
    best <- list(rounds=0L,multiplier=0,
                 worst_risk=max(base_risk),rank=1)
    history$selected <- history$rounds==0L
  }
  out <- list(mode="pu_residual",levels=c("0","1"),
    priors=priors,alpha=1,networks=final_teacher,
    encoding=full$encoding,forest=NULL,
    residual_trees=final_residual,
    residual_rate=learning_rate,
    residual_multiplier=best$multiplier,
    residual_cap=max_logit_change,
    selected_rounds=best$rounds,
    selected_candidate=if(best$rounds==0L)
      "neural_fallback" else sprintf("residual_%d_%.2f",
                                     best$rounds,best$multiplier),
    validation_pu_risk=best$worst_risk,
    selection_history=history,
    hidden=hidden,epochs=epochs,
    timings=list(selection_seconds=select_seconds,
      final_fit_seconds=final_seconds,
      total_training_seconds=now()-initial))
  class(out) <- "nf_pu_forest"
  out
}
