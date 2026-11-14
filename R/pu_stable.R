# Stable PU guidance: native C++ neural ensembles plus conservative
# out-of-fold sampling and a holdout-only robust selection gate.
#
# PU risk is meaningful under representative P and U sampling, not under
# feature-dependent confirmation or spatially hard-selected backgrounds.
# These rules are engineering safety constraints, NOT formal guarantees.

nf_pu_stable_ensemble <- function(X,y,priors,hidden,epochs,seed,restarts=2L) {
  fits <- lapply(seq_len(restarts),function(k)
    nf_pu_fit_networks(X,y,priors,hidden,epochs,
                       as.integer(seed+3011L*(k-1L))))
  list(networks=unlist(lapply(fits,function(z)z$networks),
                       recursive=FALSE),
       priors=priors)
}

nf_pu_stable_matrix <- function(models,X) {
  ans <- vapply(models$networks,function(z) {
    scaled <- nf_pu_scale(X,z$center,z$spread)$X
    as.numeric(nf_pu_cpp_predict(z$native,scaled))
  },numeric(nrow(X)))
  matrix(ans,nrow=nrow(X),ncol=length(models$networks))
}

nf_pu_stable_oof <- function(X,y,priors,hidden,epochs,seed,restarts=2L) {
  # Small class support makes cross-fit PU risk very unstable. In that
  # circumstance skip guided trees instead of using in-sample predictions.
  if(min(tabulate(y+1L,nbins=2L))<50L) return(NULL)
  half <- nf_split_indices(y,"classification",.49,seed+73L)
  chunks <- list(half$train,half$validation)
  n <- nrow(X)
  out <- matrix(NA_real_,n,as.integer(restarts)*length(priors))
  for(j in seq_along(chunks)) {
    hold <- chunks[[j]]
    training <- setdiff(seq_len(n),hold)
    if(min(tabulate(y[training]+1L,nbins=2L))<20L)
      return(NULL)
    model <- nf_pu_stable_ensemble(
      X[training,,drop=FALSE],y[training],
      priors,hidden,epochs,seed+2111L*j,restarts=restarts)
    out[hold,] <- nf_pu_stable_matrix(model,X[hold,,drop=FALSE])
  }
  if(any(!is.finite(out)))stop("Incomplete out-of-fold PU predictions")
  out
}

nf_pu_stable_risks <- function(y,p,priors,folds) {
  vapply(folds,function(rows) {
    max(vapply(priors,function(prior)
      nf_pu_risk(y[rows],p[rows],prior),numeric(1)))
  },numeric(1))
}

# Per-prior, per-fold risks avoid hiding damage to one assumed prevalence.
nf_pu_stable_risk_grid <- function(y,p,priors,folds) {
  matrix(unlist(lapply(folds,function(rows)
    vapply(priors,function(prior)
      nf_pu_risk(y[rows],p[rows],prior),numeric(1)))),
    nrow=length(priors),ncol=length(folds))
}

# Fit the monotone mapping on one holdout half and score on the other.
nf_pu_rank_oof <- function(teacher,forest,folds,student_fraction,cap) {
  out <- rep(NA_real_,length(teacher))
  for(rows in folds) {
    calibration <- setdiff(seq_along(teacher),rows)
    if(length(calibration)<2L)stop("Rank calibration needs disjoint halves")
    map <- nf_pu_cpp_rank_fit(teacher[calibration],forest[calibration])
    out[rows] <- nf_pu_cpp_rank_apply(
      teacher[rows],map,student_fraction,cap)
  }
  if(any(!is.finite(out)))stop("Incomplete rank-preserving holdout")
  out
}

#' Stability-constrained PU-guided random forest
#'
#' Two independent C++ neural nnPU fits per prior assess sensitivity to
#' both the prior and the random initialization. Held-out cross-fitted
#' teacher probabilities control the native C++ tree-sampling weights.
#' The sampler retains the positive/unlabeled class mass and imposes an
#' effective-sample-size lower bound for the unlabeled group.
#'
#' The outer PU holdout is split into two independent scoring halves.
#' A candidate forest is accepted only if its worst-case PU risk improves
#' on the neural-only teacher by at least `min_gain` on BOTH halves,
#' for the prespecified class-prior sensitivity grid. Otherwise the
#' algorithm falls back to the teacher (no forest is fitted at refit).
#' External/test labels are NEVER used to select a candidate.
#'
#' @param x Numeric or categorical training features.
#' @param y 1=confirmed positive; 0=unlabeled, not a verified negative.
#' @param priors Predeclared plausible class priors (not test-estimated).
#' @param seed Reproducible random seed.
#' @param hidden Width of the native C++ PU neural hidden layer.
#' @param epochs Maximum neural training epochs.
#' @param n_trees Number of C++ decision trees per candidate.
#' @param min_gain Minimum worst-fold PU risk improvement needed to
#'   select any forest instead of the neural-only fallback.
#' @param strengths Prespecified bounded bootstrap guidance strengths.
#' @param min_ess Minimum unlabeled effective sample size fraction.
#' @param uncertainty_limit Maximum acceptable spread in PU network
#'   probabilities before guidance is suppressed.
#' @return A fitted nf_pu_forest with selection history and diagnostics.
#' @export
nf_pu_stable_forest <- function(
  x,y,priors=c(.12,.25,.40),seed=42L,hidden=20L,
  epochs=60L,n_trees=130L,min_gain=.004,
  strengths=c(.25,.55),min_ess=.96,uncertainty_limit=.35,
  teacher_restarts=2L,fidelity_cap=NULL,min_rank_correlation=0,
  rank_lock=FALSE) {
  if(!is.numeric(y) && !is.factor(y))
    stop("PU labels must be numeric 0/1 or factor 0/1")
  raw_y <- as.character(y)
  if(anyNA(raw_y)||any(!raw_y %in% c("0","1")))
    stop("PU labels must be 0 or 1")
  y <- as.integer(raw_y)
  if(!is.data.frame(x)&&!is.matrix(x))
    stop("x must be a data.frame or matrix")
  if(nrow(x)!=length(y) ||
     min(tabulate(y+1L,nbins=2L))<20L)
    stop("Need matched predictors and at least 20 observations per PU group")
  if(length(priors)<1L||any(!is.finite(priors))||
     any(priors<=0|priors>=1))
    stop("Priors must be strictly between zero and one")
  if(length(strengths)<1L||any(!is.finite(strengths))||
     any(strengths<=0|strengths>1))
    stop("Guidance strengths must be in (0,1]")
  if(length(min_gain)!=1L||!is.finite(min_gain)||min_gain<0||
     length(min_ess)!=1L||!is.finite(min_ess)||
     min_ess<.5||min_ess>1||
     length(uncertainty_limit)!=1L||!is.finite(uncertainty_limit)||
     uncertainty_limit<=0||uncertainty_limit>1||
     length(hidden)!=1L||hidden<1||hidden>256||
     length(epochs)!=1L||epochs<1||epochs>1500||
     length(n_trees)!=1L||n_trees<1||
     length(teacher_restarts)!=1L||
       !(teacher_restarts %in% c(1L,2L))||
     length(min_rank_correlation)!=1L||
       !is.finite(min_rank_correlation)||
       min_rank_correlation<0||min_rank_correlation>1||
     length(rank_lock)!=1L||!is.logical(rank_lock)||is.na(rank_lock)||
     (rank_lock && is.null(fidelity_cap))||
     (!is.null(fidelity_cap) &&
       (length(fidelity_cap)!=1L||!is.finite(fidelity_cap)||
        fidelity_cap<=0||fidelity_cap>=1)))
    stop("Invalid stable PU model controls")
  now <- function()proc.time()[["elapsed"]]
  beginning <- now()
  split <- nf_split_indices(y,"classification",.2,seed)
  enc <- nf_encoding(x[split$train,,drop=FALSE])
  training <- enc$X
  validation <- nf_encoding(x[split$validation,,drop=FALSE],
                            fit=enc$encoding)$X
  yt <- y[split$train]
  yv <- y[split$validation]
  teacher <- nf_pu_stable_ensemble(training,yt,priors,hidden,epochs,
                                  seed,restarts=teacher_restarts)
  teacher_val <- nf_pu_ensemble_predict(teacher,validation)
  use_two_halves <- min(tabulate(yv+1L,nbins=2L))>=6L
  if(use_two_halves) {
    h <- nf_split_indices(yv,"classification",.49,seed+811L)
    folds <- list(h$train,h$validation)
  } else folds <- list(seq_along(yv))
  base_risk <- nf_pu_stable_risks(yv,teacher_val,priors,folds)
  base_grid <- nf_pu_stable_risk_grid(yv,teacher_val,priors,folds)
  selected <- list(name="neural_fallback",strength=NA_real_,
                   blend=1,risks=base_risk,rank_map=NULL)
  history <- data.frame(
    candidate="neural_fallback",strength=NA_real_,blend=1,
    worst_risk=max(base_risk),worst_gain=0,
    eligible=TRUE,stringsAsFactors=FALSE)
  crossfit <- if(use_two_halves) nf_pu_stable_oof(
    training,yt,priors,hidden,epochs,seed+119L,
    restarts=teacher_restarts) else NULL
  consider <- function(name,p,strength,blend) {
    mapping <- NULL
    if(rank_lock) {
      # Distill the original forest probabilities, not their numerical
      # reconstruction from a convex mixture (which could leave [0,1]).
      mapping <- nf_pu_cpp_rank_fit(teacher_val,p)
      p <- nf_pu_rank_oof(teacher_val,p,folds,
                          1-blend,fidelity_cap)
    } else if(blend>0) {
      p <- blend*teacher_val+(1-blend)*p
    }
    # The fidelity-limited mode only judges predictions it could actually
    # deploy. Do not select on an unconstrained validation prediction and
    # clip only at inference.
    if(!is.null(fidelity_cap))
      p <- teacher_val+pmax(-fidelity_cap,
               pmin(fidelity_cap,p-teacher_val))
    risks <- nf_pu_stable_risks(yv,p,priors,folds)
    risk_grid <- nf_pu_stable_risk_grid(yv,p,priors,folds)
    correlations <- vapply(folds,function(rows) {
      if(min_rank_correlation<=0)return(1.0)
      if(stats::sd(p[rows])<1e-12 ||
         stats::sd(teacher_val[rows])<1e-12)return(NA_real_)
      suppressWarnings(stats::cor(
        teacher_val[rows],p[rows],method="spearman"))
    },numeric(1))
    improvement <- base_risk-risks
    eligible <- use_two_halves &&
      all(is.finite(risk_grid)) &&
      all(base_grid-risk_grid>min_gain) &&
      all(is.finite(correlations)) &&
      all(correlations>=min_rank_correlation)
    history <<- rbind(history,data.frame(
      candidate=name,strength=strength,blend=blend,
      worst_risk=max(risks),worst_gain=min(improvement),
      eligible=eligible,stringsAsFactors=FALSE))
    if(eligible && max(risks)<max(selected$risks)) {
      selected <<- list(name=name,strength=strength,
                        blend=blend,risks=risks,rank_map=mapping)
    }
  }
  if(use_two_halves) {
    # Baseline controls tree complexity and random seeds; guided variants
    # differ ONLY in their *training-only* bootstrap weights.
    levels <- c(0,if(!is.null(crossfit)) strengths else numeric())
    for(j in seq_along(levels)) {
      strength <- levels[[j]]
      native_weights <- if(strength==0) rep(1,nrow(training)) else
        as.numeric(nf_pu_cpp_stable_weights(
          crossfit,as.numeric(yt),strength,.65,
          uncertainty_limit,min_ess)$weights)
      forest <- nf_pu_forest_fit(training,yt,NULL,n_trees,
                 seed+4709L,weights=native_weights)
      fp <- nf_pu_forest_predict(forest,validation)
      label <- if(strength==0) "plain" else sprintf("guided_%g",strength)
      consider(label,fp,strength,0)
      consider(paste0(label,"_blend50"),
               fp,strength,.5)
    }
  }
  history$chosen <- history$candidate==selected$name
  select_time <- now()-beginning
  final_start <- now()
  # Re-estimate the final imputation/encoding using ONLY the full training
  # sample; the two independent test cohorts are never consulted here.
  # In fidelity mode use exactly the same encoding as the legacy nnPU
  # teacher: a neural-only fallback reproduces its probabilities.
  full_enc <- if(is.null(fidelity_cap)) nf_encoding(x) else
    list(X=nf_encoding(x,fit=enc$encoding)$X,
         encoding=enc$encoding)
  fullX <- full_enc$X
  final_teacher <- nf_pu_stable_ensemble(
    fullX,y,priors,hidden,epochs,seed+989L,
    restarts=teacher_restarts)
  final_forest <- NULL
  diagnostics <- list(ess_fraction=1,guided_fraction=0,
                      mean_confidence=0,shrink_factor=1,
                      crossfit_used=!is.null(crossfit))
  if(is.finite(selected$strength) && !rank_lock) {
    if(selected$strength>0) {
      posterior <- nf_pu_stable_oof(fullX,y,priors,hidden,epochs,
                                   seed+14027L,restarts=teacher_restarts)
      if(is.null(posterior))
        stop("Selected guidance cannot be safely cross-fitted on full data")
      guidance <- nf_pu_cpp_stable_weights(
        posterior,as.numeric(y),selected$strength,.65,
        uncertainty_limit,min_ess)
      weights <- as.numeric(guidance$weights)
      diagnostics <- list(
        ess_fraction=as.numeric(guidance$ess_fraction),
        guided_fraction=as.numeric(guidance$guided_fraction),
        mean_confidence=as.numeric(guidance$mean_confidence),
        shrink_factor=as.numeric(guidance$shrink_factor),
        crossfit_used=TRUE)
    } else weights <- rep(1,nrow(fullX))
    final_forest <- nf_pu_forest_fit(
      fullX,y,NULL,n_trees,seed+4709L,weights=weights)
  }
  final_time <- now()-final_start
  out <- list(
    mode="stable",levels=c("0","1"),priors=priors,
    alpha=selected$blend,forest=final_forest,
    networks=final_teacher,encoding=full_enc$encoding,
    validation_pu_risk=max(selected$risks),
    selected_candidate=selected$name,
    selected_strength=selected$strength,
    fidelity_cap=fidelity_cap,
    min_rank_correlation=min_rank_correlation,
    rank_lock=rank_lock,rank_map=selected$rank_map,
    selection_history=history,stability_diagnostics=diagnostics,
    hidden=hidden,epochs=epochs,n_trees=n_trees,
    timings=list(selection_seconds=select_time,
                 final_fit_seconds=final_time,
                 total_training_seconds=now()-beginning))
  class(out) <- "nf_pu_forest"
  out
}

#' Fidelity-constrained native PU forest with exact neural fallback
#'
#' Builds on the stable PU forest but constrains every selected forest
#' to change the original native nnPU teacher's predictions by at most
#' `fidelity_cap` in absolute probability, both on validation and on
#' future observations. Selection additionally requires high Spearman
#' concordance separately on two validation halves. This bound limits
#' probability drift but CANNOT guarantee Boyce, AUC or AP non-inferiority,
#' especially for observations whose teacher scores are nearly tied.
#' A selected native C++ rank map is a strictly increasing function
#' of teacher probability, so it cannot invert ranking on unseen cohorts.
#' It can affect calibration but cannot improve rank-based discrimination.
#' Cross-fitted assessment uses opposite halves of the PU holdout.
#' If no candidate passes the gate, the fitted model reproduces the
#' standalone `nf_pu_forest(..., mode="neural")` exactly.
#' @inheritParams nf_pu_stable_forest
#' @param fidelity_cap Maximum absolute difference from nnPU teacher.
#' @param min_rank_correlation Minimum teacher/forest Spearman rank
#'   correlation separately on each PU holdout half.
#' @return A fitted `nf_pu_forest`.
#' @export
nf_pu_safe_forest <- function(
   x,y,priors=c(.12,.25,.40),seed=42L,hidden=20L,
   epochs=60L,n_trees=130L,min_gain=.004,
   strengths=c(.25,.55),min_ess=.96,uncertainty_limit=.35,
   fidelity_cap=.02,min_rank_correlation=.995) {
  nf_pu_stable_forest(
    x=x,y=y,priors=priors,seed=seed,hidden=hidden,
    epochs=epochs,n_trees=n_trees,min_gain=min_gain,
    strengths=strengths,min_ess=min_ess,
    uncertainty_limit=uncertainty_limit,
    teacher_restarts=1L,rank_lock=TRUE,
    fidelity_cap=fidelity_cap,
    min_rank_correlation=min_rank_correlation)
}
