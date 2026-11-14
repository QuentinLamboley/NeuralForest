# C++17 shallow non-negative positive-unlabeled (nnPU) learning.
# IMPORTANT: a case-control P/U sample is not automatically a random
# sample of the target population. Priors are scientific assumptions,
# not estimated from independent test labels.
nf_pu_risk <- function(y,p,prior) {
  y <- as.integer(y)
  p <- pmin(1-1e-10,pmax(1e-10,as.numeric(p)))
  ip <- which(y==1L); iu <- which(y==0L)
  if(!length(ip)||!length(iu)) return(Inf)
  pos <- -mean(log(p[ip]))
  neg <- -mean(log1p(-p[iu]))+prior*mean(log1p(-p[ip]))
  prior*pos+max(0,neg)
}

nf_pu_scale <- function(X,center=NULL,spread=NULL) {
  X <- as.matrix(X)
  if(is.null(center))center <- colMeans(X)
  if(is.null(spread)) {
    spread <- apply(X,2L,stats::sd)
    spread[!is.finite(spread)|spread<1e-7] <- 1
  }
  z <- sweep(sweep(X,2,center,"-"),2,spread,"/")
  # Preserve matrix dimensions: pmin/pmax with a scalar in the
  # first argument can silently return an ordinary vector in R.
  z[z>8] <- 8
  z[z< -8] <- -8
  list(X=z,center=center,spread=spread)
}

nf_pu_fit_networks <- function(X,y,priors,hidden,epochs,seed) {
  split <- nf_split_indices(y,"classification",0.15,seed+201L)
  scaled <- nf_pu_scale(X[split$train,,drop=FALSE])
  valid <- nf_pu_scale(X[split$validation,,drop=FALSE],
                       scaled$center,scaled$spread)$X
  networks <- lapply(seq_along(priors),function(i) {
    model <- nf_pu_cpp_fit(scaled$X,as.numeric(y[split$train]),
                valid,as.numeric(y[split$validation]),
                as.numeric(priors[i]),as.integer(hidden),
                as.integer(epochs),0.008,0.0004,
                as.integer(seed+i*107L))
    list(native=model,center=scaled$center,spread=scaled$spread,
         assumed_prior=priors[i])
  })
  list(networks=networks,priors=priors)
}

nf_pu_ensemble_predict <- function(ensemble,X) {
  pred <- do.call(cbind,lapply(ensemble$networks,function(network) {
    Z <- nf_pu_scale(X,network$center,network$spread)$X
    as.numeric(nf_pu_cpp_predict(network$native,Z))
  }))
  as.numeric(rowMeans(pred))
}

nf_pu_forest_weights <- function(y,posterior) {
  # Confirmed positives keep their sampling weight. Unlabeled examples
  # resembling positives are discounted rather than labelled negative
  # with full confidence. Clip prevents a complete loss of coverage.
  weights <- ifelse(y==1L,1,pmax(.15,1-.8*posterior))
  weights / mean(weights)
}

nf_pu_forest_fit <- function(X,y,posterior,n_trees,seed,weights=NULL) {
  if(is.null(weights)) weights <- nf_pu_forest_weights(y,posterior)
  if(length(weights)!=nrow(X)||any(!is.finite(weights))||any(weights<=0))
    stop("Invalid native forest sampling weights")
  config <- nf_control(n_trees=n_trees,min_leaf=2L,
     neural=FALSE,differentiable=FALSE,class_balance="none",
     sample_fraction=.8,replace=TRUE,n_threads=1L)
  profile <- nf_target_profile(y,"classification",X)
  ctrl <- nf_resolve_control(config,length(y),ncol(X),"classification",profile)
  nf_cpp_fit(X,as.numeric(y),weights,
             rep(1,ncol(X)),nf_native_cfg(ctrl),1L,2L,
             as.integer(seed+2089L))
}

nf_pu_forest_predict <- function(forest,X) {
  as.numeric(nf_cpp_predict(forest$trees,X,2L,FALSE)[,2L])
}

#' Positive-unlabeled neural-informed forest
#'
#' Runs a native C++ one-hidden-layer non-negative PU risk learner.
#' A mean prediction across externally declared class-prior assumptions
#' guides the forest's bootstrap weights for *unlabeled* cases. An outer
#' training-only holdout selects the forest/PU neural mixture via worst-case
#' PU risk. No true labels of unlabeled cases are inferred from a test set.
#' The risk estimator requires representative P and U samples (SCAR-like
#' sampling). Environmentally selected background can violate these
#' assumptions severely; evaluation on verified independent data is vital.
#' @param x Training predictors, a data frame or matrix.
#' @param y Binary labels, 1=confirmed positive and 0=unlabeled.
#' @param mode nnPU network only, or informed C++ random forest.
#' @param priors Prespecified class-prior sensitivity assumptions.
#' @param seed Training random seed.
#' @param hidden Width of the native C++ neural hidden layer.
#' @param epochs Maximum native Adam optimisation epochs.
#' @param n_trees Number of trees in the guided native forest.
#' @return Fitted model; predict(object,newdata,type="prob") gives two columns.
#' @export
nf_pu_forest <- function(x,y,mode=c("informed","neural"),
                         priors=c(.12,.25,.40),seed=42L,
                         hidden=20L,epochs=60L,n_trees=130L) {
  mode <- match.arg(mode)
  if(any(!is.finite(priors))||!length(priors)||
     any(priors<=0|priors>=1))
    stop("priors must be predefined values strictly between 0 and 1")
  if(!is.numeric(y)&&!is.factor(y))
    stop("PU labels must be numeric 0/1 or factor with levels 0/1")
  labels <- as.character(y)
  if(anyNA(labels)||any(!labels %in% c("0","1")))
    stop("PU labels must be 0 (unlabeled) or 1 (confirmed positive)")
  y <- as.integer(labels)
  if(sum(y==1L)<20||sum(y==0L)<20)
    stop("At least 20 confirmed positives and 20 unlabeled cases are required")
  if(!is.data.frame(x)&&!is.matrix(x))stop("Invalid predictors")
  if(nrow(x)!=length(y))stop("Predictors and PU labels disagree")
  if(length(hidden)!=1L||hidden<1||hidden>256||
     length(epochs)!=1L||epochs<1||epochs>1500||
     length(n_trees)!=1L||n_trees<1)
    stop("Invalid model size")
  now <- function()proc.time()[["elapsed"]]
  beginning <- now()
  split <- nf_split_indices(y,"classification",.2,seed)
  enc <- nf_encoding(x[split$train,,drop=FALSE])
  train <- enc$X
  val <- nf_encoding(x[split$validation,,drop=FALSE],
                     fit=enc$encoding)$X
  yt <- y[split$train]; yv <- y[split$validation]
  models <- nf_pu_fit_networks(train,yt,priors,hidden,epochs,seed)
  soft_val <- nf_pu_ensemble_predict(models,val)
  if(mode=="informed") {
    forest <- nf_pu_forest_fit(train,yt,
              nf_pu_ensemble_predict(models,train),n_trees,seed)
    hard_val <- nf_pu_forest_predict(forest,val)
    options <- c(0,.25,.5,.75,1)
    candidate <- vapply(options,function(alpha) {
      p <- alpha*soft_val+(1-alpha)*hard_val
      max(vapply(priors,function(prior)nf_pu_risk(yv,p,prior),0.0))
    },0.0)
    alpha <- options[which.min(candidate)]
    score <- min(candidate)
  } else {
    alpha <- 1
    score <- max(vapply(priors,function(prior)
                         nf_pu_risk(yv,soft_val,prior),0.0))
  }
  selection_seconds <- now()-beginning
  refit_start <- now()
  fullX <- nf_encoding(x,fit=enc$encoding)$X
  final <- nf_pu_fit_networks(fullX,y,priors,hidden,epochs,seed+989L)
  final_forest <- if(mode=="informed") {
    nf_pu_forest_fit(fullX,y,
      nf_pu_ensemble_predict(final,fullX),n_trees,seed+131L)
  } else NULL
  final_seconds <- now()-refit_start
  object <- list(mode=mode,levels=c("0","1"),priors=priors,
     alpha=alpha,forest=final_forest,networks=final,
     encoding=enc$encoding,validation_pu_risk=score,
     hidden=hidden,epochs=epochs,n_trees=n_trees,
     timings=list(selection_seconds=selection_seconds,
                  final_fit_seconds=final_seconds,
                  total_training_seconds=now()-beginning))
  class(object)<-"nf_pu_forest"
  object
}

#' @export
predict.nf_pu_forest <- function(object,newdata,
                type=c("prob","response","class"),...) {
  type <- match.arg(type)
  X <- nf_encoding(newdata,fit=object$encoding)$X
  p <- nf_pu_ensemble_predict(object$networks,X)
  if(!is.null(object$rank_map)) {
    # Batch-invariant, strictly monotone teacher calibration in native C++.
    p <- nf_pu_cpp_rank_apply(
      p,object$rank_map,1-object$alpha,object$fidelity_cap)
  } else if(!is.null(object$forest)) {
    pf <- nf_pu_forest_predict(object$forest,X)
    candidate <- object$alpha*p+(1-object$alpha)*pf
    if(!is.null(object$fidelity_cap)) {
      candidate <- p+pmax(-object$fidelity_cap,
                          pmin(object$fidelity_cap,candidate-p))
    }
    p <- candidate
  }
  p <- pmin(1-1e-8,pmax(1e-8,p))
  if(type=="class")return(ifelse(p>=.5,"1","0"))
  if(type=="response")return(p)
  out <- cbind(1-p,p)
  colnames(out) <- object$levels
  out
}
