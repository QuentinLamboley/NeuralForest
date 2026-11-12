#' External permutation importance
#'
#' Measures the decrease in a chosen metric after each predictor is shuffled.
#' Supply an untouched evaluation set rather than the training data for
#' interpretable, approximately unbiased importance estimates.
#' @param object A fitted neural_forest.
#' @param x Evaluation features with the training schema.
#' @param y Evaluation targets.
#' @param metric Metric to assess (defaults to the model selection metric).
#' @param repeats Number of random shuffles per variable.
#' @param seed Random seed.
#' @param groups Optional groups: permute only within each group.
#' @return Data frame ranked by average metric deterioration.
#' @export
nf_permutation_importance <- function(object,x,y,metric=object$metric,
                                      repeats=5L,seed=42L,groups=NULL) {
  stopifnot(inherits(object,"neural_forest"))
  if (length(y)!=nrow(x) || !is.numeric(repeats) || repeats<1)
    stop("Invalid evaluation rows or repeats")
  task <- object$task
  type <- if(task=="classification") "prob" else "response"
  d <- as.data.frame(x)
  names(d) <- make.unique(make.names(names(d)))
  if (!is.null(groups) && (length(groups)!=nrow(d)||anyNA(groups)))
    stop("groups must match evaluation rows and have no missing values")
  eval <- function(data) {
    p <- predict(object,data,type=type)
    as.numeric(nf_metric(y,p,metric=metric,task=task))
  }
  baseline <- eval(d)
  if (!is.finite(baseline)) stop("Baseline metric is undefined")
  direction <- nf_metric_direction(metric)
  set.seed(seed)
  impact <- matrix(NA_real_,nrow=length(d),ncol=as.integer(repeats))
  for (j in seq_along(d)) {
    for (r in seq_len(repeats)) {
      altered <- d
      z <- altered[[j]]
      if (is.null(groups)) z <- sample(z,length(z))
      else {
        for (g in split(seq_along(z),groups))
          if (length(g)>1) z[g] <- sample(z[g],length(g))
      }
      altered[[j]] <- z
      newscore <- eval(altered)
      impact[j,r] <- if (direction=="maximize")
        baseline-newscore else newscore-baseline
    }
  }
  ans <- data.frame(feature=names(d),mean_degradation=rowMeans(impact),
                    sd_degradation=if(repeats>1)
                      apply(impact,1,stats::sd) else rep(NA_real_,ncol(d)),
                    baseline=baseline,metric=metric)
  ans <- ans[order(ans$mean_degradation,decreasing=TRUE),]
  rownames(ans) <- NULL
  ans
}

#' Ensemble disagreement and predictive uncertainty proxies
#'
#' These are **un-calibrated** diagnostics. Standard deviations of tree
#' predictions and vote disagreement are NOT confidence intervals.
#' @param object A fitted neural_forest.
#' @param newdata Prediction features.
#' @param measure One of sd, entropy, margin, or vote.
#' @return Numeric vector or matrix with prediction uncertainty proxies.
#' @export
nf_uncertainty <- function(object,newdata,measure=c("sd","entropy","margin","vote")) {
  stopifnot(inherits(object,"neural_forest"))
  measure <- match.arg(measure)
  if (object$task=="regression" && measure!="sd")
    stop("Regression currently supports measure='sd' only")
  raw <- predict(object,newdata,type="trees")
  if (object$task=="regression")
    return(apply(raw,1,stats::sd))
  n <- dim(raw)[1]; K <- dim(raw)[2]; T <- dim(raw)[3]
  if (measure=="sd") {
    out <- matrix(0,n,K,dimnames=list(NULL,object$levels))
    if(T>1) for(k in seq_len(K))
      out[,k] <- apply(raw[,k,,drop=FALSE],1,stats::sd)
    return(out)
  }
  prob <- predict(object,newdata,type="prob")
  if(measure=="entropy")
    return(-rowSums(prob*log(pmax(prob,1e-12)))/log(K))
  if(measure=="margin") {
    ordered <- t(apply(prob,1,sort,decreasing=TRUE))
    return(1-(ordered[,1]-ordered[,2]))
  }
  votes <- matrix(0,n,K)
  for(t in seq_len(T)) {
    m <- matrix(raw[,,t],nrow=n,ncol=K)
    k <- max.col(m,ties.method="first")
    votes[cbind(seq_len(n),k)] <- votes[cbind(seq_len(n),k)] + 1
  }
  1-apply(votes,1,max)/T
}
