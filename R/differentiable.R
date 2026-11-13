# Differentiable routing trees. The parameters of every internal split,
# including its oblique projection and threshold, receive gradients.
# The objective is a differentiable surrogate when the chosen model
# selection metric has no useful derivative.

nf_soft_surrogate <- function(metric, task, choice="auto") {
  if (!identical(choice,"auto")) return(choice)
  if (task=="regression") return("mse")
  if (metric %in% c("pr_auc","macro_pr_auc")) return("ap")
  if (metric %in% c("auc","macro_auc","boyce","ks","roc_gini","lift")) return("auc")
  if (metric %in% c("brier","ece","mce")) return("brier")
  if (metric %in% c("logloss","macro_logloss")) return("logloss")
  "focal"
}

nf_soft_scale <- function(X, center=NULL, scale=NULL) {
  X <- as.matrix(X)
  if (is.null(center)) center <- colMeans(X)
  if (is.null(scale)) {
    scale <- apply(X,2,stats::sd)
    scale[!is.finite(scale)|scale<1e-8] <- 1
  }
  z <- sweep(sweep(X,2,center,"-"),2,scale,"/")
  z[z>10] <- 10
  z[z< -10] <- -10
  list(X=z,center=center,scale=scale)
}

nf_soft_train <- function(Xtr,ytr,Xval,yval,task,K,ctrl,metric,seed) {
  scaled <- nf_soft_scale(Xtr)
  val <- nf_soft_scale(Xval,scaled$center,scaled$scale)$X
  yc <- if(task=="regression") mean(ytr) else 0
  ys <- if(task=="regression") stats::sd(ytr) else 1
  if (!is.finite(ys)||ys<1e-8) ys <- 1
  config <- unclass(ctrl)
  config$soft_loss <- nf_soft_surrogate(metric,task,ctrl$soft_loss)
  if(task=="regression") {
    config$soft_loss <- "mse"
    k <- nf_soft_fit(scaled$X,as.numeric((ytr-yc)/ys),val,
                     as.numeric((yval-yc)/ys),config,as.integer(seed),FALSE)
    models <- list(k)
  } else if (K==2L) {
    # A binary sigmoid already models both complementary class outcomes.
    # Avoid fitting two independent OVR trees and renormalizing their outputs.
    models <- list(nf_soft_fit(
      scaled$X,as.numeric(ytr==1L),val,as.numeric(yval==1L),
      config,as.integer(seed+997L),TRUE))
  } else {
    models <- lapply(seq_len(K),function(label) {
      nf_soft_fit(scaled$X,as.numeric(ytr==label-1L),val,
                  as.numeric(yval==label-1L),config,
                  as.integer(seed+label*997L),TRUE)
    })
  }
  list(models=models,center=scaled$center,scale=scaled$scale,
       y_center=yc,y_scale=ys,task=task,K=K,
       surrogate=config$soft_loss)
}

nf_soft_probabilities <- function(model,X) {
  Z <- nf_soft_scale(X,model$center,model$scale)$X
  if(model$task=="classification" && model$K==2L) {
    p1 <- as.numeric(nf_soft_predict(model$models[[1L]],Z))
    p1 <- pmin(1-1e-8,pmax(1e-8,p1))
    return(cbind(1-p1,p1))
  }
  raw <- vapply(model$models,
                function(m) nf_soft_predict(m,Z),numeric(nrow(Z)))
  raw <- matrix(raw,nrow=nrow(Z),ncol=length(model$models))
  if(model$task=="regression")
    return(matrix(raw[,1]*model$y_scale+model$y_center,ncol=1))
  raw[raw<1e-8] <- 1e-8
  raw[raw>1-1e-8] <- 1-1e-8
  raw / pmax(rowSums(raw),1e-8)
}

nf_soft_select_blend <- function(hard,soft,yval,metric,task,labels,control) {
  candidates <- if(is.numeric(control$soft_blend))
    control$soft_blend else c(0,0.15,0.3,0.5,0.7,0.85,1)
  direction <- nf_metric_direction(metric)
  scores <- vapply(candidates,function(alpha) {
    p <- (1-alpha)*hard+alpha*soft
    if(task=="classification") colnames(p) <- labels
    nf_objective(yval,p,metric,task,labels)
  },numeric(1))
  valid <- which(is.finite(scores))
  if(!length(valid)) return(list(weight=0,score=NA_real_,candidates=candidates,scores=scores))
  winner <- if(direction=="maximize") valid[which.max(scores[valid])]
  else valid[which.min(scores[valid])]
  list(weight=candidates[winner],score=scores[winner],
       candidates=candidates,scores=scores)
}
