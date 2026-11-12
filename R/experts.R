# Ridge-regularized linear experts are fitted on the inner training fold.
# A holdout selects whether they complement the neural forest.
# No external observations are used for expert fitting or selection.

nf_expert_fit <- function(X,y,task,K,penalty=0.1) {
  X <- as.matrix(X)
  center <- colMeans(X)
  spread <- apply(X,2L,stats::sd)
  spread[!is.finite(spread) | spread < 1e-9] <- 1
  Z <- sweep(sweep(X,2L,center,"-"),2L,spread,"/")
  Z[Z > 8] <- 8
  Z[Z < -8] <- -8
  Z <- cbind(1,Z)
  if (task=="regression") {
    p <- ncol(Z)
    gram <- crossprod(Z)/nrow(Z)
    penalty_matrix <- diag(c(0,rep(penalty,p-1L)),p)
    coef <- tryCatch(solve(gram+penalty_matrix,
                           crossprod(Z,as.numeric(y))/nrow(Z)),
                     error=function(e) qr.solve(gram+penalty_matrix+diag(1e-7,p),
                                                crossprod(Z,as.numeric(y))/nrow(Z)))
    return(list(coef=matrix(coef,ncol=1),center=center,scale=spread,
                task=task,K=1L,penalty=penalty))
  }
  fit_class <- function(label) {
    response <- as.numeric(y==label)
    b0 <- c(stats::qlogis(pmin(.999,pmax(.001,mean(response)))),
            rep(0,ncol(Z)-1L))
    evaluate <- function(beta) {
      eta <- as.vector(Z%*%beta)
      sum(pmax(eta,0)+log1p(exp(-abs(eta)))-response*eta)/length(response)+
        penalty*sum(beta[-1L]^2)/2
    }
    derivative <- function(beta) {
      eta <- as.vector(Z%*%beta)
      grad <- as.vector(crossprod(Z,stats::plogis(eta)-response))/nrow(Z)
      grad[-1L] <- grad[-1L]+penalty*beta[-1L]
      grad
    }
    out <- stats::optim(b0,evaluate,derivative,method="L-BFGS-B",
                        control=list(maxit=120L,factr=5e7))
    out$par
  }
  # Binary classes need one calibrated logit rather than two separately
  # normalized one-vs-rest probabilities.
  selected <- if(K==2L) 1L else 0:(K-1L)
  coef <- vapply(selected,fit_class,numeric(ncol(Z)))
  if(is.null(dim(coef))) coef <- matrix(coef,ncol=1L)
  list(coef=coef,center=center,scale=spread,
       task=task,K=K,penalty=penalty)
}

nf_expert_predict <- function(model,X) {
  Z <- sweep(sweep(as.matrix(X),2L,model$center,"-"),
             2L,model$scale,"/")
  Z[Z > 8] <- 8
  Z[Z < -8] <- -8
  Z <- cbind(1,Z)
  p <- Z%*%model$coef
  if(model$task=="regression") return(matrix(p,ncol=1L))
  if(model$K==2L) {
    p1 <- stats::plogis(p[,1L])
    return(cbind(1-p1,p1))
  }
  raw <- stats::plogis(p)
  raw / pmax(rowSums(raw),1e-10)
}

nf_prior_probabilities <- function(y,K,task,n) {
  if (task=="regression") return(matrix(mean(y),n,1L))
  prior <- (tabulate(y+1L,nbins=K)+0.5)/(length(y)+0.5*K)
  matrix(rep(prior,each=n),nrow=n,ncol=K)
}

nf_expert_selection <- function(base,Xtr,ytr,Xval,yval,metric,task,K,labels,ctrl) {
  minimize <- nf_metric_direction(metric)=="minimize"
  best_score <- nf_objective(yval,base,metric,task,labels)
  if (!is.finite(best_score))
    best_score <- if(minimize) Inf else -Inf
  selected <- list(type="forest",weight=0,score=best_score,
                   penalty=NA_real_,model=NULL,prior=NULL)
  history <- data.frame(candidate="forest",blend=0,
                        score=best_score,penalty=NA_real_)
  update <- function(p,kind,alpha,penalty=NA_real_,model=NULL) {
    if(task=="classification") colnames(p) <- labels
    score <- nf_objective(yval,p,metric,task,labels)
    history <<- rbind(history,data.frame(candidate=kind,blend=alpha,
                                        score=score,penalty=penalty))
    threshold <- ctrl$expert_min_gain
    better <- is.finite(score) &&
      (if(minimize) score < selected$score-threshold else
         score > selected$score+threshold)
    if(better) selected <<- list(type=kind,weight=alpha,score=score,
                                 penalty=penalty,model=model,prior=NULL)
  }
  prior <- nf_prior_probabilities(ytr,K,task,nrow(Xval))
  update(prior,"prior",1)
  for(lambda in ctrl$linear_penalties) {
    expert <- tryCatch(nf_expert_fit(Xtr,ytr,task,K,lambda),
                       error=function(e) NULL)
    if(is.null(expert)) next
    pv <- nf_expert_predict(expert,Xval)
    for(alpha in ctrl$expert_blend) {
      if(alpha==0) next
      pred <- (1-alpha)*base+alpha*pv
      update(pred,"linear",alpha,lambda,expert)
    }
  }
  selected$candidates <- history
  selected
}
