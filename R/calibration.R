# Binary probability calibration using a regularized OOB Platt map.
# Fits rely on out-of-bag predictions from the inner training partition.

nf_platt_fit <- function(oob,y,penalty=0.1) {
  if(ncol(oob)!=2L) return(NULL)
  prob <- as.numeric(oob[,2L])
  ok <- is.finite(prob) & is.finite(y)
  if(sum(ok)<30L || sum(y[ok]==1L)<4L || sum(y[ok]==0L)<4L)
    return(NULL)
  p <- pmin(1-1e-6,pmax(1e-6,prob[ok]))
  z <- pmax(-12,pmin(12,stats::qlogis(p)))
  truth <- as.numeric(y[ok]==1L)
  objective <- function(par) {
    eta <- par[1L]+par[2L]*z
    mean(pmax(eta,0)+log1p(exp(-abs(eta)))-truth*eta)+
      penalty*(par[1L]^2+(par[2L]-1)^2)/2
  }
  derivative <- function(par) {
    eta <- par[1L]+par[2L]*z
    e <- stats::plogis(eta)-truth
    c(mean(e)+penalty*par[1L],
      mean(e*z)+penalty*(par[2L]-1))
  }
  fit <- tryCatch(stats::optim(c(0,1),objective,derivative,
                 method="BFGS",control=list(maxit=100L,reltol=1e-8)),
                  error=function(e) NULL)
  if(is.null(fit) || any(!is.finite(fit$par)) ||
     fit$par[2L]<=0 || fit$par[2L]>20) return(NULL)
  list(intercept=as.numeric(fit$par[1L]),slope=as.numeric(fit$par[2L]),
       penalty=penalty,oob_n=sum(ok),oob_positives=sum(truth))
}

nf_platt_apply <- function(p,calibration) {
  if(is.null(calibration)) return(p)
  if(ncol(p)!=2L) stop("Platt scaling requires binary probabilities")
  z <- stats::qlogis(pmin(1-1e-6,pmax(1e-6,p[,2L])))
  p1 <- stats::plogis(calibration$intercept+calibration$slope*z)
  result <- cbind(1-p1,p1)
  colnames(result) <- colnames(p)
  result
}

nf_platt_select <- function(native,ytr,pv,yval,metric,labels,ctrl) {
  if(!metric %in% c("brier","logloss","macro_logloss","ece","mce"))
    return(NULL)
  best_score <- nf_objective(yval,pv,metric,"classification",labels)
  if(!is.finite(best_score)) return(NULL)
  chosen <- NULL
  candidates <- data.frame(penalty=numeric(),score=numeric())
  for(lambda in ctrl$calibration_penalties) {
    model <- nf_platt_fit(native$oob_prediction,ytr,lambda)
    if(is.null(model)) next
    calibrated <- nf_platt_apply(pv,model)
    score <- nf_objective(yval,calibrated,metric,"classification",labels)
    candidates <- rbind(candidates,data.frame(penalty=lambda,score=score))
    if(is.finite(score) &&
       score<best_score-ctrl$calibration_min_gain) {
      best_score <- score
      chosen <- model
    }
  }
  if(is.null(chosen)) return(NULL)
  chosen$selected_score <- best_score
  chosen$candidates <- candidates
  chosen
}
