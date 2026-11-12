# Metric registry: metrics have explicit optimization directions. Some
# metrics are undefined on folds lacking positive or negative examples.
nf_metric_direction <- function(metric) {
  smaller <- c("logloss","macro_logloss","brier","mse","rmse","mae",
      "medae","mape","smape","wape","rmsle","msle","huber","logcosh",
      "pinball","max_error","mase","ece","mce","fpr","fnr","abs_bias")
  if (metric %in% smaller) "minimize" else "maximize"
}

#' Supported optimization and evaluation metrics
#' @param task classification, regression, or all.
#' @return Data frame of metric identifiers and optimization directions.
#' @export
nf_metrics <- function(task=c("all","classification","regression")) {
  task <- match.arg(task)
  cls <- c("accuracy","balanced_accuracy","auc","macro_auc","pr_auc",
    "macro_pr_auc","logloss","macro_logloss","brier","precision",
    "recall","sensitivity","specificity","f1","fbeta","macro_f1",
    "macro_precision","macro_recall","mcc","kappa","gmean",
    "jaccard","npv","fpr","fnr","youden","ks","roc_gini",
    "lift","top_k_accuracy","ece","mce","boyce")
  reg <- c("r2","mse","rmse","mae","medae","mape","smape","wape",
    "rmsle","msle","huber","logcosh","pinball","max_error",
    "explained_variance","nse","pearson","spearman","kendall",
    "ccc","rpd","rpiq","abs_bias","mase")
  z <- rbind(
    data.frame(task="classification",metric=cls),
    data.frame(task="regression",metric=reg))
  if (task!="all") z <- z[z$task==task,,drop=FALSE]
  z$direction <- vapply(z$metric,nf_metric_direction,character(1))
  rownames(z) <- NULL
  z
}

nf_safe_div <- function(x,y) ifelse(abs(y)<1e-14,NA_real_,x/y)
nf_binary_auc <- function(y,p) {
  pos <- sum(y==1); neg <- sum(y==0)
  if (!pos || !neg) return(NA_real_)
  ranks <- rank(p,ties.method="average")
  (sum(ranks[y==1]) - pos*(pos+1)/2)/(pos*neg)
}
nf_pr_auc <- function(y,p) {
  if (!sum(y==1) || !sum(y==0)) return(NA_real_)
  idx <- order(p,decreasing=TRUE)
  yy <- y[idx]; pp <- p[idx]
  ends <- which(!duplicated(pp,fromLast=TRUE))
  tp <- cumsum(yy)[ends]; fp <- ends-tp
  recall <- c(0,tp/sum(yy))
  precision <- c(1,tp/pmax(1,tp+fp))
  sum(diff(recall)*(head(precision,-1)+tail(precision,-1))/2)
}
nf_boyce <- function(y,p,bins=20L) {
  if (sum(y==1)<5 || length(unique(p))<3) return(NA_real_)
  cuts <- seq(min(p),max(p),length.out=bins+1L)
  hp <- tabulate(findInterval(p[y==1],cuts,all.inside=TRUE),nbins=bins)
  ha <- tabulate(findInterval(p,cuts,all.inside=TRUE),nbins=bins)
  ok <- hp>0 & ha>0
  if (sum(ok)<3) return(NA_real_)
  mids <- (head(cuts,-1)+tail(cuts,-1))/2
  ratio <- (hp/sum(hp))/pmax(ha/sum(ha),1e-12)
  suppressWarnings(stats::cor(mids[ok],ratio[ok],method="spearman"))
}
nf_confusion <- function(y,p,threshold=0.5,beta=1) {
  yy <- as.integer(p>=threshold)
  tp <- sum(yy==1 & y==1); tn <- sum(yy==0 & y==0)
  fp <- sum(yy==1 & y==0); fn <- sum(yy==0 & y==1)
  precision <- nf_safe_div(tp,tp+fp)
  recall <- nf_safe_div(tp,tp+fn)
  specificity <- nf_safe_div(tn,tn+fp)
  f1 <- nf_safe_div((1+beta^2)*tp,(1+beta^2)*tp+beta^2*fn+fp)
  pc <- (tp+tn)/length(y)
  pe <- ((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/length(y)^2
  tpr <- tp+fn; fprden <- fp+tn
  list(accuracy=pc,
       balanced_accuracy=mean(c(recall,specificity)),
       precision=precision,recall=recall,sensitivity=recall,
       specificity=specificity,f1=f1,fbeta=f1,
       mcc=nf_safe_div(tp*tn-fp*fn,sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
       kappa=nf_safe_div(pc-pe,1-pe),
       gmean=sqrt(recall*specificity),
       jaccard=nf_safe_div(tp,tp+fp+fn),
       npv=nf_safe_div(tn,tn+fn),
       fpr=nf_safe_div(fp,fprden),
       fnr=nf_safe_div(fn,tpr),
       youden=recall+specificity-1,
       lift=nf_safe_div(nf_safe_div(tp,tp+fp),mean(y)))
}
nf_calibration <- function(y,p,bins=15) {
  id <- pmin(bins,pmax(1L,ceiling(p*bins)))
  ece <- 0; mce <- 0
  for (b in unique(id)) {
    ix <- id==b
    err <- abs(mean(p[ix])-mean(y[ix]))
    ece <- ece+mean(ix)*err
    mce <- max(mce,err)
  }
  c(ece=ece,mce=mce)
}

#' Evaluate classification and regression predictions
#' @param actual Ground-truth vector.
#' @param predicted Numeric prediction vector or classification probability
#'   matrix (columns must correspond to class labels).
#' @param metric Identifier from nf_metrics().
#' @param task auto, classification, regression.
#' @param threshold Binary decision threshold.
#' @param beta F-beta coefficient.
#' @param top_k K for top-k accuracy.
#' @param quantile Quantile for pinball loss.
#' @param huber_delta Delta for Huber loss.
#' @param bins Number of bins for calibration and Boyce metrics.
#' @return Named scalar metric value.
#' @export
nf_metric <- function(actual,predicted,metric,task="auto",threshold=0.5,
                      beta=1,top_k=3L,quantile=0.5,huber_delta=1,bins=20L) {
  metric <- tolower(metric)
  if (task=="auto") task <- if (is.matrix(predicted) ||
    is.factor(actual) || is.character(actual)) "classification" else "regression"
  if (!metric %in% nf_metrics(task)$metric)
    stop("Unsupported metric for ",task,": ",metric)
  if (length(actual)<1 || anyNA(actual)) stop("Actual labels contain missing values")
  if (task=="regression") {
    y <- as.numeric(actual); p <- as.numeric(predicted)
    if (length(y)!=length(p) || any(!is.finite(y)) || any(!is.finite(p)))
      stop("Invalid regression predictions")
    e <- y-p; n <- length(y)
    mse <- mean(e^2); rmse <- sqrt(mse)
    numerator <- sum(e^2); denominator <- sum((y-mean(y))^2)
    out <- switch(metric,
      r2=1-nf_safe_div(numerator,denominator),
      nse=1-nf_safe_div(numerator,denominator),
      mse=mse,rmse=rmse,mae=mean(abs(e)),
      medae=stats::median(abs(e)),
      mape=mean(abs(e)/pmax(abs(y),1e-8))*100,
      smape=mean(2*abs(e)/pmax(abs(y)+abs(p),1e-8))*100,
      wape=nf_safe_div(sum(abs(e)),sum(abs(y)))*100,
      rmsle=sqrt(mean((log1p(pmax(0,p))-log1p(pmax(0,y)))^2)),
      msle=mean((log1p(pmax(0,p))-log1p(pmax(0,y)))^2),
      huber=mean(ifelse(abs(e)<=huber_delta, e^2/2,
                        huber_delta*(abs(e)-huber_delta/2))),
      logcosh=mean(abs(e)+log1p(exp(-2*abs(e)))-log(2)),
      pinball=mean(ifelse(e>=0,quantile*e,(quantile-1)*e)),
      max_error=max(abs(e)),
      explained_variance=1-nf_safe_div(stats::var(e),stats::var(y)),
      pearson=suppressWarnings(stats::cor(y,p)),
      spearman=suppressWarnings(stats::cor(y,p,method="spearman")),
      kendall=suppressWarnings(stats::cor(y,p,method="kendall")),
      ccc={ vy<-stats::var(y);vp<-stats::var(p)
        nf_safe_div(2*stats::cov(y,p),vy+vp+(mean(y)-mean(p))^2) },
      rpd=nf_safe_div(stats::sd(y),rmse),
      rpiq=nf_safe_div(diff(stats::quantile(y,c(.25,.75))),rmse),
      abs_bias=abs(mean(e)),
      mase={ if (n<2) NA_real_ else
        nf_safe_div(mean(abs(e)),mean(abs(diff(y)))) })
    return(stats::setNames(as.numeric(out),metric))
  }
  ytext <- as.character(actual)
  if (is.null(dim(predicted))) {
    if (!is.numeric(predicted)) stop("Classification needs probability values")
    labs <- sort(unique(ytext))
    if (length(labs)!=2L) stop("Vector probabilities require binary classes")
    p1 <- as.numeric(predicted)
    P <- cbind(1-p1,p1);colnames(P) <- labs
  } else {
    P <- as.matrix(predicted)
    labs <- colnames(P)
    if (is.null(labs)) stop("Probability matrix needs class-name columns")
  }
  if (nrow(P)!=length(actual) || any(!is.finite(P)) ||
      any(P < -1e-9) || any(P > 1+1e-9) ||
      any(!ytext %in% labs))
    stop("Invalid probability matrix or mismatched class labels")
  P <- pmax(0,pmin(1,P))
  P <- P/pmax(rowSums(P),1e-12)
  K <- ncol(P); n <- nrow(P)
  idx <- match(ytext,labs)
  ptrue <- pmax(1e-12,P[cbind(seq_len(n),idx)])
  yhat <- max.col(P,ties.method="first")
  binary <- K==2L
  y <- as.integer(idx==K)
  p <- P[,K]
  cmetric <- nf_confusion(y,p,threshold,beta)
  a <- if (binary) nf_binary_auc(y,p) else
    mean(vapply(seq_len(K),function(j)
      nf_binary_auc(as.integer(idx==j),P[,j]),numeric(1)))
  ap <- if (binary) nf_pr_auc(y,p) else
    mean(vapply(seq_len(K),function(j)
      nf_pr_auc(as.integer(idx==j),P[,j]),numeric(1)))
  mc <- function(name) mean(vapply(seq_len(K),function(j)
    nf_confusion(as.integer(idx==j),P[,j],threshold,beta)[[name]],
    numeric(1)),na.rm=TRUE)
  cal <- nf_calibration(as.integer(idx==yhat),apply(P,1L,max),bins)
  if (binary) cal <- nf_calibration(y,p,bins)
  ks <- if (!binary || !sum(y) || sum(1-y)==0) NA_real_ else {
    o <- order(p,decreasing=TRUE)
    max(abs(cumsum(y[o])/sum(y)-cumsum(1-y[o])/sum(1-y)))
  }
  brier <- if (binary) mean((p-y)^2) else {
    onehot <- matrix(0,n,K);onehot[cbind(seq_len(n),idx)] <- 1
    mean(rowSums((P-onehot)^2))
  }
  out <- switch(metric,
     accuracy=mean(idx==yhat),
     balanced_accuracy=if(binary) cmetric$balanced_accuracy else mc("recall"),
     auc=a,macro_auc=mean(vapply(seq_len(K),function(j)
       nf_binary_auc(as.integer(idx==j),P[,j]),numeric(1))),
     pr_auc=ap,macro_pr_auc=mean(vapply(seq_len(K),function(j)
       nf_pr_auc(as.integer(idx==j),P[,j]),numeric(1))),
     logloss=-mean(log(ptrue)),
     macro_logloss=mean(vapply(seq_len(K),function(j) {
       yy <- as.integer(idx==j); pp<-pmax(1e-12,pmin(1-1e-12,P[,j]))
       -mean(yy*log(pp)+(1-yy)*log(1-pp)) },numeric(1))),
     brier=brier,
     precision=if(binary) cmetric$precision else mc("precision"),
     recall=if(binary) cmetric$recall else mc("recall"),
     sensitivity=if(binary) cmetric$sensitivity else mc("recall"),
     specificity=if(binary) cmetric$specificity else mc("specificity"),
     f1=if(binary) cmetric$f1 else mc("f1"),
     fbeta=if(binary) cmetric$fbeta else mc("fbeta"),
     macro_f1=mc("f1"),macro_precision=mc("precision"),
     macro_recall=mc("recall"),
     mcc=if(binary) cmetric$mcc else NA_real_,
     kappa=if(binary) cmetric$kappa else NA_real_,
     gmean=if(binary) cmetric$gmean else NA_real_,
     jaccard=if(binary) cmetric$jaccard else mc("jaccard"),
     npv=if(binary) cmetric$npv else mc("npv"),
     fpr=if(binary) cmetric$fpr else mc("fpr"),
     fnr=if(binary) cmetric$fnr else mc("fnr"),
     youden=if(binary) cmetric$youden else mc("youden"),
     ks=ks,roc_gini=2*a-1,
     lift=if(binary) cmetric$lift else NA_real_,
     top_k_accuracy=mean(vapply(seq_len(n),function(i)
       idx[i] %in% order(P[i,],decreasing=TRUE)[seq_len(min(K,top_k))],logical(1))),
     ece=unname(cal["ece"]),mce=unname(cal["mce"]),
     boyce=if(binary) nf_boyce(y,p,bins) else NA_real_)
  stats::setNames(as.numeric(out),metric)
}
