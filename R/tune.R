#' Cross-validated search over neural-forest hyperparameters
#'
#' Each outer fold is held out completely; neural_forest() uses its own
#' inner holdout to select joint training rounds. Reported CV estimates
#' must still be confirmed on an untouched external test set.
#' @param x Predictors.
#' @param y Response.
#' @param search Named list of trial values for nf_control() arguments.
#' @param trials Number of random configurations (grid=FALSE).
#' @param folds Number of outer CV folds.
#' @param metric Metric from nf_metrics().
#' @param task auto, regression, or classification.
#' @param control Base nf_control().
#' @param seed RNG seed.
#' @param groups Optional group identifiers for grouped outer CV.
#' @param grid Evaluate every supplied combination instead of random trials.
#' @return List with results, best_control, best_score and metric.
#' @export
nf_tune <- function(x,y,search=list(n_trees=c(100L,300L,500L),
               mtry=c(2L,4L,8L),min_leaf=c(1L,3L,8L),
               neural_guidance=c(0,0.3,0.6,0.9),
               feature_guidance=c(0,0.25,0.5),
               neural_rounds=c(1L,3L,5L)),
               trials=20L,folds=5L,metric=NULL,task="auto",
               control=nf_control(),seed=42L,groups=NULL,grid=FALSE) {
  if (length(y)<folds*4 || folds<2) stop("Invalid number of folds")
  resp <- nf_response(y,task)
  task <- resp$task
  if (is.null(metric)) metric <- if(task=="regression") "rmse" else
    if(length(resp$labels)==2) "auc" else "macro_auc"
  metric <- tolower(metric)
  if (!metric %in% nf_metrics(task)$metric)
    stop("Unsupported metric")
  set.seed(seed)
  if (!is.list(search) || is.null(names(search)) ||
      any(!names(search) %in% names(control)) ||
      any(vapply(search,length,integer(1))<1))
    stop("search must be a named list of nonempty nf_control() parameters")
  design <- if (isTRUE(grid)) expand.grid(search,KEEP.OUT.ATTRS=FALSE) else
    as.data.frame(lapply(search,function(v) sample(v,trials,replace=TRUE)),
                  optional=TRUE)
  n <- length(y); fold <- integer(n)
  if (!is.null(groups)) {
    if (length(groups)!=n || anyNA(groups)) stop("Invalid groups")
    ug <- sample(unique(as.character(groups)))
    mp <- stats::setNames(rep(seq_len(folds),length.out=length(ug)),ug)
    fold <- unname(mp[as.character(groups)])
  } else if (task=="classification") {
    for (ids in split(seq_len(n),resp$y))
      fold[sample(ids)] <- rep(seq_len(folds),length.out=length(ids))
  } else fold[sample.int(n)] <- rep(seq_len(folds),length.out=n)
  results <- vector("list",nrow(design))
  direction <- nf_metric_direction(metric)
  for (i in seq_len(nrow(design))) {
    params <- as.list(design[i,,drop=FALSE])
    ctrl <- utils::modifyList(unclass(control),params)
    class(ctrl) <- "nf_control"
    values <- rep(NA_real_,folds)
    for (k in seq_len(folds)) {
      tr <- which(fold!=k);val <- which(fold==k)
      if (length(val)<1 || length(tr)<20) next
      fit <- try(neural_forest(x[tr,,drop=FALSE],y[tr],
                       task=task,control=ctrl,metric=metric,
                       validation=0.2,seed=seed+i*1000L+k,
                       groups=if (is.null(groups)) NULL else groups[tr]),
                 silent=TRUE)
      if (inherits(fit,"try-error")) next
      pred <- predict(fit,x[val,,drop=FALSE],
                      type=if (task=="classification") "prob" else "response")
      values[k] <- tryCatch(unname(nf_metric(y[val],pred,
                            metric=metric,task=task)),
                            error=function(e) NA_real_)
    }
    results[[i]] <- cbind(design[i,,drop=FALSE],score=if (all(is.na(values)))
                           NA_real_ else mean(values,na.rm=TRUE),
                         sd=if(sum(is.finite(values))>1) stats::sd(values,na.rm=TRUE) else NA_real_,
                         completed=sum(is.finite(values)))
  }
  out <- do.call(rbind,results);rownames(out)<-NULL
  valid <- which(is.finite(out$score))
  if (!length(valid)) stop("No valid configurations; verify folds and labels")
  winner <- valid[if (direction=="maximize")
    which.max(out$score[valid]) else which.min(out$score[valid])]
  chosen <- utils::modifyList(unclass(control),
                              as.list(design[winner,,drop=FALSE]))
  class(chosen) <- "nf_control"
  structure(list(results=out,best_control=chosen,
            best_score=out$score[winner],metric=metric,direction=direction,
            best_trial=winner,folds=folds),class="nf_tuning")
}

#' Timed holdout benchmark against ranger (if installed)
#' @param x Predictors.
#' @param y Response.
#' @param control NeuralForest controls.
#' @param seed Random seed.
#' @param validation Holdout fraction.
#' @return Data frame with wall-clock training, prediction and metric results.
#' @export
nf_benchmark <- function(x,y,control=nf_control(n_trees=200L,
                         neural_rounds=2L),seed=42L,validation=.2) {
  resp <- nf_response(y,"auto")
  task <- resp$task
  sp <- nf_split_indices(resp$y,task,validation,seed)
  tr <- sp$train; vl <- sp$validation
  metric <- if(task=="regression") "rmse" else if(length(resp$labels)==2) "auc" else "accuracy"
  results <- list()
  elapsed <- system.time(fit <- neural_forest(x[tr,,drop=FALSE],y[tr],
        task=task,control=control,metric=metric,seed=seed+1L))
  elapsed_predict <- system.time(pp<-predict(fit,x[vl,,drop=FALSE],
        type=if(task=="classification") "prob" else "response"))
  results[[1]] <- data.frame(model="NeuralForest",train_seconds=unname(elapsed["elapsed"]),
     predict_seconds=unname(elapsed_predict["elapsed"]),
     score=unname(nf_metric(y[vl],pp,metric,task)))
  if (requireNamespace("ranger",quietly=TRUE)) {
    d <- as.data.frame(x)
    names(d) <- make.unique(make.names(names(d)))
    d$.target <- if(task=="classification") factor(y) else as.numeric(y)
    elapsed <- system.time(rf<-ranger::ranger(.target ~ .,data=d[tr,,drop=FALSE],
                  probability=task=="classification",
                  num.trees=control$n_trees,seed=seed))
    elapsed_predict <- system.time(pr<-predict(rf,
                              data=d[vl,,drop=FALSE])$predictions)
    if(task=="classification") {
      if (is.null(dim(pr))) pr<-matrix(pr,ncol=length(resp$labels))
      colnames(pr)<-levels(d$.target)
    }
    results[[2]] <- data.frame(model="ranger",train_seconds=unname(elapsed["elapsed"]),
      predict_seconds=unname(elapsed_predict["elapsed"]),
      score=unname(nf_metric(y[vl],pr,metric,task)))
  }
  do.call(rbind,results)
}
