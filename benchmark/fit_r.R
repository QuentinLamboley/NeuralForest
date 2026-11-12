#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 5L) stop("Usage: fit_r.R train.csv test.csv output.json metric seed")
train <- utils::read.csv(args[[1]],check.names=FALSE)
test <- utils::read.csv(args[[2]],check.names=FALSE)
out <- args[[3]]
metric <- args[[4]]
seed <- as.integer(args[[5]])
set.seed(seed)
xtr <- train[,names(train)!="Vampirisme",drop=FALSE]
xte <- test[,names(test)!="Vampirisme",drop=FALSE]
ytr <- factor(train$Vampirisme,levels=c(0,1))
yte <- factor(test$Vampirisme,levels=c(0,1))
results <- list()
run_nf <- function(label, differentiable) {
  cfg <- nf_control(
    n_trees=130L, min_leaf=2L, sample_fraction=0.8,
    neural=TRUE, neural_rounds=2L, neural_epochs=12L,
    neural_hidden=c(32L,16L),
    differentiable=differentiable,
    soft_n_trees=4L,soft_depth=3L,soft_epochs=24L,
    soft_batch=96L,soft_lr=0.015,soft_patience=5L,
    soft_temperature=0.8,soft_loss="auto",
    soft_blend="auto",n_threads=1L,verbose=FALSE)
  t0 <- proc.time()[["elapsed"]]
  fit <- neural_forest(xtr,ytr,task="classification",
                       control=cfg,metric=metric,seed=seed,
                       validation=0.2,refit=TRUE)
  duration <- proc.time()[["elapsed"]]-t0
  t1 <- proc.time()[["elapsed"]]
  prediction <- predict(fit,xte,type="prob")[,"1"]
  p_duration <- proc.time()[["elapsed"]]-t1
  b <- if (!is.null(fit$soft_selection))
    fit$soft_selection$weight else 0
  list(name=label,probability=as.numeric(prediction),
       train_seconds=duration,prediction_seconds=p_duration,
       validation_score=as.numeric(fit$validation_score),
       configuration=sprintf(
          "native_trees=%d; neural_rounds=%d; soft_depth=%d; soft_trees=%d; blend=%.2f; surrogate=%s",
          cfg$n_trees,cfg$neural_rounds,cfg$soft_depth,
          if(differentiable)cfg$soft_n_trees else 0L,b,
          if(differentiable)fit$soft_model$surrogate else "none"))
}
results[[1]] <- run_nf("NeuralForest_CART",FALSE)
results[[2]] <- run_nf("NeuralForest_Differentiable",TRUE)
if(requireNamespace("ranger",quietly=TRUE)) {
  d <- xtr
  d$.target <- ytr
  split <- NeuralForest:::nf_split_indices(as.integer(ytr)-1L,"classification",
                             validation=0.2,seed=seed+17L)
  metric_value <- function(labels,p) {
    as.numeric(nf_metric(labels,p,metric=metric,task="classification"))
  }
  t0 <- proc.time()[["elapsed"]]
  best_idx <- 1L; best <- if (NeuralForest:::nf_metric_direction(metric)=="minimize") Inf else -Inf
  options <- list(list(mtry=max(1L,floor(sqrt(ncol(xtr)))),min.node.size=2L),
                  list(mtry=max(1L,floor(ncol(xtr)/3)),min.node.size=5L))
  for (i in seq_along(options)) {
    fit <- ranger::ranger(.target ~ ., data=d[split$train,,drop=FALSE],
       num.trees=160L,mtry=options[[i]]$mtry,
       min.node.size=options[[i]]$min.node.size,
       probability=TRUE,num.threads=1L,seed=seed+i)
    pv <- predict(fit,d[split$validation,,drop=FALSE])$predictions
    s <- metric_value(ytr[split$validation],pv)
    if (is.finite(s) &&
        (if (NeuralForest:::nf_metric_direction(metric)=="minimize") s<best else s>best)) {
      best <- s;best_idx <- i
    }
  }
  opt <- options[[best_idx]]
  final <- ranger::ranger(.target ~ .,data=d,num.trees=160L,
              mtry=opt$mtry,min.node.size=opt$min.node.size,
              probability=TRUE,num.threads=1L,seed=seed+200L)
  duration <- proc.time()[["elapsed"]]-t0
  t1 <- proc.time()[["elapsed"]]
  prob <- predict(final,xte)$predictions[,"1"]
  p_duration <- proc.time()[["elapsed"]]-t1
  results[[3]] <- list(name="ranger",probability=as.numeric(prob),
     train_seconds=duration,prediction_seconds=p_duration,
     validation_score=best,
     configuration=sprintf("trees=160;mtry=%d;min_leaf=%d",
                           opt$mtry,opt$min.node.size))
}
jsonlite::write_json(list(models=results),out,
                     digits=12,pretty=TRUE,auto_unbox=TRUE)
