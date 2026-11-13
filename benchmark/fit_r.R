#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 6L) stop("Usage: fit_r.R train.csv test.csv output.json metric seed external.csv")
train <- utils::read.csv(args[[1]],check.names=FALSE)
test <- utils::read.csv(args[[2]],check.names=FALSE)
external <- utils::read.csv(args[[6]],check.names=FALSE)
out <- args[[3]]
metric <- args[[4]]
seed <- as.integer(args[[5]])
set.seed(seed)
xtr <- train[,names(train)!="Vampirisme",drop=FALSE]
xte <- test[,names(test)!="Vampirisme",drop=FALSE]
xext <- external[,names(external)!="Vampirisme",drop=FALSE]
ytr <- factor(train$Vampirisme,levels=c(0,1))
yte <- factor(test$Vampirisme,levels=c(0,1))
results <- list()
run_nf <- function(label, differentiable, adaptive=FALSE,
                   calibrated=FALSE, neural=TRUE,
                   sampling="standard", fraction=0.8,
                   replace=TRUE, nrounds=2L) {
  cfg <- nf_control(
    n_trees=130L, min_leaf=2L, sample_fraction=fraction,
    replace=replace,tree_sampling=sampling,
    class_balance=if(sampling=="balanced_undersample") "none" else "auto",
    neural=neural, neural_rounds=nrounds, neural_epochs=12L,
    neural_hidden=c(32L,16L),
    differentiable=differentiable,
    soft_n_trees=4L,soft_depth=3L,soft_epochs=24L,
    soft_batch=96L,soft_lr=0.015,soft_patience=5L,
    soft_temperature=0.8,soft_loss="auto",
    soft_blend="auto",n_threads=1L,verbose=FALSE,
    sampling_weight_mode="single",
    linear_expert=if(adaptive) "auto" else "none",
    calibration=if(calibrated) "auto" else "none")
  t0 <- proc.time()[["elapsed"]]
  fit <- neural_forest(xtr,ytr,task="classification",
                       control=cfg,metric=metric,seed=seed,
                       validation=0.2,refit=TRUE)
  duration <- proc.time()[["elapsed"]]-t0
  t1 <- proc.time()[["elapsed"]]
  prediction <- predict(fit,xte,type="prob")[,"1"]
  prediction_external <- predict(fit,xext,type="prob")[,"1"]
  p_duration <- proc.time()[["elapsed"]]-t1
  b <- if (!is.null(fit$soft_selection))
    fit$soft_selection$weight else 0
  expert <- if(is.null(fit$expert_selection)) "none" else
    fit$expert_selection$type
  select_seconds <- fit$timings$preprocess_seconds+
    fit$timings$native_selection_seconds+
    fit$timings$calibration_selection_seconds+
    fit$timings$soft_selection_seconds+
    fit$timings$expert_selection_seconds
  list(name=label,probability=as.numeric(prediction),
       external_probability=as.numeric(prediction_external),
       train_seconds=duration,
       selection_seconds=select_seconds,
       final_fit_seconds=fit$timings$final_refit_seconds,
       prediction_seconds=p_duration,
       validation_score=as.numeric(fit$validation_score),
       configuration=sprintf(
          "native_trees=%d; neural_rounds=%d; soft_depth=%d; soft_trees=%d; blend=%.2f; surrogate=%s; expert=%s; expert_blend=%.2f; calibrated=%s; sampling=%s; sample_fraction=%.2f; replacement=%s",
          cfg$n_trees,cfg$neural_rounds,cfg$soft_depth,
          if(differentiable)cfg$soft_n_trees else 0L,b,
          if(differentiable)fit$soft_model$surrogate else "none",
          expert,if(is.null(fit$expert_selection)) 0 else fit$expert_selection$weight,
          !is.null(fit$calibration),sampling,fraction,replace))
}
results[[1]] <- run_nf("NeuralForest_CART",FALSE)
results[[2]] <- run_nf("NeuralForest_Differentiable",TRUE)
results[[3]] <- run_nf("NeuralForest_Adaptive",TRUE,TRUE)
results[[4]] <- run_nf("NeuralForest_Calibrated",TRUE,TRUE,TRUE)
results[[5]] <- run_nf("NeuralForest_SubSample50_Informed",
                      FALSE,neural=TRUE,sampling="standard",
                      fraction=0.5,replace=FALSE)
results[[6]] <- run_nf("NeuralForest_UndersampledPlain",
                      FALSE,neural=FALSE,sampling="balanced_undersample",
                      fraction=0.8,replace=FALSE)
results[[7]] <- run_nf("NeuralForest_UndersampledInformed",
                      FALSE,neural=TRUE,sampling="balanced_undersample",
                      fraction=0.8,replace=FALSE,nrounds=3L)
if(requireNamespace("ranger",quietly=TRUE)) {
  # Per-fold imputation: never estimate training medians on held-out
  # validation or test observations. Time both preprocessing stages.
  ranger_impute <- function(train_frame,other_frame) {
    training <- as.data.frame(train_frame)
    other <- as.data.frame(other_frame)
    for(name in names(training)) {
      tr <- as.numeric(training[[name]])
      tr[!is.finite(tr)] <- NA_real_
      med <- if(all(is.na(tr))) 0 else stats::median(tr,na.rm=TRUE)
      tr[is.na(tr)] <- med
      training[[name]] <- tr
      v <- as.numeric(other[[name]])
      v[!is.finite(v)] <- med
      other[[name]] <- v
    }
    list(training=training,other=other)
  }
  metric_value <- function(labels,p) {
    as.numeric(nf_metric(labels,p,metric=metric,task="classification"))
  }
  t0 <- proc.time()[["elapsed"]]
  split <- NeuralForest:::nf_split_indices(as.integer(ytr)-1L,"classification",
                             validation=0.2,seed=seed+17L)
  prepared <- ranger_impute(xtr[split$train,,drop=FALSE],
                            xtr[split$validation,,drop=FALSE])
  d_fit <- prepared$training
  d_val <- prepared$other
  d_fit$.target <- ytr[split$train]
  best_idx <- 1L; best <- if (NeuralForest:::nf_metric_direction(metric)=="minimize") Inf else -Inf
  options <- list(list(mtry=max(1L,floor(sqrt(ncol(xtr)))),min.node.size=2L),
                  list(mtry=max(1L,floor(ncol(xtr)/3)),min.node.size=5L))
  for (i in seq_along(options)) {
    fit <- ranger::ranger(.target ~ ., data=d_fit,
       num.trees=160L,mtry=options[[i]]$mtry,
       min.node.size=options[[i]]$min.node.size,
       probability=TRUE,num.threads=1L,seed=seed+i)
    pv <- predict(fit,d_val)$predictions
    s <- metric_value(ytr[split$validation],pv)
    if (is.finite(s) &&
        (if (NeuralForest:::nf_metric_direction(metric)=="minimize") s<best else s>best)) {
      best <- s;best_idx <- i
    }
  }
  select_seconds <- proc.time()[["elapsed"]]-t0
  fit_start <- proc.time()[["elapsed"]]
  full <- ranger_impute(xtr,xte)
  d <- full$training
  d$.target <- ytr
  ranger_test <- full$other
  ranger_external <- ranger_impute(xtr,xext)$other
  opt <- options[[best_idx]]
  final <- ranger::ranger(.target ~ .,data=d,num.trees=160L,
              mtry=opt$mtry,min.node.size=opt$min.node.size,
              probability=TRUE,num.threads=1L,seed=seed+200L)
  fit_seconds <- proc.time()[["elapsed"]]-fit_start
  duration <- proc.time()[["elapsed"]]-t0
  t1 <- proc.time()[["elapsed"]]
  prob <- predict(final,ranger_test)$predictions[,"1"]
  probext <- predict(final,ranger_external)$predictions[,"1"]
  p_duration <- proc.time()[["elapsed"]]-t1
  results[[8]] <- list(name="ranger",probability=as.numeric(prob),
     external_probability=as.numeric(probext),
     train_seconds=duration,selection_seconds=select_seconds,
     final_fit_seconds=fit_seconds,prediction_seconds=p_duration,
     validation_score=best,
     configuration=sprintf("trees=160;mtry=%d;min_leaf=%d",
                           opt$mtry,opt$min.node.size))
}
jsonlite::write_json(list(models=results),out,
                     digits=12,pretty=TRUE,auto_unbox=TRUE)
