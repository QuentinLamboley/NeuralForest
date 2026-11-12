#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=5L)
  stop("Usage: rounds_fit.R train.csv test.csv output.json metric seed")
train <- utils::read.csv(args[[1L]],check.names=FALSE)
test <- utils::read.csv(args[[2L]],check.names=FALSE)
out <- args[[3L]]
metric <- args[[4L]]
seed <- as.integer(args[[5L]])
xtr <- train[,names(train)!="Vampirisme",drop=FALSE]
xte <- test[,names(test)!="Vampirisme",drop=FALSE]
ytr <- factor(train$Vampirisme,levels=c(0,1))

fit_one <- function(rounds) {
  control <- nf_control(n_trees=130L,min_leaf=2L,sample_fraction=0.8,
          neural=TRUE,neural_rounds=rounds,neural_epochs=12L,
          neural_hidden=c(32L,16L),differentiable=TRUE,
          soft_n_trees=4L,soft_depth=3L,soft_epochs=24L,
          soft_batch=96L,soft_lr=0.015,soft_patience=5L,
          soft_temperature=0.8,soft_loss="auto",soft_blend="auto",
          n_threads=1L,sampling_weight_mode="single",
          linear_expert="auto",calibration="auto")
  start <- proc.time()[["elapsed"]]
  fit <- neural_forest(xtr,ytr,task="classification",metric=metric,
       control=control,seed=seed,validation=0.2,refit=TRUE)
  training <- proc.time()[["elapsed"]]-start
  begin <- proc.time()[["elapsed"]]
  prob <- predict(fit,xte,type="prob")[,"1"]
  prediction <- proc.time()[["elapsed"]]-begin
  selection <- fit$timings$preprocess_seconds+
      fit$timings$native_selection_seconds+
      fit$timings$calibration_selection_seconds+
      fit$timings$soft_selection_seconds+
      fit$timings$expert_selection_seconds
  list(rounds=rounds,selected_round=fit$round,
       validation_score=fit$validation_score,
       probability=as.numeric(prob),
       selection_seconds=selection,
       final_fit_seconds=fit$timings$final_refit_seconds,
       train_seconds=training,predict_seconds=prediction,
       expert=if(is.null(fit$expert_selection)) "none" else
         fit$expert_selection$type)
}
results <- list(fit_one(2L),fit_one(3L))
jsonlite::write_json(list(models=results),out,
                     digits=12,pretty=FALSE,auto_unbox=TRUE)
