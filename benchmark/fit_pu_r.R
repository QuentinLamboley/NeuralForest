#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=6L)
  stop("Usage: fit_pu_r.R train.csv test.csv output.json metric seed external.csv")
train <- utils::read.csv(args[[1]],check.names=FALSE)
test <- utils::read.csv(args[[2]],check.names=FALSE)
external <- utils::read.csv(args[[6]],check.names=FALSE)
file <- args[[3]]
seed <- as.integer(args[[5]])
X <- train[,names(train)!="Vampirisme",drop=FALSE]
Xtest <- test[,names(test)!="Vampirisme",drop=FALSE]
Xexternal <- external[,names(external)!="Vampirisme",drop=FALSE]
label <- as.integer(train$Vampirisme)
if(!all(label %in% c(0L,1L)))stop("PU training labels must be 0/1")
models <- list()
for(kind in c("neural","informed")) {
  start <- proc.time()[["elapsed"]]
  fitted <- nf_pu_forest(X,label,mode=kind,seed=seed,
          priors=c(.12,.25,.40),hidden=20L,epochs=60L,n_trees=130L)
  train_time <- proc.time()[["elapsed"]]-start
  predict_start <- proc.time()[["elapsed"]]
  prediction <- as.numeric(predict(fitted,Xtest,type="prob")[,"1"])
  ext <- as.numeric(predict(fitted,Xexternal,type="prob")[,"1"])
  prediction_time <- proc.time()[["elapsed"]]-predict_start
  models[[length(models)+1L]] <- list(
    name=if(kind=="informed") "NeuralForest_PUInformed" else "NeuralForest_PUNeural",
    probability=prediction,external_probability=ext,
    train_seconds=train_time,prediction_seconds=prediction_time,
    selection_seconds=fitted$timings$selection_seconds,
    final_fit_seconds=fitted$timings$final_fit_seconds,
    validation_score=-fitted$validation_pu_risk,
    configuration=sprintf("native_nnpu=true;mode=%s;hidden=%d;epochs=%d;prior_sensitivity=0.12,0.25,0.40;blend=%.2f;score=negative_max_PU_risk",
         kind,fitted$hidden,fitted$epochs,fitted$alpha))
}
jsonlite::write_json(list(models=models),file,pretty=TRUE,auto_unbox=TRUE,digits=12)
