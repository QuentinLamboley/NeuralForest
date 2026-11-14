#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=6L)
 stop("Usage: pu_residual_pilot_r.R train.csv test.csv output.json metric seed external.csv")
train <- utils::read.csv(args[[1]],check.names=FALSE)
test <- utils::read.csv(args[[2]],check.names=FALSE)
external <- utils::read.csv(args[[6]],check.names=FALSE)
xtr <- train[,names(train)!="Vampirisme",drop=FALSE]
xte <- test[,names(test)!="Vampirisme",drop=FALSE]
xext <- external[,names(external)!="Vampirisme",drop=FALSE]
y <- as.integer(train$Vampirisme)
seed <- as.integer(args[[5]])
start <- proc.time()[["elapsed"]]
model <- nf_pu_residual_forest(
 xtr,y,seed=seed,priors=c(.12,.25,.40),
 hidden=16L,epochs=35L,rounds=4L,trees_per_round=2L,
 max_depth=2L,learning_rate=.12,max_logit_change=.4,
 min_gain=.002,min_rank_correlation=.99,
 multipliers=c(.25,.5,1))
train_time <- proc.time()[["elapsed"]]-start
pt <- proc.time()[["elapsed"]]
p <- as.numeric(predict(model,xte,type="prob")[,"1"])
p_ext <- as.numeric(predict(model,xext,type="prob")[,"1"])
ptime <- proc.time()[["elapsed"]]-pt
teacher <- nf_pu_matched_teacher(model)
t0 <- proc.time()[["elapsed"]]
base <- as.numeric(predict(teacher,xte,type="prob")[,"1"])
base_ext <- as.numeric(predict(teacher,xext,type="prob")[,"1"])
tbase <- proc.time()[["elapsed"]]-t0
row <- function(label,pred,external_pred,pred_seconds) {
 list(name=label,probability=pred,
  external_probability=external_pred,train_seconds=train_time,
  prediction_seconds=pred_seconds,
  selection_seconds=model$timings$selection_seconds,
  final_fit_seconds=model$timings$final_fit_seconds,
  validation_score=-model$validation_pu_risk,
  selected_candidate=model$selected_candidate,
  selected_rounds=model$selected_rounds,
  eligible_count=sum(model$selection_history$eligible)-1L,
  config=sprintf("pu_grad_cpp=true;rounds=%d;selected=%s;shared_teacher_fit=true",
   model$selected_rounds,model$selected_candidate))
}
jsonlite::write_json(list(models=list(
 row("NeuralForest_PUResidual",p,p_ext,ptime),
 row("NeuralForest_PUResidualTeacher",base,base_ext,tbase)
)),args[[3]],pretty=TRUE,auto_unbox=TRUE,digits=12)
