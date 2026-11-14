#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=6L)
 stop("Usage: pu_residual_verified_r.R train.csv audit.csv test.csv external.csv out.json seed")
train <- utils::read.csv(args[[1]],check.names=FALSE)
audit <- utils::read.csv(args[[2]],check.names=FALSE)
test <- utils::read.csv(args[[3]],check.names=FALSE)
external <- utils::read.csv(args[[4]],check.names=FALSE)
features <- setdiff(names(train),"Vampirisme")
data <- function(d) d[,features,drop=FALSE]
fit_start <- proc.time()[["elapsed"]]
fit <- nf_pu_residual_forest(data(train),as.integer(train$Vampirisme),
  seed=as.integer(args[[6]]),priors=c(.12,.25,.40),
  hidden=16L,epochs=35L,rounds=4L,trees_per_round=2L,
  max_depth=2L,learning_rate=.12,max_logit_change=.4,
  min_gain=.002,min_rank_correlation=.99)
fit_sec <- proc.time()[["elapsed"]]-fit_start
t0 <- proc.time()[["elapsed"]]
gated <- nf_pu_verified_gate(fit,data(audit),
   as.integer(audit$Vampirisme),objective="auc",
   min_gain=.002,max_ap_loss=.01,max_brier_increase=.01,
   min_group_half=15L,seed=as.integer(args[[6]])+37L)
audit_sec <- proc.time()[["elapsed"]]-t0
teacher <- nf_pu_matched_teacher(fit)
models <- list(
 NeuralForest_PUResidual=fit,
 NeuralForest_PUResidualTeacher=teacher,
 NeuralForest_PUResidualVerified=gated)
results <- lapply(names(models),function(label) {
  t0 <- proc.time()[["elapsed"]]
  p <- as.numeric(predict(models[[label]],data(test),type="prob")[,"1"])
  ext <- as.numeric(predict(models[[label]],data(external),type="prob")[,"1"])
  predict_sec <- proc.time()[["elapsed"]]-t0
  is_gated <- identical(label,"NeuralForest_PUResidualVerified")
  list(name=label,probability=p,external_probability=ext,
    train_seconds=fit_sec+if(is_gated) audit_sec else 0,
    prediction_seconds=predict_sec,
    base_fit_seconds=fit_sec,audit_selection_seconds=if(is_gated) audit_sec else 0,
    selected_candidate=fit$selected_candidate,
    correction_rounds=fit$selected_rounds,
    verified_selected=if(is_gated) gated$verified_gate$selected else "not_applied",
    shared_train_cost=TRUE)
})
jsonlite::write_json(list(models=results),args[[5]],
  auto_unbox=TRUE,pretty=TRUE,digits=12)
