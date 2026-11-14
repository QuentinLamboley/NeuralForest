#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=6L)
 stop("Usage: pu_residual_legacy_r.R train.csv test.csv output.json metric seed external.csv")
train <- utils::read.csv(args[[1]],check.names=FALSE)
test <- utils::read.csv(args[[2]],check.names=FALSE)
ext <- utils::read.csv(args[[6]],check.names=FALSE)
columns <- setdiff(names(train),"Vampirisme")
seed <- as.integer(args[[5]])
started <- proc.time()[["elapsed"]]
fit <- nf_pu_residual_forest(train[,columns,drop=FALSE],
   as.integer(train$Vampirisme),priors=c(.12,.25,.40),
   seed=seed,hidden=20L,epochs=60L,rounds=4L,
   trees_per_round=2L,max_depth=2L,learning_rate=.12,
   max_logit_change=.4,min_gain=.002,min_rank_correlation=.99)
fitsec <- proc.time()[["elapsed"]]-started
teacher <- nf_pu_matched_teacher(fit)
getrow <- function(name,object) {
  pstart <- proc.time()[["elapsed"]]
  p <- as.numeric(predict(object,test[,columns,drop=FALSE],type="prob")[,"1"])
  q <- as.numeric(predict(object,ext[,columns,drop=FALSE],type="prob")[,"1"])
  psec <- proc.time()[["elapsed"]]-pstart
  list(name=name,probability=p,external_probability=q,
       train_seconds=fitsec,prediction_seconds=psec,
       selection_seconds=fit$timings$selection_seconds,
       final_fit_seconds=fit$timings$final_fit_seconds,
       validation_score=-fit$validation_pu_risk,
       configuration=sprintf("mode=pu_residual;selected=%s;rounds=%d;shared_neural_teacher=true",
          fit$selected_candidate,fit$selected_rounds))
}
rows <- list(getrow("NeuralForest_PUResidual",fit),
             getrow("NeuralForest_PUResidualTeacher",teacher))
jsonlite::write_json(list(models=rows),args[[3]],
           pretty=TRUE,auto_unbox=TRUE,digits=12)
