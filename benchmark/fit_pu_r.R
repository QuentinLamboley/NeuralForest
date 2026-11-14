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
for(kind in c("neural","informed","stable","safe")) {
  start <- proc.time()[["elapsed"]]
  fitted <- if(kind=="safe")
    nf_pu_safe_forest(X,label,seed=seed,
      priors=c(.12,.25,.40),hidden=20L,epochs=60L,n_trees=130L,
      min_gain=.004,strengths=c(.25,.55),min_ess=.96,
      uncertainty_limit=.35,fidelity_cap=.02,min_rank_correlation=.995)
  else if(kind=="stable")
    nf_pu_stable_forest(X,label,seed=seed,
      priors=c(.12,.25,.40),hidden=20L,epochs=60L,n_trees=130L,
      min_gain=.004,strengths=c(.25,.55),min_ess=.96,
      uncertainty_limit=.35)
  else nf_pu_forest(X,label,mode=kind,seed=seed,
          priors=c(.12,.25,.40),hidden=20L,epochs=60L,n_trees=130L)
  train_time <- proc.time()[["elapsed"]]-start
  predict_start <- proc.time()[["elapsed"]]
  prediction <- as.numeric(predict(fitted,Xtest,type="prob")[,"1"])
  ext <- as.numeric(predict(fitted,Xexternal,type="prob")[,"1"])
  prediction_time <- proc.time()[["elapsed"]]-predict_start
  models[[length(models)+1L]] <- list(
    name=switch(kind,neural="NeuralForest_PUNeural",
                informed="NeuralForest_PUInformed",
                stable="NeuralForest_PUStable",
                safe="NeuralForest_PUSafe"),
    probability=prediction,external_probability=ext,
    train_seconds=train_time,prediction_seconds=prediction_time,
    selection_seconds=fitted$timings$selection_seconds,
    final_fit_seconds=fitted$timings$final_fit_seconds,
    validation_score=-fitted$validation_pu_risk,
    configuration=sprintf("native_nnpu=true;mode=%s;hidden=%d;epochs=%d;prior_sensitivity=0.12,0.25,0.40;blend=%.2f;score=negative_max_PU_risk;selected=%s;guidance=%.3f;ess=%.4f;crossfit=%s",
         kind,fitted$hidden,fitted$epochs,fitted$alpha,
         if(is.null(fitted$selected_candidate)) "legacy" else
           fitted$selected_candidate,
         if(is.null(fitted$selected_strength)||
            !is.finite(fitted$selected_strength)) 0 else
           fitted$selected_strength,
         if(is.null(fitted$stability_diagnostics)) NA_real_ else
           fitted$stability_diagnostics$ess_fraction,
         if(is.null(fitted$stability_diagnostics)) "legacy" else
           fitted$stability_diagnostics$crossfit_used))
  if(kind=="stable") {
    # Matched control: the EXACT same fitted neural networks, encoding,
    # and prior ensemble as PUStable, but with the forest disabled.
    # This isolates improvements due to trees from gains due to extra
    # neural restarts. Both methods share the FULL training cost;
    # no extra training is performed for this diagnostic.
    t_teacher <- proc.time()[["elapsed"]]
    matched <- nf_pu_matched_teacher(fitted)
    teacher_test <- as.numeric(predict(matched,Xtest,type="prob")[,"1"])
    teacher_ext <- as.numeric(predict(matched,Xexternal,type="prob")[,"1"])
    dt_teacher <- proc.time()[["elapsed"]]-t_teacher
    models[[length(models)+1L]] <- list(
      name="NeuralForest_PUStableTeacher",
      probability=as.numeric(teacher_test),
      external_probability=as.numeric(teacher_ext),
      train_seconds=train_time,
      prediction_seconds=dt_teacher,
      selection_seconds=fitted$timings$selection_seconds,
      final_fit_seconds=fitted$timings$final_fit_seconds,
      validation_score=-as.numeric(
        fitted$selection_history$worst_risk[[1L]]),
      configuration=paste0(
        "native_nnpu=true;mode=matched_teacher;",
        "shares_training_with=NeuralForest_PUStable;",
        "teacher_restarts=2;forest_disabled_at_inference=true;",
        "timings_shared_training=true"))
  }
}
jsonlite::write_json(list(models=models),file,pretty=TRUE,auto_unbox=TRUE,digits=12)
