#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=6L)
  stop("Usage: pu_hidden_pilot_r.R train.csv audit.csv test.csv shift.csv out.json seed")
train <- utils::read.csv(args[[1]],check.names=FALSE)
audit <- utils::read.csv(args[[2]],check.names=FALSE)
test <- utils::read.csv(args[[3]],check.names=FALSE)
shift <- utils::read.csv(args[[4]],check.names=FALSE)
seed <- as.integer(args[[6]])
features <- setdiff(names(train),"Vampirisme")
X <- function(d) d[,features,drop=FALSE]
stopifnot(identical(names(X(train)),names(X(audit))),
          identical(names(X(train)),names(X(test))),
          identical(names(X(train)),names(X(shift))))
train_one <- function(kind) {
  start <- proc.time()[["elapsed"]]
  fit <- nf_pu_feature_forest(X(train),as.integer(train$Vampirisme),
      mode=kind,gate="candidate",priors=c(.12,.25,.40),
      seed=seed,hidden=12L,epochs=35L,
      n_trees=130L,max_depth=5L,min_leaf=5L,
      per_network=2L,latent_mass=.35,forest_fraction=.65)
  sec <- proc.time()[["elapsed"]]-start
  st <- proc.time()[["elapsed"]]
  gated <- nf_pu_verified_gate(
    fit,X(audit),as.integer(audit$Vampirisme),
    objective="auc",min_gain=.005,max_ap_loss=.015,
    max_brier_increase=.015,min_group_half=15L,seed=seed+37L)
  verify_sec <- proc.time()[["elapsed"]]-st
  list(fit=fit,gated=gated,training=sec,verified=verify_sec)
}
raw <- train_one("raw")
latent <- train_one("hidden")
teacher <- nf_pu_matched_teacher(raw$fit)
matched <- nf_pu_matched_teacher(latent$fit)
test_teach <- as.numeric(predict(teacher,X(test),type="prob")[,"1"])
test_teach2 <- as.numeric(predict(matched,X(test),type="prob")[,"1"])
shift_teach <- as.numeric(predict(teacher,X(shift),type="prob")[,"1"])
shift_teach2 <- as.numeric(predict(matched,X(shift),type="prob")[,"1"])
if(max(abs(test_teach-test_teach2))>1e-12 ||
   max(abs(shift_teach-shift_teach2))>1e-12)
  stop("Ablation invalid: raw and augmented forests did not have identical neural teachers")
record <- function(name,model,train_seconds,verification_seconds=0,
                   verified="not_used") {
  start <- proc.time()[["elapsed"]]
  p <- as.numeric(predict(model,X(test),type="prob")[,"1"])
  q <- as.numeric(predict(model,X(shift),type="prob")[,"1"])
  psec <- proc.time()[["elapsed"]]-start
  list(name=name,probability=p,external_probability=q,
    train_seconds=train_seconds+verification_seconds,
    prediction_seconds=psec,
    fit_shared_with=if(name=="NeuralForest_PUHiddenTeacher")
      "NeuralForest_PUHiddenRawCandidate" else "none",
    verification_seconds=verification_seconds,
    selected=verified,
    latent_importance_fraction=if(is.null(model$feature_diagnostics))
       NA_real_ else model$feature_diagnostics$latent_importance_fraction)
}
rows <- list(
  record("NeuralForest_PUHiddenTeacher",teacher,raw$training),
  record("NeuralForest_PUHiddenRawCandidate",raw$fit,raw$training),
  record("NeuralForest_PUHiddenLatentCandidate",latent$fit,latent$training),
  record("NeuralForest_PUHiddenRawVerified",raw$gated,
     raw$training,raw$verified,raw$gated$verified_gate$selected),
  record("NeuralForest_PUHiddenLatentVerified",latent$gated,
     latent$training,latent$verified,latent$gated$verified_gate$selected)
)
jsonlite::write_json(list(models=rows),args[[5]],
                     pretty=TRUE,auto_unbox=TRUE,digits=12)
