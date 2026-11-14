#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=8L) stop(
 "Usage: po_gradient_spatial_models.R inner_train_P.csv inner_train_BG.csv inner_val_P.csv inner_val_BG.csv test_P.csv test_grid.csv output.json seed")
pr<-utils::read.csv(args[[1]],check.names=FALSE)
bg<-utils::read.csv(args[[2]],check.names=FALSE)
vp<-utils::read.csv(args[[3]],check.names=FALSE)
vb<-utils::read.csv(args[[4]],check.names=FALSE)
pt<-utils::read.csv(args[[5]],check.names=FALSE)
grid<-utils::read.csv(args[[6]],check.names=FALSE)
seed<-as.integer(args[[8]])
stopifnot(identical(names(pr),names(bg)),
 identical(names(pr),names(vp)),identical(names(pr),names(vb)),
 identical(names(pr),names(pt)),identical(names(pr),names(grid)))
allp<-rbind(pr,vp)
allb<-rbind(bg,vb)
models<-list()
put<-function(name,model,predictor,seconds,stages=NA_integer_,
              inner_spatial=TRUE) {
 started<-proc.time()[["elapsed"]]
 pres<-as.numeric(predictor(pt))
 area<-as.numeric(predictor(grid))
 psecs<-proc.time()[["elapsed"]]-started
 if(length(pres)!=nrow(pt)||length(area)!=nrow(grid)||
    any(!is.finite(c(pres,area)))||any(c(pres,area)<=0))
    stop(paste("Invalid spatial presence-only scores:",name))
 models[[length(models)+1L]]<<-list(name=name,
  presence_prediction=pres,background_prediction=area,
  training_seconds=seconds,prediction_seconds=psecs,
  selected_stages=stages,
  used_spatial_inner_validation=inner_spatial)
}
matched_teacher<-NULL
matched_cost<-NULL
for(mode in c("raw","random","hidden")) {
 started<-proc.time()[["elapsed"]]
 fit<-nf_po_gradient_forest(pr,bg,
  validation_presences=vp,validation_background=vb,
  refit_validation=TRUE,feature_mode=mode,
  seed=seed,hidden=12L,epochs=75L,
  max_stages=4L,trees_per_stage=5L,
  max_depth=3L,min_leaf=8L,
  hidden_features=8L,latent_mass=.35,
  step_size=.12,min_gain=.002)
 secs<-proc.time()[["elapsed"]]-started
 if(!identical(fit$selection_rule,"explicit_disjoint_holdout") ||
    !isTRUE(fit$validation_refitted))
    stop("Spatial inner validation was not used or pooled incorrectly")
 if(mode=="raw") {
   matched_teacher<-fit$teacher
   matched_cost<-secs
 }
 put(paste0("NeuralForest_POGrad_",mode),fit,
   function(x) predict(fit,x,type="relative"),
   secs,fit$selected_stages,TRUE)
}
put("NeuralForest_POGrad_teacher",matched_teacher,
  function(x) predict(matched_teacher,x,type="relative"),
  matched_cost,0L,TRUE)

if(!requireNamespace("maxnet",quietly=TRUE))
 stop("MaxEnt/maxnet mandatory in spatial gradient benchmark")
train<-rbind(allp,allb)
labels<-c(rep(1L,nrow(allp)),rep(0L,nrow(allb)))
t0<-proc.time()[["elapsed"]]
form<-maxnet::maxnet.formula(labels,train,classes="lqph")
maxent<-maxnet::maxnet(labels,train,f=form,regmult=1.5)
sec<-proc.time()[["elapsed"]]-t0
put("maxnet_lqph",maxent,
 function(x) as.numeric(predict(maxent,x,type="exponential",clamp=TRUE)),
 sec,NA_integer_,FALSE)

if(!requireNamespace("mgcv",quietly=TRUE))
 stop("mgcv GAM comparator is mandatory")
t0<-proc.time()[["elapsed"]]
gamdata<-data.frame(response=labels,train)
term<-vapply(names(train),function(nm) sprintf("s(%s,k=%d)",
 nm,if(nm=="land_cover")3L else 5L),character(1))
f<-stats::as.formula(paste("response~",paste(term,collapse="+")))
gamfit<-mgcv::gam(f,data=gamdata,family=binomial(),method="REML")
sec<-proc.time()[["elapsed"]]-t0
put("GAM_smooth",gamfit,function(x) {
 p<-as.numeric(predict(gamfit,x,type="response"))
 p<-pmin(1-1e-8,pmax(1e-8,p));p/(1-p)
},sec,NA_integer_,FALSE)

jsonlite::write_json(list(models=models),args[[7]],
 pretty=TRUE,auto_unbox=TRUE,digits=12)
