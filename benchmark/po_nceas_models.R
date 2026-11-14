#!/usr/bin/env Rscript
# Fit native presence-only algorithms and ecological baselines on
# NCEAS PO + background (never the independently surveyed PA outcomes).
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=5L)stop(
  "Usage: po_nceas_models.R TRAIN_P.csv TRAIN_BG.csv SURVEY_X.csv OUTPUT.json SEED")
p<-utils::read.csv(args[[1]],check.names=FALSE)
bg<-utils::read.csv(args[[2]],check.names=FALSE)
test<-utils::read.csv(args[[3]],check.names=FALSE)
seed<-as.integer(args[[5]])
if(!identical(names(p),names(bg)) || !identical(names(p),names(test)))
  stop("NCEAS predictor columns not aligned")
if(nrow(p)<30L||nrow(bg)<100L)
  stop("Insufficient PO/available background data")
record<-list()
add<-function(name,prediction_f,seconds,implementation) {
  started<-proc.time()[["elapsed"]]
  result<-as.numeric(prediction_f(test))
  ptime<-proc.time()[["elapsed"]]-started
  if(length(result)!=nrow(test)||any(!is.finite(result))||
     any(result<=0)) stop(paste("Invalid prediction from",name))
  record[[length(record)+1L]]<<-list(
       name=name,score=result,train_seconds=seconds,
       predict_seconds=ptime,implementation=implementation)
}
for(mode in c("neural","raw","random","hidden")) {
  start<-proc.time()[["elapsed"]]
  fitted<-nf_po_forest(p,bg,mode=mode,
         seed=seed,hidden=12L,epochs=75L,
         n_trees=130L,max_depth=5L,min_leaf=5L,
         hidden_features=8L,latent_mass=.35,
         forest_fraction=.65)
  elapsed<-proc.time()[["elapsed"]]-start
  add(paste0("NeuralForest_PO_",mode),
      function(x)predict(fitted,x,type="relative"),
      elapsed,"native_Cpp17")
}
data<-rbind(p,bg)
y<-c(rep(1L,nrow(p)),rep(0L,nrow(bg)))
if(!requireNamespace("maxnet",quietly=TRUE))
 stop("maxnet is mandatory in the NCEAS public benchmark")
for(features in c("lq","lqph")) {
  t0<-proc.time()[["elapsed"]]
  formula<-maxnet::maxnet.formula(y,data,classes=features)
  fitted<-maxnet::maxnet(y,data,f=formula,regmult=1.5)
  elapsed<-proc.time()[["elapsed"]]-t0
  add(paste0("maxnet_",features),
      function(x)predict(fitted,x,type="exponential",clamp=TRUE),
      elapsed,"MaxEnt_maxnet")
}
if(!requireNamespace("ranger",quietly=TRUE))
 stop("ranger is mandatory in the NCEAS public benchmark")
t0<-proc.time()[["elapsed"]]
train<-data.frame(observed=factor(y,levels=c(0,1)),data)
forest<-ranger::ranger(observed~.,data=train,probability=TRUE,
    num.trees=200L,min.node.size=5L,
    mtry=max(1L,as.integer(sqrt(ncol(data)))),
    seed=seed,num.threads=1L)
elapsed<-proc.time()[["elapsed"]]-t0
add("ranger_RF",function(x) {
  prob<-as.numeric(predict(forest,data=x,num.threads=1L)$predictions[,"1"])
  prob<-pmin(1-1e-8,pmax(1e-8,prob))
  prob/(1-prob)
},elapsed,"ranger_Cpp")
t0<-proc.time()[["elapsed"]]
train_glm<-data.frame(observed=y,data)
glm_fit<-suppressWarnings(
    stats::glm(observed~.,data=train_glm,family=binomial()))
elapsed<-proc.time()[["elapsed"]]-t0
add("GLM_logistic",function(x) {
  prob<-as.numeric(stats::predict(glm_fit,newdata=x,type="response"))
  prob<-pmin(1-1e-8,pmax(1e-8,prob))
  prob/(1-prob)
},elapsed,"stats_GLM")
if(!requireNamespace("mgcv",quietly=TRUE))
 stop("mgcv is mandatory in the NCEAS public benchmark")
t0<-proc.time()[["elapsed"]]
# Fixed basis; if fewer than 5 unique values in a numeric covariate,
# represent it parametrically rather than with an invalid smooth.
terms<-vapply(names(data),function(nm) {
  unique_values<-length(unique(data[[nm]]))
  if(unique_values<6L)nm
  else paste0("s(",nm,",k=5)")
},character(1))
gam_formula<-as.formula(paste("observed~",paste(terms,collapse="+")))
gam_fit<-suppressWarnings(
  mgcv::gam(gam_formula,data=train_glm,
            family=binomial(),method="REML"))
elapsed<-proc.time()[["elapsed"]]-t0
add("GAM_smooth",function(x) {
  prob<-as.numeric(predict(gam_fit,newdata=x,type="response"))
  prob<-pmin(1-1e-8,pmax(1e-8,prob))
  prob/(1-prob)
},elapsed,"mgcv_GAM")
jsonlite::write_json(list(models=record),args[[4]],
                     auto_unbox=TRUE,pretty=TRUE,digits=12)
