#!/usr/bin/env Rscript
# NCEAS PO-only model fitting: outcome-blind, with a dedicated
# geographic INNER validation holdout for native IPP gradient stages.
# No survey outcomes are available to this process.
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=7L)
 stop("Usage: po_nceas_multi_models.R PO.csv BG.csv PO_XY.csv BG_XY.csv SURVEY_X.csv RESULTS.json SEED")
p<-utils::read.csv(args[[1]],check.names=FALSE)
bg<-utils::read.csv(args[[2]],check.names=FALSE)
pcoord<-utils::read.csv(args[[3]],check.names=FALSE)
bcoord<-utils::read.csv(args[[4]],check.names=FALSE)
survey<-utils::read.csv(args[[5]],check.names=FALSE)
output<-args[[6]]
seed<-as.integer(args[[7]])
if(!identical(names(p),names(bg))||
   !identical(names(p),names(survey))||
   !identical(names(pcoord),c("x","y"))||
   !identical(names(bcoord),c("x","y"))||
   nrow(p)!=nrow(pcoord)||nrow(bg)!=nrow(bcoord)||
   nrow(p)<70L||nrow(bg)<100L)
 stop("Invalid or misaligned NCEAS training covariates and coordinates")

# Train-only deterministic coordinate blocking, fixed candidate axes,
# no evaluation locations and no PA outcomes. The chosen group is an
# x/y band with support for separate IPP training and inner validation.
find_inner<-function(pxy,bxy) {
  candidates<-list()
  for(axis in c("x","y")) for(nbins in c(3L,4L,5L)) {
    edges<-unique(as.numeric(stats::quantile(bxy[[axis]],
       probs=seq(0,1,length.out=nbins+1L),na.rm=TRUE)))
    if(length(edges)<3L)next
    pi<-cut(pxy[[axis]],breaks=edges,labels=FALSE,
            include.lowest=TRUE)
    bi<-cut(bxy[[axis]],breaks=edges,labels=FALSE,
            include.lowest=TRUE)
    for(k in seq_len(length(edges)-1L)) {
      vp<-which(pi==k)
      vb<-which(bi==k)
      if(length(vp)<15L||length(vb)<20L||
         nrow(pxy)-length(vp)<50L||
         nrow(bxy)-length(vb)<80L)next
      cost<-abs(length(vp)/nrow(pxy)-.20)+
        .1*abs(length(vb)/nrow(bxy)-.20)
      candidates[[length(candidates)+1L]]<-list(
        pres=vp,bg=vb,axis=axis,bins=nbins,
        held_band=k,cost=cost)
    }
  }
  if(length(candidates)) {
    chosen<-candidates[[which.min(vapply(
                  candidates,function(z)z$cost,numeric(1)))]]
    chosen$selection<-"geographic_training_only"
    return(chosen)
  }
  # Explicitly labelled non-spatial fallback: not disguised as a
  # geographically independent inner holdout.
  set.seed(seed+113L)
  nvp<-max(15L,min(floor(.2*nrow(pxy)),nrow(pxy)-50L))
  nvb<-max(20L,min(floor(.2*nrow(bxy)),nrow(bxy)-80L))
  list(pres=sort(sample(seq_len(nrow(pxy)),nvp)),
       bg=sort(sample(seq_len(nrow(bxy)),nvb)),
       axis="none",bins=0L,held_band=NA_integer_,cost=NA_real_,
       selection="nonspatial_random_fallback")
}
inner<-find_inner(pcoord,bcoord)
train_p<-p[-inner$pres,,drop=FALSE]
train_b<-bg[-inner$bg,,drop=FALSE]
val_p<-p[inner$pres,,drop=FALSE]
val_b<-bg[inner$bg,,drop=FALSE]
if(nrow(train_p)<50||nrow(val_p)<15||
   nrow(train_b)<80||nrow(val_b)<20)
 stop("Invalid training-only inner split")
all<-rbind(p,bg)
y<-c(rep(1L,nrow(p)),rep(0L,nrow(bg)))
if(any(vapply(all,function(z)!is.numeric(z),logical(1)))||
   any(!is.finite(as.matrix(all)))||
   any(!is.finite(as.matrix(survey))))
 stop("Pre-encoded predictors must be numeric and finite")
# Remove fixed zero/constant columns exclusively using training data.
use<-vapply(all,function(z)length(unique(z))>1L,logical(1))
if(sum(use)<2L)stop("No informative available environmental variables")
p<-p[,use,drop=FALSE];bg<-bg[,use,drop=FALSE]
val_p<-val_p[,use,drop=FALSE];val_b<-val_b[,use,drop=FALSE]
train_p<-train_p[,use,drop=FALSE];train_b<-train_b[,use,drop=FALSE]
survey<-survey[,use,drop=FALSE]
data<-rbind(p,bg)
record<-list()
append_model<-function(name,model,predictor,fitsec,
           selected_stages=NA_integer_,
           inner_rule="not_applicable",shared_fit=FALSE) {
  begin<-proc.time()[["elapsed"]]
  pred<-as.numeric(predictor(survey))
  secs<-proc.time()[["elapsed"]]-begin
  if(length(pred)!=nrow(survey)||any(!is.finite(pred))||
     any(pred<=0))
     stop(paste("Invalid NCEAS relative scores",name))
  record[[length(record)+1L]]<<-list(name=name,
    prediction=pred,train_seconds=fitsec,predict_seconds=secs,
    selected_stages=selected_stages,inner_validation=inner_rule,
    shared_fit=shared_fit)
}

for(mode in c("neural","raw","random","hidden")) {
  start<-proc.time()[["elapsed"]]
  model<-nf_po_forest(p,bg,mode=mode,seed=seed,
    hidden=12L,epochs=75L,n_trees=130L,
    max_depth=5L,min_leaf=5L,hidden_features=8L,
    latent_mass=.35,forest_fraction=.65)
  sec<-proc.time()[["elapsed"]]-start
  append_model(paste0("NeuralForest_PO_",mode),model,
    function(x)predict(model,x,type="relative"),sec)
}

teacher<-NULL
for(mode in c("raw","random","hidden")) {
  start<-proc.time()[["elapsed"]]
  model<-nf_po_gradient_forest(train_p,train_b,
    validation_presences=val_p,validation_background=val_b,
    refit_validation=TRUE,feature_mode=mode,
    seed=seed,hidden=12L,epochs=75L,max_stages=4L,
    trees_per_stage=5L,max_depth=3L,min_leaf=8L,
    hidden_features=8L,latent_mass=.35,step_size=.12,min_gain=.002)
  sec<-proc.time()[["elapsed"]]-start
  if(mode=="raw")teacher<-model$teacher
  if(!isTRUE(model$validation_refitted)||
     model$selection_rule!="explicit_disjoint_holdout")
    stop("Native IPP gradient stage selection misaligned")
  append_model(paste0("NeuralForest_POGrad_",mode),model,
    function(x)predict(model,x,type="relative"),sec,
    selected_stages=model$selected_stages,
    inner_rule=inner$selection)
}
append_model("NeuralForest_POGrad_teacher",teacher,
  function(x)predict(teacher,x,type="relative"),
  teacher$training$total_seconds,0L,inner$selection,TRUE)

if(!requireNamespace("maxnet",quietly=TRUE))
  stop("maxnet required for real ecological comparison")
for(features in c("lq","lqph")) {
  start<-proc.time()[["elapsed"]]
  formula<-maxnet::maxnet.formula(y,data,classes=features)
  fitted<-maxnet::maxnet(y,data,f=formula,regmult=1.5)
  sec<-proc.time()[["elapsed"]]-start
  append_model(paste0("maxnet_",features),fitted,
    function(x)as.numeric(predict(fitted,x,
                     clamp=TRUE,type="exponential")),sec)
}

if(!requireNamespace("ranger",quietly=TRUE))
 stop("ranger required")
start<-proc.time()[["elapsed"]]
rf_data<-data.frame(response=factor(y,levels=c(0,1)),data)
rf<-ranger::ranger(response~.,data=rf_data,
  probability=TRUE,num.trees=200L,min.node.size=5L,
  mtry=max(1L,as.integer(sqrt(ncol(data)))),
  seed=seed,num.threads=1L)
sec<-proc.time()[["elapsed"]]-start
append_model("ranger_RF",rf,function(x) {
  prob<-as.numeric(predict(rf,data=x,num.threads=1L)$predictions[,"1"])
  prob<-pmin(1-1e-8,pmax(1e-8,prob))
  prob/(1-prob)
},sec)

start<-proc.time()[["elapsed"]]
glmdata<-data.frame(response=y,data)
glmfit<-suppressWarnings(stats::glm(
  response~.,data=glmdata,family=binomial()))
sec<-proc.time()[["elapsed"]]-start
append_model("GLM_logistic",glmfit,function(x) {
  prob<-suppressWarnings(as.numeric(predict(
        glmfit,newdata=x,type="response")))
  prob<-pmin(1-1e-8,pmax(1e-8,prob))
  prob/(1-prob)
},sec)

if(!requireNamespace("mgcv",quietly=TRUE))
 stop("mgcv required")
start<-proc.time()[["elapsed"]]
gam_terms<-vapply(names(data),function(name) {
  if(length(unique(data[[name]]))<6L)name
  else sprintf("s(%s,k=5)",name)
},character(1))
gam_formula<-stats::as.formula(
   paste("response~",paste(gam_terms,collapse="+")))
gamfit<-suppressWarnings(mgcv::gam(
    gam_formula,data=glmdata,
    family=binomial(),method="REML"))
sec<-proc.time()[["elapsed"]]-start
append_model("GAM_smooth",gamfit,function(x) {
  prob<-suppressWarnings(as.numeric(predict(
        gamfit,newdata=x,type="response")))
  prob<-pmin(1-1e-8,pmax(1e-8,prob))
  prob/(1-prob)
},sec)

jsonlite::write_json(list(models=record,preprocessing=list(
  n_original_columns=ncol(all),n_active_columns=ncol(data),
  removed_train_constant_features=names(use)[!use],
  inner_validation_type=inner$selection,inner_axis=inner$axis,
  inner_number_bands=inner$bins,
  n_inner_train_presence=nrow(train_p),
  n_inner_validation_presence=nrow(val_p),
  n_inner_train_background=nrow(train_b),
  n_inner_validation_background=nrow(val_b))),
  output,pretty=TRUE,auto_unbox=TRUE,digits=12)
