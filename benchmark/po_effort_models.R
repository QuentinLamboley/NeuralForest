#!/usr/bin/env Rscript
# Native IPP effort-offset challenge worker.
# The true simulated ecological suitability / test effort is NEVER
# passed to fitting. q_true and q_proxy represent externally supplied
# observation effort, NOT a response label.
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=14L)
 stop("Expected 14 args: ptrain btrain pval bval ptest bgtest qBtrain qBval proxyBtrain proxyBval qPtrain qPval outfile seed")
read_features<-function(i)utils::read.csv(args[[i]],check.names=FALSE)
ptr<-read_features(1L); btr<-read_features(2L)
pval<-read_features(3L); bval<-read_features(4L)
ptest<-read_features(5L); btest<-read_features(6L)
if(!all(vapply(list(btr,pval,bval,ptest,btest),
                  function(x)identical(names(x),names(ptr)),logical(1))))
 stop("Spatial effort challenge feature columns mismatch")
read_effort<-function(i,n) {
 x<-utils::read.csv(args[[i]])$effort
 if(length(x)!=n||any(!is.finite(x))||any(x<=0))
    stop(paste("Invalid independent effort vector",i))
 as.numeric(x)
}
q_btr<-read_effort(7L,nrow(btr))
q_bval<-read_effort(8L,nrow(bval))
proxy_btr<-read_effort(9L,nrow(btr))
proxy_bval<-read_effort(10L,nrow(bval))
q_ptr<-read_effort(11L,nrow(ptr))
q_pval<-read_effort(12L,nrow(pval))
outfile<-args[[13L]]
seed<-as.integer(args[[14L]])
pa<-rbind(ptr,pval); ba<-rbind(btr,bval)
results<-list()
add<-function(name,fit,seconds,stages=NA_integer_,
              effort="none",mode="") {
 t0<-proc.time()[["elapsed"]]
 pr<-as.numeric(predict(fit,ptest,type="relative"))
 br<-as.numeric(predict(fit,btest,type="relative"))
 sec<-proc.time()[["elapsed"]]-t0
 if(length(pr)!=nrow(ptest)||length(br)!=nrow(btest)||
    any(!is.finite(c(pr,br)))||any(c(pr,br)<=0))
    stop(paste("Invalid relative intensity",name))
 results[[length(results)+1L]]<<-list(
   model=name,presence=pr,background=br,
   train_seconds=seconds,predict_seconds=sec,
   selected_stages=stages,effort_assumption=effort,
   forest_features=mode)
}
fit_neural<-function(which) {
 t0<-proc.time()[["elapsed"]]
 if(which=="naive"){
  fit<-nf_po_forest(pa,ba,mode="neural",
      hidden=12L,epochs=65L,seed=seed+989L)
 }else{
  effort<-if(which=="true")c(q_btr,q_bval)
          else c(proxy_btr,proxy_bval)
  fit<-nf_po_effort_forest(pa,ba,
      effort_background=effort,method="neural",
      hidden=12L,epochs=65L,seed=seed+989L)
 }
 sec<-proc.time()[["elapsed"]]-t0
 add(paste0("IPP_neural_",which),fit,sec,0L,which,"neural")
}
for(which in c("naive","true","proxy"))fit_neural(which)
for(which in c("naive","true","proxy")) {
 modes<-if(which=="true")c("raw","random","hidden")else "hidden"
 for(mode in modes) {
  t0<-proc.time()[["elapsed"]]
  if(which=="naive") {
   fit<-nf_po_gradient_forest(ptr,btr,
      validation_presences=pval,
      validation_background=bval,refit_validation=TRUE,
      feature_mode=mode,seed=seed,hidden=12L,epochs=65L,
      max_stages=3L,trees_per_stage=4L,
      max_depth=3L,min_leaf=8L,
      hidden_features=8L,latent_mass=.35,
      step_size=.12,min_gain=.002)
  } else {
   effort<-if(which=="true")q_btr else proxy_btr
   effort_val<-if(which=="true")q_bval else proxy_bval
   fit<-nf_po_effort_forest(ptr,btr,
      effort_background=effort,
      validation_presences=pval,
      validation_background=bval,
      validation_effort_background=effort_val,
      method="gradient",feature_mode=mode,
      seed=seed,hidden=12L,epochs=65L,
      max_stages=3L,trees_per_stage=4L,
      max_depth=3L,min_leaf=8L,
      hidden_features=8L,latent_mass=.35,
      step_size=.12,min_gain=.002)
  }
  sec<-proc.time()[["elapsed"]]-t0
  add(paste0("IPP_gradient_",which,"_",mode),
      fit,sec,fit$selected_stages,which,mode)
 }
}
# Baseline logistic density ratio, either naive or with an independently
# supplied known effort offset. The latter predicts ecological scores
# by setting the additive log effort to ZERO on test sites.
data<-rbind(pa,ba)
binary<-c(rep(1,nrow(pa)),rep(0,nrow(ba)))
term<-paste(sprintf("\x60%s\x60",names(data)),collapse="+")
for(which in c("naive","true","proxy")) {
  logeff<-if(which=="naive")rep(0,nrow(data))
     else if(which=="true")log(c(q_ptr,q_pval,q_btr,q_bval))
     else log(c(q_ptr,q_pval,proxy_btr,proxy_bval))
  # Approximate proxy at positive sites by the known effort in this
  # simulation; proxy model's comparison is sensitivity-only.
  if(which=="proxy")next
  frame<-data.frame(response=binary,data,log_effort=logeff,
                    check.names=FALSE)
  formula<-stats::as.formula(paste("response ~",term,
                         "+ offset(log_effort)"))
  t0<-proc.time()[["elapsed"]]
  fitted<-suppressWarnings(stats::glm(
       formula,data=frame,family=stats::binomial()))
  sec<-proc.time()[["elapsed"]]-t0
  predict_ecology<-function(x) {
    new<-x
    new$log_effort<-0
    link<-suppressWarnings(stats::predict(
       fitted,newdata=new,type="link"))
    exp(pmin(25,pmax(-25,link)))
  }
  add(paste0("GLM_",which),list(),0L)
  # replace the inert preceding row with the actual R logistic score
  results[[length(results)]]<-NULL
  t1<-proc.time()[["elapsed"]]
  x1<-predict_ecology(ptest)
  x0<-predict_ecology(btest)
  psec<-proc.time()[["elapsed"]]-t1
  if(any(!is.finite(c(x1,x0)))||any(c(x1,x0)<=0))
      stop("Invalid ecological GLM offset predictions")
  results[[length(results)+1L]]<-list(model=paste0("GLM_",which),
       presence=as.numeric(x1),background=as.numeric(x0),
       train_seconds=sec,predict_seconds=psec,
       selected_stages=NA_integer_,effort_assumption=which,
       forest_features="linear")
}
if(!requireNamespace("maxnet",quietly=TRUE))
 stop("maxnet reference required")
t0<-proc.time()[["elapsed"]]
maxfit<-maxnet::maxnet(binary,data,
    f=maxnet::maxnet.formula(binary,data,classes="lqph"),regmult=1.5)
sec<-proc.time()[["elapsed"]]-t0
t1<-proc.time()[["elapsed"]]
pp<-as.numeric(predict(maxfit,ptest,type="exponential",clamp=TRUE))
bp<-as.numeric(predict(maxfit,btest,type="exponential",clamp=TRUE))
psec<-proc.time()[["elapsed"]]-t1
if(any(!is.finite(c(pp,bp)))||any(c(pp,bp)<=0))
 stop("Invalid MaxEnt presence-only reference")
results[[length(results)+1L]]<-list(model="MaxEnt_lqph",
    presence=pp,background=bp,train_seconds=sec,
    predict_seconds=psec,selected_stages=NA_integer_,
    effort_assumption="naive",forest_features="regularized_features")
jsonlite::write_json(list(models=results),outfile,
    pretty=FALSE,auto_unbox=TRUE,digits=13)
