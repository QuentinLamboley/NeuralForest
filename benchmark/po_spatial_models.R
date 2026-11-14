#!/usr/bin/env Rscript
# Reusable native/R ecological comparator worker.
# No test labels or simulated suitability enter this process.
suppressPackageStartupMessages(library(NeuralForest))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=6L)
 stop("Usage: po_spatial_models.R presence_train.csv background_train.csv presence_test.csv grid_test.csv output.json seed")
p<-utils::read.csv(args[[1]],check.names=FALSE)
b<-utils::read.csv(args[[2]],check.names=FALSE)
ptest<-utils::read.csv(args[[3]],check.names=FALSE)
grid<-utils::read.csv(args[[4]],check.names=FALSE)
stopifnot(identical(names(p),names(b)),
          identical(names(p),names(ptest)),
          identical(names(p),names(grid)))
seed<-as.integer(args[[6]])
training<-rbind(p,b)
y<-c(rep(1L,nrow(p)),rep(0L,nrow(b)))
result<-list()
add<-function(name,predict_f,train_seconds) {
 begin<-proc.time()[["elapsed"]]
 pp<-as.numeric(predict_f(ptest))
 pb<-as.numeric(predict_f(grid))
 predict_seconds<-proc.time()[["elapsed"]]-begin
 if(length(pp)!=nrow(ptest)||length(pb)!=nrow(grid)||
    any(!is.finite(c(pp,pb)))||any(c(pp,pb)<=0))
   stop(paste("Invalid relative intensity prediction",name))
 result[[length(result)+1L]]<<-list(name=name,
      presence_prediction=pp,background_prediction=pb,
      training_seconds=train_seconds,prediction_seconds=predict_seconds)
}
for(mode in c("neural","raw","random","hidden")) {
 t0<-proc.time()[["elapsed"]]
 fit<-nf_po_forest(p,b,mode=mode,seed=seed,
       hidden=12L,epochs=75L,learning_rate=.01,
       decay=.0005,n_trees=130L,max_depth=5L,min_leaf=5L,
       hidden_features=8L,latent_mass=.35,forest_fraction=.65)
 seconds<-proc.time()[["elapsed"]]-t0
 add(paste0("NeuralForest_PO_",mode),
     function(x) predict(fit,x,type="relative"),seconds)
}

if(!requireNamespace("maxnet",quietly=TRUE))
  stop("MaxEnt/maxnet is mandatory, not optional, for this benchmark")
for(classes in c("lq","lqph")) {
 t0<-proc.time()[["elapsed"]]
 formula<-maxnet::maxnet.formula(y,training,classes=classes)
 fitted<-maxnet::maxnet(y,training,f=formula,regmult=1.5)
 seconds<-proc.time()[["elapsed"]]-t0
 add(paste0("maxnet_",classes),
     function(x) as.numeric(stats::predict(
        fitted,x,clamp=TRUE,type="exponential")),seconds)
}
if(!requireNamespace("ranger",quietly=TRUE))
  stop("ranger mandatory for classic Random Forest control")
t0<-proc.time()[["elapsed"]]
forest_df<-data.frame(response=factor(y,levels=c(0,1)),training)
forest<-ranger::ranger(response~.,data=forest_df,
       probability=TRUE,num.trees=200L,
       min.node.size=5L,mtry=max(1L,as.integer(sqrt(ncol(training)))),
       seed=seed,num.threads=1L)
sec<-proc.time()[["elapsed"]]-t0
add("ranger_RF",function(x) {
 prob<-as.numeric(predict(forest,data=x,
                         num.threads=1L)$predictions[,"1"])
 prob<-pmax(1e-8,pmin(1-1e-8,prob))
 prob/(1-prob)
},sec)

t0<-proc.time()[["elapsed"]]
glm_df<-data.frame(response=y,training)
glm_fit<-stats::glm(response~.,data=glm_df,
                  family=stats::binomial())
sec<-proc.time()[["elapsed"]]-t0
add("GLM_logistic",function(x) {
 prob<-stats::predict(glm_fit,newdata=x,type="response")
 prob<-pmax(1e-8,pmin(1-1e-8,prob))
 prob/(1-prob)
},sec)

# GAM is an important ecological competitor, not a softmax proxy.
if(!requireNamespace("mgcv",quietly=TRUE))
 stop("mgcv mandatory for GAM reference")
t0<-proc.time()[["elapsed"]]
smoothers<-vapply(names(training),function(nm)
        sprintf("s(%s,k=5)",nm),character(1))
gam_f<-stats::as.formula(paste("response~",paste(smoothers,collapse="+")))
gam_fit<-mgcv::gam(gam_f,data=glm_df,
             family=stats::binomial(),method="REML")
sec<-proc.time()[["elapsed"]]-t0
add("GAM_smooth",function(x) {
 prob<-stats::predict(gam_fit,newdata=x,type="response")
 prob<-pmax(1e-8,pmin(1-1e-8,prob))
 prob/(1-prob)
},sec)
jsonlite::write_json(list(models=result),args[[5]],
                     pretty=TRUE,auto_unbox=TRUE,digits=12)
