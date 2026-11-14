# Native presence-only point-process gradient forests.
# A tree stage approximates the NEGATIVE GRADIENT OF THE SAME IPP
# OBJECTIVE as the neural teacher; no unlabeled location is a
# "verified negative". Standard outer-test labels never select stages.

nf_po_ipp_risk <- function(presence_score,background_score,
                           background_weights) {
  if(!length(presence_score)||!length(background_score))
    stop("IPP risk needs both presence and availability groups")
  nf_po_logmeanexp(as.numeric(background_score),background_weights)-
    mean(as.numeric(presence_score))
}

nf_po_boost_setup <- function(teacher,feature_mode,hidden_features,seed) {
  x<-teacher
  x$mode<-feature_mode
  x$hidden_features<-as.integer(hidden_features)
  if(feature_mode=="random")
    x$random_network<-nf_pu_cpp_random_projection(
      nrow(teacher$network$W),as.integer(hidden_features),
      as.integer(seed+19051L))
  x
}

nf_po_boost_tree_fit <- function(X,y,background_weights,
  teacher,feature_mode,hidden_features,latent_mass,
  stage_trees,max_depth,min_leaf,rate,max_stages,seed) {
  features_model<-nf_po_boost_setup(
     teacher,feature_mode,hidden_features,seed)
  Z<-nf_po_features(features_model,X)
  nraw<-ncol(X);nl<-ncol(Z)-nraw
  featureweights<-rep(1,ncol(Z))
  if(nl>0)
    featureweights[(nraw+1L):ncol(Z)]<-
       nraw*latent_mass/(nl*(1-latent_mass))
  controls<-nf_control(n_trees=as.integer(stage_trees),
      min_leaf=as.integer(min_leaf),max_depth=as.integer(max_depth),
      neural=FALSE,class_balance="none",n_threads=1L,
      sample_fraction=.8,replace=TRUE,
      split_search="histogram",histogram_bins=64L,
      mtry=max(2L,as.integer(ceiling(sqrt(ncol(Z))))))
  # Same native engine and hyperparameters for raw/hidden/random.
  totalweights<-c(rep(1,sum(y==1L)),background_weights)
  if(length(totalweights)!=nrow(X))
    stop("Background importance weights not aligned with rows")
  current<-nf_po_raw_score(teacher,X)
  out<-vector("list",max_stages)
  stages<-vector("list",max_stages)
  for(j in seq_len(max_stages)) {
    grad<-nf_po_cpp_gradient(current,as.numeric(y),
       as.numeric(totalweights),5.0)
    target<-as.numeric(grad$target)
    profile<-nf_target_profile(target,"regression",Z)
    ctrl<-nf_resolve_control(controls,nrow(Z),ncol(Z),
                             "regression",profile)
    tree<-nf_cpp_fit(Z,target,rep(1,nrow(Z)),
       as.numeric(featureweights),nf_native_cfg(ctrl),
       0L,1L,as.integer(seed+4019L+307L*j))
    step<-as.numeric(nf_cpp_predict(tree$trees,Z,1L,FALSE)[,1L])
    if(any(!is.finite(step)))stop("Nonfinite native IPP tree regression")
    out[[j]]<-tree$trees
    current<-current+rate*step
    stages[[j]]<-list(train_risk=nf_po_ipp_risk(
       current[y==1L],current[y==0L],background_weights),
       target_min=min(target),target_max=max(target))
  }
  list(trees=out,training=stages,feature_model=features_model)
}

nf_po_boost_raw_scores <- function(object,newdata) {
  X<-nf_encoding(newdata,fit=object$teacher$encoding)$X
  out<-nf_po_raw_score(object$teacher,X)
  if(length(object$trees)>0L) {
    features<-nf_po_features(object$feature_model,X)
    for(stage in object$trees)
      out<-out+object$step_size*as.numeric(
         nf_cpp_predict(stage,features,1L,FALSE)[,1L])
  }
  out
}

#' Native objective-aligned presence-only gradient forest
#'
#' Improves on the previous presence/background classification forest:
#' instead of estimating pseudo-class probability, each regression tree
#' stage is trained on the full-batch negative gradient of the *same*
#' inhomogeneous Poisson point-process objective as the C++ neural
#' teacher. Learned, raw and untrained random split coordinates are
#' compared with equal native forest budgets.
#'
#' The number of stages is determined using a separate presence/
#' background validation cohort. If no explicit validation cohort is
#' supplied, a training-only stratified 20% split is used: this is
#' NOT a substitute for spatial validation in an SDM publication.
#' If no stage passes, inference equals the final native IPP teacher
#' exactly. No output is an occupancy probability.
#'
#' @param presences,background Training observations and availability.
#' @param validation_presences,validation_background Optional disjoint
#'   TRAINING-ONLY inner validation cohorts (ideally spatially blocked).
#' @param refit_validation Refit the chosen stage count on the pooled
#'   outer-training data (including the inner holdout) AFTER inner
#'   model selection. Never use an outer evaluation/test fold here.
#' @param background_weights Design weights for training availability.
#' @param validation_background_weights Equivalent held-out weights.
#' @param feature_mode "raw", "hidden" or "random" (negative control).
#' @param seed Predefined random seed.
#' @param hidden,epochs Base C++ point-process neural settings.
#' @param max_stages Upper bound on native forest boosting stages.
#' @param trees_per_stage Native C++ regression trees per stage.
#' @param max_depth,min_leaf Native CART structure controls.
#' @param hidden_features,latent_mass Hidden split-feature controls.
#' @param step_size Fixed additive log-intensity forest rate.
#' @param min_gain Minimal improvement of inner held-out IPP risk.
#' @return Fitted \code{nf_po_gradient_forest}; see predict().
#' @export
nf_po_gradient_forest <- function(presences,background,
   validation_presences=NULL,validation_background=NULL,
   background_weights=NULL,validation_background_weights=NULL,
   refit_validation=TRUE,
   feature_mode=c("hidden","raw","random"),seed=42L,
   hidden=12L,epochs=75L,max_stages=4L,
   trees_per_stage=5L,max_depth=3L,min_leaf=8L,
   hidden_features=8L,latent_mass=.35,step_size=.12,
   min_gain=.002) {
  feature_mode<-match.arg(feature_mode)
  if(!is.data.frame(presences)&&!is.matrix(presences))
    stop("Presence predictors must be a matrix or data frame")
  if(!is.data.frame(background)&&!is.matrix(background))
    stop("Background predictors must be a matrix or data frame")
  np<-nrow(presences);nb<-nrow(background)
  if(np<50||nb<80 ||
     !identical(colnames(presences),colnames(background)))
    stop("Insufficient or incompatible presence/background training data")
  if(is.null(background_weights))background_weights<-rep(1,nb)
  if(length(background_weights)!=nb||
     any(!is.finite(background_weights))||
     any(background_weights<=0))
    stop("Invalid positive background weights")
  if(length(max_stages)!=1L||!is.finite(max_stages)||
     max_stages<1||max_stages>12||max_stages!=as.integer(max_stages)||
     length(trees_per_stage)!=1L||!is.finite(trees_per_stage)||
     trees_per_stage<2||trees_per_stage>50||
     trees_per_stage!=as.integer(trees_per_stage)||
     length(step_size)!=1L||!is.finite(step_size)||
     step_size<=0||step_size>1||
     length(min_gain)!=1L||!is.finite(min_gain)||min_gain<0||
     length(latent_mass)!=1L||!is.finite(latent_mass)||
     latent_mass<.05||latent_mass>.49)
    stop("Invalid native IPP boosting controls")
  if(xor(is.null(validation_presences),is.null(validation_background)))
    stop("Supply both or neither of inner validation cohorts")
  if(length(refit_validation)!=1L || is.na(refit_validation) ||
     !is.logical(refit_validation))
    stop("refit_validation must be TRUE or FALSE")
  external<-!is.null(validation_presences)
  if(external) {
    if(!identical(colnames(validation_presences),colnames(presences)) ||
       !identical(colnames(validation_background),colnames(background)) ||
       nrow(validation_presences)<15||nrow(validation_background)<20)
      stop("Incompatible or insufficient disjoint inner validation")
    if(is.null(validation_background_weights))
      validation_background_weights<-rep(1,nrow(validation_background))
    if(length(validation_background_weights)!=nrow(validation_background)||
       any(!is.finite(validation_background_weights))||
       any(validation_background_weights<=0))
      stop("Invalid validation availability weights")
    trp<-presences;trb<-background;trw<-background_weights
    vap<-validation_presences;vab<-validation_background
    vaw<-validation_background_weights
    selection_rule<-"explicit_disjoint_holdout"
  } else {
    all<-rbind(as.data.frame(presences),as.data.frame(background))
    binary<-c(rep(1L,np),rep(0L,nb))
    half<-nf_split_indices(binary,"classification",.2,seed+337L)
    selected<-half$train
    val<-half$validation
    trp<-all[intersect(selected,seq_len(np)),,drop=FALSE]
    trb<-all[intersect(selected,np+seq_len(nb)),,drop=FALSE]
    vab<-all[intersect(val,np+seq_len(nb)),,drop=FALSE]
    vap<-all[intersect(val,seq_len(np)),,drop=FALSE]
    trw<-background_weights[intersect(selected,np+seq_len(nb))-np]
    vaw<-background_weights[intersect(val,np+seq_len(nb))-np]
    selection_rule<-"random_training_only_split_not_spatial"
  }
  start<-proc.time()[["elapsed"]]
  fit_one<-function(p,b,weights,seedval,stages) {
    teacher<-nf_po_forest(p,b,background_weights=weights,
         mode="neural",seed=as.integer(seedval),
         hidden=hidden,epochs=epochs)
    inp<-rbind(as.data.frame(p),as.data.frame(b))
    X<-nf_encoding(inp,fit=teacher$encoding)$X
    y<-c(rep(1L,nrow(p)),rep(0L,nrow(b)))
    native<-nf_po_boost_tree_fit(X,y,weights,teacher,feature_mode,
       as.integer(hidden_features),latent_mass,
       as.integer(trees_per_stage),as.integer(max_depth),
       as.integer(min_leaf),step_size,as.integer(stages),seedval)
    list(teacher=teacher,native=native)
  }
  # Learn candidate stage directions on INNER training data only.
  partial<-fit_one(trp,trb,trw,as.integer(seed),as.integer(max_stages))
  te<-partial$teacher
  vp<-nf_encoding(vap,fit=te$encoding)$X
  vb<-nf_encoding(vab,fit=te$encoding)$X
  pv<-nf_po_raw_score(te,vp)
  bv<-nf_po_raw_score(te,vb)
  baseline<-nf_po_ipp_risk(pv,bv,vaw)
  history<-data.frame(stage=0L,holdout_ipp_risk=baseline,
           improvement=0,selected=FALSE)
  chosen<-0L
  previous<-baseline
  clone<-partial$native$feature_model
  Zv_p<-nf_po_features(clone,vp)
  Zv_b<-nf_po_features(clone,vb)
  for(j in seq_along(partial$native$trees)) {
    one<-partial$native$trees[[j]]
    pv<-pv+step_size*as.numeric(nf_cpp_predict(
          one,Zv_p,1L,FALSE)[,1L])
    bv<-bv+step_size*as.numeric(nf_cpp_predict(
          one,Zv_b,1L,FALSE)[,1L])
    risk<-nf_po_ipp_risk(pv,bv,vaw)
    allowed<-is.finite(risk) && risk<previous-min_gain
    history<-rbind(history,data.frame(
       stage=j,holdout_ipp_risk=risk,
       improvement=previous-risk,selected=FALSE))
    if(allowed && chosen==j-1L) {
      chosen<-as.integer(j)
      previous<-risk
    }
  }
  history$selected<-history$stage==chosen
  selection_seconds<-proc.time()[["elapsed"]]-start
  # Once stages are chosen strictly from the INNER holdout, refit on
  # all OUTER TRAINING rows. An externally supplied audit should be
  # pooled only when explicitly marked as an inner training-only fold.
  # There is NO inference about which rows belong to a final test set.
  finalp<-if(external && refit_validation)
    rbind(as.data.frame(presences),
          as.data.frame(validation_presences)) else presences
  finalb<-if(external && refit_validation)
    rbind(as.data.frame(background),
          as.data.frame(validation_background)) else background
  finalw<-if(external && refit_validation)
    c(background_weights,validation_background_weights)
    else background_weights
  final<-if(external && !refit_validation)partial else {
    if(chosen>0L)fit_one(finalp,finalb,
      finalw,as.integer(seed+989L),chosen)
    else list(teacher=nf_po_forest(finalp,finalb,
             background_weights=finalw,mode="neural",
             seed=as.integer(seed+989L),hidden=hidden,epochs=epochs),
              native=list(trees=list(),feature_model=NULL))
  }
  trees<-if(chosen>0L)
     final$native$trees[seq_len(chosen)] else list()
  model<-list(teacher=final$teacher,trees=trees,
    feature_model=if(chosen>0L)
      final$native$feature_model else NULL,
    step_size=step_size,selected_stages=chosen,
    selected_mode=feature_mode,
    validation=history,selection_rule=selection_rule,
    n_presence=nrow(finalp),n_background=nrow(finalb),
    validation_refitted=(external&&refit_validation),
    score_definition="point_process_relative_intensity",
    training=list(selection_seconds=selection_seconds,
       total_seconds=proc.time()[["elapsed"]]-start))
  class(model)<-"nf_po_gradient_forest"
  # Final relative intensity normalizer is fitted on TRAINING
  # availability only, not on the outer test region or any oracle.
  logscore<-nf_po_boost_raw_scores(model,finalb)
  model$reference_log_mean_intensity<-nf_po_logmeanexp(
           logscore,finalw)
  model
}

#' @export
predict.nf_po_gradient_forest <- function(object,newdata,
       type=c("relative","log_relative","suitability"),...) {
  type<-match.arg(type)
  score<-nf_po_boost_raw_scores(object,newdata)-
          object$reference_log_mean_intensity
  if(type=="log_relative")return(as.numeric(score))
  relative<-exp(pmin(25,pmax(-25,score)))
  if(type=="relative")return(as.numeric(relative))
  as.numeric(relative/(1+relative))
}
