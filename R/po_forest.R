# Spatially aware presence-only SDM research API.
# Presence records vs a sampled environment accessible to the organism.
# BACKGROUND means availability, NOT a verified negative outcome.
# Relative intensity is not occupancy probability or prevalence.

nf_po_logmeanexp <- function(s,weights) {
  m<-max(s)
  m+log(sum(weights*exp(s-m))/sum(weights))
}

nf_po_features <- function(model,X) {
  if(identical(model$mode,"neural")||identical(model$mode,"raw"))
    return(X)
  Xs<-nf_pu_scale(X,model$center,model$spread)$X
  if(identical(model$mode,"hidden")) {
    rank<-order(-abs(as.numeric(model$network$output)),
                seq_along(model$network$output))
    units<-rank[seq_len(min(as.integer(model$hidden_features),length(rank)))]
    extra<-nf_pu_cpp_hidden_units(model$network,Xs,as.integer(units))
  } else if(identical(model$mode,"random")) {
    extra<-nf_pu_cpp_hidden_units(model$random_network,Xs,
           as.integer(seq_len(model$hidden_features)))
  } else stop("Unexpected presence-only feature mode")
  colnames(extra)<-paste0("po_",model$mode,"_",seq_len(ncol(extra)))
  cbind(X,extra)
}

nf_po_raw_score <- function(model,X) {
  Xs<-nf_pu_scale(X,model$center,model$spread)$X
  neural<-as.numeric(nf_po_cpp_predict(model$network,Xs))
  if(is.null(model$forest))return(neural)
  feat<-nf_po_features(model,X)
  p<-as.numeric(nf_cpp_predict(model$forest$trees,feat,2L,FALSE)[,2L])
  calibrated<-nf_pu_cpp_align_apply(p,model$forest_alignment)
  other<-stats::qlogis(pmin(1-1e-8,pmax(1e-8,calibrated)))
  (1-model$forest_fraction)*neural+model$forest_fraction*other
}

#' Native presence-only inhomogeneous point-process neural forest
#'
#' A class-prior-free C++17 one-hidden-layer neural network maximizes
#' mean score at presence sites relative to log-mean-exp score over
#' environmental availability. Background points are NOT negatives.
#' The optional native forest classifies sampled presence/background
#' status only as a density-ratio surrogate and uses either raw,
#' frozen learned hidden, or equally sized *untrained* random features.
#' The probability-like forest output is aligned with the teacher
#' using only OOB predictions; final scores remain RELATIVE intensity.
#'
#' This research method does not infer absolute species prevalence or
#' occupancy, and requires independent spatial validation of sampling
#' effort, availability area, transport and map stability.
#'
#' @param presences Data frame or numeric matrix for observed presences.
#' @param background Same predictor columns for available environment.
#' @param background_weights Positive background design/inclusion weights,
#'   if known. Does NOT automatically correct observation bias.
#' @param mode Native IPP neural network, native raw-feature forest,
#'   learned-hidden feature forest or untrained random projection forest.
#' @param seed Fixed RNG seed.
#' @param hidden C++ hidden network width.
#' @param epochs Number of native full-batch Adam training iterations.
#' @param learning_rate Adam learning rate, fixed before validation.
#' @param decay L2 regularization on neural weights.
#' @param n_trees Native forest size.
#' @param max_depth Maximum CART depth.
#' @param min_leaf Minimum number of observations per terminal node.
#' @param hidden_features Additional neural/random tanh split coordinates.
#' @param latent_mass Candidate split feature sampling mass allocated
#'   to hidden/random coordinates.
#' @param forest_fraction Fixed network/forest log-intensity mixing factor.
#' @return Fitted nf_po_forest, with predict(type="relative") describing
#'   relative suitability (NOT occupancy).
#' @export
nf_po_forest <- function(presences,background,
      background_weights=NULL,
      mode=c("hidden","raw","random","neural"),seed=42L,
      hidden=16L,epochs=90L,learning_rate=.01,decay=.0005,
      n_trees=130L,max_depth=5L,min_leaf=5L,
      hidden_features=8L,latent_mass=.35,forest_fraction=.65) {
  mode<-match.arg(mode)
  if(!(is.data.frame(presences)||is.matrix(presences)) ||
     !(is.data.frame(background)||is.matrix(background)))
    stop("Presences and background must be data frames or matrices")
  if(ncol(presences)!=ncol(background) ||
     !identical(colnames(presences),colnames(background)))
    stop("Presence and background predictors must have identical columns")
  np<-nrow(presences); nb<-nrow(background)
  if(np<30L||nb<40L)
    stop("Presence-only training needs >=30 presences and >=40 background rows")
  if(is.null(background_weights))background_weights<-rep(1,nb)
  if(length(background_weights)!=nb ||
     any(!is.finite(background_weights)) ||
     any(background_weights<=0))
    stop("Background inclusion weights must be positive and match rows")
  if(length(seed)!=1L||!is.finite(seed) ||
     length(hidden)!=1L||!is.finite(hidden)||hidden<2||hidden>128||
     length(epochs)!=1L||!is.finite(epochs)||epochs<1||epochs>2000||
     length(learning_rate)!=1L||!is.finite(learning_rate)||
     learning_rate<=0||learning_rate>.1||
     length(decay)!=1L||!is.finite(decay)||decay<0||decay>.2||
     length(n_trees)!=1L||!is.finite(n_trees)||
     n_trees<25||n_trees>2000||
     length(max_depth)!=1L||!is.finite(max_depth)||
     max_depth<2||max_depth>12||
     length(min_leaf)!=1L||!is.finite(min_leaf)||
     min_leaf<2||min_leaf>100||
     length(hidden_features)!=1L||!is.finite(hidden_features)||
     hidden_features<1||hidden_features>64||
     length(latent_mass)!=1L||!is.finite(latent_mass)||
     latent_mass<.05||latent_mass>.49||
     length(forest_fraction)!=1L||!is.finite(forest_fraction)||
     forest_fraction<0||forest_fraction>1)
    stop("Invalid presence-only forest hyperparameters")
  original<-rbind(as.data.frame(presences),as.data.frame(background))
  enc<-nf_encoding(original)
  X<-enc$X
  y<-c(rep(1,np),rep(0,nb))
  weights<-c(rep(1,np),as.numeric(background_weights))
  center_scale<-nf_pu_scale(X)
  begin<-proc.time()[["elapsed"]]
  network<-nf_po_cpp_fit(center_scale$X,y,weights,
       as.integer(hidden),as.integer(epochs),
       learning_rate,decay,as.integer(seed))
  network_seconds<-proc.time()[["elapsed"]]-begin
  model<-list(mode=mode,encoding=enc$encoding,
       center=center_scale$center,spread=center_scale$spread,
       network=network,forest=NULL,forest_alignment=NULL,
       forest_fraction=if(mode=="neural")0 else forest_fraction,
       hidden_features=as.integer(hidden_features),
       random_network=NULL,
       n_presence=np,n_background=nb,
       background_design_weights=TRUE,
       score_definition="relative_point_process_intensity",
       training=list(neural_seconds=network_seconds))
  class(model)<-"nf_po_forest"
  if(mode=="random")
    model$random_network<-nf_pu_cpp_random_projection(
       ncol(X),as.integer(hidden_features),as.integer(seed+19051L))
  if(mode!="neural") {
    start<-proc.time()[["elapsed"]]
    XX<-nf_po_features(model,X)
    p<-ncol(X); nnew<-ncol(XX)-p
    feature_weights<-rep(1,ncol(XX))
    if(nnew>0L)
      feature_weights[(p+1L):ncol(XX)]<-
          p*latent_mass/(nnew*(1-latent_mass))
    # Sampling weights balance the two design groups, NOT actual
    # occurrence prevalence. For inverse-inclusion weighted background
    # set weights separately within the group.
    w<-c(rep(1,np),as.numeric(background_weights)/
           mean(background_weights))
    w<-w/mean(w)
    ctrl<-nf_control(n_trees=as.integer(n_trees),
        max_depth=as.integer(max_depth),min_leaf=as.integer(min_leaf),
        neural=FALSE,class_balance="none",n_threads=1L,
        sample_fraction=.8,replace=TRUE,
        split_search="histogram",histogram_bins=64L,
        mtry=as.integer(max(2,ceiling(sqrt(ncol(XX))))))
    profile<-nf_target_profile(y,"classification",XX)
    spec<-nf_resolve_control(ctrl,nrow(XX),ncol(XX),
                             "classification",profile)
    forest<-nf_cpp_fit(XX,as.numeric(y),as.numeric(w),
        as.numeric(feature_weights),nf_native_cfg(spec),
        1L,2L,as.integer(seed+2089L))
    teacher_score<-as.numeric(nf_po_cpp_predict(network,center_scale$X))
    oob<-as.numeric(forest$oob_prediction[,2L])
    alignment<-nf_pu_cpp_align_fit(oob,stats::plogis(teacher_score))
    model$forest<-forest
    model$forest_alignment<-alignment
    model$forest_diagnostics<-list(
       importance=forest$importance,
       raw_dimensions=p,extra_dimensions=nnew,
       extra_split_weight_mass=if(nnew>0)latent_mass else 0,
       oob_coverage=alignment$n_oob/alignment$n_total)
    model$training$forest_seconds<-proc.time()[["elapsed"]]-start
  } else model$training$forest_seconds<-0
  model$training$total_seconds<-proc.time()[["elapsed"]]-begin
  # Scale every method to mean relative intensity = 1 over its OWN
  # training/background accessible region. Never test-background
  # normalize; the resulting normalization is not a prevalence.
  raw<-nf_po_raw_score(model,X[(np+1L):(np+nb),,drop=FALSE])
  model$reference_log_mean_intensity<-nf_po_logmeanexp(
      raw,background_weights)
  model
}

#' @export
predict.nf_po_forest <- function(object,newdata,
         type=c("relative","log_relative","suitability"),...) {
  type<-match.arg(type)
  X<-nf_encoding(newdata,fit=object$encoding)$X
  scores<-nf_po_raw_score(object,X)-
          object$reference_log_mean_intensity
  if(type=="log_relative")return(as.numeric(scores))
  relative<-exp(pmax(-25,pmin(25,scores)))
  if(type=="relative")return(as.numeric(relative))
  as.numeric(relative/(1+relative))
}

#' Presence-background validation without treating background as absence
#'
#' AUC_pb discriminates held-out PRESENCES from sampled BACKGROUND.
#' Boyce is an uncalibrated observed/expected ranking diagnostic;
#' neither is occupancy accuracy or independent biological truth.
#' Supply geographically disjoint validation records when possible.
#' @param presence_scores Scores at held-out presence sites.
#' @param background_scores Scores at held-out available environment.
#' @return AUC_pb, Boyce and means for presence/background scores.
#' @export
nf_po_metrics <- function(presence_scores,background_scores) {
  a<-as.numeric(presence_scores);b<-as.numeric(background_scores)
  if(length(a)<10L||length(b)<10L||
     any(!is.finite(c(a,b))) ||
     any(c(a,b)<0))
    stop("Need finite, nonnegative, held-out intensity scores")
  y<-c(rep(1L,length(a)),rep(0L,length(b)))
  p<-c(a,b)
  c(auc_pb=nf_binary_auc(y,p),
    boyce=nf_boyce(y,p),
    mean_presence=mean(a),
    mean_background=mean(b),
    presence_background_ratio=mean(a)/pmax(mean(b),1e-12))
}
