# Frozen native neural hidden representations can augment the split
# coordinates of a native C++ CART ensemble. All transformations and
# alignments are fitted without looking at verification/test outcomes.
#
# This remains an experimental P/U-observation classifier. P/U labels
# are NOT necessarily true positive/negative states, especially under
# SAR/SNAR and spatially selected pseudoabsences.

nf_pu_hidden_spec <- function(networks,per_network=2L,mode="hidden") {
  if(identical(mode,"raw"))return(list(mode="raw",selected=list()))
  if(!identical(mode,"hidden"))stop("Unknown frozen hidden feature mode")
  if(length(per_network)!=1L||!is.finite(per_network)||
     per_network<1||per_network>12||
     per_network!=as.integer(per_network))
    stop("Invalid number of hidden units per fitted network")
  chosen <- lapply(networks$networks,function(net) {
    score <- abs(as.numeric(net$native$output))
    as.integer(order(-score,seq_along(score))[
      seq_len(min(as.integer(per_network),length(score)))])
  })
  list(mode="hidden",selected=chosen)
}

nf_pu_hidden_matrix <- function(networks,X,spec) {
  if(!is.matrix(X)||!is.numeric(X)||any(!is.finite(X)))
    stop("Expected finite numeric encoded features for hidden forest")
  if(identical(spec$mode,"raw"))return(X)
  if(!identical(spec$mode,"hidden")||
     length(spec$selected)!=length(networks$networks))
    stop("Frozen hidden feature specification does not match teacher")
  parts <- lapply(seq_along(networks$networks),function(j) {
    net <- networks$networks[[j]]
    scaled <- nf_pu_scale(X,net$center,net$spread)$X
    nf_pu_cpp_hidden_units(net$native,scaled,spec$selected[[j]])
  })
  Z <- do.call(cbind,parts)
  colnames(Z) <- paste0("h",seq_len(ncol(Z)))
  cbind(X,Z)
}

nf_pu_feature_fit <- function(X,y,networks,spec,
    n_trees,max_depth,min_leaf,latent_mass,seed) {
  features <- nf_pu_hidden_matrix(networks,X,spec)
  nr <- ncol(X); ne <- ncol(features)-nr
  # Constrain latent candidate-feature mass, leaving a majority of
  # split opportunities for the original ecological/clinical variables.
  fw <- rep(1,ncol(features))
  if(ne>0) fw[(nr+1L):length(fw)] <-
    (nr*latent_mass)/(ne*(1-latent_mass))
  cfg <- nf_control(n_trees=as.integer(n_trees),
    min_leaf=as.integer(min_leaf),max_depth=as.integer(max_depth),
    neural=FALSE, class_balance="none",
    sample_fraction=.8,replace=TRUE,
    split_search="histogram",histogram_bins=64L,
    mtry=as.integer(max(2L,ceiling(sqrt(ncol(features))))),
    n_threads=1L,adaptive=FALSE)
  profile <- nf_target_profile(y,"classification",features)
  ctrl <- nf_resolve_control(cfg,nrow(features),ncol(features),
                             "classification",profile)
  trained <- nf_cpp_fit(features,as.numeric(y),rep(1,nrow(features)),
                as.numeric(fw),nf_native_cfg(ctrl),1L,2L,
                as.integer(seed+2089L))
  oob <- as.numeric(trained$oob_prediction[,2L])
  teacher_score <- nf_pu_ensemble_predict(networks,X)
  align <- nf_pu_cpp_align_fit(oob,teacher_score)
  importance <- as.numeric(trained$importance)
  raw_gain <- sum(importance[seq_len(nr)])
  hidden_gain <- if(ne>0) sum(importance[(nr+1L):length(importance)]) else 0
  list(forest=trained,alignment=align,
       diagnostics=list(raw_columns=nr,latent_columns=ne,
         raw_importance=raw_gain,latent_importance=hidden_gain,
         latent_importance_fraction=if(raw_gain+hidden_gain>0)
           hidden_gain/(raw_gain+hidden_gain) else 0,
         forest_oob_covered=align$n_oob/align$n_total,
         latent_split_sampling_mass=if(ne>0) latent_mass else 0))
}

nf_pu_feature_predict <- function(networks,X,forest,spec,alignment,
                                   forest_fraction) {
  p <- nf_pu_ensemble_predict(networks,X)
  augmented <- nf_pu_hidden_matrix(networks,X,spec)
  forest_score <- nf_pu_forest_predict(forest$forest,augmented)
  aligned <- nf_pu_cpp_align_apply(forest_score,alignment)
  (1-forest_fraction)*p+forest_fraction*aligned
}

#' Neural hidden-feature-guided native PU forest
#'
#' A matched native nnPU ensemble is frozen, and its strongest hidden
#' units are concatenated with the original features as CART split
#' candidates. A C++ native forest is fitted with constrained hidden
#' feature-sampling mass. An OOB-only, label-free monotone logit map
#' aligns the raw observed-P/U forest scores with teacher scores.
#'
#' If gate="pu", the forest is accepted only after improvements in
#' every fixed class-prior PU risk across TWO training-only holdout
#' halves, plus teacher-ranking stability. Under SAR/SNAR the PU
#' surrogate is not a valid outcome metric.
#'
#' If gate="verified", the *already trained* forest can be selected
#' on a genuinely separate cohort of verified positives AND verified
#' negatives using nf_pu_verified_gate(). This cohort must NOT have
#' been used in neural/forest fitting or later model tuning; a third
#' untouched test cohort is still needed. Returning the matched
#' neural teacher is the safe fallback when this evidence is weak.
#'
#' @param x PU training predictors.
#' @param y Observed 1=confirmed positive, 0=unlabeled (NOT negative).
#' @param mode "hidden" for frozen hidden features; "raw" for
#'   same-network native forest using the original predictors only.
#' @param gate "pu" surrogate, "verified" independent audit, or
#'   "candidate" for UNGATED research-only ablation (never deployment).
#' @param audit_x,audit_y Reserved verification cohort and real 0/1 outcomes,
#'   required when gate="verified"; not usable as fitting data.
#' @param priors Prespecified class prevalences, not test-estimated.
#' @param seed RNG seed for reproducibility.
#' @param hidden Native nnPU network hidden width.
#' @param epochs Native nnPU training epochs.
#' @param n_trees Number of native C++ trees.
#' @param max_depth Maximum tree depth for interaction learning.
#' @param min_leaf Minimum native tree terminal node size.
#' @param per_network Number of high-contribution hidden units per network.
#' @param latent_mass Fraction of per-node feature sampling mass
#'   assigned to neural features, restricted to 0.05-0.49.
#' @param forest_fraction Prespecified aligned forest fraction
#'   in the neural/forest prediction blend.
#' @param min_pu_gain Strict gain on EVERY assumed prior/holdout half.
#' @param min_rank_correlation Minimum rank concordance in PU gate.
#' @param verified_min_gain Required AUC improvement on EACH audit half.
#' @return Fitted nf_pu_forest with exact matched neural baseline.
#' @export
nf_pu_feature_forest <- function(x,y,mode=c("hidden","raw"),
    gate=c("pu","verified","candidate"),audit_x=NULL,audit_y=NULL,
    priors=c(.12,.25,.40),seed=42L,hidden=20L,epochs=60L,
    n_trees=130L,max_depth=5L,min_leaf=5L,
    per_network=2L,latent_mass=.35,forest_fraction=.65,
    min_pu_gain=.003,min_rank_correlation=.99,
    verified_min_gain=.005) {
  mode <- match.arg(mode);gate <- match.arg(gate)
  if(!is.matrix(x)&&!is.data.frame(x))stop("Training features must be matrix or data frame")
  yy <- as.character(y)
  if(length(y)!=nrow(x) || anyNA(yy)||any(!yy %in% c("0","1")))
    stop("Observed PU labels must contain only 0/1 with matching length")
  yy <- as.integer(yy)
  if(min(tabulate(yy+1L,nbins=2L))<40L)
    stop("Need at least 40 PU positives and 40 unlabeled observations")
  if(!length(priors)||any(!is.finite(priors))||
     any(priors<=0|priors>=1)||
     length(hidden)!=1L||!is.finite(hidden)||hidden<2||hidden>128||
     length(epochs)!=1L||!is.finite(epochs)||epochs<1||epochs>1500||
     length(n_trees)!=1L||!is.finite(n_trees)||n_trees<25||
     n_trees>2000||n_trees!=as.integer(n_trees)||
     length(max_depth)!=1L||!is.finite(max_depth)||
     max_depth<2||max_depth>12||max_depth!=as.integer(max_depth)||
     length(min_leaf)!=1L||!is.finite(min_leaf)||
     min_leaf<2||min_leaf>100||min_leaf!=as.integer(min_leaf)||
     length(latent_mass)!=1L||!is.finite(latent_mass)||
     latent_mass<.05||latent_mass>.49||
     length(forest_fraction)!=1L||!is.finite(forest_fraction)||
     forest_fraction<=0||forest_fraction>1||
     length(min_pu_gain)!=1L||!is.finite(min_pu_gain)||
     min_pu_gain<0||
     length(min_rank_correlation)!=1L||
     !is.finite(min_rank_correlation)||
     min_rank_correlation<0||min_rank_correlation>1||
     length(verified_min_gain)!=1L||!is.finite(verified_min_gain)||
     verified_min_gain<0)
    stop("Invalid frozen neural feature forest configuration")
  if(gate=="verified" &&
     (is.null(audit_x)||is.null(audit_y)))
    stop("Verified selection requires a disjoint 0/1 audit cohort")
  now <- function()proc.time()[["elapsed"]]
  start <- now()
  history <- data.frame(candidate="matched_teacher",
       risk=NA_real_,min_gain=NA_real_,rank_min=1,eligible=TRUE)
  selection_time <- 0
  chosen_fraction <- 0
  if(gate=="pu") {
    outer <- nf_split_indices(yy,"classification",.2,seed)
    enc <- nf_encoding(x[outer$train,,drop=FALSE])
    tr <- enc$X
    v <- nf_encoding(x[outer$validation,,drop=FALSE],
                     fit=enc$encoding)$X
    yt <- yy[outer$train];yv <- yy[outer$validation]
    teacher <- nf_pu_stable_ensemble(tr,yt,priors,hidden,epochs,
                                    seed,restarts=2L)
    spec <- nf_pu_hidden_spec(teacher,per_network,mode)
    teacher_p <- nf_pu_ensemble_predict(teacher,v)
    split <- nf_split_indices(yv,"classification",.49,seed+811L)
    halves <- list(split$train,split$validation)
    baseline <- nf_pu_stable_risk_grid(yv,teacher_p,priors,halves)
    selected <- list(risk=max(baseline),fraction=0)
    train_candidate <- nf_pu_feature_fit(tr,yt,teacher,spec,
                n_trees,max_depth,min_leaf,latent_mass,seed+4709L)
    # Exactly one prespecified blending strength; no multiple-testing
    # advantage from sweeping many strengths on the same holdout.
    p <- nf_pu_feature_predict(teacher,v,train_candidate,spec,
               train_candidate$alignment,forest_fraction)
    risk <- nf_pu_stable_risk_grid(yv,p,priors,halves)
    gain <- baseline-risk
    correlation <- vapply(halves,function(ix) {
       suppressWarnings(stats::cor(teacher_p[ix],p[ix],method="spearman"))
    },numeric(1))
    accepted <- all(is.finite(c(risk,gain,correlation))) &&
      all(gain>min_pu_gain) &&
      all(correlation>=min_rank_correlation)
    history <- rbind(history,data.frame(
      candidate=paste0("native_",mode),
      risk=max(risk),min_gain=min(gain),
      rank_min=min(correlation),eligible=accepted))
    if(accepted)chosen_fraction <- forest_fraction
    history$risk[[1]] <- max(baseline)
    history$min_gain[[1]] <- 0
    selection_time <- now()-start
  }
  refit_start <- now()
  full <- nf_encoding(x)
  full_teacher <- nf_pu_stable_ensemble(full$X,yy,priors,
                     hidden,epochs,seed+989L,restarts=2L)
  full_spec <- nf_pu_hidden_spec(full_teacher,per_network,mode)
  candidate <- NULL
  if(gate!="pu" || chosen_fraction>0) {
    candidate <- nf_pu_feature_fit(full$X,yy,full_teacher,full_spec,
                        n_trees,max_depth,min_leaf,latent_mass,seed+4709L)
  }
  final_fit <- now()-refit_start
  selected <- gate=="pu" && chosen_fraction>0
  out <- list(mode=paste0("feature_",mode),
       levels=c("0","1"),priors=priors,
       alpha=1,networks=full_teacher,encoding=full$encoding,
       forest=if(is.null(candidate)) NULL else candidate$forest,
       feature_spec=if(is.null(candidate)) NULL else full_spec,
       forest_align=if(is.null(candidate)) NULL else candidate$alignment,
       feature_fraction=if(gate!="pu") forest_fraction
                        else chosen_fraction,
       feature_diagnostics=if(is.null(candidate)) NULL
                        else candidate$diagnostics,
       selected_candidate=if(gate!="pu") paste0("unverified_",mode)
         else if(selected) paste0("native_",mode) else "neural_fallback",
       validation_pu_risk=if(gate=="pu")
         if(selected) history$risk[[2L]] else history$risk[[1L]]
         else NA_real_,
       selection_history=history,hidden=hidden,epochs=epochs,n_trees=n_trees,
       timings=list(selection_seconds=selection_time,
            final_fit_seconds=final_fit,
            total_training_seconds=now()-start))
  class(out)<-"nf_pu_forest"
  if(gate=="verified") {
    verified <- nf_pu_verified_gate(out,audit_x,audit_y,
          objective="auc",min_gain=verified_min_gain,
          max_ap_loss=.015,max_brier_increase=.015,
          min_group_half=15L,seed=seed+37L)
    verified$selected_candidate <- if(identical(
           verified$verified_gate$selected,"candidate"))
             paste0("verified_",mode) else "neural_fallback"
    verified$timings$verification_seconds <-
       verified$verified_gate$selection_seconds
    verified$timings$total_training_seconds <-
       verified$timings$total_training_seconds +
       verified$verified_gate$selection_seconds
    out <- verified
  }
  out
}
