# The verified audit must be reserved BEFORE training any PU candidate.
# Its true outcomes are used only in the final two-half selection gate.

#' Extract the EXACT fitted teacher from a PU forest
#' @param object A native nf_pu_forest, including PUStable.
#' @return A predictor with the same networks and encoder but no forest.
#' @export
nf_pu_matched_teacher <- function(object) {
  if(!inherits(object,"nf_pu_forest") || is.null(object$networks))
    stop("Expected a fitted nf_pu_forest")
  ans <- object
  ans$forest <- NULL
  ans$forest_align <- NULL
  ans$feature_spec <- NULL
  ans$feature_fraction <- NULL
  ans$feature_diagnostics <- NULL
  ans$rank_map <- NULL
  ans$residual_trees <- NULL
  ans$alpha <- 1
  ans$mode <- "neural_matched"
  ans
}

#' Optional verified-outcome stability gate for PU forest
#'
#' A separate set of verified binary outcomes, never used to fit any
#' candidate, decides whether to deploy the guided forest or its own
#' exactly matched neural teacher. Disjoint stratified audit halves
#' must BOTH meet an objective gain and calibration guardrails. When
#' evidence is weak the teacher is selected. Subsequent test outcomes
#' MUST remain untouched: this is not a generalization guarantee.
#'
#' @param object Previously fitted PU forest.
#' @param audit_x Reserved disjoint predictors, same columns as fitting.
#' @param audit_y Verified positive (1) and verified negative (0) outcomes,
#'   NOT positive/unlabeled PU indicators.
#' @param objective Prespecified "auc" or "pr_auc".
#' @param min_gain Minimum strict primary-metric gain on each half.
#' @param max_ap_loss Maximum permissible AP loss on each half.
#' @param max_brier_increase Maximum permissible Brier increase per half.
#' @param min_group_half Minimum verified positives and negatives per half.
#' @param seed Prespecified random seed for stratified audit subdivision.
#' @return Model with selection metadata. No retraining is performed.
#' @export
nf_pu_verified_gate <- function(object,audit_x,audit_y,
   objective=c("auc","pr_auc"),min_gain=.002,max_ap_loss=.01,
   max_brier_increase=.01,min_group_half=15L,seed=42L) {
  objective <- match.arg(objective)
  if(!inherits(object,"nf_pu_forest") ||
     !(is.matrix(audit_x)||is.data.frame(audit_x)) ||
     nrow(audit_x)!=length(audit_y))
    stop("Need a fitted PU model and matched audit predictors")
  yy <- as.character(audit_y)
  if(anyNA(yy)||any(!yy %in% c("0","1")))
    stop("Audit labels must be verified binary outcomes 0 or 1")
  y <- as.integer(yy)
  controls <- c(min_gain,max_ap_loss,max_brier_increase,min_group_half,seed)
  if(any(!is.finite(controls))||min_gain<0||max_ap_loss<0||
     max_brier_increase<0||min_group_half<5L||seed<0||
     length(min_gain)!=1L||length(max_ap_loss)!=1L||
     length(max_brier_increase)!=1L||length(min_group_half)!=1L||
     length(seed)!=1L)
    stop("Invalid verified-gate controls")
  beginning <- proc.time()[["elapsed"]]
  teacher <- nf_pu_matched_teacher(object)
  enough <- min(tabulate(y+1L,nbins=2L))>=2L*min_group_half
  result <- teacher
  history <- data.frame(half=integer(),teacher_auc=numeric(),
      candidate_auc=numeric(),teacher_ap=numeric(),candidate_ap=numeric(),
      teacher_brier=numeric(),candidate_brier=numeric(),
      primary_gain=numeric(),eligible=logical())
  if(enough) {
    p0 <- as.numeric(predict(teacher,audit_x,type="response"))
    p1 <- as.numeric(predict(object,audit_x,type="response"))
    split <- nf_split_indices(y,"classification",.49,as.integer(seed)+819L)
    folds <- list(split$train,split$validation)
    for(j in seq_along(folds)) {
      ii <- folds[[j]]
      if(min(tabulate(y[ii]+1L,nbins=2L))<min_group_half) {
        enough <- FALSE
        break
      }
      score <- function(p,metric) as.numeric(
        nf_metric(y[ii],p[ii],metric,task="classification"))
      auc0 <- score(p0,"auc");auc1 <- score(p1,"auc")
      ap0 <- score(p0,"pr_auc");ap1 <- score(p1,"pr_auc")
      brier0 <- score(p0,"brier");brier1 <- score(p1,"brier")
      delta <- if(objective=="auc") auc1-auc0 else ap1-ap0
      pass <- all(is.finite(c(auc0,auc1,ap0,ap1,brier0,brier1))) &&
        delta>min_gain && ap1>=ap0-max_ap_loss &&
        brier1<=brier0+max_brier_increase
      history <- rbind(history,data.frame(half=j,teacher_auc=auc0,
         candidate_auc=auc1,teacher_ap=ap0,candidate_ap=ap1,
         teacher_brier=brier0,candidate_brier=brier1,
         primary_gain=delta,eligible=pass))
    }
    if(enough && nrow(history)==2L && all(history$eligible))
      result <- object
  }
  accepted <- enough && nrow(history)==2L && all(history$eligible)
  result$verified_gate <- list(
      selected=if(accepted) "candidate" else "matched_teacher",
      objective=objective,verified_n=length(y),
      verified_positive=sum(y==1L),verified_negative=sum(y==0L),
      sufficient_class_support=enough,
      min_gain=min_gain,max_ap_loss=max_ap_loss,
      max_brier_increase=max_brier_increase,
      half_comparisons=history,
      selection_seconds=proc.time()[["elapsed"]]-beginning)
  result
}
