# Explicit observation-effort offset for presence-only ecological SDMs.
#
# For observed presences from intensity lambda_obs(s)=effort(s)*lambda_habitat(s)
# and a quadrature background sampled with inclusion density r(s), the
# habitat log-relative-intensity f(s) is estimated by:
#
#   L(f) = log(sum_bg w_avail(s) * effort(s) * exp(f(s))
#                / sum_bg w_avail(s) * effort(s)) - mean_P f(s)
#
# where w_avail is a known, independent available-area inverse-inclusion
# weight. The term mean_P log(effort) is constant in f and is omitted.
#
# Reusing the native IPP C++ optimizer/gradient with the product
# availability_weights * effort is MATHEMATICALLY EXACT for this loss.
# The model's reported habitat suitability is normalized with
# availability_weights ALONE (NOT with detection-effort weights).
#
# Effort must be genuinely independently measured or prespecified.
# Estimating it from the same focal presence response can confound
# detectability and ecology. Never use survey evaluation outcomes.

nf_po_effort_weights <- function(effort,availability,n,scale=NULL) {
  if(!is.numeric(effort)||length(effort)!=n||
     any(!is.finite(effort))||any(effort<=0))
    stop("Independent background effort must be finite and positive at every available site")
  if(is.null(availability))availability<-rep(1,n)
  if(!is.numeric(availability)||length(availability)!=n||
     any(!is.finite(availability))||any(availability<=0))
    stop("Background area inclusion weights must be positive and match available sites")
  if(is.null(scale))scale<-sum(availability*effort)/sum(availability)
  if(!is.finite(scale)||scale<=0)
    stop("Invalid detection-effort reference scale")
  # Constant effort cancels analytically from the IPP partition
  # function. Preserve EXACT native teacher equivalence across
  # platforms instead of injecting round-off from q / mean(q).
  weights<-if(all(effort==effort[[1L]]))
    availability else availability*(effort/scale)
  if(any(!is.finite(weights))||any(weights<=0))
    stop("Non-finite effort-corrected IPP quadrature weights")
  list(weights=weights,availability=availability,
       effort_scale=scale,
       effort_range=range(effort),
       effective_background_samples=(sum(weights)^2/sum(weights^2)))
}

#' Presence-only forest with an independently measured effort offset
#'
#' Fits relative ECOLOGICAL intensity while explicitly separating a
#' known relative observation-effort surface from accessible habitat.
#' For unbiased uniformly sampled background, \`availability_weights\`
#' may be all ones; for nonuniform quadrature it must approximate
#' inverse background inclusion probability independently of the
#' focal species. Effort must be known or estimated from data external
#' to this focal species (survey effort logs, documented target-group
#' reporting intensity), not inferred using the held-out test label.
#'
#' The C++ native network and IPP-gradient forest use the exact Poisson
#' objective obtained by multiplying available-area weights by the
#' independent observation-effort offset on each background row.
#' Scores remain HABITAT relative intensity; the offset is excluded
#' from prediction and from background habitat score normalization.
#' It is not an absolute occupancy, prevalence or disease probability.
#'
#' Unknown effort is NOT identifiable from focal presence-only data
#' without additional assumptions. This is an experimental research
#' interface, not automatic bias correction or a proven improvement.
#'
#' @param presences,background Training presence and accessible-area
#'   predictors, identically named and ordered.
#' @param effort_background Positive independently documented relative
#'   sampling/detection effort at training background locations.
#' @param availability_weights Positive inverse-inclusion area weights
#'   for the background sampling design, not survey detection effort.
#' @param method Native \`neural\` IPP teacher or \`gradient\` IPP forest.
#' @param feature_mode For gradient tree splits: learned native hidden,
#'   raw predictors or equal-sized untrained random-tanh control.
#' @param validation_presences,validation_background Disjoint inner
#'   training-only validation data for gradient stage selection.
#' @param validation_effort_background Independently measured effort
#'   at the disjoint inner background locations; required if supplied.
#' @param validation_availability_weights Inner background quadrature
#'   inverse-inclusion weights, if any.
#' @param seed,hidden,epochs Shared native C++ network controls.
#' @param max_stages,trees_per_stage,max_depth,min_leaf,hidden_features,
#'   latent_mass,step_size,min_gain Gradient-forest controls.
#' @return Fitted \`nf_po_forest\` or \`nf_po_gradient_forest\`, with
#'   \`effort_design\` metadata and habitat-only relative scores.
#' @export
nf_po_effort_forest <- function(presences,background,
  effort_background,availability_weights=NULL,
  method=c("gradient","neural"),
  feature_mode=c("hidden","raw","random"),
  validation_presences=NULL,validation_background=NULL,
  validation_effort_background=NULL,
  validation_availability_weights=NULL,
  seed=42L,hidden=12L,epochs=75L,
  max_stages=4L,trees_per_stage=5L,
  max_depth=3L,min_leaf=8L,hidden_features=8L,
  latent_mass=.35,step_size=.12,min_gain=.002) {
  method<-match.arg(method)
  feature_mode<-match.arg(feature_mode)
  if(!(is.matrix(presences)||is.data.frame(presences))||
     !(is.matrix(background)||is.data.frame(background))||
     !identical(colnames(presences),colnames(background)))
    stop("Presence and available-area predictors must be matching named matrices or data frames")
  if(nrow(presences)<50L||nrow(background)<80L)
    stop("Effort-aware native IPP needs >=50 presences and >=80 background sites")
  if(length(seed)!=1L||!is.finite(seed))
    stop("Invalid reproducible effort-aware model seed")
  if(xor(is.null(validation_presences),is.null(validation_background)))
    stop("Provide BOTH or neither of the inner validation predictor groups")
  valid_inner<-!is.null(validation_background)
  if(valid_inner && (
    !identical(colnames(presences),colnames(validation_presences))||
    !identical(colnames(background),colnames(validation_background))))
    stop("Inner validation predictor columns differ from training columns")
  if(method=="neural"&&
      (valid_inner||!is.null(validation_effort_background)||
       !is.null(validation_availability_weights)))
    stop("Neural-only native IPP does not use an inner gradient-stage validation cohort")
  if(method=="gradient" && valid_inner &&
     is.null(validation_effort_background))
    stop("Independent observation effort is required for every inner validation background site")
  if(!valid_inner && (
    !is.null(validation_effort_background)||
    !is.null(validation_availability_weights)))
    stop("Inner effort or inclusion weights require a disjoint validation background")
  training<-nf_po_effort_weights(
    effort_background,availability_weights,nrow(background))
  vweights<-NULL
  if(valid_inner) {
    vweights<-nf_po_effort_weights(
       validation_effort_background,
       validation_availability_weights,
       nrow(validation_background),training$effort_scale)
  }
  begin<-proc.time()[["elapsed"]]
  if(method=="neural") {
    model<-nf_po_forest(presences,background,
       background_weights=training$weights,mode="neural",
       seed=as.integer(seed),hidden=hidden,epochs=epochs)
    X<-nf_encoding(background,fit=model$encoding)$X
    raw<-nf_po_raw_score(model,X)
  } else {
    model<-nf_po_gradient_forest(
       presences,background,
       validation_presences=validation_presences,
       validation_background=validation_background,
       background_weights=training$weights,
       validation_background_weights=if(valid_inner)vweights$weights else NULL,
       refit_validation=TRUE,
       feature_mode=feature_mode,
       seed=as.integer(seed),hidden=hidden,epochs=epochs,
       max_stages=max_stages,trees_per_stage=trees_per_stage,
       max_depth=max_depth,min_leaf=min_leaf,
       hidden_features=hidden_features,latent_mass=latent_mass,
       step_size=step_size,min_gain=min_gain)
    raw<-nf_po_boost_raw_scores(model,background)
  }
  # Unlike the OBSERVATION-intensity partition function, the
  # HABITAT-relative intensity denominator is area-based only.
  # With an explicit inner CV refit on all outer-training groups,
  # normalizing on *initial* training background still defines a
  # valid fixed accessible reference without using a test location.
  model$reference_log_mean_intensity<-nf_po_logmeanexp(
    raw,training$availability)
  model$effort_design<-list(
    observation_effort_offset="independent_relative_background_effort",
    area_quadrature="inverse_inclusion_weights_if_known",
    effort_scale=training$effort_scale,
    effort_min=training$effort_range[[1]],
    effort_max=training$effort_range[[2]],
    background_ess=training$effective_background_samples,
    validation_background_ess=if(valid_inner)
      vweights$effective_background_samples else NA_real_,
    normalization="area_only_not_detection_effort",
    label_semantics="opportunistic_presences_vs_available_environment",
    biological_absence_supplied=FALSE,
    effort_estimated_from_training_target=FALSE,
    test_outcomes_used=FALSE,
    method=method)
  model$training$effort_wrapper_total_seconds<-
    proc.time()[["elapsed"]]-begin
  model
}
