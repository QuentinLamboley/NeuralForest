#' Configure an adaptive neural forest
#'
#' Controls tree growth, sampling, neural guidance, and compute resources.
#' Explicit values override distribution-aware defaults. Not every parameter
#' is appropriate for every sample size; use held-out evaluation.
#' @param n_trees Number of native C++ trees per round.
#' @param mtry Number of candidate features per split; NULL chooses automatically.
#' @param min_leaf Minimum observations per terminal node.
#' @param min_split Minimum observations to consider splitting.
#' @param max_depth Maximum tree depth (NULL means an adaptive default).
#' @param max_nodes Maximum total nodes per tree.
#' @param max_candidates Approximate maximum evaluated split positions per feature.
#' @param sample_fraction Fraction of training rows sampled per tree.
#' @param replace Whether to sample with replacement.
#' @param split_rule One of auto, gini, entropy, variance.
#' @param split_search Auto, exact (sorted), or histogram (binned splits).
#' @param histogram_bins Number of bins for histogram splits.
#' @param extra_random Randomly offset candidate split positions.
#' @param min_gain Minimum impurity gain to split.
#' @param laplace Additive class-probability smoothing.
#' @param n_threads Native OpenMP threads, when available.
#' @param neural Enable neural-guided resampling and feature selection.
#' @param neural_rounds Number of joint forest/MLP training rounds.
#' @param neural_hidden Hidden MLP widths; NULL adapts to input dimension.
#' @param neural_epochs Epochs of MLP training per round.
#' @param neural_batch MLP minibatch size.
#' @param neural_lr Adam learning rate.
#' @param neural_weight_decay AdamW-style decoupled weight decay.
#' @param neural_patience Neural validation early stopping patience.
#' @param neural_guidance Strength of learned observation sampling.
#' @param feature_guidance Strength of learned feature sampling.
#' @param neural_temperature Temperature used for difficulty weighting.
#' @param max_weight_ratio Upper bound on sample weight multiplier.
#' @param class_balance none, balanced or auto.
#' @param adaptive Adapt unset hyperparameters using the target profile.
#' @param patience Outer validation-based early stopping.
#' @param tree_sampling Standard sampling or balanced_undersample. The latter
#'   draws an equal number per class without replacement in every tree;
#'   the per-class draw is floor(sample_fraction * minority_class_count).
#'   Only classification supports balanced undersampling.
#' @param sampling_weight_mode Single applies sampling weights once; double
#'   preserves the legacy split-weighted bootstrap scheme.
#' @param linear_expert None or auto, selecting a ridge logistic or linear
#'   expert using a training-only validation set.
#' @param linear_penalties Positive ridge regularization strengths.
#' @param expert_blend Mixture coefficients with the linear expert.
#' @param expert_min_gain Minimum validation improvement for expert selection.
#' @param calibration None or auto, using regularized OOB Platt scaling for
#'   proper-scoring metrics and independent holdout selection.
#' @param calibration_penalties Positive regularization strengths for calibration.
#' @param calibration_min_gain Minimum holdout improvement required to calibrate.
#' @param differentiable Enable gradient-trained oblique soft trees.
#' @param soft_n_trees Differentiable trees in the auxiliary ensemble.
#' @param soft_depth Complete binary soft-tree depth (1 to 7).
#' @param soft_epochs Gradient training epochs.
#' @param soft_batch Gradient minibatch size.
#' @param soft_lr Adam step size for split and leaf parameters.
#' @param soft_temperature Soft sigmoid routing temperature.
#' @param soft_loss Auto or logloss, brier, focal, auc, mse surrogate.
#' @param soft_blend Auto validation-based blending or numeric weight.
#' @param soft_weight_decay Weight decay for soft trees.
#' @param soft_patience Gradient early-stopping patience.
#' @param soft_clip Global gradient norm clip.
#' @param verbose Print progress.
#' @return A control object.
#' @export
nf_control <- function(
  n_trees = 300L, mtry = NULL, min_leaf = NULL, min_split = NULL,
  max_depth = NULL, max_nodes = 100000L, max_candidates = 128L,
  sample_fraction = 0.8, replace = TRUE, split_rule = "auto",
  split_search = "auto", histogram_bins = 128L,
  extra_random = FALSE, min_gain = 0, laplace = 1, n_threads = 1L,
  neural = TRUE, neural_rounds = 3L, neural_hidden = NULL,
  neural_epochs = 35L, neural_batch = 256L, neural_lr = 0.002,
  neural_weight_decay = 1e-4, neural_patience = 6L,
  neural_guidance = 0.6, feature_guidance = 0.25,
  neural_temperature = 0.8, max_weight_ratio = 6,
  class_balance = "auto", adaptive = TRUE, patience = 2L,
  sampling_weight_mode = "single", tree_sampling="standard",
  linear_expert = "none",
  linear_penalties = c(0.0005,0.005,0.05,0.5,3), expert_blend = c(0.25, 0.5, 0.75, 1),
  expert_min_gain = 0.001,
  calibration = "none", calibration_penalties=c(0.02,0.15,1),
  calibration_min_gain=0.0001,
  differentiable = FALSE, soft_n_trees = 6L, soft_depth = 3L,
  soft_epochs = 35L, soft_batch = 128L, soft_lr = 0.015,
  soft_temperature = 0.8, soft_loss = "auto", soft_blend = "auto",
  soft_weight_decay = 1e-4, soft_patience = 7L, soft_clip = 4,
  verbose = FALSE
) {
  z <- as.list(environment())
  z$split_rule <- match.arg(split_rule, c("auto","gini","entropy","variance"))
  z$split_search <- match.arg(split_search, c("auto","exact","histogram"))
  z$class_balance <- match.arg(class_balance, c("auto","none","balanced"))
  z$sampling_weight_mode <- match.arg(sampling_weight_mode,c("single","double"))
  z$tree_sampling <- match.arg(tree_sampling,c("standard","balanced_undersample"))
  z$linear_expert <- match.arg(linear_expert,c("none","auto"))
  z$calibration <- match.arg(calibration,c("none","auto"))
  if(!is.numeric(calibration_penalties) || !length(calibration_penalties) ||
     any(!is.finite(calibration_penalties) | calibration_penalties<=0))
    stop("calibration_penalties must be positive and finite")
  if(length(calibration_min_gain)!=1L ||
     !is.finite(calibration_min_gain) || calibration_min_gain<0)
    stop("calibration_min_gain must be nonnegative and finite")
  if (!length(linear_penalties) || !is.numeric(linear_penalties) ||
      any(!is.finite(linear_penalties) | linear_penalties <= 0))
    stop("linear_penalties must be finite and positive")
  if (!length(expert_blend) || !is.numeric(expert_blend) ||
      any(!is.finite(expert_blend) | expert_blend <= 0 | expert_blend > 1))
    stop("expert_blend must lie in (0,1]")
  if (!is.finite(expert_min_gain) || expert_min_gain<0)
    stop("expert_min_gain must be nonnegative")
  z$soft_loss <- match.arg(soft_loss,
      c("auto","logloss","brier","focal","auc","ap","mse"))
  if (!(identical(soft_blend,"auto") ||
        (is.numeric(soft_blend) && length(soft_blend)>0L &&
         all(is.finite(soft_blend)) &&
         all(soft_blend>=0 & soft_blend<=1))))
    stop("soft_blend must be auto or numeric weights in [0,1]")
  if (!is.numeric(n_trees) || n_trees < 1 ||
      neural_rounds < 1 || min_gain < 0 || laplace < 0 ||
      sample_fraction <= 0 || (!replace && sample_fraction > 1) ||
      neural_guidance < 0 || neural_guidance > 1 ||
      feature_guidance < 0 || feature_guidance > 1 ||
      neural_temperature <= 0 || max_weight_ratio < 1 ||
      neural_epochs < 1 || neural_batch < 1 || n_threads < 1 ||
      max_candidates < 0 || histogram_bins < 8 ||
      max_nodes < 3 || patience < 1 ||
      soft_n_trees < 1 || soft_depth < 1 || soft_depth > 7 ||
      soft_epochs < 1 || soft_batch < 2 || soft_lr <= 0 ||
      soft_temperature <= 0 || soft_weight_decay < 0 ||
      soft_patience < 1 || soft_clip <= 0)
    stop("Invalid neural forest control parameters", call. = FALSE)
  if (!is.null(neural_hidden) &&
      (!length(neural_hidden) || any(!is.finite(neural_hidden)) ||
       any(neural_hidden < 1)))
    stop("neural_hidden must contain positive integers")
  class(z) <- "nf_control"
  z
}

nf_target_profile <- function(y, task, X) {
  n <- length(y)
  # Bounded deterministic sampling keeps diagnostics inexpensive on large data.
  subrows <- unique(as.integer(seq(1,n,length.out=min(n,4096L))))
  Z <- X[subrows,,drop=FALSE]
  feature_profile <- data.frame(
    feature=colnames(X),
    mean=colMeans(Z),
    sd=apply(Z,2,stats::sd),
    unique_fraction=apply(Z,2,function(v) length(unique(v))/length(v)),
    zero_fraction=colMeans(abs(Z)<1e-12),
    row.names=NULL
  )
  if (task == "classification") {
    counts <- tabulate(y + 1L, nbins = max(y) + 1L)
    prob <- counts / sum(counts)
    entropy <- -sum(prob[prob > 0] * log(prob[prob > 0]))
    return(list(task=task, n=n, p=ncol(X), counts=counts,
                prevalence=prob, entropy=entropy,
                imbalance=max(counts)/max(1,min(counts)),
                missing_fraction=attr(X,"missing_fraction"),
                feature_profile=feature_profile))
  }
  q <- as.numeric(stats::quantile(y, c(.05,.25,.5,.75,.95), names=FALSE))
  m <- mean(y); v <- stats::sd(y)
  list(task=task, n=n, p=ncol(X), quantiles=q,
       skewness=if (v > 0) mean(((y-m)/v)^3) else 0,
       kurtosis=if (v > 0) mean(((y-m)/v)^4) - 3 else 0,
       zero_fraction=mean(y==0), missing_fraction=attr(X,"missing_fraction"),
       feature_profile=feature_profile)
}

nf_resolve_control <- function(ctrl, n, p, task, profile) {
  if (is.null(ctrl$mtry))
    ctrl$mtry <- max(1L, if (task=="classification") floor(sqrt(p))
                      else floor(p/3))
  if (is.null(ctrl$min_leaf))
    ctrl$min_leaf <- if (isTRUE(ctrl$adaptive))
      max(2L, min(12L, floor(sqrt(n)/20))) else 2L
  if (is.null(ctrl$min_split)) ctrl$min_split <- max(4L, 2L * ctrl$min_leaf)
  if (is.null(ctrl$max_depth))
    ctrl$max_depth <- if (isTRUE(ctrl$adaptive))
      min(30L, max(6L, ceiling(log2(n)))) else 30L
  if (is.null(ctrl$neural_hidden)) {
    a <- max(12L, min(128L, 2L * p))
    b <- max(8L, min(64L, p))
    if (isTRUE(ctrl$adaptive) && task == "regression" &&
        abs(profile$skewness) > 2) a <- min(192L, a + 32L)
    if (isTRUE(ctrl$adaptive) && task == "classification") {
      a <- min(192L, a + min(48L,8L*max(0L,length(profile$counts)-2L)))
      if (profile$imbalance > 8) b <- min(96L,b+16L)
    }
    ctrl$neural_hidden <- c(a, b)
  }
  if (ctrl$split_rule == "auto")
    ctrl$split_rule <- if (task == "classification") "gini" else "variance"
  if (ctrl$split_search == "auto")
    ctrl$split_search <- if (n >= 5000) "histogram" else "exact"
  if (task == "regression" && ctrl$split_rule != "variance")
    stop("Regression supports split_rule='variance' only")
  if (task == "regression" && ctrl$tree_sampling=="balanced_undersample")
    stop("Balanced undersampling requires classification")
  if (task == "classification" && ctrl$split_rule == "variance")
    stop("Classification supports gini or entropy splits")
  ctrl$mtry <- max(1L, min(p, as.integer(ctrl$mtry)))
  for (nm in c("n_trees","min_leaf","min_split","max_depth","max_nodes",
               "max_candidates","histogram_bins","n_threads",
               "neural_rounds","neural_epochs",
               "neural_batch","neural_patience","patience",
               "soft_n_trees","soft_depth","soft_epochs","soft_batch",
               "soft_patience"))
    ctrl[[nm]] <- as.integer(ctrl[[nm]])
  if (ctrl$min_leaf < 1 || ctrl$min_split < 2 || ctrl$max_depth < 1)
    stop("Invalid tree shape parameters")
  ctrl
}

nf_encoding <- function(x, fit = NULL) {
  if (!is.data.frame(x) && !is.matrix(x))
    stop("x must be a data.frame or matrix")
  d <- as.data.frame(x, optional=TRUE, stringsAsFactors=FALSE)
  if (!ncol(d) || !nrow(d)) stop("Predictors cannot be empty")
  nm <- names(d)
  if (is.null(nm) || length(nm) != ncol(d)) nm <- rep("",ncol(d))
  nm[is.na(nm) | !nzchar(nm)] <- paste0("V",which(is.na(nm) | !nzchar(nm)))
  names(d) <- make.unique(make.names(nm))
  raw_missing <- vapply(d, function(v)
    mean(if (is.numeric(v)) !is.finite(v) else is.na(v)), numeric(1))
  if (is.null(fit)) {
    numeric_medians <- numeric(0)
    for (j in seq_along(d)) {
      if (inherits(d[[j]], c("Date","POSIXt"))) d[[j]] <- as.numeric(d[[j]])
      if (is.character(d[[j]]) || is.factor(d[[j]]) || is.logical(d[[j]])) {
        v <- as.character(d[[j]])
        v[is.na(v)] <- "(Missing)"
        d[[j]] <- factor(v, levels=unique(c(sort(unique(v)), "(Other)")))
      } else if (is.numeric(d[[j]])) {
        z <- as.numeric(d[[j]])
        z[!is.finite(z)] <- NA_real_
        m <- if (all(is.na(z))) 0 else stats::median(z,na.rm=TRUE)
        z[is.na(z)] <- m
        d[[j]] <- z
        numeric_medians[names(d)[j]] <- m
      } else stop("Unsupported predictor type: ",names(d)[j])
    }
    terms <- stats::terms(~ . - 1, data=d)
    xlev <- lapply(d[vapply(d, is.factor, logical(1))], levels)
    mat <- stats::model.matrix(terms, data=d, na.action=stats::na.pass)
    med <- apply(mat, 2L, function(z) {
      z[!is.finite(z)] <- NA_real_
      if (all(is.na(z))) 0 else stats::median(z, na.rm=TRUE)
    })
    structure <- list(input_names=names(d), terms=terms, xlev=xlev,
                      columns=colnames(mat), medians=med,
                      numeric_medians=numeric_medians)
  } else {
    structure <- fit
    if (!identical(names(d),structure$input_names))
      stop("newdata predictors must have the same names and order as training data")
    for (j in seq_along(d)) {
      nm <- names(d)[j]
      if (nm %in% names(structure$xlev)) {
        v <- as.character(d[[j]])
        v[is.na(v)] <- "(Missing)"
        v[!v %in% structure$xlev[[nm]]] <- "(Other)"
        d[[j]] <- factor(v, levels=structure$xlev[[nm]])
      } else {
        if (inherits(d[[j]], c("Date","POSIXt"))) d[[j]] <- as.numeric(d[[j]])
        if (!is.numeric(d[[j]])) stop("Numeric predictor has changed type: ",nm)
        z <- as.numeric(d[[j]])
        z[!is.finite(z)] <- structure$numeric_medians[[nm]]
        d[[j]] <- z
      }
    }
    mat <- stats::model.matrix(structure$terms, data=d, na.action=stats::na.pass,
                               xlev=structure$xlev)
    missing_cols <- setdiff(structure$columns,colnames(mat))
    if (length(missing_cols)) {
      addon <- matrix(0, nrow(mat), length(missing_cols),
                      dimnames=list(NULL,missing_cols))
      mat <- cbind(mat, addon)
    }
    mat <- mat[,structure$columns,drop=FALSE]
  }
  miss <- mean(raw_missing)
  for (j in seq_len(ncol(mat))) {
    bad <- !is.finite(mat[,j])
    if (any(bad)) mat[bad,j] <- structure$medians[j]
  }
  storage.mode(mat) <- "double"
  attr(mat,"missing_fraction") <- miss
  list(X=mat, encoding=structure)
}

nf_response <- function(y, task) {
  if (anyNA(y)) stop("Missing response values are not supported")
  if (task == "auto")
    task <- if (is.factor(y) || is.character(y) || is.logical(y) ||
                (is.numeric(y) && length(unique(y))==2L &&
                 all(sort(unique(y)) == c(0,1))))
      "classification" else "regression"
  if (!task %in% c("classification","regression"))
    stop("task must be 'auto', 'classification', or 'regression'")
  if (task == "classification") {
    f <- factor(y)
    if (nlevels(f) < 2L) stop("Classification requires at least two classes")
    return(list(y=as.numeric(f)-1, labels=levels(f), task=task))
  }
  if (!is.numeric(y) || any(!is.finite(y)))
    stop("Regression requires a finite numeric response")
  list(y=as.numeric(y), labels=NULL, task=task)
}

nf_split_indices <- function(y, task, validation, seed, groups=NULL) {
  n <- length(y)
  if (validation <= 0 || validation >= 0.5)
    stop("validation must be in (0,0.5); use groups for grouped holdout")
  set.seed(seed)
  if (!is.null(groups)) {
    if (length(groups)!=n || anyNA(groups)) stop("Invalid groups")
    u <- sample(unique(as.character(groups)))
    nval <- max(1, floor(length(u)*validation))
    if (nval >= length(u)) stop("At least two groups are needed")
    val <- which(as.character(groups) %in% u[seq_len(nval)])
  } else if (task=="classification") {
    val <- unlist(lapply(split(seq_len(n), y), function(i) {
      if (length(i)<=2L) integer(0) else
        sample(i,max(1L,min(length(i)-2L,round(validation*length(i)))))
    }), use.names=FALSE)
    if (!length(val)) val <- sample.int(n,max(1L, floor(validation*n)))
  } else val <- sample.int(n, max(1L, floor(validation*n)))
  list(train=setdiff(seq_len(n), val), validation=sort(val))
}

nf_native_cfg <- function(ctrl) {
  list(n_trees=ctrl$n_trees, mtry=ctrl$mtry, min_leaf=ctrl$min_leaf,
       min_split=ctrl$min_split, max_depth=ctrl$max_depth,
       max_nodes=ctrl$max_nodes, max_candidates=ctrl$max_candidates,
       sample_fraction=ctrl$sample_fraction, replace=ctrl$replace,
       split_rule=if (ctrl$split_rule=="entropy") 1L else 0L,
       histogram_mode=if (ctrl$split_search=="histogram") 1L else 0L,
       histogram_bins=ctrl$histogram_bins,
       extra_random=ctrl$extra_random, min_gain=ctrl$min_gain,
       laplace=ctrl$laplace, n_threads=ctrl$n_threads,
       split_weighted=identical(ctrl$sampling_weight_mode,"double"),
       sampling_mode=if(ctrl$tree_sampling=="balanced_undersample") 1L else 0L)
}
nf_base_weights <- function(y, task, ctrl) {
  w <- rep(1, length(y))
  if (task=="classification") {
    prop <- tabulate(y+1L)/length(y)
    ratio <- max(prop)/max(min(prop),1e-8)
    alpha <- if (ctrl$class_balance=="balanced") 1 else
      if (ctrl$class_balance=="auto") {
        if (ratio<=1.5) 0 else if (ratio<=3) 0.15 else
          if (ratio<=10) 0.30 else 0.40
      } else 0
    w <- (1 / pmax(prop[y+1L],1e-8))^alpha
  }
  w/mean(w)
}
nf_objective <- function(y, prediction, metric, task, labels) {
  actual <- if (task=="classification") labels[y+1L] else y
  score <- nf_metric(actual, prediction, metric=metric, task=task)
  if (length(score)!=1L || !is.finite(score)) NA_real_ else unname(score)
}
nf_hardness <- function(y, prediction, task) {
  if (task=="classification") {
    p <- prediction[cbind(seq_along(y),y+1L)]
    p <- pmin(1-1e-8,pmax(1e-8,p))
    raw <- -log(p)
  } else {
    raw <- abs(y-prediction[,1])
  }
  ok <- is.finite(raw)
  if (sum(ok) < 10) return(rep(NA_real_,length(y)))
  z <- rank(raw[ok],ties.method="average")/(sum(ok)+1)
  ans <- rep(NA_real_,length(y)); ans[ok] <- z
  ans
}

#' Train a neural-guided C++ random forest
#'
#' A fresh out-of-bag difficulty target is estimated after each forest round.
#' An MLP predicts which feature patterns are difficult, and its activations
#' and input weights guide the next forest's bootstraps and feature proposals.
#' Validation data are never used to fit the MLP or the candidate trees.
#' Performance superiority over other algorithms is NOT assumed.
#' @param x Data frame or numerical matrix of predictors.
#' @param y Numeric regression target, or a factor/class target.
#' @param task auto, classification or regression.
#' @param control Settings from nf_control().
#' @param metric Selection metric; see nf_metrics().
#' @param validation Fraction reserved for selection.
#' @param seed Reproducible pseudo-random seed.
#' @param groups Optional group IDs for leakage-resistant grouped holdout.
#' @param refit If TRUE, refit the selected structure on all observations.
#'   The reported holdout score is still a model-selection score, not
#'   an independent test-set estimate.
#' @return Object of class neural_forest.
#' @export
neural_forest <- function(x, y, task="auto", control=nf_control(),
                          metric=NULL, validation=0.2, seed=42L,
                          groups=NULL, refit=FALSE) {
  now <- function() proc.time()[["elapsed"]]
  t_initial <- now()
  if (!is.data.frame(x) && !is.matrix(x))
    stop("x must be a data.frame or matrix")
  n <- nrow(x)
  if (length(y)!=n) stop("Response length differs from predictor rows")
  resp <- nf_response(y,task); y <- resp$y; task <- resp$task
  if (n < 20) stop("At least 20 training observations are required")
  if (is.null(metric))
    metric <- if (task=="regression") "rmse" else
      if (length(resp$labels)==2) "auc" else "macro_auc"
  metric <- tolower(metric)
  if (!metric %in% nf_metrics(task)$metric)
    stop("Unsupported metric for this task: ", metric)
  split <- nf_split_indices(y, task, validation, seed, groups)
  tr <- split$train; vl <- split$validation
  ytr <- y[tr]; yval <- y[vl]
  if (task=="classification" && length(unique(ytr)) != length(resp$labels))
    stop("Training split does not contain every response class")
  enc <- nf_encoding(x[tr,,drop=FALSE])
  Xtr <- enc$X
  Xval <- nf_encoding(x[vl,,drop=FALSE],fit=enc$encoding)$X
  profile <- nf_target_profile(ytr,task,Xtr)
  ctrl <- nf_resolve_control(control,length(tr),ncol(Xtr),task,profile)
  K <- if (task=="regression") 1L else length(resp$labels)
  preparation_seconds <- now()-t_initial
  native_start <- now()
  w0 <- nf_base_weights(ytr,task,ctrl)
  w <- w0
  fw <- rep(1,ncol(Xtr))
  best <- NULL; best_nn <- NULL
  best_score <- if (nf_metric_direction(metric)=="maximize") -Inf else Inf
  stale <- 0L
  history <- data.frame(round=integer(),score=double(),oob_coverage=double(),
                        mean_nodes=double())
  nn <- NULL
  rounds <- if (isTRUE(ctrl$neural)) ctrl$neural_rounds else 1L
  for (round in seq_len(rounds)) {
    native <- nf_cpp_fit(Xtr, ytr, as.numeric(w), as.numeric(fw),
                         nf_native_cfg(ctrl), as.integer(task=="classification"),
                         K, as.integer(seed+round*104729L))
    pv <- nf_cpp_predict(native$trees,Xval,K,FALSE)
    if (task=="classification") colnames(pv) <- resp$labels
    score <- nf_objective(yval,pv,metric,task,resp$labels)
    better <- if (is.na(score)) FALSE else
      if (nf_metric_direction(metric)=="maximize")
        score > best_score else score < best_score
    if (is.null(best) || better) {
      best <- native; best_nn <- nn
      if (!is.na(score)) best_score <- score
      best_round <- round; stale <- 0L
    } else stale <- stale+1L
    coverage <- mean(native$oob_count>0)
    history <- rbind(history,
      data.frame(round=round,score=score,oob_coverage=coverage,
        mean_nodes=mean(vapply(native$trees, function(t) length(t$feature),numeric(1)))))
    if (isTRUE(ctrl$verbose))
      message(sprintf("round %d/%d | %s=%s | OOB %.1f%%",
        round,rounds,metric,format(score,digits=4),100*coverage))
    if (round>=rounds || stale >= ctrl$patience) break
    hardness <- nf_hardness(ytr,native$oob_prediction,task)
    available <- which(is.finite(hardness))
    if (length(available)<max(15,2*ncol(Xtr))) next
    nn <- nf_nn_train(Xtr[available,,drop=FALSE],hardness[available],
                      ctrl,seed+round,nn)
    difficulty <- nf_nn_predict(nn,Xtr)
    logits <- (difficulty - mean(difficulty))/ctrl$neural_temperature
    logits <- pmin(log(ctrl$max_weight_ratio),
                   pmax(-log(ctrl$max_weight_ratio),logits))
    guided <- exp(logits)
    guided <- guided/mean(guided)
    w <- w0 * ((1-ctrl$neural_guidance) +
                 ctrl$neural_guidance * guided)
    w <- pmax(1e-8,w/mean(w))
    strengths <- sqrt(rowSums(nn$layers[[1L]]$W^2))
    strengths <- strengths/mean(strengths)
    fw <- (1-ctrl$feature_guidance) +
      ctrl$feature_guidance * strengths
    fw <- pmax(1e-8,fw)
  }
  native_seconds <- now()-native_start
  soft_model <- NULL
  soft_selection <- NULL
  original_validation_score <- best_score
  hard_val <- nf_cpp_predict(best$trees,Xval,K,FALSE)
  if(task=="classification") colnames(hard_val) <- resp$labels
  calibration_start <- now()
  calibration <- NULL
  if(identical(ctrl$calibration,"auto") &&
     task=="classification" && K==2L) {
    calibration <- nf_platt_select(
      best,ytr,hard_val,yval,metric,resp$labels,ctrl)
    if(!is.null(calibration)) {
      hard_val <- nf_platt_apply(hard_val,calibration)
      best_score <- calibration$selected_score
    }
  }
  calibration_seconds <- now()-calibration_start
  soft_start <- now()
  if (isTRUE(ctrl$differentiable)) {
    # The outer holdout selects blend weights and experts. It cannot also
    # train/early-stop soft trees: that would reuse validation labels twice.
    # A distinct inner partition drawn *only* from training observations
    # provides gradient-training early stopping, including for rare cases.
    inner_soft <- nf_split_indices(ytr,task,0.15,seed+70313L)
    soft_model <- nf_soft_train(
      Xtr[inner_soft$train,,drop=FALSE],ytr[inner_soft$train],
      Xtr[inner_soft$validation,,drop=FALSE],ytr[inner_soft$validation],
      task,K,ctrl,metric,seed=seed+1919L)
    soft_val <- nf_soft_probabilities(soft_model,Xval)
    soft_selection <- nf_soft_select_blend(
      hard_val,soft_val,yval,metric,task,resp$labels,ctrl)
    if (is.finite(soft_selection$score)) best_score <- soft_selection$score
    alpha <- soft_selection$weight
    if(alpha>0) hard_val <- (1-alpha)*hard_val+alpha*soft_val
  }
  soft_seconds <- now()-soft_start
  expert_start <- now()
  expert_selection <- NULL
  if (identical(ctrl$linear_expert,"auto")) {
    expert_selection <- nf_expert_selection(
      hard_val,Xtr,ytr,Xval,yval,metric,task,K,resp$labels,ctrl)
    if(is.finite(expert_selection$score))
      best_score <- expert_selection$score
  }
  expert_seconds <- now()-expert_start
  refit_start <- now()
  if (isTRUE(refit)) {
    # Refit uses all training observations only after inner selection.
    X <- nf_encoding(x,fit=enc$encoding)$X
    full_w <- nf_base_weights(y,task,ctrl)
    full_fw <- rep(1,ncol(X))
    if (!is.null(best_nn)) {
      difficulty <- nf_nn_predict(best_nn,X)
      logits <- pmin(log(ctrl$max_weight_ratio),
        pmax(-log(ctrl$max_weight_ratio),
             (difficulty-mean(difficulty))/ctrl$neural_temperature))
      guided <- exp(logits); guided <- guided/mean(guided)
      full_w <- full_w * ((1-ctrl$neural_guidance)+ctrl$neural_guidance*guided)
      full_w <- full_w/mean(full_w)
      strength <- sqrt(rowSums(best_nn$layers[[1]]$W^2))
      full_fw <- (1-ctrl$feature_guidance)+
        ctrl$feature_guidance*strength/mean(strength)
    }
    best <- nf_cpp_fit(X,y,as.numeric(full_w),as.numeric(full_fw),
                       nf_native_cfg(ctrl),as.integer(task=="classification"),
                       K,as.integer(seed+999983L))
    if(!is.null(calibration)) {
      full_cal <- nf_platt_fit(best$oob_prediction,y,calibration$penalty)
      if(!is.null(full_cal)) calibration <- full_cal
    }
    if(!is.null(expert_selection) &&
       identical(expert_selection$type,"linear")) {
      expert_selection$model <- nf_expert_fit(
        X,y,task,K,expert_selection$penalty)
    }
  }
  refit_seconds <- now()-refit_start
  colnames(best$oob_prediction) <- if (task=="classification")
    resp$labels else "response"
  names(best$importance) <- colnames(Xtr)
  timings <- list(preprocess_seconds=preparation_seconds,
                  native_selection_seconds=native_seconds,
                  calibration_selection_seconds=calibration_seconds,
                  soft_selection_seconds=soft_seconds,
                  expert_selection_seconds=expert_seconds,
                  final_refit_seconds=refit_seconds,
                  total_training_seconds=now()-t_initial)
  z <- list(native=best,task=task,levels=resp$labels,encoding=enc$encoding,
            control=ctrl,metric=metric,validation_score=best_score,
            validation_rows=vl,trained_on=if(refit) n else length(tr),
            refit=refit,round=best_round,history=history,
            target_profile=profile,neural=best_nn,
            soft_model=soft_model,soft_selection=soft_selection,
            calibration=calibration,
            expert_selection=expert_selection,prior_target=if(refit)y else ytr,
            timings=timings,
            native_validation_score=original_validation_score)
  class(z) <- "neural_forest"
  z
}

#' @export
predict.neural_forest <- function(object,newdata,type=c("response","prob","class","trees"),
                                  ...) {
  type <- match.arg(type)
  X <- nf_encoding(newdata,fit=object$encoding)$X
  K <- if (object$task=="regression") 1L else length(object$levels)
  if (type=="trees") return(nf_cpp_predict(object$native$trees,X,K,TRUE))
  p <- nf_cpp_predict(object$native$trees,X,K,FALSE)
  if (!is.null(object$calibration))
    p <- nf_platt_apply(p,object$calibration)
  if (!is.null(object$soft_model) && !is.null(object$soft_selection)) {
    alpha <- object$soft_selection$weight
    if (alpha>0) {
      soft <- nf_soft_probabilities(object$soft_model,X)
      p <- (1-alpha)*p + alpha*soft
    }
  }
  if (!is.null(object$expert_selection)) {
    selected <- object$expert_selection
    if (selected$type=="linear") {
      expert <- nf_expert_predict(selected$model,X)
      p <- (1-selected$weight)*p+selected$weight*expert
    } else if (selected$type=="prior") {
      p <- nf_prior_probabilities(object$prior_target,K,object$task,nrow(X))
    }
  }
  if (object$task=="regression") {
    if (type %in% c("class","prob")) stop("Regression provides response or trees")
    return(as.numeric(p[,1]))
  }
  colnames(p) <- object$levels
  if (type=="prob") return(p)
  object$levels[max.col(p,ties.method="first")]
}

#' @export
print.neural_forest <- function(x,...) {
  cat(sprintf("NeuralForest %s | %d trees | %d rounds | selected round %d\n",
      x$task,length(x$native$trees),nrow(x$history),x$round))
  cat(sprintf("Holdout %s: %s | refit: %s\n",x$metric,
      format(x$validation_score,digits=4),x$refit))
  if (!is.null(x$soft_selection))
    cat(sprintf("Differentiable trees: %d | blend: %.2f | surrogate: %s\n",
       length(x$soft_model$models)*x$control$soft_n_trees,
       x$soft_selection$weight,x$soft_model$surrogate))
  invisible(x)
}
#' @export
summary.neural_forest <- function(object,...) {
  list(task=object$task,settings=object$control,
       target_profile=object$target_profile,
       validation_score=object$validation_score,
       history=object$history,
       importance=nf_importance(object))
}
#' Native impurity-decrease variable importance
#' @param object Fitted neural_forest object.
#' @param normalize Divide by total importance.
#' @export
nf_importance <- function(object,normalize=TRUE) {
  stopifnot(inherits(object,"neural_forest"))
  x <- object$native$importance
  if (normalize && sum(x)>0) x <- x/sum(x)
  sort(x,decreasing=TRUE)
}
#' Model diagnostics
#' @param object Fitted neural_forest.
#' @export
nf_diagnostics <- function(object) {
  stopifnot(inherits(object,"neural_forest"))
  list(history=object$history,target_profile=object$target_profile,
       validation_metric=object$metric,
       validation_score=object$validation_score,
       oob_coverage=mean(object$native$oob_count>0),
       feature_importance=nf_importance(object),
       gradient_surrogate=if (!is.null(object$soft_model))
         object$soft_model$surrogate else NULL,
       differentiable_blend=if (!is.null(object$soft_selection))
         object$soft_selection else NULL,
       selected_expert=if(!is.null(object$expert_selection))
         object$expert_selection$type else NULL,
       calibration=if(!is.null(object$calibration))
         object$calibration else NULL,
       timings=object$timings)
}
