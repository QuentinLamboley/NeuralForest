test_that("training-only feature preparation does not inspect validation covariates", {
  set.seed(201)
  n <- 150L
  x <- data.frame(a=rnorm(n),b=rnorm(n))
  x$a[c(4L,8L)] <- NA_real_
  y <- factor(rep(c("no","yes"),length.out=n))
  fit <- neural_forest(x,y,metric="auc",seed=201,
             control=nf_control(n_trees=12L,neural=FALSE))
  modified <- x
  modified$a[fit$validation_rows] <- 99999
  modified$b[fit$validation_rows] <- -99999
  fit2 <- neural_forest(modified,y,metric="auc",seed=201,
             control=nf_control(n_trees=12L,neural=FALSE))
  expect_equal(fit$encoding$numeric_medians,
               fit2$encoding$numeric_medians)
  expect_equal(fit$target_profile$feature_profile,
               fit2$target_profile$feature_profile)
  expect_equal(fit$native$importance,fit2$native$importance)
})
test_that("single and legacy double weighting produce finite native outputs", {
  set.seed(202)
  n <- 150L
  x <- data.frame(a=rnorm(n),b=runif(n))
  y <- factor(ifelse(x$a>0,"yes","no"))
  for(mode in c("single","double")) {
    fit <- neural_forest(x,y,metric="auc",seed=202,
        control=nf_control(neural=FALSE,n_trees=20L,
                           sampling_weight_mode=mode))
    prob <- predict(fit,x[1:10,,drop=FALSE],type="prob")
    expect_equal(dim(prob),c(10L,2L))
    expect_true(all(is.finite(prob)))
  }
})
test_that("adaptive experts are selected without external test observations", {
  set.seed(203)
  n <- 350L
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=rnorm(n))
  odds <- stats::plogis(0.1+1.5*x$a-0.6*x$b)
  y <- factor(ifelse(stats::runif(n)<odds,"yes","no"))
  cfg <- nf_control(n_trees=25L,neural=FALSE,
                    differentiable=TRUE,
                    soft_n_trees=2L,soft_depth=2L,soft_epochs=6L,
                    linear_expert="auto",
                    linear_penalties=c(0.05,0.5))
  fit <- neural_forest(x,y,metric="logloss",control=cfg,seed=203,
                       refit=TRUE)
  p <- predict(fit,x[1:12,,drop=FALSE],type="prob")
  expect_equal(dim(p),c(12L,2L))
  expect_true(all(is.finite(p)))
  expect_true(all(abs(rowSums(p)-1)<1e-6))
  expect_true(fit$expert_selection$type %in% c("prior","forest","linear"))
  expect_true(all(c("native_selection_seconds","soft_selection_seconds",
                    "expert_selection_seconds","final_refit_seconds")
                  %in% names(fit$timings)))
})
test_that("rare events use a differentiable average-precision objective", {
  set.seed(204)
  n <- 300L
  x <- data.frame(a=rnorm(n),b=rnorm(n))
  y <- factor(ifelse(seq_len(n) %% 20L == 0L,"yes","no"))
  cfg <- nf_control(n_trees=12L,neural=FALSE,
                    differentiable=TRUE,
                    soft_n_trees=2L,soft_depth=2L,soft_epochs=5L,
                    soft_batch=64L)
  fit <- neural_forest(x,y,metric="pr_auc",control=cfg,seed=204)
  expect_equal(fit$soft_model$surrogate,"ap")
  expect_true(is.finite(fit$validation_score))
})
