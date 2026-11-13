test_that("soft training uses no outer validation outcomes", {
  set.seed(831)
  n <- 160L
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=runif(n))
  original <- factor(ifelse(x$a+0.5*x$b+rnorm(n,sd=.4)>0,
                            "positive","negative"),
                     levels=c("negative","positive"))
  groups <- seq_len(n) # Keep outer partition identical if labels change.
  indices <- NeuralForest:::nf_split_indices(
      as.numeric(original)-1L,"classification",0.2,831L,groups)
  altered <- original
  altered[indices$validation] <- factor(
    ifelse(original[indices$validation]=="positive",
           "negative","positive"),
    levels=levels(original))
  control <- nf_control(n_trees=15L,neural=FALSE,
    differentiable=TRUE,soft_n_trees=2L,soft_depth=2L,
    soft_epochs=7L,soft_patience=3L)
  before <- neural_forest(x,original,control=control,metric="auc",
                          seed=831L,groups=groups)
  after <- neural_forest(x,altered,control=control,metric="auc",
                         seed=831L,groups=groups)
  X <- NeuralForest:::nf_encoding(x[1:20,,drop=FALSE],
                                  fit=before$encoding)$X
  trained_a <- NeuralForest:::nf_soft_probabilities(before$soft_model,X)
  trained_b <- NeuralForest:::nf_soft_probabilities(after$soft_model,X)
  expect_equal(trained_a,trained_b,tolerance=1e-11)
  expect_equal(before$validation_rows,after$validation_rows)
  expect_true(is.finite(before$timings$soft_selection_seconds))
})

test_that("calibration and soft timing stages are nonoverlapping", {
  set.seed(832)
  x <- data.frame(a=rnorm(180),b=rnorm(180))
  y <- factor(ifelse(x$a+0.2*x$b+rnorm(180)>.4,"yes","no"))
  model <- neural_forest(x,y,metric="brier",seed=832L,
    control=nf_control(n_trees=16L,neural=FALSE,
      calibration="auto",differentiable=TRUE,soft_n_trees=2L,
      soft_depth=2L,soft_epochs=5L,soft_patience=2L))
  t <- model$timings
  accounted <- t$preprocess_seconds+t$native_selection_seconds+
    t$calibration_selection_seconds+t$soft_selection_seconds+
    t$expert_selection_seconds+t$final_refit_seconds
  expect_lte(accounted,t$total_training_seconds+0.025)
  expect_true(is.finite(t$calibration_selection_seconds))
})
