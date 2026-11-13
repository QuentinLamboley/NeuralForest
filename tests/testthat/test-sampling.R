test_that("balanced undersampling draws equal classes without replacement", {
  set.seed(701)
  n <- 250L
  x <- data.frame(a=rnorm(n), b=runif(n), c=rnorm(n))
  y <- factor(c(rep("negative",225L),rep("positive",25L)))
  control <- nf_control(
    n_trees=32L,neural=FALSE,tree_sampling="balanced_undersample",
    class_balance="none",sample_fraction=0.8,replace=FALSE)
  model <- neural_forest(x,y,metric="auc",seed=701L,
                         control=control,refit=FALSE)
  counts <- model$native$sampled_class_counts
  expect_equal(dim(counts),c(32L,2L))
  expect_true(all(counts[,1L]==16L))
  expect_true(all(counts[,2L]==16L))
  expect_gt(min(model$native$oob_count),0)
  prob <- predict(model,x[1:10,,drop=FALSE],type="prob")
  expect_true(all(is.finite(prob)))
  expect_equal(rowSums(prob),rep(1,10),tolerance=1e-12)
})

test_that("regular forest sampling preserves its original class distribution", {
  set.seed(702)
  x <- data.frame(a=rnorm(200L),b=rnorm(200L))
  y <- factor(c(rep("negative",180L),rep("positive",20L)))
  fit <- neural_forest(x,y,metric="auc",seed=702L,
        control=nf_control(n_trees=45L,neural=FALSE,
                           tree_sampling="standard",
                           sample_fraction=0.8,replace=TRUE))
  n_draws <- fit$native$sampled_class_counts
  expect_true(all(rowSums(n_draws)==128L))
  expect_true(mean(n_draws[,2]) < mean(n_draws[,1]))
})

test_that("balanced sampling rejects regression and invalid modes", {
  expect_error(nf_control(tree_sampling="unknown"),"arg")
  x <- data.frame(a=seq_len(50),b=rnorm(50))
  y <- 0.1*seq_len(50)
  expect_error(
    neural_forest(x,y,task="regression",metric="rmse",
       control=nf_control(n_trees=10L,neural=FALSE,
                          tree_sampling="balanced_undersample")),
    "classification")
})

test_that("binary soft forest trains one normalized probability model", {
  set.seed(703)
  n <- 180L
  x <- data.frame(a=rnorm(n),b=rnorm(n))
  y <- factor(ifelse(x$a+0.4*x$b>0,"yes","no"))
  model <- neural_forest(x,y,metric="auc",seed=703L,
      control=nf_control(n_trees=15L,neural=FALSE,
          differentiable=TRUE,soft_n_trees=2L,
          soft_depth=2L,soft_epochs=7L,soft_patience=3L))
  expect_length(model$soft_model$models,1L)
  p <- NeuralForest:::nf_soft_probabilities(
     model$soft_model,
     NeuralForest:::nf_encoding(x[1:12,,drop=FALSE],
                               fit=model$encoding)$X)
  expect_equal(dim(p),c(12L,2L))
  expect_equal(rowSums(p),rep(1,12),tolerance=1e-12)
  expect_true(all(is.finite(p)))
})

test_that("balanced informed forest executes additional neural rounds", {
  set.seed(704)
  x <- data.frame(a=rnorm(250),b=rnorm(250))
  y <- factor(ifelse(seq_len(250)%%10==0,"positive","negative"))
  m <- neural_forest(x,y,metric="auc",seed=704L,
       control=nf_control(n_trees=20L,neural=TRUE,
          neural_rounds=3L,patience=3L,
          neural_hidden=c(12L,6L),neural_epochs=3L,
          tree_sampling="balanced_undersample",
          sample_fraction=0.75))
  expect_equal(nrow(m$history),3L)
  expect_true(is.finite(m$validation_score))
})
