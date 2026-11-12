test_that("binary forest learns a clear signal", {
  set.seed(11)
  n <- 160
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=sample(letters[1:3],n,TRUE))
  y <- factor(ifelse(x$a+x$b>0,"yes","no"))
  fit <- neural_forest(x,y,metric="auc",seed=1,
    control=nf_control(n_trees=35,neural_rounds=2,neural_epochs=5,
                       neural_hidden=c(12,6),min_leaf=2,max_depth=8))
  pred <- predict(fit,x[1:12,,drop=FALSE],type="prob")
  expect_equal(dim(pred),c(12L,2L))
  expect_true(all(abs(rowSums(pred)-1)<1e-6))
  expect_true(is.finite(fit$validation_score))
  expect_true(length(nf_importance(fit))>=2)
  expect_true(nrow(nf_diagnostics(fit)$history)>=1)
})
test_that("multiclass and factor levels work in predict", {
  set.seed(13)
  id <- sample(seq_len(nrow(iris)),120)
  fit <- neural_forest(iris[id,1:4],iris$Species[id],
       metric="macro_auc",seed=13,
       control=nf_control(n_trees=30,neural_rounds=1,min_leaf=2))
  p <- predict(fit,iris[-id,1:4],type="prob")
  expect_identical(colnames(p),levels(iris$Species))
  expect_true(all(is.finite(p)))
  expect_equal(nrow(p),30L)
  expect_equal(length(predict(fit,iris[-id,1:4])),30L)
})
test_that("regression, missing values, and native OOB predictions", {
  set.seed(31)
  n <- 110
  x <- data.frame(x=rnorm(n),z=rnorm(n))
  y <- 4*x$x-1.5*x$z+rnorm(n,sd=.1)
  x$x[c(1,8)] <- NA
  fit <- neural_forest(x,y,task="regression",seed=31,
     control=nf_control(n_trees=35,neural_rounds=2,neural_epochs=5,
                        max_depth=8,min_leaf=2))
  expect_equal(length(predict(fit,x[1:11,,drop=FALSE])),11L)
  expect_true(is.finite(fit$validation_score))
  expect_true(any(fit$native$oob_count>0))
  expect_equal(dim(predict(fit,x[1:3,,drop=FALSE],type="trees")),
               c(3L,1L,length(fit$native$trees)))
})
test_that("control validation catches malformed requests", {
  expect_error(nf_control(neural_guidance=1.5))
  expect_error(nf_control(sample_fraction=1.2,replace=FALSE))
  expect_error(neural_forest(matrix(1,5,2),1:5), "At least 20")
})

test_that("native histogram splitting provides valid predictions", {
  set.seed(70)
  x <- data.frame(a=rnorm(110),b=runif(110))
  y <- factor(ifelse(x$a>0,"yes","no"))
  fit <- neural_forest(x,y,metric="auc",seed=70,
       control=nf_control(neural=FALSE,n_trees=15,
                          split_search="histogram",histogram_bins=24,
                          max_candidates=0,min_leaf=2))
  p <- predict(fit,x[1:10,,drop=FALSE],type="prob")
  expect_equal(dim(p),c(10L,2L))
  expect_true(all(abs(rowSums(p)-1)<1e-6))
})

test_that("differentiable splits receive analytically correct gradients", {
  expect_lt(NeuralForest:::nf_soft_gradient_check(),1e-6)
})
test_that("soft trees train and blend without accessing the external test data", {
  set.seed(101)
  n <- 120L
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=rnorm(n))
  y <- factor(ifelse(x$a + 0.5*x$b > 0,"vampire","human"))
  fit <- neural_forest(x,y,task="classification",metric="auc",seed=101,
    control=nf_control(n_trees=20,neural=FALSE,
      differentiable=TRUE,soft_n_trees=3,soft_depth=2,
      soft_epochs=8,soft_batch=32,soft_patience=4))
  expect_true(!is.null(fit$soft_model))
  expect_true(is.finite(fit$soft_selection$score))
  p <- predict(fit,x[1:12,,drop=FALSE],type="prob")
  expect_equal(dim(p),c(12L,2L))
  expect_true(all(is.finite(p)))
  expect_true(all(abs(rowSums(p)-1)<1e-6))
  expect_true(all(fit$soft_selection$candidates>=0))
})
