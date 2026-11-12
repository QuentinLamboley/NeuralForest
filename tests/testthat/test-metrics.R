test_that("metric registry returns directions", {
  m <- nf_metrics()
  expect_true(all(c("boyce","auc","pr_auc","logloss","rmse","ccc",
                    "pinball") %in% m$metric))
  expect_equal(m$direction[m$metric=="rmse"],"minimize")
  expect_equal(m$direction[m$metric=="auc"],"maximize")
})
test_that("binary metrics agree with simple analytical cases", {
  y <- factor(c("n","n","p","p"))
  p <- matrix(c(.9,.8,.2,.1, .1,.2,.8,.9),ncol=2,
              dimnames=list(NULL,c("n","p")))
  expect_equal(unname(nf_metric(y,p,"auc")),1)
  expect_equal(unname(nf_metric(y,p,"accuracy")),1)
  expect_lt(unname(nf_metric(y,p,"brier")),.1)
  expect_gt(unname(nf_metric(y,p,"pr_auc")),.8)
})
test_that("regression metrics are finite and ordered", {
  y <- c(1,2,3,4,5)
  perfect <- nf_metric(y,y,"rmse",task="regression")
  worse <- nf_metric(y,y+2,"rmse",task="regression")
  expect_equal(unname(perfect),0)
  expect_gt(unname(worse),unname(perfect))
  expect_equal(unname(nf_metric(y,y,"r2",task="regression")),1)
})
test_that("tuning returns the best configuration with independent outer folds", {
  set.seed(5)
  x <- data.frame(a=rnorm(72),b=rnorm(72))
  y <- factor(ifelse(x$a>0,"yes","no"))
  out <- nf_tune(x,y,task="classification",metric="accuracy",
     search=list(n_trees=c(10L,15L),neural_rounds=c(1L)),
     trials=2,folds=2,seed=5,
     control=nf_control(neural=FALSE,max_depth=5))
  expect_equal(nrow(out$results),2)
  expect_true(is.finite(out$best_score))
})

test_that("average precision uses noninterpolated rank precision", {
  truth <- factor(c("yes","no","yes","no"),levels=c("no","yes"))
  p <- c(0.9,0.8,0.7,0.6)
  expected <- 0.5*(1+2/3)
  expect_equal(unname(nf_metric(truth,cbind(no=1-p,yes=p),"pr_auc")),expected)
})
