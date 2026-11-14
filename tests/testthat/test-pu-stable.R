test_that("native uncertainty sampling keeps class mass and ESS safe", {
  set.seed(2401)
  n <- 240L
  p <- matrix(runif(n*6L,.05,.95),nrow=n,ncol=6L)
  y <- c(rep(1L,80L),rep(0L,160L))
  original <- NeuralForest:::nf_pu_cpp_stable_weights(
    p,as.numeric(y),0,.65,.35,.96)
  expect_equal(as.numeric(original$weights),rep(1,n),tolerance=1e-12)
  guided <- NeuralForest:::nf_pu_cpp_stable_weights(
    p,as.numeric(y),.8,.65,.35,.96)
  expect_equal(as.numeric(guided$weights[y==1L]),rep(1,80))
  expect_equal(mean(guided$weights[y==0L]),1,tolerance=1e-10)
  expect_true(all(is.finite(guided$weights)))
  expect_true(all(guided$weights>0))
  expect_gte(as.numeric(guided$ess_fraction),.96-1e-10)
  expect_error(NeuralForest:::nf_pu_cpp_stable_weights(
    p,as.numeric(y),1,.65,.35,1.1),"Invalid")
  bad <- p
  bad[1,1] <- NA_real_
  expect_error(NeuralForest:::nf_pu_cpp_stable_weights(
    bad,as.numeric(y),.5,.65,.35,.96),"finite")
})

test_that("crossfitted probabilities have finite held-out values", {
  set.seed(2402)
  x <- matrix(rnorm(280*3L),ncol=3L)
  y <- c(rep(1L,120L),rep(0L,160L))
  probs <- NeuralForest:::nf_pu_stable_oof(
    x,y,c(.12,.25,.40),hidden=8L,epochs=8L,seed=2402L)
  expect_equal(dim(probs),c(280L,6L))
  expect_true(all(is.finite(probs)))
  expect_true(all(probs>0 & probs<1))
  no_oof <- NeuralForest:::nf_pu_stable_oof(
    x,c(rep(1L,35L),rep(0L,245L)),
    c(.12,.25,.40),8L,8L,2402L)
  expect_null(no_oof)
})

test_that("stable PU selection can safely fall back to teacher-only", {
  set.seed(2403)
  n <- 300L
  x <- data.frame(a=rnorm(n),b=runif(n),c=rnorm(n))
  y <- c(rep(1L,125L),rep(0L,175L))
  model <- nf_pu_stable_forest(
    x,y,seed=2403L,hidden=8L,epochs=9L,n_trees=12L,
    min_gain=1e6,min_ess=.97)
  expect_identical(model$selected_candidate,"neural_fallback")
  expect_null(model$forest)
  expect_equal(model$alpha,1)
  expect_true(is.data.frame(model$selection_history))
  expect_equal(sum(model$selection_history$chosen),1L)
  expect_true(model$stability_diagnostics$crossfit_used)
  pred <- predict(model,x[1:20,,drop=FALSE],type="prob")
  expect_equal(dim(pred),c(20L,2L))
  expect_equal(rowSums(pred),rep(1,20),tolerance=1e-10)
  expect_true(all(is.finite(pred)))
  expect_true(model$timings$selection_seconds>=0)
  expect_true(model$timings$final_fit_seconds>=0)
})

test_that("PU stability gate never accepts deterioration on either half", {
  y <- c(1L,0L,1L,0L,1L,0L,1L,0L)
  folds <- list(1:4,5:8)
  priors <- c(.12,.25,.4)
  teacher <- c(.9,.1,.8,.2,.7,.3,.6,.4)
  good <- c(.95,.05,.85,.15,.75,.25,.65,.35)
  bad <- c(.95,.05,.85,.15,.05,.95,.05,.95)
  a <- NeuralForest:::nf_pu_stable_risks(y,teacher,priors,folds)
  b <- NeuralForest:::nf_pu_stable_risks(y,good,priors,folds)
  c <- NeuralForest:::nf_pu_stable_risks(y,bad,priors,folds)
  expect_equal(length(a),2L)
  expect_true(all(b<a))
  expect_true(any(c>=a))
})


test_that("PU gating accounts for each prior on each holdout half", {
  y <- c(1L,0L,1L,0L,1L,0L,1L,0L)
  folds <- list(1:4,5:8)
  priors <- c(.1,.2,.4)
  p <- c(.9,.2,.8,.3,.7,.4,.6,.5)
  grid <- NeuralForest:::nf_pu_stable_risk_grid(y,p,priors,folds)
  expect_equal(dim(grid),c(3L,2L))
  expect_equal(as.numeric(apply(grid,2,max)),
    as.numeric(NeuralForest:::nf_pu_stable_risks(y,p,priors,folds)))
  expect_true(all(is.finite(grid)))
})
