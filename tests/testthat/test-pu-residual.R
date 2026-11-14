test_that("C++ PU residual gradients match finite-difference nnPU loss", {
  y <- c(rep(1L,60L),rep(0L,100L))
  set.seed(9101)
  p <- plogis(rnorm(160,sd=.55))
  pi <- .25
  ans <- NeuralForest:::nf_pu_cpp_residual_targets(
    p,as.numeric(y),pi,gradient_cap=100)
  expect_equal(length(ans$target),length(y))
  expect_true(all(is.finite(ans$target)))
  eps <- 1e-5
  for(i in c(2L,59L,68L,140L)) {
    z <- qlogis(p)
    plus <- z;plus[i] <- plus[i]+eps
    minus <- z;minus[i] <- minus[i]-eps
    d <- (NeuralForest:::nf_pu_risk(y,plogis(plus),pi)-
          NeuralForest:::nf_pu_risk(y,plogis(minus),pi))/(2*eps)
    expect_equal(as.numeric(ans$target[i]),-length(y)*d,
                 tolerance=3e-5)
  }
  expect_error(NeuralForest:::nf_pu_cpp_residual_targets(
    c(p[-1],NaN),as.numeric(y),pi),"finite teacher")
  expect_error(NeuralForest:::nf_pu_cpp_residual_targets(
    p,as.numeric(y),1.1),"prespecified PU")
})

test_that("native clipped logit correction is stable and order safe at zero", {
  p <- c(0,.000001,.1,.5,.9,.999999,1)
  correction <- c(-100,100,-.3,.3,-10,10,0)
  q <- NeuralForest:::nf_pu_cpp_logit_adjust(p,correction,1,.4)
  expect_equal(length(q),length(p))
  expect_true(all(is.finite(q) & q>0 & q<1))
  untouched <- NeuralForest:::nf_pu_cpp_logit_adjust(
    p,correction,0,.4)
  expect_equal(untouched,pmax(1e-10,pmin(1-1e-10,p)),
               tolerance=1e-12)
  expect_lte(max(abs(qlogis(q)-qlogis(untouched))),.4+1e-6)
  expect_error(NeuralForest:::nf_pu_cpp_logit_adjust(
    p,c(correction[-1],NA_real_),1,.4),"finite")
})

test_that("residual forest with impossible gain returns exact matched teacher", {
  set.seed(9102)
  n <- 310L
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=runif(n))
  y <- c(rep(1L,140L),rep(0L,170L))
  fit <- nf_pu_residual_forest(
    x,y,seed=9102L,hidden=6L,epochs=7L,
    rounds=2L,trees_per_round=1L,max_depth=2L,
    min_gain=1e6,multipliers=c(.5,1))
  expect_identical(fit$selected_candidate,"neural_fallback")
  expect_null(fit$residual_trees)
  expect_identical(fit$selected_rounds,0L)
  teacher <- nf_pu_matched_teacher(fit)
  sample <- x[1:30,,drop=FALSE]
  expect_equal(predict(fit,sample,type="prob"),
               predict(teacher,sample,type="prob"),
               tolerance=1e-12)
  expect_equal(rowSums(predict(fit,sample,type="prob")),
               rep(1,30),tolerance=1e-12)
  expect_true(is.data.frame(fit$selection_history))
  expect_equal(sum(fit$selection_history$selected),1L)
  expect_true(all(fit$selection_history$rounds>=0L))
  expect_true(fit$timings$selection_seconds>=0)
})

test_that("residual model handles input constraints and immutable teacher", {
  set.seed(9103)
  x <- matrix(rnorm(200),ncol=2)
  y <- c(rep(1L,45L),rep(0L,55L))
  expect_error(nf_pu_residual_forest(
    x,y,rounds=0L),"Invalid")
  expect_error(nf_pu_residual_forest(
    x,y,max_logit_change=10),"Invalid")
  expect_error(nf_pu_residual_forest(
    x,rep(0L,100L)),"at least 30")
})
