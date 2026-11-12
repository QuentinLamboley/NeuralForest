test_that("transported neural warm starts preserve predictions", {
  set.seed(501)
  X <- matrix(rnorm(180,sd=0.8),nrow=60,ncol=3)
  net <- NeuralForest:::nf_nn_init(3L,c(6L,3L),501L)
  net$center <- c(-0.2,0.1,0.05)
  net$scale <- c(1.1,0.8,0.9)
  old_p <- NeuralForest:::nf_nn_predict(net,X)
  moved <- NeuralForest:::nf_nn_transport(net,c(0.25,-0.3,0.2),
                                          c(0.65,1.25,0.75))
  new_p <- NeuralForest:::nf_nn_predict(moved,X)
  expect_equal(new_p,old_p,tolerance=1e-11)
  expect_equal(moved$t,0L)
  expect_equal(sum(abs(moved$state[[1]]$mW)),0)
})

test_that("OOB calibration improves a deliberately miscalibrated predictor", {
  set.seed(502)
  z <- rnorm(1600)
  p <- stats::plogis(2*z)
  true <- stats::plogis(-1+0.7*z)
  y <- rbinom(length(p),1,true)
  ix <- seq_len(1200L)
  oob <- cbind(1-p[ix],p[ix])
  model <- NeuralForest:::nf_platt_fit(oob,y[ix],penalty=0.02)
  expect_true(!is.null(model))
  expect_gt(model$slope,0)
  held <- 1201:1600
  calibrated <- NeuralForest:::nf_platt_apply(cbind(1-p[held],p[held]),
                                               model)
  before <- mean((p[held]-y[held])^2)
  after <- mean((calibrated[,2]-y[held])^2)
  expect_lt(after,before)
  expect_equal(rowSums(calibrated),rep(1,length(held)),tolerance=1e-12)
})

test_that("calibration can be selected without touching a test set", {
  set.seed(503)
  X <- data.frame(a=rnorm(400),b=runif(400))
  y <- factor(ifelse(runif(400) <
      stats::plogis(-0.7+0.8*X$a),"yes","no"))
  cfg <- nf_control(n_trees=30L,neural=FALSE,
          linear_expert="auto",calibration="auto",
          calibration_penalties=c(0.02,0.1))
  fit <- neural_forest(X,y,metric="brier",control=cfg,
                       seed=503L,refit=TRUE)
  pred <- predict(fit,X[1:18,,drop=FALSE],type="prob")
  expect_true(all(is.finite(pred)))
  expect_equal(rowSums(pred),rep(1,18),tolerance=1e-8)
  expect_true("calibration_selection_seconds" %in% names(fit$timings))
  expect_true(is.null(fit$calibration) ||
              is.finite(fit$calibration$intercept))
})

test_that("average precision can be explicitly chosen as a soft loss", {
  ctrl <- nf_control(soft_loss="ap",calibration="auto")
  expect_equal(ctrl$soft_loss,"ap")
  expect_error(nf_control(calibration_penalties=c(-1,0.1)),"positive")
})
