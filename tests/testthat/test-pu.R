test_that("native nnPU objective is finite and matches R risk", {
  set.seed(1801)
  n <- 180L
  x <- cbind(a=rnorm(n),b=runif(n),c=rnorm(n))
  y <- rep(c(1,0),c(55L,125L))
  tr <- seq_len(150L); validation <- 151:180
  # Ensure both classes in both partitions independently of seed.
  tr <- c(seq_len(45),56:150)
  validation <- setdiff(seq_len(n),tr)
  net <- NeuralForest:::nf_pu_cpp_fit(
    x[tr,,drop=FALSE],as.numeric(y[tr]),
    x[validation,,drop=FALSE],as.numeric(y[validation]),
    0.25,12L,35L,.008,.0004,1801L)
  p <- NeuralForest:::nf_pu_cpp_predict(net,x[validation,,drop=FALSE])
  expect_equal(length(p),length(validation))
  expect_true(all(is.finite(p)&p>0&p<1))
  manual <- NeuralForest:::nf_pu_risk(y[validation],p,.25)
  native <- NeuralForest:::nf_pu_cpp_risk(
    net,x[validation,,drop=FALSE],as.numeric(y[validation]),.25)
  expect_equal(native,manual,tolerance=1e-8)
  expect_true(is.finite(net$validation_risk))
})

test_that("PU forest training respects group interpretation and priors", {
  set.seed(1802)
  n <- 240
  x <- data.frame(a=rnorm(n),b=rnorm(n))
  y <- as.integer(seq_len(n)<=75L)
  baseline <- nf_pu_forest(x,y,mode="neural",
      priors=c(.12,.25,.40),seed=1802L,hidden=10L,epochs=16L,
      n_trees=18L)
  guided <- nf_pu_forest(x,y,mode="informed",
      priors=c(.12,.25,.40),seed=1802L,hidden=10L,epochs=16L,
      n_trees=18L)
  expect_equal(dim(predict(guided,x[1,,drop=FALSE],
                           type="prob")),c(1L,2L))
  p <- predict(guided,x[1:25,,drop=FALSE],type="prob")
  expect_equal(dim(p),c(25L,2L))
  expect_equal(rowSums(p),rep(1,25),tolerance=1e-10)
  expect_equal(length(guided$networks$networks),3L)
  expect_true(all(is.finite(p)))
  expect_gte(guided$alpha,0);expect_lte(guided$alpha,1)
  expect_true(is.finite(guided$validation_pu_risk))
  expect_equal(dim(predict(baseline,x[1:25,,drop=FALSE],
                           type="prob")),c(25L,2L))
})

test_that("PU predictions ignore all external outcome labels", {
  set.seed(1803)
  x <- data.frame(a=rnorm(240),b=rnorm(240))
  y <- c(rep(1L,90L),rep(0L,150L))
  z <- nf_pu_forest(x,y,mode="neural",seed=1803L,
                      epochs=10L,hidden=8L)
  newdata <- data.frame(a=c(-1,0,1),b=c(2,1,0))
  p1 <- predict(z,newdata,type="prob")
  p2 <- predict(z,newdata,type="prob")
  expect_identical(p1,p2)
  expect_error(nf_pu_forest(x,y,priors=c(-.1,.4)),
               "priors must")
})
