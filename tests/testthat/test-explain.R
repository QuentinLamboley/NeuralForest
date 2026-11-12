test_that("external permutation importance and uncertainty return well-shaped outputs", {
  set.seed(51)
  id <- sample.int(nrow(iris),120)
  fit <- neural_forest(iris[id,1:4],iris$Species[id],
     metric="accuracy",seed=51,
     control=nf_control(neural=FALSE,n_trees=12,min_leaf=2))
  x <- iris[-id,1:4]; y <- iris$Species[-id]
  pm <- nf_permutation_importance(fit,x,y,metric="accuracy",
                                  repeats=2,seed=21)
  expect_equal(nrow(pm),4L)
  expect_true(all(is.finite(pm$mean_degradation)))
  expect_equal(dim(nf_uncertainty(fit,x[1:5,,drop=FALSE],measure="sd")),
               c(5L,3L))
  expect_equal(length(nf_uncertainty(fit,x[1:5,,drop=FALSE],
                                     measure="entropy")),5L)
  expect_true(all(nf_uncertainty(fit,x[1:5,,drop=FALSE],
                                 measure="vote") >= 0))
  expect_equal(nrow(fit$target_profile$feature_profile),4L)
})
