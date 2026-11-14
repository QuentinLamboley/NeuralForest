test_that("matched PU teacher removes trees but preserves the fitted neural model", {
  set.seed(3001)
  x <- data.frame(a=rnorm(310),b=rnorm(310))
  y <- c(rep(1L,140L),rep(0L,170L))
  fit <- nf_pu_stable_forest(x,y,seed=3001L,hidden=6L,
                 epochs=7L,n_trees=9L,min_gain=1e6)
  teacher <- nf_pu_matched_teacher(fit)
  expect_null(teacher$forest)
  expect_null(teacher$rank_map)
  expect_equal(teacher$alpha,1)
  expect_equal(predict(teacher,x[1:25,,drop=FALSE],type="prob"),
               predict(fit,x[1:25,,drop=FALSE],type="prob"),
               tolerance=1e-12)
  expect_error(nf_pu_matched_teacher(1), "Expected")
})

test_that("independent verified audit conservatively falls back", {
  set.seed(3002)
  x <- data.frame(a=rnorm(320),b=runif(320))
  y <- c(rep(1L,140L),rep(0L,180L))
  model <- nf_pu_stable_forest(x,y,seed=3002L,
             hidden=6L,epochs=6L,n_trees=8L,min_gain=1e6)
  audit <- data.frame(a=rnorm(100),b=runif(100))
  truth <- rep(c(0L,1L),each=50L)
  selected <- nf_pu_verified_gate(model,audit,truth,
            min_gain=.002,min_group_half=15L,seed=3002L)
  expect_identical(selected$verified_gate$selected,"matched_teacher")
  expect_equal(nrow(selected$verified_gate$half_comparisons),2L)
  expect_equal(predict(selected,audit,type="prob"),
               predict(nf_pu_matched_teacher(model),audit,type="prob"),
               tolerance=1e-12)
  insufficient <- nf_pu_verified_gate(model,audit,c(rep(1L,8),rep(0L,92)),
                           min_group_half=15L)
  expect_identical(insufficient$verified_gate$selected,"matched_teacher")
  expect_false(insufficient$verified_gate$sufficient_class_support)
  expect_error(nf_pu_verified_gate(model,audit,rep(NA,100)), "verified binary")
  expect_error(nf_pu_verified_gate(model,audit,rep(1L,100),
                  max_ap_loss=-1), "Invalid verified-gate")
})
