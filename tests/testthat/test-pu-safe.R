test_that("fidelity-constrained fallback exactly reproduces legacy PU network", {
  set.seed(2701)
  n <- 300L
  x <- data.frame(a=rnorm(n),b=runif(n),c=rnorm(n))
  y <- c(rep(1L,125L),rep(0L,175L))
  legacy <- nf_pu_forest(x,y,mode="neural",seed=2701L,
                         priors=c(.12,.25,.40),hidden=8L,
                         epochs=9L,n_trees=12L)
  safe <- nf_pu_safe_forest(x,y,seed=2701L,
                           priors=c(.12,.25,.40),hidden=8L,
                           epochs=9L,n_trees=12L,
                           min_gain=1e6,fidelity_cap=.02)
  expect_identical(safe$selected_candidate,"neural_fallback")
  expect_null(safe$forest)
  expect_equal(predict(safe,x[1:35,,drop=FALSE],type="prob"),
               predict(legacy,x[1:35,,drop=FALSE],type="prob"),
               tolerance=1e-12)
  expect_true(all(safe$selection_history$chosen %in% c(TRUE,FALSE)))
  expect_equal(sum(safe$selection_history$chosen),1L)
})

test_that("fidelity cap is enforced for all future predictions", {
  set.seed(2702)
  x <- data.frame(a=rnorm(310),b=rnorm(310))
  y <- c(rep(1L,140L),rep(0L,170L))
  safe <- nf_pu_safe_forest(x,y,seed=2702L,hidden=8L,
    epochs=9L,n_trees=12L,min_gain=0,
    fidelity_cap=.02,min_rank_correlation=.7)
  legacy <- nf_pu_forest(x,y,mode="neural",seed=2702L,
    hidden=8L,epochs=9L,n_trees=12L)
  p <- predict(safe,x,type="prob")[,"1"]
  teacher <- predict(legacy,x,type="prob")[,"1"]
  expect_true(all(abs(p-teacher)<=.02000002))
  expect_true(all(is.finite(p) & p>=0 & p<=1))
  expect_equal(as.numeric(rowSums(predict(safe,x,type="prob"))),
               rep(1,nrow(x)),tolerance=1e-12)
  expect_error(nf_pu_safe_forest(x,y,fidelity_cap=-.2),
               "Invalid stable PU model controls")
  expect_error(nf_pu_safe_forest(x,y,min_rank_correlation=1.2),
               "Invalid stable PU model controls")
})


test_that("native rank mapping is monotone and batch invariant", {
  teacher <- seq(.003,.997,length.out=500L)
  forest <- rev(teacher)^2
  mapping <- NeuralForest:::nf_pu_cpp_rank_fit(teacher,forest)
  for(fraction in c(0,.3,.5,1)) {
    p <- NeuralForest:::nf_pu_cpp_rank_apply(teacher,mapping,fraction,.02)
    expect_true(all(diff(p)>0))
    expect_identical(order(p),order(teacher))
    expect_lte(max(abs(p-teacher)),.020000001)
    expect_true(all(p>0 & p<1))
  }
  rows <- c(301L,2L,499L,3L)
  allp <- NeuralForest:::nf_pu_cpp_rank_apply(teacher,mapping,.75,.02)
  part <- NeuralForest:::nf_pu_cpp_rank_apply(teacher[rows],mapping,.75,.02)
  expect_equal(allp[rows],part,tolerance=1e-14)
  expect_error(NeuralForest:::nf_pu_cpp_rank_fit(teacher,c(forest[-1],NA)),
               "finite probability")
  expect_error(NeuralForest:::nf_pu_cpp_rank_apply(teacher,mapping,1,1.1),
               "Invalid")
})

test_that("safe model enables rank lock and retains exact fallback", {
  set.seed(2801)
  x <- data.frame(a=rnorm(320),b=rnorm(320))
  y <- c(rep(1L,150L),rep(0L,170L))
  model <- nf_pu_safe_forest(x,y,seed=2801L,
                             hidden=8L,epochs=9L,n_trees=12L,
                             min_gain=1e6)
  expect_true(model$rank_lock)
  expect_null(model$forest)
  expect_null(model$rank_map)
})
