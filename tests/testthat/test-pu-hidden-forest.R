test_that("C++ hidden units equal native tanh activations", {
  set.seed(9410)
  X <- matrix(rnorm(200*4),ncol=4)
  y <- c(rep(1L,85),rep(0L,115))
  model <- NeuralForest:::nf_pu_stable_ensemble(
    X,y,c(.12,.25),hidden=5L,epochs=6L,seed=9410L,
    restarts=1L)
  z <- model$networks[[1]]
  scaled <- NeuralForest:::nf_pu_scale(X,z$center,z$spread)$X
  units <- c(1L,3L)
  got <- NeuralForest:::nf_pu_cpp_hidden_units(z$native,scaled,units)
  expected <- tanh(sweep(scaled %*% z$native$W[,units,drop=FALSE],
    2,z$native$bias[units],"+"))
  expect_equal(got,expected,tolerance=1e-10,ignore_attr=TRUE)
  expect_error(NeuralForest:::nf_pu_cpp_hidden_units(
    z$native,scaled,c(1L,100L)),"out of range")
  spec <- NeuralForest:::nf_pu_hidden_spec(model,2L,"hidden")
  aug <- NeuralForest:::nf_pu_hidden_matrix(model,X,spec)
  expect_equal(nrow(aug),nrow(X))
  expect_equal(ncol(aug),ncol(X)+4L)
  expect_equal(unname(aug[,1:4,drop=FALSE]),unname(X))
  expect_equal(NeuralForest:::nf_pu_hidden_matrix(model,X,
      NeuralForest:::nf_pu_hidden_spec(model,2L,"raw")),X)
})

test_that("OOB calibration ignores missing OOB rows and has positive slope", {
  p <- seq(.04,.96,length.out=120)
  q <- plogis(1.2*qlogis(p)+.3)
  p[c(2,11)] <- NA_real_
  fit <- NeuralForest:::nf_pu_cpp_align_fit(p,q)
  expect_true(fit$n_oob==118L)
  expect_gt(fit$slope,0)
  expect_true(is.finite(fit$intercept))
  pred <- NeuralForest:::nf_pu_cpp_align_apply(c(.01,.2,.5,.8,.99),fit)
  expect_true(all(diff(pred)>0))
  expect_true(all(pred>0 & pred<1))
  expect_error(NeuralForest:::nf_pu_cpp_align_fit(
    rep(NA_real_,120),q),"Too few")
})

test_that("PU feature forest rejects all candidates under impossible min gain", {
  set.seed(9411)
  n <- 320L
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=runif(n))
  y <- c(rep(1L,130L),rep(0L,190L))
  fit <- nf_pu_feature_forest(x,y,
    mode="hidden",gate="pu",seed=9411L,hidden=7L,epochs=7L,
    n_trees=28L,max_depth=4L,
    min_pu_gain=10000)
  expect_null(fit$forest)
  expect_identical(fit$selected_candidate,"neural_fallback")
  expect_equal(nrow(fit$selection_history),2L)
  expect_false(fit$selection_history$eligible[[2]])
  teacher <- nf_pu_matched_teacher(fit)
  p <- predict(fit,x[1:24,,drop=FALSE],type="prob")
  expect_equal(p,predict(teacher,x[1:24,,drop=FALSE],type="prob"),
               tolerance=1e-12)
  expect_equal(rowSums(p),rep(1,24),tolerance=1e-12)
})

test_that("raw vs hidden candidates have identical frozen nnPU teacher", {
  set.seed(9412)
  n <- 310L
  x <- data.frame(a=rnorm(n),b=rnorm(n),c=rnorm(n))
  y <- c(rep(1L,120L),rep(0L,190L))
  args <- list(x=x,y=y,gate="candidate",seed=9412L,hidden=6L,
      epochs=7L,n_trees=28L,max_depth=4L)
  raw <- do.call(nf_pu_feature_forest,c(args,list(mode="raw")))
  hidden <- do.call(nf_pu_feature_forest,c(args,list(mode="hidden")))
  random <- do.call(nf_pu_feature_forest,c(args,list(mode="random")))
  expect_false(is.null(raw$forest))
  expect_false(is.null(hidden$forest))
  expect_false(is.null(random$forest))
  expect_identical(raw$feature_diagnostics$latent_columns,0L)
  expect_gte(hidden$feature_diagnostics$latent_columns,2L)
  sample <- x[1:24,,drop=FALSE]
  baseline1 <- predict(nf_pu_matched_teacher(raw),sample,type="prob")
  baseline2 <- predict(nf_pu_matched_teacher(hidden),sample,type="prob")
  expect_equal(baseline1,baseline2,tolerance=1e-12)
  expect_equal(baseline1,
      predict(nf_pu_matched_teacher(random),sample,type="prob"),
      tolerance=1e-12)
  expect_identical(random$feature_diagnostics$latent_columns,
                   hidden$feature_diagnostics$latent_columns)
  expect_true(!isTRUE(random$feature_spec$random$trained))
  expect_equal(random$feature_spec$random$W,
               NeuralForest:::nf_pu_cpp_random_projection(
                 nrow(random$feature_spec$random$W),
                 ncol(random$feature_spec$random$W),
                 9412L+19051L)$W,
               tolerance=1e-12)
  for(m in list(raw,hidden,random)) {
    p <- predict(m,sample,type="prob")
    expect_equal(dim(p),c(24L,2L))
    expect_true(all(is.finite(p)))
    expect_equal(rowSums(p),rep(1,24),tolerance=1e-11)
    expect_true(m$feature_diagnostics$forest_oob_covered>.9)
    expect_true(m$feature_diagnostics$latent_importance_fraction>=0)
    expect_true(m$feature_diagnostics$latent_importance_fraction<=1)
  }
  expect_error(nf_pu_feature_forest(x,y,gate="verified"),
               "disjoint 0/1 audit")
})

test_that("independent verified labels can select or reject fitted forest", {
  set.seed(9413)
  x <- data.frame(a=rnorm(300),b=rnorm(300))
  y <- c(rep(1L,120),rep(0L,180))
  audit <- data.frame(a=rnorm(110),b=rnorm(110))
  truth <- rep(c(0L,1L),55L)
  fitted <- nf_pu_feature_forest(x,y,mode="hidden",gate="candidate",
        seed=9413L,hidden=6L,epochs=7L,n_trees=28L)
  chosen <- nf_pu_verified_gate(fitted,audit,truth,min_gain=1e6,
           min_group_half=15L)
  expect_identical(chosen$verified_gate$selected,"matched_teacher")
  expect_equal(predict(chosen,audit,type="prob"),
               predict(nf_pu_matched_teacher(fitted),audit,type="prob"),
               tolerance=1e-12)
})
