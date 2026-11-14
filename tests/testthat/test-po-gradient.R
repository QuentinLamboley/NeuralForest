test_that("native IPP risk gradients match numerical derivatives", {
  set.seed(10011L)
  s<-rnorm(104,sd=.3)
  y<-c(rep(1,42),rep(0,62))
  w<-c(rep(1,42),runif(62,.4,1.8))
  z<-NeuralForest:::nf_po_cpp_gradient(s,y,w,1e6)
  loss<-function(score) {
    NeuralForest:::nf_po_logmeanexp(score[y==0],w[y==0])-
      mean(score[y==1])
  }
  expect_equal(z$objective,loss(s),tolerance=1e-12)
  expect_equal(sum(z$target),0,tolerance=1e-9)
  delta<-1e-6
  for(k in c(2L,30L,49L,101L)) {
    sp<-s;sm<-s
    sp[k]<-sp[k]+delta;sm[k]<-sm[k]-delta
    derivative<-(loss(sp)-loss(sm))/(2*delta)
    expect_equal(z$target[k],-length(s)*derivative,
                 tolerance=2e-5)
  }
  expect_error(NeuralForest:::nf_po_cpp_gradient(
      s,y,rep(0,length(y)),5),"inclusion weight")
  expect_error(NeuralForest:::nf_po_cpp_gradient(
      s,rep(1,length(y)),w,5),"Insufficient")
})

test_that("no reliable heldout gain is an EXACT native neural fallback", {
  set.seed(10012L)
  p<-data.frame(a=rnorm(80,1),b=rnorm(80),c=runif(80))
  bg<-data.frame(a=rnorm(140),b=rnorm(140),c=runif(140))
  fit<-nf_po_gradient_forest(
       p,bg,feature_mode="hidden",seed=10012L,
       hidden=7L,epochs=10L,max_stages=2L,
       trees_per_stage=3L,max_depth=2L,
       min_gain=1e8)
  expect_s3_class(fit,"nf_po_gradient_forest")
  expect_identical(fit$selected_stages,0L)
  expect_equal(sum(fit$validation$selected),1L)
  expect_identical(fit$selection_rule,
                   "random_training_only_split_not_spatial")
  expect_equal(length(fit$trees),0L)
  teacher<-nf_po_forest(p,bg,mode="neural",
    seed=10012L+989L,hidden=7L,epochs=10L)
  scores<-predict(fit,p[1:25,,drop=FALSE])
  expect_equal(scores,predict(teacher,p[1:25,,drop=FALSE]),
               tolerance=1e-9)
  expect_equal(mean(predict(fit,bg)),1,tolerance=1e-8)
  expect_true(all(is.finite(fit$validation$holdout_ipp_risk)))
  expect_gte(fit$training$total_seconds,0)
})

test_that("raw/random/learned IPP gradient trees use consistent score scale", {
  set.seed(10013L)
  p<-data.frame(a=rnorm(90,1.1),b=rnorm(90),c=rnorm(90))
  bg<-data.frame(a=rnorm(160),b=rnorm(160),c=rnorm(160))
  ap<-data.frame(a=rnorm(35,1),b=rnorm(35),c=rnorm(35))
  ab<-data.frame(a=rnorm(55),b=rnorm(55),c=rnorm(55))
  models<-lapply(c("raw","random","hidden"),function(mode)
    nf_po_gradient_forest(p,bg,validation_presences=ap,
      validation_background=ab,feature_mode=mode,
      seed=10013L,hidden=7L,epochs=11L,
      max_stages=2L,trees_per_stage=3L,max_depth=2L,
      min_gain=1e8))
  for(m in models) {
    expect_identical(m$selection_rule,"explicit_disjoint_holdout")
    expect_identical(m$selected_stages,0L)
    pp<-predict(m,ap)
    bb<-predict(m,ab)
    expect_true(all(is.finite(c(pp,bb)) & c(pp,bb)>0))
    expect_equal(predict(m,ab,type="log_relative"),
                 log(bb),tolerance=1e-9)
    expect_true(isTRUE(m$validation_refitted))
    # After selecting stage count on the disjoint inner block, the final
    # normalizer is fitted on ALL outer-training background, including
    # the inner fold, not on the original training subset alone.
    expect_equal(mean(predict(m,rbind(bg,ab))),1,tolerance=1e-8)
  }
  expect_error(nf_po_gradient_forest(p,bg,
      validation_presences=ap),"both or neither")
  expect_error(nf_po_gradient_forest(p,bg,
      validation_presences=ap,validation_background=ab,
      validation_background_weights=rep(0,55)),"validation availability")
})
