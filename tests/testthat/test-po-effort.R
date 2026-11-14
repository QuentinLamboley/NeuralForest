test_that("uniform independent effort reproduces exact native IPP teacher", {
  set.seed(13301L)
  p<-data.frame(temp=rnorm(90,1),host=rnorm(90,.5))
  b<-data.frame(temp=rnorm(165),host=rnorm(165))
  teacher<-nf_po_forest(p,b,mode="neural",seed=13301L,
          hidden=7L,epochs=30L)
  effort<-nf_po_effort_forest(p,b,
          effort_background=rep(2.3,nrow(b)),
          method="neural",seed=13301L,hidden=7L,epochs=30L)
  expect_equal(teacher$network$W,effort$network$W,tolerance=0)
  expect_equal(teacher$network$bias,effort$network$bias,tolerance=0)
  expect_equal(predict(teacher,p),predict(effort,p),
               tolerance=1e-11)
  expect_equal(mean(predict(effort,b)),1,tolerance=1e-8)
  expect_identical(effort$effort_design$biological_absence_supplied,FALSE)
  expect_identical(effort$effort_design$test_outcomes_used,FALSE)
})

test_that("IPP effort offset exactly implements observation likelihood", {
  set.seed(13302L)
  n=125
  yy<-c(rep(1,45),rep(0,80))
  scores<-rnorm(n,sd=.6)
  available<-exp(runif(80,-.4,.4))
  detected<-exp(runif(80,-1,1))
  area<-c(rep(1,45),available*detected)
  # Explicit habitat-only objective with known observation effort.
  direct<-function(s) {
    nf<-NeuralForest:::nf_po_logmeanexp(
       s[yy==0]+log(detected),available)
    nf-mean(s[yy==1])
  }
  result<-NeuralForest:::nf_po_cpp_gradient(
        scores,yy,area,1e10)
  expect_equal(result$objective+log(
      sum(available*detected)/sum(available)),
      direct(scores),tolerance=1e-12)
  h<-1e-6
  for(k in c(3L,37L,54L,100L)) {
    sp<-scores;sm<-scores
    sp[k]<-sp[k]+h;sm[k]<-sm[k]-h
    num<-(direct(sp)-direct(sm))/(2*h)
    expect_equal(result$target[[k]],-n*num,tolerance=5e-6)
  }
})

test_that("habitat score is normalized by available AREA, not effort", {
  set.seed(13303L)
  p<-data.frame(a=rnorm(75,1),b=rnorm(75))
  bg<-data.frame(a=rnorm(145),b=rnorm(145))
  availability<-seq(.5,1.5,length.out=nrow(bg))
  q<-exp(1.1*bg$a)
  model<-nf_po_effort_forest(p,bg,
      effort_background=q,availability_weights=availability,
      method="neural",hidden=7L,epochs=24L,seed=13303L)
  prediction<-predict(model,bg)
  expect_equal(sum(prediction*availability)/sum(availability),
               1,tolerance=1e-8)
  expect_true(model$effort_design$background_ess>0)
  expect_true(model$effort_design$background_ess<=nrow(bg))
  expect_gt(model$effort_design$effort_max,
            model$effort_design$effort_min)
  expect_false(is.null(model$effort_design))
  expect_error(nf_po_effort_forest(p,bg,
      effort_background=rep(0,nrow(bg)),method="neural"),
      "effort")
  expect_error(nf_po_effort_forest(p,bg,
      effort_background=rep(1,nrow(bg)-1),method="neural"),
      "effort")
  expect_error(nf_po_effort_forest(p,bg,
      effort_background=q,availability_weights=rep(0,nrow(bg)),
      method="neural"),"inclusion")
})

test_that("effort-aware native IPP gradient uses independent inner effort", {
  set.seed(13304L)
  p<-data.frame(a=rnorm(85,1),b=rnorm(85),c=runif(85))
  bg<-data.frame(a=rnorm(155),b=rnorm(155),c=runif(155))
  ap<-data.frame(a=rnorm(30,1),b=rnorm(30),c=runif(30))
  ab<-data.frame(a=rnorm(50),b=rnorm(50),c=runif(50))
  q<-exp(.3*bg$a)
  qv<-exp(.3*ab$a)
  fit<-nf_po_effort_forest(p,bg,effort_background=q,
     validation_presences=ap,validation_background=ab,
     validation_effort_background=qv,
     method="gradient",seed=13304L,
     hidden=7L,epochs=11L,
     max_stages=2L,trees_per_stage=3L,
     max_depth=2L,min_leaf=3L,min_gain=1e8)
  expect_s3_class(fit,"nf_po_gradient_forest")
  expect_identical(fit$selected_stages,0L)
  expect_equal(sum(fit$validation$selected),1)
  expect_equal(mean(predict(fit,bg)),1,tolerance=1e-8)
  expect_identical(fit$effort_design$method,"gradient")
  expect_true(is.finite(fit$effort_design$validation_background_ess))
  expect_error(nf_po_effort_forest(p,bg,
       effort_background=q,
       method="gradient",validation_presences=ap,
       validation_background=ab,max_stages=2L),
       "effort")
  expect_error(nf_po_effort_forest(p,bg,
       effort_background=q,method="neural",
       validation_presences=ap,validation_background=ab),
       "does not use")
})
