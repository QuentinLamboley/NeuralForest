test_that("native IPP learns presence-vs-availability without negatives", {
  set.seed(10001L)
  p<-data.frame(temp=rnorm(100,1.15,1),
                host=rnorm(100,.6,1),
                moisture=rnorm(100))
  b<-data.frame(temp=rnorm(200,-.1,1),
                host=rnorm(200),moisture=rnorm(200))
  mod<-nf_po_forest(p,b,mode="neural",seed=10001L,
            hidden=7L,epochs=65L)
  expect_s3_class(mod,"nf_po_forest")
  expect_identical(mod$score_definition,"relative_point_process_intensity")
  expect_null(mod$forest)
  expect_equal(length(mod$network$objective_history),65L)
  expect_true(all(is.finite(mod$network$objective_history)))
  expect_lt(min(mod$network$objective_history),
             mod$network$objective_history[[1L]])
  pp<-predict(mod,p,type="relative")
  pb<-predict(mod,b,type="relative")
  expect_true(all(is.finite(c(pp,pb)) & c(pp,pb)>0))
  expect_equal(mean(pb),1,tolerance=1e-6)
  expect_gt(mean(pp),mean(pb))
  expect_equal(predict(mod,b,type="suitability"),
    pb/(1+pb),tolerance=1e-12)
  expect_equal(predict(mod,b,type="log_relative"),
    log(pb),tolerance=1e-11)
})

test_that("raw, trained neural, and untrained random forests all run", {
  set.seed(10002L)
  p<-data.frame(a=rnorm(88,1),b=rnorm(88,1))
  bg<-data.frame(a=rnorm(165),b=rnorm(165))
  args<-list(presences=p,background=bg,
            seed=10002L,hidden=6L,epochs=12L,
            n_trees=30L,max_depth=3L,min_leaf=4L,
            hidden_features=3L)
  neural<-do.call(nf_po_forest,c(args,list(mode="neural")))
  raw<-do.call(nf_po_forest,c(args,list(mode="raw")))
  hidden<-do.call(nf_po_forest,c(args,list(mode="hidden")))
  random<-do.call(nf_po_forest,c(args,list(mode="random")))
  expect_equal(neural$network$W,raw$network$W,tolerance=0)
  expect_equal(raw$network$W,hidden$network$W,tolerance=0)
  expect_equal(hidden$network$W,random$network$W,tolerance=0)
  expect_identical(raw$forest_diagnostics$extra_dimensions,0L)
  expect_identical(hidden$forest_diagnostics$extra_dimensions,3L)
  expect_identical(random$forest_diagnostics$extra_dimensions,3L)
  for(fit in list(neural,raw,hidden,random)) {
    x<-predict(fit,p[1:19,,drop=FALSE])
    y<-predict(fit,bg[1:21,,drop=FALSE])
    expect_true(all(is.finite(c(x,y))))
    expect_true(all(c(x,y)>0))
    expect_true(fit$training$total_seconds>=0)
    expect_equal(mean(predict(fit,bg)),1,tolerance=1e-5)
  }
  expect_identical(is.null(hidden$random_network),TRUE)
  expect_identical(is.null(random$random_network),FALSE)
})

test_that("background is availability not a verified negative outcome", {
  set.seed(10003L)
  p<-data.frame(a=rnorm(70,.7),b=rnorm(70))
  bg<-data.frame(a=rnorm(120),b=rnorm(120))
  weighted<-nf_po_forest(p,bg,background_weights=seq(.5,1.5,length.out=120),
        mode="neural",hidden=5L,epochs=12L)
  ratio<-predict(weighted,bg)
  w<-seq(.5,1.5,length.out=120)
  expect_equal(weighted.mean<-sum(w*ratio)/sum(w),1,tolerance=1e-6)
  expect_error(nf_po_forest(p,bg[,1,drop=FALSE]),"identical columns")
  expect_error(nf_po_forest(p,bg,background_weights=rep(0,120)),
               "inclusion weights")
  expect_error(nf_po_forest(p,bg,mode="unknown"))
  expect_error(nf_po_forest(p[1:5,,drop=FALSE],bg),"needs")
  expect_equal(length(nf_po_metrics(c(1,2,3,4,5,6,7,8,9,10),
                           rep(1,20))),5L)
  met<-nf_po_metrics(1:20,seq(.1,2,length.out=30))
  expect_gte(met[["auc_pb"]],.5)
  expect_lte(met[["auc_pb"]],1)
  expect_error(nf_po_metrics(rep(-1,20),rep(1,30)),
               "nonnegative")
})
