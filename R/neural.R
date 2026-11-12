# Neural observation-priority learner. All training targets are OOB, and
# selection uses an independent holdout. This is predictive reweighting,
# not end-to-end differentiable optimization of CART split decisions.
nf_nn_init <- function(p, hidden, seed) {
  set.seed(seed)
  dims <- c(p,as.integer(hidden),1L)
  layers <- lapply(seq_len(length(dims)-1L),function(j)
    list(W=matrix(stats::rnorm(dims[j]*dims[j+1],
              sd=sqrt(2/dims[j])),nrow=dims[j]),b=rep(0,dims[j+1])))
  state <- lapply(layers,function(a)
    list(mW=a$W*0,vW=a$W*0,mb=a$b*0,vb=a$b*0))
  list(layers=layers,state=state,t=0L)
}
nf_nn_scale <- function(X, center=NULL, scale=NULL) {
  if (is.null(center)) center <- colMeans(X)
  if (is.null(scale)) {
    scale <- apply(X,2L,stats::sd)
    scale[!is.finite(scale)|scale<1e-8] <- 1
  }
  z <- sweep(sweep(X,2L,center,"-"),2L,scale,"/")
  z[z > 8] <- 8
  z[z < -8] <- -8
  list(X=z,center=center,scale=scale)
}
nf_nn_forward <- function(net,X) {
  activations <- list(X)
  pre <- vector("list",length(net$layers))
  for (j in seq_along(net$layers)) {
    layer <- net$layers[[j]]
    z <- activations[[j]] %*% layer$W +
      matrix(rep(layer$b,each=nrow(X)),nrow=nrow(X))
    pre[[j]] <- z
    activations[[j+1L]] <- if (j==length(net$layers))
      1/(1+exp(-pmax(-30,pmin(30,z)))) else pmax(z,0)
  }
  list(A=activations,Z=pre)
}
nf_nn_step <- function(net,forward,target,lr,wd) {
  L <- length(net$layers)
  n <- length(target)
  # BCE with soft targets and a sigmoid output: d/dlogit = p - target.
  dz <- matrix((as.numeric(forward$A[[L+1L]])-target)/n,ncol=1)
  net$t <- net$t+1L
  for (j in seq.int(L,1L)) {
    Wold <- net$layers[[j]]$W
    gradW <- crossprod(forward$A[[j]],dz)
    gradb <- colSums(dz)
    if (j>1L) {
      next_dz <- dz %*% t(Wold)
      next_dz[forward$Z[[j-1L]]<=0] <- 0
    }
    st <- net$state[[j]]
    st$mW <- 0.9*st$mW + 0.1*gradW
    st$vW <- 0.999*st$vW + 0.001*gradW^2
    st$mb <- 0.9*st$mb + 0.1*gradb
    st$vb <- 0.999*st$vb + 0.001*gradb^2
    mw <- st$mW/(1-0.9^net$t); vw <- st$vW/(1-0.999^net$t)
    mb <- st$mb/(1-0.9^net$t); vb <- st$vb/(1-0.999^net$t)
    net$layers[[j]]$W <- (1-lr*wd)*Wold -
      lr*mw/(sqrt(vw)+1e-8)
    net$layers[[j]]$b <- net$layers[[j]]$b -
      lr*mb/(sqrt(vb)+1e-8)
    net$state[[j]] <- st
    if (j>1L) dz <- next_dz
  }
  net
}
nf_nn_train <- function(X,target,ctrl,seed,previous=NULL) {
  set.seed(seed)
  scaled <- nf_nn_scale(X)
  xx <- scaled$X
  n <- nrow(xx)
  ord <- sample.int(n)
  nv <- if (n>=30) max(4L,floor(0.15*n)) else 0L
  val <- if (nv>0) ord[seq_len(nv)] else integer()
  train <- if (nv>0) ord[-seq_len(nv)] else ord
  net <- if (!is.null(previous) &&
             nrow(previous$layers[[1]]$W)==ncol(xx) &&
             identical(vapply(previous$layers,ncol,integer(1)),
                       as.integer(c(ctrl$neural_hidden,1L))))
    previous else nf_nn_init(ncol(xx),ctrl$neural_hidden,seed)
  # The centering is recomputed for each round, and the previous net only
  # provides warm-start weights; all hard-example targets stay OOB.
  best <- net; best_loss <- Inf; stale <- 0L
  for (epoch in seq_len(ctrl$neural_epochs)) {
    shuffled <- sample(train)
    batches <- split(shuffled,ceiling(seq_along(shuffled)/ctrl$neural_batch))
    for (ids in batches) {
      fw <- nf_nn_forward(net,xx[ids,,drop=FALSE])
      net <- nf_nn_step(net,fw,target[ids],ctrl$neural_lr,
                        ctrl$neural_weight_decay)
    }
    check <- if (nv>0) val else train
    p <- as.numeric(nf_nn_forward(net,xx[check,,drop=FALSE])$A[[length(net$layers)+1L]])
    loss <- -mean(target[check]*log(pmax(p,1e-9))+
                      (1-target[check])*log(pmax(1-p,1e-9)))
    if (is.finite(loss) && loss < best_loss-1e-6) {
      best_loss <- loss;best <- net;stale <- 0L
    } else stale <- stale+1L
    if (stale>=ctrl$neural_patience) break
  }
  best$center <- scaled$center; best$scale <- scaled$scale
  best
}
nf_nn_predict <- function(net,X) {
  z <- nf_nn_scale(X,net$center,net$scale)$X
  as.numeric(nf_nn_forward(net,z)$A[[length(net$layers)+1L]])
}
