#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace Rcpp;

namespace {
double sigmoid(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-std::min(40.0,x)));
  const double z = std::exp(std::max(-40.0,x));
  return z / (1.0+z);
}
double clip_prob(double x) {
  return std::min(1.0-1e-7,std::max(1e-7,x));
}
struct SoftForest {
  int p, depth, trees, nodes, leaves, stride;
  bool classification;
  double temperature;
  std::vector<double> theta;
  SoftForest(int p_, int d_, int trees_, bool classification_, double tau)
      : p(p_), depth(d_), trees(trees_), nodes((1<<d_)-1),
        leaves(1<<d_), stride(nodes*(p_+1)+leaves),
        classification(classification_),temperature(tau),
        theta(static_cast<size_t>(trees_)*stride,0.0) {}
  int woffset(int tree,int node) const {
    return tree*stride+node*(p+1);
  }
  int leaf_offset(int tree,int leaf) const {
    return tree*stride+nodes*(p+1)+leaf;
  }
  double forward_row(const NumericMatrix& X,int i,
      std::vector<double>* gradients=nullptr,
      double upstream=0.0) const {
    double output=0.0;
    const double invT=1.0/trees;
    std::vector<double> gate(nodes);
    for (int t=0;t<trees;++t) {
      for (int nd=0;nd<nodes;++nd) {
        const int k=woffset(t,nd);
        double z=theta[k+p];
        for (int j=0;j<p;++j) z+=theta[k+j]*X(i,j);
        gate[nd]=sigmoid(z/temperature);
      }
      for (int leaf=0;leaf<leaves;++leaf) {
        double route=1.0;
        int nd=0;
        for (int d=0;d<depth;++d) {
          const int go_right=(leaf>>(depth-d-1))&1;
          route*=go_right?gate[nd]:(1.0-gate[nd]);
          nd=2*nd+1+go_right;
        }
        const int lk=leaf_offset(t,leaf);
        const double value=classification?sigmoid(theta[lk]):theta[lk];
        output+=invT*route*value;
        if (gradients) {
          const double scale=upstream*invT*route;
          (*gradients)[lk]+=scale*(classification?value*(1.0-value):1.0);
          nd=0;
          for (int d=0;d<depth;++d) {
            const int go_right=(leaf>>(depth-d-1))&1;
            const double dz=scale*value*(go_right-gate[nd])/temperature;
            const int offset=woffset(t,nd);
            for (int j=0;j<p;++j) (*gradients)[offset+j]+=dz*X(i,j);
            (*gradients)[offset+p]+=dz;
            nd=2*nd+1+go_right;
          }
        }
      }
    }
    return output;
  }
};
double surrogate_loss(const std::vector<double>& pred,
                      const NumericVector& y,
                      const std::vector<int>& ids,
                      int loss,int task,
                      std::vector<double>* derivatives=nullptr) {
  const size_t n=ids.size();
  std::vector<double> dp(n,0.0);
  double value=0.0;
  if (task==1 && loss==3) {
    // Differentiable pairwise ranking surrogate for AUC and similar metrics.
    int count=0;
    for (size_t a=0;a<n;++a) {
      if (y[ids[a]]<0.5) continue;
      for (size_t b=0;b<n;++b) {
        if (y[ids[b]]>0.5) continue;
        const double z=(pred[a]-pred[b])/0.15;
        value+=std::log1p(std::exp(-std::max(-40.0,std::min(40.0,z))));
        const double d=-sigmoid(-z)/0.15;
        dp[a]+=d; dp[b]-=d; ++count;
      }
    }
    if (count>0) {
      value/=count;
      for (double& v:dp) v/=count;
      if (derivatives) *derivatives=std::move(dp);
      return value;
    }
    // A homogeneous minibatch cannot define pairwise AUC.
    loss=0;
  }
  for (size_t a=0;a<n;++a) {
    const double target=y[ids[a]];
    const double p=task==1?clip_prob(pred[a]):pred[a];
    double d=0.0;
    if (task==0) {
      const double e=p-target;
      value+=e*e; d=2*e;
    } else if (loss==1) {
      const double e=p-target;
      value+=e*e; d=2*e;
    } else if (loss==2) {
      // Exact first derivative of gamma=2 focal loss, alpha=0.5.
      if (target>0.5) {
        value+=-0.5*(1-p)*(1-p)*std::log(p);
        d=0.5*(2*(1-p)*std::log(p)-(1-p)*(1-p)/p);
      } else {
        value+=-0.5*p*p*std::log(1-p);
        d=0.5*(-2*p*std::log(1-p)+p*p/(1-p));
      }
    } else {
      value+=-(target*std::log(p)+(1-target)*std::log(1-p));
      d=(p-target)/(p*(1-p));
    }
    dp[a]=d/n;
  }
  if (derivatives) *derivatives=std::move(dp);
  return value/n;
}
}

// [[Rcpp::export]]
List nf_soft_fit(NumericMatrix X,NumericVector y,NumericMatrix Xvalid,
                 NumericVector yvalid,List config,int seed,bool classification) {
  const int n=X.nrow(), p=X.ncol();
  const int nt=as<int>(config["soft_n_trees"]);
  const int depth=as<int>(config["soft_depth"]);
  const int epochs=as<int>(config["soft_epochs"]);
  const int batch=as<int>(config["soft_batch"]);
  const double tau=as<double>(config["soft_temperature"]);
  const double lr=as<double>(config["soft_lr"]);
  const double wd=as<double>(config["soft_weight_decay"]);
  const int patience=as<int>(config["soft_patience"]);
  const double clip=as<double>(config["soft_clip"]);
  const std::string loss_name=as<std::string>(config["soft_loss"]);
  if(n<10 || p<1 || nt<1 || depth<1 || depth>7 ||
     epochs<1 || batch<2 || tau<=0 || lr<=0 || wd<0 ||
     Xvalid.ncol()!=p || y.size()!=n || Xvalid.nrow()!=yvalid.size())
    stop("Invalid differentiable forest settings");
  for(int i=0;i<n;++i) {
    if (!std::isfinite(y[i])) stop("Non-finite target");
    if(classification && y[i]!=0.0 && y[i]!=1.0)
      stop("Differentiable forest currently accepts binary class labels");
    for(int j=0;j<p;++j) if(!std::isfinite(X(i,j))) stop("Non-finite predictor");
  }
  SoftForest forest(p,depth,nt,classification,tau);
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::normal_distribution<double> normal(0,1);
  const double pos=std::accumulate(y.begin(),y.end(),0.0)/n;
  const double prior=std::log(clip_prob(pos)/(1-clip_prob(pos)));
  for(int t=0;t<nt;++t) {
    for(int nd=0;nd<forest.nodes;++nd) {
      const int k=forest.woffset(t,nd);
      for(int j=0;j<p;++j) forest.theta[k+j]=normal(rng)*0.65/std::sqrt(p);
      forest.theta[k+p]=normal(rng)*0.25;
    }
    for(int leaf=0;leaf<forest.leaves;++leaf)
      forest.theta[forest.leaf_offset(t,leaf)]=(classification?prior:pos)+normal(rng)*0.3;
  }
  const int loss=loss_name=="brier"?1:loss_name=="focal"?2:
                 loss_name=="auc"?3:0;
  std::vector<double> m(forest.theta.size(),0),v(m);
  std::vector<double> best_theta=forest.theta;
  double best_loss=std::numeric_limits<double>::infinity();
  int best_epoch=0,stale=0,updates=0;
  NumericVector train_curve(epochs),val_curve(epochs);
  std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);
  std::vector<int> vid(Xvalid.nrow());std::iota(vid.begin(),vid.end(),0);
  for(int ep=0;ep<epochs;++ep) {
    std::shuffle(ids.begin(),ids.end(),rng);
    for(int start=0;start<n;start+=batch) {
      std::vector<int> chunk(ids.begin()+start,ids.begin()+std::min(n,start+batch));
      std::vector<double> predictions(chunk.size());
      for(size_t i=0;i<chunk.size();++i)
        predictions[i]=forest.forward_row(X,chunk[i]);
      std::vector<double> derivative;
      surrogate_loss(predictions,y,chunk,loss,classification?1:0,&derivative);
      std::vector<double> grad(forest.theta.size(),0.0);
      for(size_t i=0;i<chunk.size();++i)
        forest.forward_row(X,chunk[i],&grad,derivative[i]);
      double norm=0.0;
      for(double g:grad) norm+=g*g;
      const double factor=norm>clip*clip?clip/std::sqrt(norm):1.0;
      ++updates;
      const double c1=1-std::pow(0.9,updates);
      const double c2=1-std::pow(0.999,updates);
      for(size_t k=0;k<grad.size();++k) {
        const double g=grad[k]*factor;
        m[k]=0.9*m[k]+0.1*g;
        v[k]=0.999*v[k]+0.001*g*g;
        forest.theta[k]=(1-lr*wd)*forest.theta[k]-
          lr*(m[k]/c1)/(std::sqrt(v[k]/c2)+1e-8);
      }
    }
    std::vector<double> pt(n),pv(vid.size());
    std::vector<int> original(n);std::iota(original.begin(),original.end(),0);
    for(int i=0;i<n;++i) pt[i]=forest.forward_row(X,i);
    for(size_t i=0;i<vid.size();++i) pv[i]=forest.forward_row(Xvalid,vid[i]);
    train_curve[ep]=surrogate_loss(pt,y,original,loss,classification?1:0);
    val_curve[ep]=pv.empty()?train_curve[ep]:
      surrogate_loss(pv,yvalid,vid,loss,classification?1:0);
    if(std::isfinite(val_curve[ep]) && val_curve[ep]<best_loss-1e-8) {
      best_loss=val_curve[ep];best_theta=forest.theta;
      best_epoch=ep+1;stale=0;
    } else ++stale;
    if(stale>=patience) {
      train_curve=train_curve[Range(0,ep)];
      val_curve=val_curve[Range(0,ep)];
      break;
    }
  }
  if(best_epoch==0) stop("Differentiable training failed to converge to finite loss");
  forest.theta=std::move(best_theta);
  return List::create(_["theta"]=wrap(forest.theta),
    _["depth"]=depth, _["trees"]=nt, _["n_features"]=p,
    _["temperature"]=tau,_["classification"]=classification,
    _["best_epoch"]=best_epoch,_["train_loss"]=train_curve,
    _["validation_loss"]=val_curve,_["loss"]=loss_name);
}

// [[Rcpp::export]]
NumericVector nf_soft_predict(List model,NumericMatrix X) {
  const int p=as<int>(model["n_features"]);
  if(X.ncol()!=p) stop("Predictor count does not match differentiable model");
  const int depth=as<int>(model["depth"]);
  const int trees=as<int>(model["trees"]);
  const bool classification=as<bool>(model["classification"]);
  const double tau=as<double>(model["temperature"]);
  SoftForest model0(p,depth,trees,classification,tau);
  model0.theta=as<std::vector<double>>(model["theta"]);
  if(model0.theta.size()!=static_cast<size_t>(model0.stride*trees))
    stop("Malformed differentiable forest");
  NumericVector out(X.nrow());
  for(int i=0;i<X.nrow();++i) out[i]=model0.forward_row(X,i);
  return out;
}


// [[Rcpp::export]]
double nf_soft_gradient_check() {
  SoftForest forest(3,2,1,true,0.8);
  for(size_t j=0;j<forest.theta.size();++j)
    forest.theta[j]=0.13*std::sin(static_cast<double>(j+1));
  NumericMatrix X(1,3);
  X(0,0)=-0.6;X(0,1)=0.7;X(0,2)=1.2;
  std::vector<double> gradient(forest.theta.size(),0.0);
  forest.forward_row(X,0,&gradient,1.0);
  const double step=1e-5;
  double error=0.0;
  for(size_t k=0;k<forest.theta.size();++k) {
    forest.theta[k]+=step;
    const double above=forest.forward_row(X,0);
    forest.theta[k]-=2*step;
    const double below=forest.forward_row(X,0);
    forest.theta[k]+=step;
    error=std::max(error,std::abs(gradient[k]-(above-below)/(2*step)));
  }
  return error;
}
