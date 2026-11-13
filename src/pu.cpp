// Positive-unlabeled shallow neural estimator, Rcpp/C++17.
// Non-negative PU risk (Kiryo et al., NeurIPS 2017), with an *assumed*
// class prior. This requires representative P and U samples and is not
// guaranteed for environmentally selected hard background samples.
// All optimisation is native C++, with no user-supplied labels from tests.
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
using namespace Rcpp;
namespace {
inline double sigmoid(double x) {
  if(x>=0){const double z=std::exp(-x);return 1.0/(1+z);}
  const double z=std::exp(x);return z/(1+z);
}
inline double softplus(double x) {
  return std::max(0.0,x)+std::log1p(std::exp(-std::abs(x)));
}
struct PUWeights {
  int p,h;
  std::vector<double> W,b,V;
  double c=0;
  PUWeights(int p_,int h_):p(p_),h(h_),W(p_*h_,0),b(h_,0),V(h_,0){}
};
void forward(const PUWeights& net,const NumericMatrix& X,int i,
             std::vector<double>& hidden,double& z,double& prob) {
  z=net.c;
  for(int j=0;j<net.h;++j){
    double v=net.b[j];
    for(int k=0;k<net.p;++k)v+=X(i,k)*net.W[k*net.h+j];
    hidden[j]=std::tanh(v);
    z+=hidden[j]*net.V[j];
  }
  prob=sigmoid(z);
}
double pu_risk(const PUWeights& net,const NumericMatrix& X,
               const NumericVector& y,double prior) {
  double pp=0,pn=0,un=0;
  int np=0,nu=0;
  std::vector<double> hidden(net.h);
  for(int i=0;i<X.nrow();++i){
    double z,p;forward(net,X,i,hidden,z,p);
    if(y[i]>0.5){pp+=softplus(-z);pn+=softplus(z);++np;}
    else {un+=softplus(z);++nu;}
  }
  if(np==0||nu==0)stop("PU loss needs both positives and unlabeled samples");
  return prior*pp/np+std::max(0.0,un/nu-prior*pn/np);
}
void adam_update(std::vector<double>& val,std::vector<double>& moment,
                 std::vector<double>& second,const std::vector<double>& grad,
                 double lr,double decay,int t) {
  double inv1=1.0-std::pow(.9,t),inv2=1.0-std::pow(.999,t);
  for(size_t i=0;i<val.size();++i){
    const double g=std::max(-5.0,std::min(5.0,grad[i]));
    moment[i]=.9*moment[i]+.1*g;
    second[i]=.999*second[i]+.001*g*g;
    val[i]=(1-lr*decay)*val[i]-
      lr*(moment[i]/inv1)/(std::sqrt(second[i]/inv2)+1e-8);
  }
}
} // namespace

// [[Rcpp::export]]
List nf_pu_cpp_fit(NumericMatrix X,NumericVector y,
                   NumericMatrix Xvalid,NumericVector yvalid,
                   double prior,int hidden,int epochs,double learning_rate,
                   double weight_decay,int seed) {
  const int n=X.nrow(),p=X.ncol(),nv=Xvalid.nrow();
  if(p<1||n<30||nv<10||Xvalid.ncol()!=p||hidden<1||hidden>256||
     epochs<1||epochs>1500||!std::isfinite(prior)||
     prior<=0||prior>=1||learning_rate<=0||learning_rate>.5||
     weight_decay<0||y.size()!=n||yvalid.size()!=nv)
    stop("Invalid native PU training configuration");
  int np=0,nu=0,npv=0,nuv=0;
  for(int i=0;i<n;++i){
    if(y[i]!=0&&y[i]!=1)stop("PU labels must be exactly 0 or 1");
    if(y[i]==1)++np;else ++nu;
  }
  for(int i=0;i<nv;++i){
    if(yvalid[i]!=0&&yvalid[i]!=1)stop("Invalid PU validation labels");
    if(yvalid[i]==1)++npv;else ++nuv;
  }
  if(np<5||nu<5||npv<2||nuv<2)stop("PU data lack class support");
  PUWeights net(p,hidden);
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::normal_distribution<double> gaussian(0,std::sqrt(1.0/p));
  for(double& v:net.W)v=gaussian(rng);
  for(double& v:net.V)v=gaussian(rng)/std::sqrt(static_cast<double>(hidden));
  std::vector<double> mW(net.W.size()),vW(net.W.size());
  std::vector<double> mV(net.V.size()),vV(net.V.size());
  std::vector<double> mb(hidden),vb(hidden),mc(1),vc(1);
  std::vector<double> h(hidden),pred(n),logit(n),gW(net.W.size()),
                      gV(net.V.size()),gb(hidden),gc(1);
  double best=std::numeric_limits<double>::infinity();
  PUWeights chosen=net;
  int stale=0,best_epoch=0;
  for(int epoch=1;epoch<=epochs;++epoch){
    double pp=0,pn=0,un=0;
    for(int i=0;i<n;++i){
      double z,pr;forward(net,X,i,h,z,pr);
      pred[i]=pr;logit[i]=z;
      if(y[i]==1){pp+=softplus(-z);pn+=softplus(z);}
      else un+=softplus(z);
    }
    const bool negative_active=(un/nu-prior*pn/np)>0;
    std::fill(gW.begin(),gW.end(),0.0);
    std::fill(gV.begin(),gV.end(),0.0);
    std::fill(gb.begin(),gb.end(),0.0);gc[0]=0.0;
    for(int i=0;i<n;++i){
      // Positive term pi*E_P[l+], corrected negative term
      // max(0, E_U[l-] - pi*E_P[l-]).
      double gz=0;
      if(y[i]==1) {
        gz=prior*(pred[i]-1.0)/np;
        if(negative_active)gz-=prior*pred[i]/np;
      } else if(negative_active)gz=pred[i]/nu;
      if(gz==0)continue;
      double z,pr;forward(net,X,i,h,z,pr);
      gc[0]+=gz;
      for(int j=0;j<hidden;++j){
        gV[j]+=gz*h[j];
        const double dh=gz*net.V[j]*(1.0-h[j]*h[j]);
        gb[j]+=dh;
        for(int k=0;k<p;++k)gW[k*hidden+j]+=dh*X(i,k);
      }
    }
    adam_update(net.W,mW,vW,gW,learning_rate,weight_decay,epoch);
    adam_update(net.V,mV,vV,gV,learning_rate,weight_decay,epoch);
    adam_update(net.b,mb,vb,gb,learning_rate,0,epoch);
    std::vector<double> c(1,net.c);
    adam_update(c,mc,vc,gc,learning_rate,0,epoch);
    net.c=c[0];
    const double validation=pu_risk(net,Xvalid,yvalid,prior);
    if(!std::isfinite(validation))stop("Non-finite PU validation risk");
    if(validation<best-1e-6){
      chosen=net;best=validation;best_epoch=epoch;stale=0;
    }else ++stale;
    if(stale>=12)break;
  }
  NumericMatrix W(p,hidden);
  for(int k=0;k<p;++k)for(int j=0;j<hidden;++j)
    W(k,j)=chosen.W[k*hidden+j];
  return List::create(_["W"]=W,_["bias"]=wrap(chosen.b),
       _["output"]=wrap(chosen.V),_["intercept"]=chosen.c,
       _["prior"]=prior,_["validation_risk"]=best,
       _["best_epoch"]=best_epoch);
}

// [[Rcpp::export]]
NumericVector nf_pu_cpp_predict(List model,NumericMatrix X) {
  NumericMatrix W=as<NumericMatrix>(model["W"]);
  NumericVector b=as<NumericVector>(model["bias"]);
  NumericVector V=as<NumericVector>(model["output"]);
  const double intercept=as<double>(model["intercept"]);
  const int p=W.nrow(),h=W.ncol();
  if(X.ncol()!=p||b.size()!=h||V.size()!=h)
    stop("PU prediction input dimensions changed");
  NumericVector result(X.nrow());
  for(int i=0;i<X.nrow();++i){
    double z=intercept;
    for(int j=0;j<h;++j){
      double v=b[j];
      for(int k=0;k<p;++k)v+=X(i,k)*W(k,j);
      z+=V[j]*std::tanh(v);
    }
    result[i]=sigmoid(z);
  }
  return result;
}

// [[Rcpp::export]]
double nf_pu_cpp_risk(List model,NumericMatrix X,NumericVector y,double prior) {
  NumericVector probs=nf_pu_cpp_predict(model,X);
  double pp=0,pn=0,un=0;
  int np=0,nu=0;
  if(y.size()!=probs.size()||prior<=0||prior>=1)stop("Invalid PU risk input");
  for(int i=0;i<probs.size();++i){
    const double p=std::min(1-1e-12,std::max(1e-12,static_cast<double>(probs[i])));
    if(y[i]==1){pp-=std::log(p);pn-=std::log1p(-p);++np;}
    else if(y[i]==0){un-=std::log1p(-p);++nu;}
    else stop("PU labels must be 0/1");
  }
  if(np==0||nu==0)stop("PU risk needs both groups");
  return prior*pp/np+std::max(0.0,un/nu-prior*pn/np);
}
