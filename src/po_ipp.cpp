// Presence-only inhomogeneous Poisson-point-process (IPP) neural scorer.
// Observed presence records versus sampled *environmental availability*.
// Background rows are NOT declared true absences; no class prevalence is used.
//
// Objective: log( sum_j w_j exp(f(b_j)) / sum_j w_j )
//              - mean_i f(p_i) + lambda/2 ||theta||^2
// Here w_j are inverse-design inclusion weights for available space,
// if known; sampling from the same observation-effort surface as recorded
// presences is a separate, explicit experimental design assumption.
//
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>
using namespace Rcpp;

namespace {
inline double tanh_clip(double x) {
  return std::tanh(std::max(-30.,std::min(30.,x)));
}
inline double stable_weight(double z) {
  return std::exp(std::max(-60.,std::min(0.,z)));
}
}

 // [[Rcpp::export]]
List nf_po_cpp_fit(NumericMatrix X, NumericVector y,
                  NumericVector background_weights,
                  int hidden=12,int epochs=100,double rate=.01,
                  double decay=.0005,int seed=42) {
  const int n=X.nrow(),p=X.ncol();
  if(n<60 || p<1 || p>10000 || y.size()!=n ||
     background_weights.size()!=n ||
     hidden<1 || hidden>256 || epochs<1 || epochs>3000 ||
     !std::isfinite(rate) || rate<=0. || rate>.3 ||
     !std::isfinite(decay) || decay<0 || decay>1.)
    stop("Invalid IPP neural training arguments");
  int np=0,nb=0;
  double bg_weight_sum=0.;
  for(int i=0;i<n;++i) {
    if(y[i]!=0. && y[i]!=1.)stop("Presence-only y must be 0/1 indicators");
    if(y[i]==1.)++np;
    else {++nb;bg_weight_sum+=background_weights[i];}
    if(!std::isfinite(background_weights[i]) ||
       background_weights[i]<=0.)
      stop("Background sampling weights must be positive finite");
    for(int j=0;j<p;++j)
      if(!std::isfinite(X(i,j)))
        stop("IPP neural input contains non-finite predictor");
  }
  if(np<20 || nb<20 || !std::isfinite(bg_weight_sum) ||
     bg_weight_sum<=0.)stop("Insufficient presence/background support");
  const int wh=p*hidden,offV=wh+hidden,parameter_count=offV+hidden;
  std::vector<double> pars(parameter_count),m(parameter_count,0.),
                       v(parameter_count,0.),grad(parameter_count);
  std::mt19937_64 rng(static_cast<uint64_t>(static_cast<uint32_t>(seed)));
  std::normal_distribution<double> norm(0.,1.);
  for(int j=0;j<wh;++j)
    pars[j]=norm(rng)*std::sqrt(1./p);
  for(int j=0;j<hidden;++j) {
    pars[wh+j]=0.;
    pars[offV+j]=norm(rng)*std::sqrt(1./hidden);
  }
  std::vector<double> scores(n),activ(n*hidden),mass(n,0.);
  NumericVector history(epochs);
  for(int iter=1;iter<=epochs;++iter) {
    double max_bg=-1e300;
    for(int i=0;i<n;++i) {
      double score=0.;
      for(int h=0;h<hidden;++h) {
        double z=pars[wh+h];
        for(int j=0;j<p;++j)z+=X(i,j)*pars[j*hidden+h];
        const double a=tanh_clip(z);
        activ[i*hidden+h]=a;
        score+=a*pars[offV+h];
      }
      scores[i]=score;
      if(y[i]==0.)max_bg=std::max(max_bg,score);
    }
    double exp_bg=0.,pos_sum=0.;
    for(int i=0;i<n;++i) {
      if(y[i]==0.) {
        mass[i]=background_weights[i]*stable_weight(scores[i]-max_bg);
        exp_bg+=mass[i];
      } else pos_sum+=scores[i];
    }
    if(!(exp_bg>0.) || !std::isfinite(exp_bg))
      stop("Numerically invalid Poisson normalizer");
    const double objective=max_bg+std::log(exp_bg/bg_weight_sum)-
                           pos_sum/static_cast<double>(np);
    history[iter-1]=objective;
    std::fill(grad.begin(),grad.end(),0.);
    for(int i=0;i<n;++i) {
      const double dscore=(y[i]==1.)?-1./static_cast<double>(np):
                            mass[i]/exp_bg;
      for(int h=0;h<hidden;++h) {
        const double a=activ[i*hidden+h];
        grad[offV+h]+=dscore*a;
        const double dh=dscore*pars[offV+h]*(1.-a*a);
        grad[wh+h]+=dh;
        for(int j=0;j<p;++j)
          grad[j*hidden+h]+=dh*X(i,j);
      }
    }
    double norm2=0.;
    for(int k=0;k<parameter_count;++k) {
      // Do not penalize hidden biases, which are not identifiable
      // under arbitrary translations of the covariate distribution.
      if(k<wh || k>=offV)grad[k]+=decay*pars[k];
      norm2+=grad[k]*grad[k];
    }
    const double scale=norm2>25.?5./std::sqrt(norm2):1.;
    const double b1corr=1.-std::pow(.9,iter);
    const double b2corr=1.-std::pow(.999,iter);
    for(int k=0;k<parameter_count;++k) {
      const double g=grad[k]*scale;
      m[k]=.9*m[k]+.1*g;
      v[k]=.999*v[k]+.001*g*g;
      pars[k]-=rate*(m[k]/b1corr)/
                 (std::sqrt(v[k]/b2corr)+1e-8);
      if(!std::isfinite(pars[k]))
        stop("IPP neural optimizer diverged");
      pars[k]=std::max(-20.,std::min(20.,pars[k]));
    }
  }
  NumericMatrix W(p,hidden);
  NumericVector bias(hidden),output(hidden);
  for(int j=0;j<p;++j)
    for(int h=0;h<hidden;++h)
      W(j,h)=pars[j*hidden+h];
  for(int h=0;h<hidden;++h) {
    bias[h]=pars[wh+h];output[h]=pars[offV+h];
  }
  return List::create(_["W"]=W, _["bias"]=bias,
      _["output"]=output,_["objective_history"]=history,
      _["n_presence"]=np,_["n_background"]=nb,
      _["objective"]="log_mean_background_exp_minus_mean_presence",
      _["seed"]=seed);
}

// [[Rcpp::export]]
NumericVector nf_po_cpp_predict(List model, NumericMatrix X) {
  NumericMatrix W=as<NumericMatrix>(model["W"]);
  NumericVector bias=as<NumericVector>(model["bias"]);
  NumericVector output=as<NumericVector>(model["output"]);
  const int n=X.nrow(),p=X.ncol(),h=W.ncol();
  if(p!=W.nrow() || bias.size()!=h || output.size()!=h)
    stop("IPP neural prediction dimension mismatch");
  NumericVector scores(n);
  for(int i=0;i<n;++i) {
    double score=0.;
    for(int k=0;k<h;++k) {
      double z=bias[k];
      for(int j=0;j<p;++j) {
        if(!std::isfinite(X(i,j)))
          stop("Non-finite IPP predictor");
        z+=X(i,j)*W(j,k);
      }
      score+=output[k]*tanh_clip(z);
    }
    scores[i]=score;
  }
  return scores;
}
