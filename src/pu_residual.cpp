// PU residual-gradient target construction and stable logit corrections.
//
// A neural teacher supplies initial probabilities. Rather than fitting
// positive-vs-unlabeled classification trees (which wrongly treat U as
// verified negatives), regress shallow trees on gradients of the nnPU
// log-loss for a PREDECLARED grid of possible class prevalences.
//
// This is a research heuristic. Under SAR, case-control shifts or
// incorrect class priors these gradients need not identify true risk.
// Neither function reads independent verification or test labels.
//
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace Rcpp;

namespace {
inline double bounded_probability(double p) {
  return std::min(1.0-1e-10,std::max(1e-10,p));
}
inline double pu_logistic(double z) {
  if(z>=0) return 1.0/(1.0+std::exp(-z));
  const double ez=std::exp(z);
  return ez/(1.0+ez);
}
}

// [[Rcpp::export]]
List nf_pu_cpp_residual_targets(NumericVector probability,NumericVector labels,
                              NumericVector priors,double gradient_cap=3.0) {
  const int n=probability.size(),k=priors.size();
  if(n<40 || labels.size()!=n || k<1 ||
     !std::isfinite(gradient_cap) || gradient_cap<=0 ||
     gradient_cap>100.0)
    stop("Invalid PU residual target dimensions or cap");
  int np=0,nu=0;
  double up_neg=0.0,pos_neg=0.0;
  for(int i=0;i<n;++i) {
    if(labels[i]!=0.0 && labels[i]!=1.0)
      stop("PU residual needs exactly 0/1 training indicators");
    const double p=probability[i];
    if(!std::isfinite(p) || p<0.0 || p>1.0)
      stop("PU residual needs finite teacher probabilities");
    if(labels[i]==1.0) {
      ++np;
      pos_neg-=std::log1p(-bounded_probability(p));
    } else {
      ++nu;
      up_neg-=std::log1p(-bounded_probability(p));
    }
  }
  if(np<15 || nu<15)stop("Insufficient PU group support for residual fit");
  pos_neg/=np;up_neg/=nu;
  NumericVector target(n),negative_active(k);
  for(int j=0;j<k;++j) {
    const double prior=priors[j];
    if(!std::isfinite(prior)||prior<=0.0||prior>=1.0)
      stop("Invalid prespecified PU class prior");
    const bool active=(up_neg-prior*pos_neg)>0;
    negative_active[j]=active?1.0:0.0;
    for(int i=0;i<n;++i) {
      const double p=bounded_probability(probability[i]);
      const double grad=labels[i]==1.0?
        (prior*(p-1.0)-(active?prior*p:0.0))/np:
        (active?p/nu:0.0);
      // Multiply by sample count because PU risk uses group MEANS.
      target[i]-=static_cast<double>(n)*grad/k;
    }
  }
  for(int i=0;i<n;++i) {
    if(!std::isfinite(target[i]))stop("Unstable PU gradient");
    target[i]=std::max(-gradient_cap,std::min(gradient_cap,
                       static_cast<double>(target[i])));
  }
  return List::create(_["target"]=target,
    _["active_negative"]=negative_active,
    _["n_positive"]=np,_["n_unlabeled"]=nu);
}

// [[Rcpp::export]]
NumericVector nf_pu_cpp_logit_adjust(NumericVector teacher,
                         NumericVector logit_correction,
                         double multiplier,double logit_cap) {
  const int n=teacher.size();
  if(logit_correction.size()!=n ||
     !std::isfinite(multiplier) || multiplier<0 || multiplier>1 ||
     !std::isfinite(logit_cap) || logit_cap<=0 || logit_cap>5)
    stop("Invalid PU residual logit correction");
  NumericVector result(n);
  for(int i=0;i<n;++i) {
    if(!std::isfinite(teacher[i]) || teacher[i]<0 || teacher[i]>1 ||
       !std::isfinite(logit_correction[i]))
      stop("Residual correction requires finite teacher and tree scores");
    const double p=bounded_probability(teacher[i]);
    const double offset=std::max(-logit_cap,
             std::min(logit_cap,multiplier*logit_correction[i]));
    const double logit=std::log(p)-std::log1p(-p);
    result[i]=pu_logistic(logit+offset);
  }
  return result;
}
