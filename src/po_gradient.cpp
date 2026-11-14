// Native presence-only point-process negative gradients for boosting
// native C++ CART regressors against the SAME IPP objective as the
// neural teacher. The background group is availability, not absence.
//
// For f = log relative intensity:
// risk = log(sum_BG w_j exp(f_j) / sum_BG w_j) - mean_P f_i
//   d risk / d f_P = -1/n_P
//   d risk / d f_BG_j = w_j exp(f_j) / sum_BG w exp(f).
//
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
using namespace Rcpp;

// [[Rcpp::export]]
List nf_po_cpp_gradient(NumericVector scores,NumericVector labels,
     NumericVector weights,double target_cap=5.0) {
  const int n=scores.size();
  if(n<50 || labels.size()!=n || weights.size()!=n ||
     !std::isfinite(target_cap) || target_cap<=0. ||
     target_cap>1e10)
    stop("Invalid point-process gradient dimensions/cap");
  int np=0,nb=0;
  double maxbg=-1e300, weight_sum=0.,positive_sum=0.;
  for(int i=0;i<n;++i) {
    if(labels[i]!=0. && labels[i]!=1.)
      stop("IPP derivative requires presence=1, available background=0");
    if(!std::isfinite(scores[i]) || !std::isfinite(weights[i]) ||
       weights[i]<=0.)
      stop("Non-finite IPP score or invalid inclusion weight");
    if(labels[i]==1.) {
      ++np;positive_sum+=scores[i];
    } else {
      ++nb;maxbg=std::max(maxbg,static_cast<double>(scores[i]));
      weight_sum+=weights[i];
    }
  }
  if(np<20 || nb<20)stop("Insufficient IPP presence or background support");
  double exp_total=0.;
  NumericVector mass(n),target(n);
  for(int i=0;i<n;++i)if(labels[i]==0.) {
    mass[i]=weights[i]*std::exp(std::max(-60.,
           std::min(0.,scores[i]-maxbg)));
    exp_total+=mass[i];
  }
  if(!std::isfinite(exp_total) || exp_total<=0)
    stop("Invalid IPP log-normalizer");
  const double objective=maxbg+std::log(exp_total/weight_sum)-
                        positive_sum/np;
  const double nscale=static_cast<double>(n);
  for(int i=0;i<n;++i) {
    const double derivative=labels[i]==1.?
         -1./static_cast<double>(np):
         mass[i]/exp_total;
    const double proposed=-nscale*derivative;
    target[i]=std::min(target_cap,std::max(-target_cap,proposed));
  }
  return List::create(_["target"]=target,
     _["objective"]=objective,_["n_presence"]=np,
     _["n_background"]=nb,_["gradient_sum_unclipped"]=0.0,
     _["score_semantics"]="relative_point_process_intensity");
}
