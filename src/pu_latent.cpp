// Hidden-layer representations and label-free forest-score alignment.
// All hidden activations are computed from frozen native C++ nnPU networks;
// never derive features from held-out test outcomes.
//
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdint>
using namespace Rcpp;

namespace {
inline double clipprob(const double p) {
  return std::max(1e-8, std::min(1.0-1e-8, p));
}
inline double sigmoid_safe(const double z) {
  if(z>=0)return 1.0/(1.0+std::exp(-z));
  const double v=std::exp(z);
  return v/(1.0+v);
}
inline double logit_safe(const double p) {
  const double q=clipprob(p);
  return std::log(q)-std::log1p(-q);
}
}

// [[Rcpp::export]]
NumericMatrix nf_pu_cpp_hidden_units(List model, NumericMatrix X,
                                    IntegerVector units) {
  NumericMatrix W=as<NumericMatrix>(model["W"]);
  NumericVector b=as<NumericVector>(model["bias"]);
  const int p=W.nrow(),h=W.ncol(),n=X.nrow();
  if(X.ncol()!=p || b.size()!=h || units.size()<1 ||
     units.size()>h)stop("Invalid native hidden feature dimensions");
  const int k=units.size();
  for(int j=0;j<k;++j)
    if(units[j]<1 || units[j]>h)
      stop("Selected hidden unit index is out of range");
  NumericMatrix Z(n,k);
  for(int i=0;i<n;++i) {
    for(int j=0;j<k;++j) {
      const int u=units[j]-1;
      double value=b[u];
      for(int col=0;col<p;++col) {
        if(!std::isfinite(X(i,col)))
          stop("Hidden input must have finite numeric features");
        value+=X(i,col)*W(col,u);
      }
      Z(i,j)=std::tanh(value);
    }
  }
  return Z;
}

// [[Rcpp::export]]
List nf_pu_cpp_align_fit(NumericVector forest, NumericVector teacher) {
  // Align the unlabeled-score distribution to a frozen nnPU teacher
  // without using audit/validation/test outcomes. Prefer OOB predictions
  // for forest, so the map is not fit against inbag overconfidence.
  const int n=forest.size();
  if(n<30 || teacher.size()!=n)
    stop("Aligning forest needs 30 paired training probabilities");
  double mx=0.0,my=0.0;int ok=0;
  for(int i=0;i<n;++i) {
    if(!std::isfinite(forest[i]))continue; // rare no-OOB cases
    if(!std::isfinite(teacher[i]) ||
       teacher[i]<0 || teacher[i]>1 ||
       forest[i]<0 || forest[i]>1)
      stop("Invalid teacher/OOB alignment probabilities");
    mx+=logit_safe(forest[i]);my+=logit_safe(teacher[i]);++ok;
  }
  if(ok<30)stop("Too few valid OOB pairs for score alignment");
  mx/=ok;my/=ok;
  double vx=0.0,cxy=0.0;
  for(int i=0;i<n;++i)if(std::isfinite(forest[i])) {
    const double dx=logit_safe(forest[i])-mx;
    const double dy=logit_safe(teacher[i])-my;
    vx+=dx*dx;cxy+=dx*dy;
  }
  // Positive slope is mandatory; no rank reversal is learned by the
  // alignment itself. The teacher/forest blend can change rankings.
  const double ridge=std::max(3.0,0.04*vx);
  double slope=(cxy+ridge)/(vx+ridge);
  slope=std::max(0.25,std::min(3.0,slope));
  const double intercept=std::max(-3.0,std::min(3.0,my-slope*mx));
  return List::create(_["slope"]=slope,_["intercept"]=intercept,
    _["n_oob"]=ok,_["n_total"]=n,
    _["ridge"]=ridge,_["source"]="training_oob_to_teacher_no_labels");
}

// [[Rcpp::export]]
NumericVector nf_pu_cpp_align_apply(NumericVector forest,List mapping) {
  const double slope=as<double>(mapping["slope"]);
  const double intercept=as<double>(mapping["intercept"]);
  if(!std::isfinite(slope)||slope<=0 ||
     !std::isfinite(intercept))
    stop("Invalid forest score alignment");
  NumericVector result(forest.size());
  for(int i=0;i<forest.size();++i) {
    if(!std::isfinite(forest[i]) ||
       forest[i]<0 || forest[i]>1)
      stop("Invalid aligned forest inference probabilities");
    result[i]=sigmoid_safe(slope*logit_safe(forest[i])+intercept);
  }
  return result;
}


// [[Rcpp::export]]
List nf_pu_cpp_random_projection(int input_columns,int hidden_units,int seed) {
  // Explicit negative control: these features have the SAME frozen tanh
  // form and number of columns, but no positive/unlabeled supervision.
  if(input_columns<1 || input_columns>20000 ||
     hidden_units<1 || hidden_units>256)
    stop("Invalid unsupervised random projection dimension");
  std::mt19937_64 rng(static_cast<uint64_t>(static_cast<uint32_t>(seed)));
  std::normal_distribution<double> gaussian(0.0,
                     1.0/std::sqrt(static_cast<double>(input_columns)));
  NumericMatrix W(input_columns,hidden_units);
  NumericVector bias(hidden_units),output(hidden_units);
  for(int i=0;i<input_columns;++i)
    for(int j=0;j<hidden_units;++j)
      W(i,j)=gaussian(rng);
  for(int j=0;j<hidden_units;++j) {
    bias[j]=0.25*gaussian(rng);
    output[j]=1.0;
  }
  return List::create(_["W"]=W,_["bias"]=bias,_["output"]=output,
                      _["seed"]=seed,_["trained"]=false);
}
