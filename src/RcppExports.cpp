// Generated compatibility wrappers for Rcpp exported native routines.
#include <Rcpp.h>
using namespace Rcpp;
List nf_cpp_fit(NumericMatrix, NumericVector, NumericVector, NumericVector,
                List, int, int, int);
NumericVector nf_cpp_predict(List, NumericMatrix, int, bool);
RcppExport SEXP _NeuralForest_nf_cpp_fit(SEXP XSEXP, SEXP ySEXP,
                                         SEXP weightsSEXP, SEXP fwSEXP,
                                         SEXP cfgSEXP, SEXP taskSEXP,
                                         SEXP kSEXP, SEXP seedSEXP) {
  BEGIN_RCPP
  return wrap(nf_cpp_fit(as<NumericMatrix>(XSEXP), as<NumericVector>(ySEXP),
                         as<NumericVector>(weightsSEXP), as<NumericVector>(fwSEXP),
                         as<List>(cfgSEXP), as<int>(taskSEXP), as<int>(kSEXP),
                         as<int>(seedSEXP)));
  END_RCPP
}
RcppExport SEXP _NeuralForest_nf_cpp_predict(SEXP treesSEXP, SEXP XSEXP,
                                              SEXP kSEXP, SEXP ptSEXP) {
  BEGIN_RCPP
  return wrap(nf_cpp_predict(as<List>(treesSEXP), as<NumericMatrix>(XSEXP),
                             as<int>(kSEXP), as<bool>(ptSEXP)));
  END_RCPP
}
static const R_CallMethodDef CallEntries[] = {
  {"_NeuralForest_nf_cpp_fit", (DL_FUNC) &_NeuralForest_nf_cpp_fit, 8},
  {"_NeuralForest_nf_cpp_predict", (DL_FUNC) &_NeuralForest_nf_cpp_predict, 4},
  {NULL, NULL, 0}
};
RcppExport void R_init_NeuralForest(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
