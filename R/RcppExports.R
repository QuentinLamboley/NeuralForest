# Internal native entry points; regenerated with Rcpp::compileAttributes().
nf_cpp_fit <- function(X, y, weights, feature_weights, cfg, task, K, seed) {
  .Call('_NeuralForest_nf_cpp_fit', PACKAGE = 'NeuralForest',
        X, y, weights, feature_weights, cfg, task, K, seed)
}
nf_cpp_predict <- function(trees, X, K, per_tree = FALSE) {
  .Call('_NeuralForest_nf_cpp_predict', PACKAGE = 'NeuralForest',
        trees, X, K, per_tree)
}
