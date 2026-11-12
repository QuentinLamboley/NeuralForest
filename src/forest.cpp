// NeuralForest: self-contained CART ensemble engine (C++17).
// Rcpp handles the ABI; all training inner loops are native C++.
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

struct NFNode {
  int feature = -1, left = -1, right = -1, count = 0;
  double threshold = 0.0, gain = 0.0;
  std::vector<double> estimate;
};
struct NFRow {
  double x, y, w;
  bool operator<(const NFRow& other) const { return x < other.x; }
};
struct NFConfig {
  int min_leaf, min_split, max_depth, max_nodes, mtry, max_candidates, rule;
  int histogram_bins, histogram_mode;
  double min_gain, smoothing;
  bool extra_random, split_weighted;
};
struct NFModelTree {
  std::vector<NFNode> nodes;
  std::vector<unsigned char> inbag;
  std::vector<double> importance;
};
static inline double nf_unit(std::mt19937_64& gen) {
  return std::generate_canonical<double, 53>(gen);
}

class NFTreeBuilder {
  const NumericMatrix& X_;
  const NumericVector& y_;
  const NumericVector& weights_;
  const NumericVector& feature_weights_;
  int task_, K_, p_, min_leaf_, min_split_, max_depth_, max_nodes_;
  int mtry_, max_candidates_, rule_, histogram_bins_, histogram_mode_;
  double min_gain_, smoothing_;
  bool extra_random_, split_weighted_;
  std::mt19937_64& gen_;
  NFModelTree& out_;

  inline double obs_weight(int i) const { return split_weighted_ ? weights_[i] : 1.0; }
  void moments(const std::vector<int>& idx, std::vector<double>& s, double& w) const {
    s.assign(task_ == 0 ? 2 : K_, 0.0);
    w = 0.0;
    for (int i : idx) {
      const double wi = obs_weight(i), yi = y_[i];
      w += wi;
      if (task_ == 0) { s[0] += wi * yi; s[1] += wi * yi * yi; }
      else s[static_cast<int>(yi)] += wi;
    }
  }
  double impurity(const std::vector<double>& s, double w) const {
    if (w <= 0.0) return 0.0;
    if (task_ == 0) {
      const double m = s[0] / w;
      return std::max(0.0, s[1] / w - m * m);
    }
    double z = 0.0;
    for (double v : s) {
      const double q = v / w;
      if (rule_ == 1) { if (q > 0.0) z -= q * std::log(q); }
      else z += q * (1.0 - q);
    }
    return z;
  }
  std::vector<int> sample_features() {
    std::vector<int> pool(p_);
    std::iota(pool.begin(), pool.end(), 0);
    std::vector<int> chosen;
    chosen.reserve(mtry_);
    for (int i = 0; i < mtry_ && !pool.empty(); ++i) {
      std::vector<double> w;
      w.reserve(pool.size());
      for (int j : pool) w.push_back(std::max(1e-9, feature_weights_[j]));
      std::discrete_distribution<int> d(w.begin(), w.end());
      const int q = d(gen_);
      chosen.push_back(pool[q]);
      pool.erase(pool.begin() + q);
    }
    return chosen;
  }
  int grow(const std::vector<int>& idx, int depth) {
    const int node_id = static_cast<int>(out_.nodes.size());
    out_.nodes.emplace_back();
    NFNode& node = out_.nodes.back();
    node.count = static_cast<int>(idx.size());
    std::vector<double> total;
    double totalw = 0.0;
    moments(idx, total, totalw);
    node.estimate.assign(K_, 0.0);
    if (task_ == 0) {
      node.estimate[0] = totalw > 0 ? total[0] / totalw : 0.0;
    } else {
      const double den = totalw + smoothing_ * K_;
      for (int k = 0; k < K_; ++k)
        node.estimate[k] = den > 0 ? (total[k] + smoothing_) / den : 1.0 / K_;
    }
    if (static_cast<int>(idx.size()) < min_split_ || depth >= max_depth_ ||
        static_cast<int>(out_.nodes.size()) + 2 > max_nodes_ ||
        static_cast<int>(idx.size()) < 2 * min_leaf_ || totalw <= 0.0)
      return node_id;
    const double parent_imp = impurity(total, totalw);
    if (parent_imp <= 1e-14) return node_id;
    int best_feature = -1;
    double best_gain = min_gain_, best_cut = 0.0;
    for (int f : sample_features()) {
      // Histogram splitting avoids the per-node sort for large datasets.
      // Bounds and histograms are computed on the *current node* only.
      if (histogram_mode_ == 1) {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -lo;
        for (int i : idx) {
          lo = std::min(lo, static_cast<double>(X_(i,f)));
          hi = std::max(hi, static_cast<double>(X_(i,f)));
        }
        const int B = histogram_bins_;
        const double step = (hi-lo)/B;
        if (!(step > 0) || !std::isfinite(step)) continue;
        std::vector<int> bin_count(B,0);
        std::vector<double> bin_weights(B,0.0);
        std::vector<std::vector<double> > bin_stats(
          B,std::vector<double>(total.size(),0.0));
        for (int i : idx) {
          const double z = (X_(i,f)-lo)/step;
          const int b = std::min(B-1,std::max(0,static_cast<int>(z)));
          const double w = obs_weight(i), yy = y_[i];
          ++bin_count[b];
          bin_weights[b] += w;
          if (task_ == 0) {
            bin_stats[b][0] += w*yy;
            bin_stats[b][1] += w*yy*yy;
          } else bin_stats[b][static_cast<int>(yy)] += w;
        }
        std::vector<double> ls(total.size(),0.0), rs=total;
        double lw=0, rw=totalw;
        int lc=0;
        const int stride=max_candidates_>0 ?
          std::max(1,(B-1)/max_candidates_):1;
        const int random_pick=extra_random_ ?
          static_cast<int>(gen_()%stride):0;
        for (int b=0; b<B-1; ++b) {
          lc += bin_count[b]; lw += bin_weights[b]; rw -= bin_weights[b];
          for (size_t k=0; k<ls.size(); ++k) {
            ls[k] += bin_stats[b][k];
            rs[k] -= bin_stats[b][k];
          }
          if (lc<min_leaf_ || static_cast<int>(idx.size())-lc<min_leaf_ ||
              lw<=0 || rw<=0) continue;
          if (stride>1 && (b-random_pick)%stride != 0) continue;
          const double gain=parent_imp -
            (lw*impurity(ls,lw)+rw*impurity(rs,rw))/totalw;
          if (gain>best_gain+1e-14) {
            best_gain=gain;best_feature=f;
            const double cut=lo+(b+1)*step;
            best_cut=std::nextafter(cut,-std::numeric_limits<double>::infinity());
          }
        }
        continue;
      }
      std::vector<NFRow> rows;
      rows.reserve(idx.size());
      for (int i : idx) rows.push_back({X_(i, f), y_[i], obs_weight(i)});
      std::sort(rows.begin(), rows.end());
      if (rows.front().x == rows.back().x) continue;
      std::vector<double> left(total.size(), 0.0), right = total;
      double wl = 0.0, wr = totalw;
      const int n = static_cast<int>(rows.size());
      const int stride = max_candidates_ > 0 ?
        std::max(1, (n - 1) / max_candidates_) : 1;
      const int random_pick = extra_random_ ? static_cast<int>(gen_() % stride) : 0;
      for (int j = 0; j < n - 1; ++j) {
        const NFRow& r = rows[j];
        wl += r.w; wr -= r.w;
        if (task_ == 0) {
          left[0] += r.w * r.y; left[1] += r.w * r.y * r.y;
          right[0] -= r.w * r.y; right[1] -= r.w * r.y * r.y;
        } else {
          const int c = static_cast<int>(r.y);
          left[c] += r.w; right[c] -= r.w;
        }
        if (j + 1 < min_leaf_ || n - j - 1 < min_leaf_ ||
            rows[j].x >= rows[j + 1].x || wl <= 0 || wr <= 0) continue;
        if (stride > 1 && ((j - random_pick) % stride != 0)) continue;
        const double gain = parent_imp -
          (wl * impurity(left, wl) + wr * impurity(right, wr)) / totalw;
        if (gain > best_gain + 1e-14) {
          best_gain = gain;
          best_feature = f;
          best_cut = rows[j].x + 0.5 * (rows[j + 1].x - rows[j].x);
        }
      }
    }
    if (best_feature < 0) return node_id;
    std::vector<int> li, ri;
    li.reserve(idx.size()); ri.reserve(idx.size());
    for (int i : idx) {
      if (X_(i, best_feature) <= best_cut) li.push_back(i);
      else ri.push_back(i);
    }
    if (static_cast<int>(li.size()) < min_leaf_ || static_cast<int>(ri.size()) < min_leaf_)
      return node_id;
    out_.importance[best_feature] += best_gain * totalw;
    const int l = grow(li, depth + 1);
    const int r = grow(ri, depth + 1);
    // Reacquire by index: recursion can invalidate std::vector references.
    out_.nodes[node_id].feature = best_feature;
    out_.nodes[node_id].threshold = best_cut;
    out_.nodes[node_id].gain = best_gain;
    out_.nodes[node_id].left = l;
    out_.nodes[node_id].right = r;
    return node_id;
  }

public:
  NFTreeBuilder(const NumericMatrix& X, const NumericVector& y,
                const NumericVector& weights, const NumericVector& feature_weights,
                int task, int K, const NFConfig& cfg, std::mt19937_64& gen,
                NFModelTree& out)
    : X_(X), y_(y), weights_(weights), feature_weights_(feature_weights),
      task_(task), K_(K), p_(X.ncol()),
      min_leaf_(cfg.min_leaf), min_split_(cfg.min_split),
      max_depth_(cfg.max_depth), max_nodes_(cfg.max_nodes),
      mtry_(cfg.mtry), max_candidates_(cfg.max_candidates),
      rule_(cfg.rule), histogram_bins_(cfg.histogram_bins),
      histogram_mode_(cfg.histogram_mode), min_gain_(cfg.min_gain),
      smoothing_(cfg.smoothing),
      extra_random_(cfg.extra_random), split_weighted_(cfg.split_weighted), gen_(gen), out_(out) {}
  void train(const std::vector<int>& ids) { grow(ids, 0); }
};
static int leaf_of(const NFModelTree& tree, const NumericMatrix& X, int i) {
  int v = 0;
  while (tree.nodes[v].feature >= 0) {
    const NFNode& node = tree.nodes[v];
    v = (X(i, node.feature) <= node.threshold) ? node.left : node.right;
  }
  return v;
}
static int leaf_of_list(const IntegerVector& feature, const NumericVector& threshold,
                        const IntegerVector& left, const IntegerVector& right,
                        const NumericMatrix& X, int i) {
  int v = 0;
  while (feature[v] > 0) {
    v = X(i, feature[v] - 1) <= threshold[v] ? left[v] - 1 : right[v] - 1;
  }
  return v;
}

// [[Rcpp::export]]
List nf_cpp_fit(NumericMatrix X, NumericVector y, NumericVector weights,
                NumericVector feature_weights, List cfg, int task, int K, int seed) {
  const int n = X.nrow(), p = X.ncol();
  const int ntree = as<int>(cfg["n_trees"]);
  if (n < 4 || p < 1 || ntree < 1 || K < 1 || weights.size() != n ||
      feature_weights.size() != p) stop("Invalid native forest inputs");
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(y[i]) || !std::isfinite(weights[i]) || weights[i] < 0)
      stop("Targets and training weights must be finite and nonnegative");
    if (task == 1 && (y[i] < 0 || y[i] >= K || std::floor(y[i]) != y[i]))
      stop("Classification labels must be encoded as 0,...,K-1");
    for (int j = 0; j < p; ++j)
      if (!std::isfinite(X(i, j))) stop("Non-finite predictor passed to C++");
  }
  const double total_sampling_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (!(total_sampling_weight > 0.0) || !std::isfinite(total_sampling_weight))
    stop("Sampling weights must have a finite positive sum");
  const bool replace = as<bool>(cfg["replace"]);
  const double fraction = as<double>(cfg["sample_fraction"]);
  const int take = std::max(2, static_cast<int>(std::ceil(fraction * n)));
  if (!replace && take > n) stop("Sample fraction exceeds 1 without replacement");
  const int threads = std::max(1, as<int>(cfg["n_threads"]));
  if (as<int>(cfg["min_leaf"]) < 1 || as<int>(cfg["min_split"]) < 2 ||
      as<int>(cfg["max_depth"]) < 1 || as<int>(cfg["mtry"]) < 1 ||
      as<int>(cfg["max_nodes"]) < 3) stop("Invalid native tree settings");
  const NFConfig native_cfg = {
    as<int>(cfg["min_leaf"]), as<int>(cfg["min_split"]),
    as<int>(cfg["max_depth"]), as<int>(cfg["max_nodes"]),
    as<int>(cfg["mtry"]), as<int>(cfg["max_candidates"]),
    as<int>(cfg["split_rule"]), as<int>(cfg["histogram_bins"]),
    as<int>(cfg["histogram_mode"]), as<double>(cfg["min_gain"]),
    as<double>(cfg["laplace"]), as<bool>(cfg["extra_random"]),
    as<bool>(cfg["split_weighted"])
  };
  std::vector<NFModelTree> forest(ntree);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads)
#endif
  for (int t = 0; t < ntree; ++t) {
    std::mt19937_64 gen(static_cast<uint64_t>(seed) + 0x9e3779b97f4a7c15ULL *
                       static_cast<uint64_t>(t + 1));
    NFModelTree& tree = forest[t];
    tree.inbag.assign(n, 0);
    tree.importance.assign(p, 0.0);
    std::vector<int> ids;
    ids.reserve(take);
    if (replace) {
      std::discrete_distribution<int> draw(weights.begin(), weights.end());
      for (int a = 0; a < take; ++a) {
        const int i = draw(gen);
        ids.push_back(i); tree.inbag[i] = 1;
      }
    } else {
      std::vector<std::pair<double, int> > keys;
      keys.reserve(n);
      for (int i = 0; i < n; ++i) {
        const double w = std::max(1e-12, static_cast<double>(weights[i]));
        keys.emplace_back(-std::log(std::max(1e-16, nf_unit(gen))) / w, i);
      }
      if (take < n) std::nth_element(keys.begin(), keys.begin() + take, keys.end());
      for (int a = 0; a < take; ++a) {
        const int i = keys[a].second;
        ids.push_back(i); tree.inbag[i] = 1;
      }
    }
    NFTreeBuilder builder(X, y, weights, feature_weights, task, K, native_cfg, gen, tree);
    builder.train(ids);
  }
  List trees(ntree);
  NumericMatrix oob_sum(n, K);
  IntegerVector oob_count(n);
  NumericVector importance(p);
  for (int t = 0; t < ntree; ++t) {
    const NFModelTree& tree = forest[t];
    const int nn = static_cast<int>(tree.nodes.size());
    IntegerVector f(nn), l(nn), r(nn), count(nn);
    NumericVector threshold(nn), gain(nn);
    NumericMatrix prediction(nn, K);
    for (int k = 0; k < p; ++k) importance[k] += tree.importance[k];
    for (int v = 0; v < nn; ++v) {
      const NFNode& node = tree.nodes[v];
      f[v] = node.feature + 1; // Leaves are zero; other features are 1-indexed.
      l[v] = node.left + 1; r[v] = node.right + 1;
      threshold[v] = node.threshold; gain[v] = node.gain; count[v] = node.count;
      for (int k = 0; k < K; ++k) prediction(v, k) = node.estimate[k];
    }
    trees[t] = List::create(_["feature"] = f, _["threshold"] = threshold,
                           _["left"] = l, _["right"] = r,
                           _["prediction"] = prediction, _["gain"] = gain,
                           _["count"] = count);
    for (int i = 0; i < n; ++i) {
      if (tree.inbag[i]) continue;
      const int v = leaf_of(tree, X, i);
      ++oob_count[i];
      for (int k = 0; k < K; ++k)
        oob_sum(i, k) += tree.nodes[v].estimate[k];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k)
      oob_sum(i, k) = oob_count[i] > 0 ?
        oob_sum(i, k) / oob_count[i] : NA_REAL;
  }
  return List::create(_["trees"] = trees, _["oob_prediction"] = oob_sum,
                      _["oob_count"] = oob_count, _["importance"] = importance);
}

// [[Rcpp::export]]
NumericVector nf_cpp_predict(List trees, NumericMatrix X, int K,
                             bool per_tree = false) {
  const int n = X.nrow(), T = trees.size();
  if (K < 1 || T < 1) stop("Empty forest");
  NumericVector answer(per_tree ? n * K * T : n * K);
  if (per_tree) answer.attr("dim") = IntegerVector::create(n, K, T);
  else answer.attr("dim") = IntegerVector::create(n, K);
  for (int t = 0; t < T; ++t) {
    List tree = trees[t];
    IntegerVector f = tree["feature"], l = tree["left"], r = tree["right"];
    NumericVector cut = tree["threshold"];
    NumericMatrix values = tree["prediction"];
    for (int i = 0; i < n; ++i) {
      int v = leaf_of_list(f, cut, l, r, X, i);
      for (int k = 0; k < K; ++k) {
        const size_t offset = i + static_cast<size_t>(n) * k +
          (per_tree ? static_cast<size_t>(n) * K * t : 0);
        answer[offset] += values(v, k) / (per_tree ? 1 : T);
      }
    }
  }
  return answer;
}
