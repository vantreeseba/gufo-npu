#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "core/sampling.hpp"
#include "models/qwen35/generate.hpp"

// Speculative sampling on the NPU: sampled output with MTP drafts must follow
// the same distribution as without them. Each mode generates `runs` short
// continuations of one prompt with independent seeds; per position, and for
// the first two generated tokens jointly, a two-sample chi-square test
// (homogeneity, p = 1e-6) compares the modes. Shared by the models' NPU
// sampling tests.
namespace gufo_npu::qwen35::testing {

// Generates `tokens` tokens (no stop ids) with or without a draft.
using SampleFn = std::function<std::vector<int>(bool draft, int tokens,
                                                const SamplingParams& sampling,
                                                SpeculativeStats* stats)>;

// Chi-square critical value at p = 1e-6 (Wilson-Hilferty).
inline double ChiSquareLimit(int df) {
  const double z = 4.753;
  const double a = 2.0 / (9.0 * df);
  return df * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

// Pearson's homogeneity statistic for two samples over the same categories;
// categories with an expected count under 5 in either sample are pooled.
inline double Homogeneity(const std::map<long, long>& a,
                          const std::map<long, long>& b, int* df) {
  std::map<long, std::pair<long, long>> both;
  long na = 0;
  long nb = 0;
  for (const auto& [k, n] : a) {
    both[k].first = n;
    na += n;
  }
  for (const auto& [k, n] : b) {
    both[k].second = n;
    nb += n;
  }
  const double total = static_cast<double>(na + nb);
  double stat = 0;
  int cells = 0;
  long pool_a = 0;
  long pool_b = 0;
  auto add = [&](long ca, long cb) {
    const double ea = (ca + cb) * na / total;
    const double eb = (ca + cb) * nb / total;
    stat += (ca - ea) * (ca - ea) / ea + (cb - eb) * (cb - eb) / eb;
    ++cells;
  };
  for (const auto& [k, n] : both) {
    const double e = (n.first + n.second) * std::min(na, nb) / total;
    if (e < 5) {
      pool_a += n.first;
      pool_b += n.second;
    } else {
      add(n.first, n.second);
    }
  }
  if ((pool_a + pool_b) * std::min(na, nb) / total >= 5) {
    add(pool_a, pool_b);
  }
  *df = std::max(1, cells - 1);
  return cells > 1 ? stat : 0.0;
}

// Returns the number of failed comparisons.
inline int CheckSpeculativeSampling(const SampleFn& sample, int runs,
                                    int tokens) {
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 20;
  std::vector<std::vector<std::vector<int>>> outs(2);
  for (int mode = 0; mode < 2; ++mode) {
    const bool draft = mode == 1;
    long steps = 0;
    long accepted = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int run = 0; run < runs; ++run) {
      params.seed = static_cast<std::uint64_t>(run) + (draft ? 1000000 : 1);
      SpeculativeStats stats;
      outs[static_cast<std::size_t>(mode)].push_back(
          sample(draft, tokens, params, &stats));
      steps += stats.steps;
      accepted += stats.accepted;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf(
        "  %s: %d runs of %d tokens at temperature 1, top_k 20 in %.1f s; "
        "%s\n",
        draft ? "MTP draft" : "no draft ", runs, tokens, seconds,
        draft ? (std::to_string(100 * accepted / std::max(1L, steps)) +
                 "% of drafts accepted")
                    .c_str()
              : "");
  }
  int failures = 0;
  auto compare = [&](const char* what, auto key) {
    std::map<long, long> a;
    std::map<long, long> b;
    for (const auto& seq : outs[0]) {
      ++a[key(seq)];
    }
    for (const auto& seq : outs[1]) {
      ++b[key(seq)];
    }
    int df = 0;
    const double stat = Homogeneity(a, b, &df);
    const double limit = ChiSquareLimit(df);
    const bool ok = stat < limit;
    failures += ok ? 0 : 1;
    std::printf(
        "%s %s: chi-square %.1f (df %d, limit %.1f), %zu / %zu "
        "distinct\n",
        ok ? "ok  " : "FAIL", what, stat, df, limit, a.size(), b.size());
  };
  for (int k = 0; k < tokens; ++k) {
    char what[64];
    std::snprintf(what, sizeof what, "token %d, MTP draft against none", k);
    compare(what, [k](const std::vector<int>& seq) {
      return static_cast<long>(seq.at(static_cast<std::size_t>(k)));
    });
  }
  compare("tokens 1 and 2 jointly", [](const std::vector<int>& seq) {
    return static_cast<long>(seq.at(1)) * 1000000L + seq.at(2);
  });
  return failures;
}

}  // namespace gufo_npu::qwen35::testing
