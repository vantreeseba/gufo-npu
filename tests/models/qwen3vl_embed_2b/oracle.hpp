#pragma once

// The HF oracle's cases (tools/reference/qwen3vl_embedding_oracle.py) and the
// checks the model tests share. Needs GUFO_NPU_QWEN3VL_EMBED_2B_GGUF (the
// model, BF16) and GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE (the oracle's JSON).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "core/tokenizer.hpp"
#include "models/qwen3vl_embed_2b/embedder.hpp"
#include "models/qwen3vl_embed_2b/input.hpp"

namespace qwen3vl_test {

struct Case {
  std::string name;
  std::string text;
  std::string instruction;
  std::vector<int> ids;
  std::vector<float> embedding;
};

struct Oracle {
  std::string gguf;
  std::vector<Case> cases;
  std::vector<int> queries, documents, best;
};

// The oracle and the GGUF path, or nullopt (after printing why) when the
// variables are unset: the test then exits 77.
inline std::optional<Oracle> Load() {
  const char* gguf = std::getenv("GUFO_NPU_QWEN3VL_EMBED_2B_GGUF");
  const char* path = std::getenv("GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE");
  if (gguf == nullptr || *gguf == '\0' || path == nullptr || *path == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN3VL_EMBED_2B_GGUF and "
        "GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE");
    return std::nullopt;
  }
  std::ifstream in(path);
  const auto j = nlohmann::json::parse(in);
  Oracle o;
  o.gguf = gguf;
  for (const auto& c : j.at("cases")) {
    o.cases.push_back({c.at("name").get<std::string>(),
                       c.at("text").get<std::string>(),
                       c.at("instruction").get<std::string>(),
                       c.at("ids").get<std::vector<int>>(),
                       c.at("embedding").get<std::vector<float>>()});
  }
  const auto& r = j.at("retrieval");
  o.queries = r.at("queries").get<std::vector<int>>();
  o.documents = r.at("documents").get<std::vector<int>>();
  o.best = r.at("best").get<std::vector<int>>();
  return o;
}

inline double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0;
  double na = 0;
  double nb = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return a.size() == b.size() && na > 0 && nb > 0 ? dot / std::sqrt(na * nb)
                                                  : 0.0;
}

// The oracle's inputs as the package formats them; counts id mismatches.
inline std::vector<std::vector<int>> Inputs(const gufo_npu::Tokenizer& tok,
                                            const Oracle& o, int& failures) {
  std::vector<std::vector<int>> inputs;
  for (const auto& c : o.cases) {
    auto ids = gufo_npu::qwen3vl_embed_2b::InputIds(tok, c.text, c.instruction);
    if (ids != c.ids) {
      std::fprintf(stderr, "FAIL: %s: token ids differ from HF (%zu vs %zu)\n",
                   c.name.c_str(), ids.size(), c.ids.size());
      ++failures;
    }
    inputs.push_back(std::move(ids));
  }
  return inputs;
}

// Every embedding against the oracle (cosine >= min_cosine) and the
// retrieval set's first-ranked documents; returns failures.
inline int CheckEmbeddings(const Oracle& o,
                           const std::vector<std::vector<float>>& got,
                           double min_cosine, const char* who) {
  int failures = 0;
  double worst = 1;
  double sum = 0;
  for (std::size_t i = 0; i < o.cases.size(); ++i) {
    const double cos = Cosine(got[i], o.cases[i].embedding);
    worst = std::min(worst, cos);
    sum += cos;
    if (cos < min_cosine) {
      std::fprintf(stderr, "FAIL: %s %s: cosine %.6f to HF < %.4f\n", who,
                   o.cases[i].name.c_str(), cos, min_cosine);
      ++failures;
    }
  }
  std::printf("%s: cosine to HF min %.6f, mean %.6f over %zu inputs\n", who,
              worst, sum / static_cast<double>(o.cases.size()), o.cases.size());
  for (std::size_t q = 0; q < o.queries.size(); ++q) {
    int best = -1;
    double top = -2;
    for (std::size_t d = 0; d < o.documents.size(); ++d) {
      const double s = Cosine(got[static_cast<std::size_t>(o.queries[q])],
                              got[static_cast<std::size_t>(o.documents[d])]);
      if (s > top) {
        top = s;
        best = static_cast<int>(d);
      }
    }
    if (best != o.best[q]) {
      std::fprintf(stderr,
                   "FAIL: %s query %zu ranks document %d first, HF %d\n", who,
                   q, best, o.best[q]);
      ++failures;
    }
  }
  return failures;
}

// Each input embedded alone matches the batched result (packing must not
// leak between inputs); returns failures.
inline int CheckBatchInvariance(gufo_npu::qwen3vl_embed_2b::Embedder& e,
                                const std::vector<std::vector<int>>& inputs,
                                const std::vector<std::vector<float>>& batched,
                                double min_cosine, const char* who) {
  int failures = 0;
  double worst = 1;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto alone = e.Embed({inputs[i]});
    const double cos = Cosine(alone.front(), batched[i]);
    worst = std::min(worst, cos);
    if (cos < min_cosine) {
      std::fprintf(stderr, "FAIL: %s input %zu alone vs batched: %.8f\n", who,
                   i, cos);
      ++failures;
    }
  }
  std::printf("%s: alone vs batched min cosine %.8f\n", who, worst);
  return failures;
}

}  // namespace qwen3vl_test
