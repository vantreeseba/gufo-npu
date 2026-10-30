// Qwen3-VL-Embedding-2B's CPU pieces without a model: the input template and
// instruction rule, packing, and the attention mixer (online softmax over
// key blocks, grouped heads, one segment per input) against a direct
// softmax, pooling and the Matryoshka cut.
#include "models/qwen3vl_embed_2b/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/vec.hpp"
#include "models/qwen3vl_embed_2b/input.hpp"

namespace {

namespace e = gufo_npu::qwen3vl_embed_2b;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

void TestInput() {
  Check(e::NormalizeInstruction("") == e::kDefaultInstruction,
        "empty instruction: the default");
  Check(e::NormalizeInstruction("  \n") == e::kDefaultInstruction,
        "blank instruction: the default");
  Check(
      e::NormalizeInstruction("Represent the query") == "Represent the query.",
      "a '.' after a letter");
  Check(e::NormalizeInstruction("  Classify the topic!  ") ==
            "Classify the topic!",
        "trimmed, punctuation kept");
  Check(e::NormalizeInstruction("Find (receipts)") == "Find (receipts)",
        "closing bracket is punctuation");
  Check(e::NormalizeInstruction("日本語で検索。") == "日本語で検索。",
        "ideographic full stop is punctuation");
  Check(e::NormalizeInstruction("Find 42") == "Find 42.",
        "a '.' after a digit");
  Check(e::FormatInput("hi", "") ==
            "<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        "the chat template");
  Check(e::FormatInput("", "Task") ==
            "<|im_start|>system\nTask.<|im_end|>\n"
            "<|im_start|>user\nNULL<|im_end|>\n<|im_start|>assistant\n",
        "empty text embeds NULL");
}

void TestPack() {
  const auto s = e::Pack({{1, 2, 3}, {4}, {5, 6}}, 10);
  Check(s.size() == 3 && s[0].begin == 0 && s[0].length == 3 &&
            s[1].begin == 3 && s[1].length == 1 && s[2].begin == 4 &&
            s[2].length == 2,
        "segments back to back");
  bool threw = false;
  try {
    e::Pack({{1}, {}}, 10);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "an empty input throws");
  threw = false;
  try {
    e::Pack({{10}}, 10);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "an id outside the vocabulary throws");
}

// The mixer against a direct implementation: per segment, per query head,
// normalized and rotated q and k, a full causal softmax, the weighted sum.
// Runs each kernel the CPU has (the AVX-512 one and the plain loop).
void TestAttention(const e::Config& cfg, const std::vector<int>& lengths,
                   const std::string& name) {
  std::vector<e::Segment> segments;
  int rows = 0;
  int longest = 0;
  for (const int length : lengths) {
    segments.push_back({rows, length});
    rows += length;
    longest = std::max(longest, length);
  }
  const int stride = cfg.InRows();
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> in(static_cast<std::size_t>(rows) * stride);
  for (auto& x : in) {
    x = dist(rng);
  }
  std::vector<float> q_norm(cfg.head_dim), k_norm(cfg.head_dim);
  for (int i = 0; i < cfg.head_dim; ++i) {
    q_norm[i] = 1.0f + 0.1f * static_cast<float>(i % 8);
    k_norm[i] = 1.0f - 0.05f * static_cast<float>(i % 8);
  }
  e::RopeTable rope(cfg.head_dim, cfg.rope_theta);
  rope.Reserve(longest);

  // Direct: q and k prepared on a copy.
  std::vector<float> prepared = in;
  for (const auto& s : segments) {
    for (int t = 0; t < s.length; ++t) {
      float* row =
          prepared.data() + static_cast<std::size_t>(s.begin + t) * stride;
      for (int h = 0; h < cfg.heads; ++h) {
        float* q = row + h * cfg.head_dim;
        gufo_npu::vec::RmsNorm(q, q_norm.data(), cfg.head_dim, cfg.rms_eps, q);
        rope.Apply(q, t);
      }
      for (int h = 0; h < cfg.kv_heads; ++h) {
        float* k = row + cfg.QDim() + h * cfg.head_dim;
        gufo_npu::vec::RmsNorm(k, k_norm.data(), cfg.head_dim, cfg.rms_eps, k);
        rope.Apply(k, t);
      }
    }
  }
  std::vector<double> want(static_cast<std::size_t>(rows) * cfg.QDim());
  const double scale = 1.0 / std::sqrt(static_cast<double>(cfg.head_dim));
  const int group = cfg.heads / cfg.kv_heads;
  for (const auto& s : segments) {
    for (int t = 0; t < s.length; ++t) {
      for (int h = 0; h < cfg.heads; ++h) {
        const float* q = prepared.data() +
                         static_cast<std::size_t>(s.begin + t) * stride +
                         h * cfg.head_dim;
        const int kv = h / group;
        std::vector<double> score(static_cast<std::size_t>(t) + 1);
        double mx = -INFINITY;
        for (int j = 0; j <= t; ++j) {
          const float* k = prepared.data() +
                           static_cast<std::size_t>(s.begin + j) * stride +
                           cfg.QDim() + kv * cfg.head_dim;
          double dot = 0;
          for (int d = 0; d < cfg.head_dim; ++d) {
            dot += static_cast<double>(q[d]) * k[d];
          }
          score[j] = dot * scale;
          mx = std::max(mx, score[j]);
        }
        double total = 0;
        for (auto& x : score) {
          x = std::exp(x - mx);
          total += x;
        }
        double* out = want.data() +
                      static_cast<std::size_t>(s.begin + t) * cfg.QDim() +
                      h * cfg.head_dim;
        for (int j = 0; j <= t; ++j) {
          const float* v = prepared.data() +
                           static_cast<std::size_t>(s.begin + j) * stride +
                           cfg.QDim() + cfg.KvDim() + kv * cfg.head_dim;
          for (int d = 0; d < cfg.head_dim; ++d) {
            out[d] += score[j] / total * v[d];
          }
        }
      }
    }
  }

  const e::ParallelRun serial = [](std::size_t n,
                                   const std::function<void(std::size_t)>& fn) {
    for (std::size_t i = 0; i < n; ++i) {
      fn(i);
    }
  };
  std::vector<bool> kernels = {false};
  if (e::detail::HasAvx512()) {
    kernels.push_back(true);
  } else {
    std::puts("note: no AVX-512, only the plain attention loop is checked");
  }
  for (const bool avx512 : kernels) {
    std::vector<float> scratch = in;  // Attention prepares q and k in place
    std::vector<float> got(want.size());
    e::detail::Attention(cfg, rope, q_norm.data(), k_norm.data(), segments,
                         scratch.data(), got.data(), serial, avx512);
    double worst = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      worst = std::max(worst, std::fabs(got[i] - want[i]));
    }
    Check(worst < 1e-4, name + (avx512 ? ", AVX-512" : ", plain") +
                            ": attention matches the direct softmax (max "
                            "error " +
                            std::to_string(worst) + ")");
  }
}

void TestAttention() {
  e::Config cfg;
  cfg.heads = 4;
  cfg.kv_heads = 2;
  cfg.head_dim = 8;
  cfg.hidden = 32;
  cfg.rms_eps = 1e-6f;
  cfg.rope_theta = 10000.0f;
  // Lengths that cross the 64-key and 16-query blocks.
  TestAttention(cfg, {70, 1, 150}, "small heads");
  // The model's shape: 16 query heads over 8 kv heads of 128.
  cfg.heads = 16;
  cfg.kv_heads = 8;
  cfg.head_dim = 128;
  cfg.hidden = 2048;
  cfg.rope_theta = 5000000.0f;
  TestAttention(cfg, {1, 63, 64, 65, 300}, "model shape");
}

void TestPoolAndTruncate() {
  e::Config cfg;
  cfg.hidden = 16;
  cfg.rms_eps = 1e-6f;
  std::vector<float> h(3 * 16);
  for (std::size_t i = 0; i < h.size(); ++i) {
    h[i] = static_cast<float>(i % 7) - 3.0f;
  }
  std::vector<float> w(16, 1.5f);
  const auto out = e::Pool(cfg, w.data(), {{0, 2}, {2, 1}}, h.data());
  Check(out.size() == 2, "one vector per segment");
  for (const auto& v : out) {
    double n = 0;
    for (const float x : v) {
      n += static_cast<double>(x) * x;
    }
    Check(std::fabs(n - 1.0) < 1e-5, "pooled vectors are unit length");
  }
  // Segment 0 pools its last row (row 1), segment 1 row 2.
  std::vector<float> row1(h.begin() + 16, h.begin() + 32);
  double n1 = 0;
  for (const float x : row1) {
    n1 += static_cast<double>(x) * x;
  }
  Check(std::fabs(out[0][0] - row1[0] / std::sqrt(n1)) < 1e-5,
        "the last row of the segment");
  auto cut = out[0];
  e::Truncate(cut, 4);
  double nc = 0;
  for (const float x : cut) {
    nc += static_cast<double>(x) * x;
  }
  Check(cut.size() == 4 && std::fabs(nc - 1.0) < 1e-5,
        "a cut vector is renormalized");
  auto full = out[0];
  e::Truncate(full, 0);
  Check(full == out[0], "0 keeps the vector");
}

}  // namespace

int main() {
  TestInput();
  TestPack();
  TestAttention();
  TestPoolAndTruncate();
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: qwen3vl_embed_2b ops");
  return EXIT_SUCCESS;
}
