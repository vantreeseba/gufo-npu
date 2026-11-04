#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/chat.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35_2b/npu_decoder.hpp"

// How often a second MTP draft would be kept (docs/models/qwen3.5-2b/
// EXPERIMENTS.md, "Drafting two tokens ahead"). Per prompt: greedy decode
// one token at a time, keeping every final residual; then over the greedy
// sequence, at each generated position p, the first draft d1 of token p + 1
// (the decoder's Draft(), as SpeculativeGenerate drafts) and a second draft
// d2 of token p + 2 from the MTP block's own final residual and d1, its
// cache entry dropped after. The residual is fed through output_norm (as
// the main model's) or shared_head_norm (as the MTP head's own output).
// Prints a1 = P(d1 kept) and a2 = P(d2 kept | d1 kept) per variant.
// usage: qwen35_mtp_depth <gguf> <kernel-dir> [max-tokens]
namespace {

namespace qwen35 = gufo_npu::qwen35;
using gufo_npu::qwen35_2b::NpuDecoder;

const char* const kPrompts[] = {
    "Explain in a few sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number, with a "
    "docstring and a short example.",
    "Write a short story about a lighthouse keeper who finds a message in a "
    "bottle.",
};

int Argmax(const float* v, int n) {
  return static_cast<int>(std::max_element(v, v + n) - v);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: qwen35_mtp_depth <gguf> <kernel-dir> [max-tokens]\n");
    return 2;
  }
  const int max_tokens = argc > 3 ? std::atoi(argv[3]) : 256;
  const auto file = gufo_npu::gguf::File::Open(argv[1]);
  const gufo_npu::Tokenizer tokenizer(*file);
  const std::vector<int> stop = {tokenizer.Find(qwen35::kImEnd),
                                 tokenizer.Find(qwen35::kEndOfText)};
  NpuDecoder npu(*file, argv[2]);
  const qwen35::Config& c = npu.config();
  const auto d = static_cast<std::size_t>(c.hidden);
  const std::size_t kv = static_cast<std::size_t>(c.kv_heads) * c.head_dim;

  long all_n = 0, all_a1 = 0, all_out = 0, all_shared = 0;
  for (const char* prompt : kPrompts) {
    std::vector<int> seq =
        tokenizer.Encode(qwen35::RenderChat({{"user", prompt}}, false));
    const int n = static_cast<int>(seq.size());
    auto state = npu.NewState();
    std::vector<float> hidden;  // final residual after the token at p
    const float* logits = nullptr;
    for (int p = 0;; ++p) {
      logits = npu.Step(&seq[p], 1, state);
      hidden.insert(hidden.end(), npu.hidden(0), npu.hidden(0) + d);
      if (p + 1 < static_cast<int>(seq.size())) {
        continue;
      }
      const int next = Argmax(logits, c.vocab);
      if (std::find(stop.begin(), stop.end(), next) != stop.end() ||
          static_cast<int>(seq.size()) - n >= max_tokens) {
        break;
      }
      seq.push_back(next);
    }
    const int len = static_cast<int>(seq.size());

    auto mtp = npu.NewMtpState();
    auto& cache = mtp.layers[static_cast<std::size_t>(c.layers)];
    auto drop_last = [&] {
      cache.keys.resize(cache.keys.size() - kv);
      cache.values.resize(cache.values.size() - kv);
    };
    long count = 0, a1 = 0, out_kept = 0, shared_kept = 0;
    std::vector<float> r1(d);
    for (int p = 1; p < len; ++p) {
      const float* hs[1] = {hidden.data() + static_cast<std::size_t>(p - 1) * d};
      // Drafts of generated tokens only, with two tokens after them known.
      if (p + 1 < n || p + 2 >= len) {
        npu.FillMtp(hs, &seq[p], 1, p, mtp);
        continue;
      }
      const int d1 = npu.Draft(hs, &seq[p], 1, p, mtp);
      std::copy_n(npu.mtp_hidden(0), d, r1.data());
      const float* rs[1] = {r1.data()};
      const int d2_out = npu.Draft(rs, &d1, 1, p + 1, mtp, false);
      drop_last();
      const int d2_shared = npu.Draft(rs, &d1, 1, p + 1, mtp, true);
      drop_last();
      ++count;
      if (d1 == seq[p + 1]) {
        ++a1;
        out_kept += d2_out == seq[p + 2];
        shared_kept += d2_shared == seq[p + 2];
      }
    }
    std::printf(
        "%-40.40s %3d generated: a1 %5.1f%% (%ld/%ld), a2|a1 output_norm "
        "%5.1f%%, shared_head_norm %5.1f%%\n",
        prompt, len - n, 100.0 * a1 / count, a1, count,
        a1 ? 100.0 * out_kept / a1 : 0.0, a1 ? 100.0 * shared_kept / a1 : 0.0);
    all_n += count;
    all_a1 += a1;
    all_out += out_kept;
    all_shared += shared_kept;
  }
  std::printf("all: a1 %.1f%%, a2|a1 output_norm %.1f%%, shared_head_norm %.1f%%\n",
              100.0 * all_a1 / all_n, 100.0 * all_out / all_a1,
              100.0 * all_shared / all_a1);
  return 0;
}
