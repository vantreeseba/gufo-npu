// The CPU reference against HF in float32 (tools/reference/
// qwen3vl_embedding_oracle.py): the token ids the package builds, every
// embedding (cosine >= 0.999), the retrieval set's rankings, and batching
// (each input alone = batched). Exits 77 without the model's variables.
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen3vl_embed_2b/reference.hpp"
#include "oracle.hpp"

int main() {
  const auto oracle = qwen3vl_test::Load();
  if (!oracle) {
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(oracle->gguf);
  const gufo_npu::Tokenizer tokenizer(*file);
  gufo_npu::qwen3vl_embed_2b::Reference reference(*file);
  int failures = 0;
  const auto inputs = qwen3vl_test::Inputs(tokenizer, *oracle, failures);
  const auto start = std::chrono::steady_clock::now();
  const auto got = reference.Embed(inputs);
  std::printf(
      "reference: %zu inputs in %.1f s\n", inputs.size(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  failures += qwen3vl_test::CheckEmbeddings(*oracle, got, 0.999, "reference");
  // Alone vs batched: fp32 on both sides, the same arithmetic per row.
  failures += qwen3vl_test::CheckBatchInvariance(
      reference, {inputs[0]}, {got[0]}, 0.999999, "reference");
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: reference matches HF");
  return EXIT_SUCCESS;
}
