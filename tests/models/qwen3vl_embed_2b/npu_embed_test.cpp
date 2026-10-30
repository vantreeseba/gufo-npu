// The NPU embedder against HF in float32 (tools/reference/
// qwen3vl_embedding_oracle.py): every embedding (cosine >= 0.99; the GEMMs
// round weights and activations to bfp16 and outputs to bf16), the
// retrieval set's rankings, a long input across GEMM chunks, and batching
// (each input alone = batched, so packing never leaks between inputs).
// Exits 77 without an NPU or the model's variables.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen3vl_embed_2b/npu_embedder.hpp"
#include "oracle.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr,
                 "usage: qwen3vl_embed_2b_npu_embed_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto oracle = qwen3vl_test::Load();
  if (!oracle) {
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(oracle->gguf);
  const gufo_npu::Tokenizer tokenizer(*file);
  gufo_npu::qwen3vl_embed_2b::NpuEmbedder embedder(*file, argv[1]);
  int failures = 0;
  const auto inputs = qwen3vl_test::Inputs(tokenizer, *oracle, failures);
  embedder.Embed(inputs);  // warm-up
  const auto start = std::chrono::steady_clock::now();
  const auto got = embedder.Embed(inputs);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  long tokens = 0;
  for (const auto& ids : inputs) {
    tokens += static_cast<long>(ids.size());
  }
  std::printf("npu: %zu inputs, %ld tokens in %.3f s (%.0f tok/s, warm)\n",
              inputs.size(), tokens, seconds, tokens / seconds);
  failures += qwen3vl_test::CheckEmbeddings(*oracle, got, 0.99, "npu");
  // The GEMM's rows are independent, so an input's result does not depend
  // on its neighbours or its place in a chunk; allow bf16-level noise only.
  failures +=
      qwen3vl_test::CheckBatchInvariance(embedder, inputs, got, 0.9999, "npu");
  const auto& p = embedder.profile();
  std::printf(
      "profile: %ld GEMM calls; gemm %.2f s, sync %.2f s, attention %.2f s, "
      "other %.2f s of %.2f s\n",
      p.gemms, p.gemm_seconds, p.sync_seconds, p.attention_seconds,
      p.other_seconds, p.total_seconds);
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: NPU embeddings match HF");
  return EXIT_SUCCESS;
}
