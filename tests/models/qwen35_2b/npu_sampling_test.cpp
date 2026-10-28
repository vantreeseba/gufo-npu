#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "../qwen35/sampling_check.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/chat.hpp"
#include "models/qwen35/npu_decoder.hpp"

// Qwen3.5-2B: sampled output with MTP drafts follows the same distribution as
// without them (tests/models/qwen35/sampling_check.hpp).
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  UD-Q4_K_XL GGUF of Qwen3.5-2B
// Exits 77 (skip) when it is unset or there is no NPU.
int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::fprintf(stderr,
                 "usage: qwen35_npu_sampling_test <kernel-dir> [runs]\n");
    return 2;
  }
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_Q4_GGUF");
  if (model == nullptr || *model == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN35_2B_Q4_GGUF");
    return 77;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  namespace qwen35 = gufo_npu::qwen35;
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  const auto prompt = tokenizer.Encode(qwen35::RenderChat(
      {{"user", "Write a short poem about the sea."}}, false));
  qwen35::NpuDecoder npu(*file, argv[1]);
  const int runs = argc == 3 ? std::atoi(argv[2]) : 1000;
  const auto sample = [&](bool draft, int tokens,
                          const gufo_npu::SamplingParams& sampling,
                          qwen35::SpeculativeStats* stats) {
    return qwen35::SpeculativeGenerate(npu, prompt, tokens, {}, draft, {},
                                       stats, {}, sampling);
  };
  const int failures =
      qwen35::testing::CheckSpeculativeSampling(sample, runs, 5);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
