// The prompt cache on Qwen3.6-35B-A3B (tests/models/qwen35/
// prompt_cache_conversation.hpp has what it checks), with the prefill's
// experts on the CPU (the default).
//   GUFO_NPU_QWEN36_MODEL  UD-Q4_K_XL GGUF of Qwen3.6-35B-A3B
// Exits 77 (skip) when it is unset or there is no NPU.

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "../qwen35/prompt_cache_conversation.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen36_35b_a3b/npu_prefill.hpp"

int main(int argc, char** argv) {
  namespace qwen35 = gufo_npu::qwen35;
  namespace moe = gufo_npu::qwen36_35b_a3b;
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen36_prompt_cache_test <kernel-dir>\n");
    return 2;
  }
  const char* path = std::getenv("GUFO_NPU_QWEN36_MODEL");
  if (path == nullptr || *path == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN36_MODEL");
    return 77;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(path);
  qwen35::NpuDecoder npu(*file, argv[1]);
  moe::NpuPrefill prefill(*file, npu, argv[1]);
  prompt_cache_test::Model m;
  m.name = "Qwen3.6-35B-A3B";
  m.file = file.get();
  m.layers = npu.config().layers;
  m.hidden = npu.config().hidden;
  m.cache_model = [&] { return moe::MakePromptCacheModel(npu, prefill); };
  m.plain = moe::PrefillPrompt(prefill);
  m.generate = [&](const std::vector<int>& prompt, int max_tokens,
                   const std::vector<int>& stop,
                   const qwen35::PromptPrefill& hook,
                   qwen35::SpeculativeStats* stats) {
    return qwen35::SpeculativeGenerate(npu, prompt, max_tokens, stop, true, {},
                                       stats, hook);
  };
  m.new_state = [&] { return npu.NewState(); };
  m.new_mtp_state = [&] { return npu.NewMtpState(); };
  // The prefill-vs-decode limits of qwen36_npu_prefill_test.
  m.max_hidden_error = 0.15;
  m.max_state_error = 0.2;
  return prompt_cache_test::Run(m);
}
