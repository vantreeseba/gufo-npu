// The prompt cache on Qwen3.5-2B (tests/models/qwen35/
// prompt_cache_conversation.hpp has what it checks).
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  UD-Q4_K_XL GGUF of Qwen3.5-2B
// Exits 77 (skip) when it is unset or there is no NPU.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "../qwen35/prompt_cache_conversation.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen35/npu_prefill.hpp"

int main(int argc, char** argv) {
  namespace qwen35 = gufo_npu::qwen35;
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen35_prompt_cache_test <kernel-dir>\n");
    return 2;
  }
  const char* path = std::getenv("GUFO_NPU_QWEN35_2B_Q4_GGUF");
  if (path == nullptr || *path == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN35_2B_Q4_GGUF");
    return 77;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(path);
  qwen35::NpuDecoder npu(*file, argv[1]);
  qwen35::NpuPrefill prefill(*file, argv[1]);
  prompt_cache_test::Model m;
  m.name = "Qwen3.5-2B";
  m.file = file.get();
  m.layers = npu.config().layers;
  m.hidden = npu.config().hidden;
  m.cache_model = [&] { return qwen35::MakePromptCacheModel(npu, prefill); };
  m.plain = qwen35::PrefillPrompt(prefill);
  m.decode_path = [&](const std::vector<int>& p, qwen35::DecodeState& s,
                      qwen35::DecodeState* mtp, float* hidden) {
    const int k = static_cast<int>(p.size()) - 0;
    const int h = npu.config().hidden;
    qwen35::StepPrompt(
        npu,
        [&](const int* t, int n, qwen35::DecodeState& st) {
          npu.Step(t, n, st, false);
        },
        h, p.data(), k, s, hidden, nullptr);
    if (mtp != nullptr) {
      const int pairs = std::min(k, static_cast<int>(p.size()) - 1);
      qwen35::StepMtp(npu, h, hidden, p.data() + 1, pairs, 1, *mtp);
    }
    return k;
  };
  m.generate = [&](const std::vector<int>& prompt, int max_tokens,
                   const std::vector<int>& stop,
                   const qwen35::PromptPrefill& hook,
                   qwen35::SpeculativeStats* stats) {
    return qwen35::SpeculativeGenerate(npu, prompt, max_tokens, stop, true, {},
                                       stats, hook);
  };
  m.new_state = [&] { return npu.NewState(); };
  m.new_mtp_state = [&] { return npu.NewMtpState(); };
  // The prefill-vs-decode limits of qwen35_npu_prefill_test.
  m.max_hidden_error = 6e-2;
  m.max_state_error = 8e-2;
  return prompt_cache_test::Run(m);
}
