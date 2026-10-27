#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/chat.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/reference.hpp"

// End to end against the HF oracle: chat template + tokenizer give the same
// prompt ids, and greedy decoding gives the same 32 tokens.
//   GUFO_NPU_QWEN35_2B_GGUF    BF16 GGUF of Qwen3.5-2B
//   GUFO_NPU_QWEN35_2B_ORACLE  oracle output directory
int main() {
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_GGUF");
  const char* oracle = std::getenv("GUFO_NPU_QWEN35_2B_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN35_2B_GGUF and GUFO_NPU_QWEN35_2B_ORACLE");
    return 77;
  }
  namespace qwen35 = gufo_npu::qwen35;
  const std::filesystem::path dir = oracle;
  auto read_ids = [&](const char* name) {
    std::ifstream in(dir / name);
    return nlohmann::json::parse(in).get<std::vector<int>>();
  };
  const auto want_prompt = read_ids("chat.ids.json");
  const auto want_greedy = read_ids("chat.greedy.json");

  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  const qwen35::Reference reference(*file);
  const auto prompt = tokenizer.Encode(qwen35::RenderChat(
      {{"user", "Write one sentence about the moon."}}, false));
  if (prompt != want_prompt) {
    std::fprintf(stderr,
                 "FAIL: prompt ids differ from HF\n  want %s\n  got  %s\n",
                 nlohmann::json(want_prompt).dump().c_str(),
                 nlohmann::json(prompt).dump().c_str());
    return 1;
  }
  const auto got = qwen35::GreedyGenerate(
      reference, prompt, static_cast<int>(want_greedy.size()), {}, nullptr);
  std::size_t same = 0;
  while (same < got.size() && same < want_greedy.size() &&
         got[same] == want_greedy[same]) {
    ++same;
  }
  std::printf("%s: prompt ids match; greedy tokens match %zu/%zu\n  %s\n",
              same == want_greedy.size() ? "PASS" : "FAIL", same,
              want_greedy.size(), tokenizer.Decode(got).c_str());
  return same == want_greedy.size() ? EXIT_SUCCESS : EXIT_FAILURE;
}
