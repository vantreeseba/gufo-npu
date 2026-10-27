#include "models/qwen35/generate.hpp"

#include <chrono>
#include <stdexcept>

namespace gufo_npu::qwen35 {
namespace {

int Argmax(const std::vector<float>& logits, std::size_t offset,
           std::size_t vocab) {
  const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(offset);
  return static_cast<int>(
      std::max_element(begin, begin + static_cast<std::ptrdiff_t>(vocab)) -
      begin);
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

std::vector<int> GreedyGenerate(const Reference& model,
                                const std::vector<int>& prompt, int max_tokens,
                                const std::vector<int>& stop,
                                const std::function<void(int)>& on_token,
                                GenerateStats* stats) {
  if (prompt.empty()) {
    throw std::invalid_argument("generate: empty prompt");
  }
  const auto vocab = static_cast<std::size_t>(model.config().vocab);
  auto state = model.NewState();
  std::vector<int> generated;
  GenerateStats local;
  local.prompt_tokens = static_cast<int>(prompt.size());

  auto start = std::chrono::steady_clock::now();
  auto out = model.Forward(prompt, state);
  local.prefill_seconds = SecondsSince(start);
  int next = Argmax(out.logits, (prompt.size() - 1) * vocab, vocab);

  start = std::chrono::steady_clock::now();
  while (static_cast<int>(generated.size()) < max_tokens) {
    generated.push_back(next);
    if (on_token) {
      on_token(next);
    }
    if (std::find(stop.begin(), stop.end(), next) != stop.end() ||
        static_cast<int>(generated.size()) == max_tokens) {
      break;
    }
    out = model.Forward({next}, state);
    next = Argmax(out.logits, 0, vocab);
  }
  local.decode_seconds = SecondsSince(start);
  local.generated_tokens = static_cast<int>(generated.size());
  if (stats != nullptr) {
    *stats = local;
  }
  return generated;
}

}  // namespace gufo_npu::qwen35
