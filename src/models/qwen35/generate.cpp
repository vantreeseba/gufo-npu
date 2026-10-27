#include "models/qwen35/generate.hpp"

#include <chrono>
#include <stdexcept>

namespace gufo_npu::qwen35 {
namespace {

int Argmax(const float* logits, int vocab) {
  return static_cast<int>(std::max_element(logits, logits + vocab) - logits);
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

std::vector<int> GreedyGenerate(const ForwardFn& forward, int vocab,
                                const std::vector<int>& prompt, int max_tokens,
                                const std::vector<int>& stop,
                                const std::function<void(int)>& on_token,
                                GenerateStats* stats) {
  if (prompt.empty()) {
    throw std::invalid_argument("generate: empty prompt");
  }
  std::vector<int> generated;
  GenerateStats local;
  local.prompt_tokens = static_cast<int>(prompt.size());

  auto start = std::chrono::steady_clock::now();
  int next = Argmax(forward(prompt), vocab);
  local.prefill_seconds = SecondsSince(start);

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
    next = Argmax(forward({next}), vocab);
  }
  local.decode_seconds = SecondsSince(start);
  local.generated_tokens = static_cast<int>(generated.size());
  if (stats != nullptr) {
    *stats = local;
  }
  return generated;
}

std::vector<int> GreedyGenerate(const Reference& model,
                                const std::vector<int>& prompt, int max_tokens,
                                const std::vector<int>& stop,
                                const std::function<void(int)>& on_token,
                                GenerateStats* stats,
                                const Projections* projections) {
  const int vocab = model.config().vocab;
  auto state = model.NewState();
  std::vector<float> logits;
  const ForwardFn forward = [&](const std::vector<int>& ids) {
    logits = model.Forward(ids, state, true, false, projections).logits;
    return logits.data() + (ids.size() - 1) * static_cast<std::size_t>(vocab);
  };
  return GreedyGenerate(forward, vocab, prompt, max_tokens, stop, on_token,
                        stats);
}

}  // namespace gufo_npu::qwen35
