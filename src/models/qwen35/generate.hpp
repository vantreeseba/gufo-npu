#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#include "models/qwen35/reference.hpp"

namespace gufo_npu::qwen35 {

struct GenerateStats {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  double prefill_seconds = 0;
  double decode_seconds = 0;
};

// Greedy decoding: runs the prompt, then appends the argmax token until a
// stop id or `max_tokens`. `on_token` sees each generated id (including the
// stop id) as it is produced. Returns the generated ids.
std::vector<int> GreedyGenerate(const Reference& model,
                                const std::vector<int>& prompt, int max_tokens,
                                const std::vector<int>& stop,
                                const std::function<void(int)>& on_token,
                                GenerateStats* stats = nullptr);

}  // namespace gufo_npu::qwen35
