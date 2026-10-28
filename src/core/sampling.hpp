#pragma once

#include <cstdint>
#include <optional>

// Token sampling settings, shared by the models' generate loops and the
// server. Temperature 0 is greedy decoding (argmax), bit-identical to the
// models' greedy paths.
namespace gufo_npu {

struct SamplingParams {
  float temperature = 0.0f;  // 0: greedy
  float top_p = 1.0f;        // nucleus mass kept; 1: off
  int top_k = 0;             // 0: off
  float min_p = 0.0f;        // drop tokens below min_p * max prob; 0: off
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  float repetition_penalty = 1.0f;  // 1: off
  std::optional<std::uint64_t> seed;

  bool greedy() const { return temperature <= 0.0f; }
};

}  // namespace gufo_npu
