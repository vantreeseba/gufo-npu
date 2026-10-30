#pragma once

#include <vector>

#include "models/qwen3vl_embed_2b/config.hpp"

// Text in (as InputIds builds it, input.hpp), unit embeddings out.
namespace gufo_npu::qwen3vl_embed_2b {

class Embedder {
public:
  virtual ~Embedder() = default;
  virtual const Config& config() const = 0;
  // One cfg.hidden unit vector per input, in order. Each input is a full
  // token sequence (template and <|endoftext|> included) and is embedded on
  // its own: batching changes the speed, not the result. Throws
  // std::invalid_argument for an empty input or an id outside the
  // vocabulary.
  virtual std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs) = 0;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
