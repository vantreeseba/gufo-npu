#pragma once

#include <vector>

#include "models/qwen3vl_embed_2b/config.hpp"
#include "models/qwen3vl_embed_2b/vision.hpp"

// Token ids in (as InputIds or ImageInputIds build them, input.hpp), unit
// embeddings out: Reference (CPU) or NpuEmbedder.
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
  // The unit embedding of one input holding images: `ids` as ImageInputIds
  // builds them (one `image_pad_id` per image token), `images` from a
  // VisionEncoder in order of appearance. Throws std::invalid_argument
  // when ids and images disagree.
  virtual std::vector<float> EmbedImage(
      const std::vector<int>& ids, int image_pad_id,
      const std::vector<ImageFeatures>& images) = 0;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
