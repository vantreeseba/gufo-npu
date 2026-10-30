#pragma once

#include <memory>

#include "core/gguf.hpp"
#include "models/qwen3vl_embed_2b/embedder.hpp"
#include "models/qwen3vl_embed_2b/vision.hpp"

// The CPU reference: the model in fp32 straight from the GGUF's weights
// (each weight row dequantized as it is used), the attention and pooling of
// ops.hpp. Slow; the oracle for the NPU embedder and `--device cpu`.
namespace gufo_npu::qwen3vl_embed_2b {

class Reference final : public Embedder {
public:
  // Keeps pointers into `file`, which must outlive it.
  explicit Reference(const gguf::File& file);
  ~Reference() override;
  Reference(const Reference&) = delete;
  Reference& operator=(const Reference&) = delete;

  const Config& config() const override;
  std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs) override;

  // The unit embedding of one input holding images: `ids` as ImageInputIds
  // builds them (input.hpp; one `image_pad_id` per image token), `images`
  // from VisionReference::Encode in order of appearance. The image tokens'
  // rows take the images' embeddings, the text model rotates by
  // MropePositions, and DeepStack block k is added to the image rows after
  // layer k. Throws std::invalid_argument when ids and images disagree.
  std::vector<float> EmbedImage(const std::vector<int>& ids, int image_pad_id,
                                const std::vector<ImageFeatures>& images);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
