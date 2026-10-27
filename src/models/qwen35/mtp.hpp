#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"

// The Qwen3.5 MTP (multi-token prediction) head on the CPU, in fp32 over the
// same q4nx weights the NPU uses. One step drafts the token after next:
//   x = eh_proj [RMS(embed(token)) * enorm ; RMS(h) * hnorm]
//   one full-attention block (blk.<layers>, its own KV cache in the state)
//   logits = lm_head(RMS(x) * shared_head_norm)
// where h is the main model's final hidden state at the position before
// `token`. docs/models/qwen3.5-2b/REFERENCE.md.
namespace gufo_npu::qwen35 {

class MtpHead {
public:
  struct Options {
    // h passes through the main model's output_norm before hnorm.
    bool hidden_after_output_norm = true;
    // RoPE position of a step: the position of `token` (true) or of h.
    bool position_of_token = true;
  };

  MtpHead(const gguf::File& file, Options options);
  ~MtpHead();
  MtpHead(const MtpHead&) = delete;
  MtpHead& operator=(const MtpHead&) = delete;

  // `h`: the main model's final residual (before output_norm) at position
  // `position - 1`; `token`: the token at `position`. Appends to the MTP
  // layer's cache in `state` and returns the logits for position + 1.
  const std::vector<float>& Step(const float* h, int token, int position,
                                 DecodeState& state);

  DecodeState NewState() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen35
