#pragma once

#include <array>
#include <vector>

#include "models/qwen35/config.hpp"

// RoPE positions of the Qwen3.5 family's full-attention layers, with the
// interleaved multi-axis RoPE (MRoPE) of image prompts: each token has a
// time, height and width position, and each of the rope_dim / 2 frequencies
// rotates by one of them. Text tokens have all three equal, which is plain
// RoPE. The rotation itself (rotate-half pairs i, i + rope_dim / 2 of each
// head's leading rope_dim dims) stays in each mixer. Rules and the HF
// transformers code they follow: docs/models/qwen3.5-2b/REFERENCE.md.
namespace gufo_npu::qwen35 {

// A token's (t, h, w) RoPE position.
struct RopePosition {
  int t = 0;
  int h = 0;
  int w = 0;
  friend bool operator==(const RopePosition&, const RopePosition&) = default;
};

// A text token at RoPE position `p` on every axis.
inline RopePosition TextPosition(int p) {
  return {p, p, p};
}

// theta^(-2i / rope_dim) for i < rope_dim / 2, rounded to float as every
// mixer has used them.
std::vector<float> RopeInvFreq(const Config& c);

// The axis (0 t, 1 h, 2 w) frequency `i` rotates by, HF Qwen3.5's
// recomposition_frequencies: h when i % 3 == 1 and i < 3 * sections[1], w
// when i % 3 == 2 and i < 3 * sections[2], t otherwise.
int MropeAxis(const std::array<int, 3>& sections, int i);

// cos and sin of each of the rope_dim / 2 frequencies at `p`: the angle
// (double) p_axis * inv_freq[i], rounded to float.
void RopeAngles(const Config& c, const std::vector<float>& inv_freq,
                RopePosition p, float* cos, float* sin);

// The largest position on any axis of `positions` plus one, minus `end`
// (the KV index after them): the offset (HF's rope_deltas) at which text
// after the last of them continues, as DecodeState::rope_delta.
int RopeDeltaAfter(const std::vector<RopePosition>& positions, int end);

// An image's patch grid before the 2 x 2 merge (HF image_grid_thw): it
// covers t * (h / 2) * (w / 2) consecutive image tokens of a prompt.
struct ImageGrid {
  int t = 1;
  int h = 0;
  int w = 0;
};

// The RoPE positions of a prompt with images (HF Qwen3.5 get_rope_index):
// text tokens count up on all axes; the image token at merged grid cell
// (f, r, c) of an image starting at position s gets (s + f, s + r, s + c);
// text after the image resumes at s + max(h, w) / 2. Each run of
// `image_token` ids takes the next of `grids`. Throws std::invalid_argument
// when the runs and the grids do not match.
struct PromptPositions {
  std::vector<RopePosition> positions;  // one per token
  int rope_delta = 0;                   // RopeDeltaAfter(positions, size)
};
PromptPositions ImagePromptPositions(const std::vector<int>& tokens,
                                     int image_token,
                                     const std::vector<ImageGrid>& grids,
                                     int merge = 2);

}  // namespace gufo_npu::qwen35
