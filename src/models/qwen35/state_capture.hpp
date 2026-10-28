#pragma once

#include <algorithm>
#include <cstddef>

#include "models/qwen35/decoder.hpp"

// Copying a prefill's DeltaNet state part-way through a chunk. The prompt
// cache (prompt_cache.hpp) takes its checkpoints this way, inside the pass
// that runs past them, so a checkpoint costs a copy and no extra pass. The
// DeltaNet layers' recurrent state and conv window cannot be recovered
// after later tokens; the attention layers' caches can (their first rows),
// so the cache slices those afterwards.
namespace gufo_npu::qwen35 {

// A request to copy the DeltaNet layers' state (conv window and recurrent
// state) as it is after the prompt token at `position - 1`, while a
// prefill's Run goes past it: into the same layers of `state`, whose conv
// and ssm are sized as a NewState()'s.
struct StateCapture {
  int position = 0;
  DecodeState* state = nullptr;
};

// Where one chunk's DeltaNet copies its state: into `layer` after the
// chunk's first `offset` tokens (1 .. count). An empty one copies nothing.
struct DeltaCapture {
  DecodeState::Layer* layer = nullptr;
  int offset = 0;
};

// Layer `il`'s DeltaCapture for a chunk of `count` tokens from `position`:
// empty unless `capture` asks for a position inside (position, position +
// count].
inline DeltaCapture CaptureIn(const StateCapture* capture, int il, int position,
                              int count) {
  if (capture == nullptr || capture->position <= position ||
      capture->position > position + count) {
    return {};
  }
  return {&capture->state->layers[static_cast<std::size_t>(il)],
          capture->position - position};
}

// The conv window after a chunk's first `offset` tokens (input rows `in`,
// `in_stride` apart; the window before the chunk in `before`): their last
// `window` inputs, oldest first, into `out` ([window][channels]).
inline void ConvWindowAfter(const float* in, std::size_t in_stride,
                            const float* before, int window, int channels,
                            int offset, float* out) {
  for (int s = 0; s < window; ++s) {
    const int src = offset - window + s;
    std::copy_n(
        src >= 0 ? in + static_cast<std::size_t>(src) * in_stride
                 : before + static_cast<std::size_t>(src + window) * channels,
        channels, out + static_cast<std::size_t>(s) * channels);
  }
}

}  // namespace gufo_npu::qwen35
