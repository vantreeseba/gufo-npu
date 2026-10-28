#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <list>
#include <vector>

#include "models/qwen35/decoder.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/state_capture.hpp"

// Reusing earlier requests' prompt processing (a multi-turn chat resends
// its whole history every turn). The DeltaNet layers' state is recurrent:
// it cannot be cut back to an earlier position, so reuse starts from saved
// checkpoints, each the full decode state (and the MTP layer's cache) after
// a prompt prefix.
//
// Where the checkpoint goes: the Qwen chat template renders earlier
// assistant turns without the think block the generation prompt opens
// (`<|im_start|>assistant\n<think>\n\n</think>\n\n` for a direct answer),
// so the next request's prompt matches the previous one only up to the
// generation prompt's `<|im_start|>`. A request's checkpoint is taken there,
// before the last `boundary_token` of its prompt: the history through the
// last user message. The next turn restores it and runs only the previous
// reply, the new message and the generation prompt. The DeltaNet state is
// copied inside the prefill pass that runs past the boundary
// (StateCapture), so taking a checkpoint costs a copy, not a pass.
namespace gufo_npu::qwen35 {

struct PromptCacheOptions {
  int max_checkpoints = 2;    // least recently used ones go first
  std::size_t max_bytes = 0;  // 0: no limit besides max_checkpoints
  // Checkpoint before the prompt's last of these tokens (the chat
  // template's `<|im_start|>`); -1: at the end of what the cache runs.
  int boundary_token = -1;
  // Shorter prefixes are not worth a checkpoint.
  int min_tokens = 16;
};

// How one model runs prompt tokens, for the cache (each model's
// PromptCacheModel builds it).
struct PromptCacheModel {
  int hidden = 0;
  int vocab = 0;
  int layers = 0;  // decoder layers; the MTP layer's cache follows them
  // Prompt tokens left to the decoder's own first step (1 for
  // Qwen3.6-35B-A3B, whose decoder runs the prompt's last token).
  int keep_last = 0;
  // A prompt with no checkpoint to restore that is shorter than this goes
  // to the decoder untouched (the hook returns 0), as the model's own
  // PrefillPrompt leaves it.
  int min_prefill = 1;
  // Runs `count` (>= 1) tokens from state.position and advances the state,
  // writing their final residuals (count rows); with a capture, copies the
  // DeltaNet layers' state at capture->position on the way.
  std::function<void(const int* tokens, int count, DecodeState& state,
                     float* hidden, const StateCapture* capture)>
      run;
  // The MTP cache: pairs (hidden row i, tokens[i]) at position + i.
  std::function<void(const float* hidden, const int* tokens, int count,
                     int position, DecodeState& mtp)>
      fill_mtp;
  // A fresh decode state (the checkpoints' DeltaNet buffers take its
  // sizes).
  std::function<DecodeState()> new_state;
};

class PromptCache {
public:
  explicit PromptCache(PromptCacheModel model, PromptCacheOptions options = {});
  PromptCache(const PromptCache&) = delete;
  PromptCache& operator=(const PromptCache&) = delete;

  // The prompt hook for the model's SpeculativeGenerate: restores the
  // longest checkpoint whose tokens begin the prompt, runs the rest (all of
  // it on a miss) and takes the prompt's checkpoint. Valid while the cache
  // lives; one generation at a time.
  PromptPrefill Hook();

  // The last Hook call's prompt tokens restored from a checkpoint (0: none).
  int last_reused() const { return last_reused_; }

  struct Stats {
    long hits = 0;
    long misses = 0;
    long checkpoints = 0;  // taken
    long evictions = 0;
  };
  const Stats& stats() const { return stats_; }
  int size() const { return static_cast<int>(entries_.size()); }
  std::size_t bytes() const { return bytes_; }
  // The tokens of each checkpoint, least recently used first.
  std::vector<int> checkpoint_lengths() const;
  void Clear();

private:
  struct Entry {
    std::vector<int> tokens;
    DecodeState state;       // main model, at tokens.size()
    DecodeState::Layer mtp;  // the MTP layer's cache (positions 1..)
    bool has_mtp = false;
    std::vector<float> hidden;  // final residual of the last token
    std::size_t bytes = 0;
  };

  int Run(const std::vector<int>& prompt, DecodeState& state, DecodeState* mtp,
          float* hidden);
  std::list<Entry>::iterator Find(const std::vector<int>& prompt, int end,
                                  bool need_mtp);
  int Boundary(const std::vector<int>& prompt, int end) const;
  void Insert(Entry entry);
  static std::size_t Bytes(const Entry& entry);

  PromptCacheModel model_;
  PromptCacheOptions options_;
  std::list<Entry> entries_;  // least recently used first
  std::size_t bytes_ = 0;
  int last_reused_ = 0;
  Stats stats_;
};

// Runs prompt tokens through an NPU decoder two per pass (both kept), as
// SpeculativeGenerate does, for prompt parts too short for a prefill pass:
// passes split so one ends at capture->position, where the DeltaNet layers'
// state is copied. `step(tokens, count, state)` runs one pass;
// `decoder.hidden(t)` reads its final residuals.
template<typename Decoder, typename StepFn>
void StepPrompt(Decoder& decoder, const StepFn& step, int hidden_size,
                const int* tokens, int count, DecodeState& state, float* hidden,
                const StateCapture* capture) {
  const auto d = static_cast<std::size_t>(hidden_size);
  for (int i = 0; i < count;) {
    int pass = std::min(2, count - i);
    if (capture != nullptr && capture->position > state.position &&
        capture->position < state.position + pass) {
      pass = capture->position - state.position;
    }
    step(tokens + i, pass, state);
    if (pass == 2) {
      decoder.Resolve(state, true);
    }
    for (int t = 0; t < pass; ++t) {
      std::copy_n(decoder.hidden(t), d, hidden + (i + t) * d);
    }
    i += pass;
    if (capture != nullptr && capture->position == state.position) {
      for (std::size_t il = 0; il < state.layers.size(); ++il) {
        const auto& from = state.layers[il];
        if (!from.ssm.empty()) {
          capture->state->layers[il].conv = from.conv;
          capture->state->layers[il].ssm = from.ssm;
        }
      }
    }
  }
}

// The MTP cache through the decoder, two pairs per call: pairs (hidden row
// i, tokens[i]) at position + i.
template<typename Decoder>
void StepMtp(Decoder& decoder, int hidden_size, const float* hidden,
             const int* tokens, int count, int position, DecodeState& mtp) {
  const auto d = static_cast<std::size_t>(hidden_size);
  for (int i = 0; i < count; i += 2) {
    const int pass = std::min(2, count - i);
    const float* hs[2] = {hidden + i * d,
                          pass == 2 ? hidden + (i + 1) * d : nullptr};
    decoder.FillMtp(hs, tokens + i, pass, position + i, mtp);
  }
}

}  // namespace gufo_npu::qwen35
