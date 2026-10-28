#pragma once

#include <optional>
#include <utility>

#include "core/sampling.hpp"

// Logprobs for the server: the generate loops' Sampler reports every pick
// with the logits it came from; the recorder keeps the last one's log
// probabilities until the server's token callback takes them. Each emitted
// token is the pick just before it (an accepted MTP draft is the verify
// pass's pick of that same token), so the two always pair up.
namespace gufo_npu::serve {

class LogprobRecorder : public SampleObserver {
public:
  explicit LogprobRecorder(int top) : top_(top) {}

  void Sampled(const float* logits, int vocab, int id) override {
    last_ = ComputeLogprobs(logits, vocab, id, top_);
    has_ = true;
  }

  // The logprobs of `id` if it was the last pick.
  std::optional<TokenLogprobs> Take(int id) {
    if (!has_ || last_.id != id) {
      return std::nullopt;
    }
    has_ = false;
    return std::move(last_);
  }

private:
  int top_;
  TokenLogprobs last_;
  bool has_ = false;
};

}  // namespace gufo_npu::serve
