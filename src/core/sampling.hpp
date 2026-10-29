#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

// Token sampling settings, shared by the models' generate loops and the
// server. Temperature 0 is greedy decoding (argmax), bit-identical to the
// models' greedy paths.
namespace gufo_npu {

// Sees every token a Sampler picks, with the logits it picked from (the
// model's own, before penalties, logit_bias and temperature). The server
// uses it for logprobs; Sample calls it after each pick.
class SampleObserver {
public:
  virtual ~SampleObserver() = default;
  virtual void Sampled(const float* logits, int vocab, int id) = 0;
};

// Restricts which tokens may come next, e.g. to keep the output valid JSON
// (structured outputs). One object serves one generation at a time: the
// Sampler resets it when constructed, asks it which tokens are allowed
// before each draw and advances it with each generated token (Accept).
class TokenConstraint {
public:
  virtual ~TokenConstraint() = default;
  // Back to the start of a generation.
  virtual void Reset() = 0;
  // mask[id] = 1 when `id` may come next, else 0, for ids in [0, vocab).
  virtual void Allowed(int vocab, std::vector<std::uint8_t>& mask) = 0;
  // Whether `id` may come next (one token; cheaper than Allowed).
  virtual bool IsAllowed(int id) = 0;
  // `id` was generated.
  virtual void Advance(int id) = 0;
};

struct SamplingParams {
  float temperature = 0.0f;  // 0: greedy
  float top_p = 1.0f;        // nucleus mass kept; 1: off
  int top_k = 0;             // 0: off
  float min_p = 0.0f;        // drop tokens below min_p * max prob; 0: off
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  float repetition_penalty = 1.0f;  // 1: off
  std::optional<std::uint64_t> seed;
  // Added to a token's logit before sampling (OpenAI logit_bias: -100
  // bans a token, 100 all but forces it). Ids outside the vocab are ignored.
  std::vector<std::pair<int, float>> logit_bias;
  SampleObserver* observer = nullptr;  // not owned
  // Optional: tokens it disallows are never drawn (their logits count as
  // -inf before penalties, temperature and the filters). Without one,
  // sampling is exactly as before.
  std::shared_ptr<TokenConstraint> constraint;

  bool greedy() const { return temperature <= 0.0f; }
};

// A token's log probability under softmax(logits), and the `top` most
// likely tokens with theirs (best first, equal logits in id order).
struct TokenLogprobs {
  int id = -1;
  float logprob = 0;
  std::vector<std::pair<int, float>> top;
};
TokenLogprobs ComputeLogprobs(const float* logits, int vocab, int id, int top);

// Picks tokens from logits under SamplingParams. Penalties apply to the
// tokens generated so far (Accept), not to the prompt:
//   repetition: a seen token's logit is divided by the penalty when
//     positive and multiplied by it when negative (CTRL / llama.cpp);
//   presence / frequency: subtract presence + count * frequency (OpenAI).
// Then, unless greedy: logits / temperature, top_k and min_p (both keep the
// best tokens, so their order does not matter), softmax, top_p, and one
// draw. Tokens more than kTailNats below the best scaled logit are never
// candidates: on a 248k vocab they hold at most ~3.5e-6 of the best token's
// probability in all.
//
// Greedy without penalties is std::max_element over the logits (the first
// maximum), exactly as the models' greedy paths pick.
class Sampler {
public:
  static constexpr float kTailNats = 25.0f;

  // A seed in `params` makes the draws reproducible; without one the
  // sampler seeds from std::random_device.
  explicit Sampler(const SamplingParams& params = {});

  const SamplingParams& params() const { return params_; }
  bool greedy() const { return params_.greedy(); }

  // The next token for `logits` (`vocab` values, not modified).
  int Sample(const float* logits, int vocab);

  // Records a generated token for the penalties (and the constraint).
  void Accept(int id);

  // The candidates Sample draws from and their probabilities (summing to
  // 1), best first; greedy gives the one token it picks. For tests.
  std::vector<std::pair<int, double>> Distribution(const float* logits,
                                                   int vocab);

  // A uniform draw in [0, 1) from the sampler's generator.
  double Uniform();

private:
  bool HasPenalties() const;
  // Penalties or logit_bias change the logits before the pick.
  bool Adjusts() const;
  int Pick(const float* logits, int vocab);
  // The logit after penalties.
  float Penalized(int id, float logit) const;
  // The logits after penalties: `logits` itself when none apply.
  const float* Source(const float* logits, int vocab);
  // The candidates and their unnormalized weights: dense_ (weights_ over
  // the vocab, zero for non-candidates) or list_ (id, weight); total_ sums
  // them.
  void Candidates(const float* logits, int vocab);
  // With a constraint: `logits` with the disallowed tokens at -inf.
  const float* Masked(const float* logits, int vocab);

  SamplingParams params_;
  std::mt19937_64 rng_;
  std::unordered_map<int, int> counts_;  // generated token -> occurrences
  std::vector<float> penalized_;
  std::vector<float> masked_;
  std::vector<std::uint8_t> allowed_;
  std::vector<float> weights_;
  std::vector<std::pair<int, double>> list_;
  std::vector<int> ids_;
  std::vector<double> mass_;
  bool dense_ = false;
  double total_ = 0;
};

}  // namespace gufo_npu
