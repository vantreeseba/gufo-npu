// PromptCache (src/models/qwen35/prompt_cache.hpp) on a toy model: one
// recurrent layer (an order-dependent hash, as a DeltaNet state cannot be
// cut back), one attention-like layer (a row per token) and an MTP layer.
// A prompt run through a restored checkpoint must leave exactly the state,
// hidden rows and MTP cache the uncached run leaves.

#include "models/qwen35/prompt_cache.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace qwen35 = gufo_npu::qwen35;
using qwen35::DecodeState;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

constexpr int kVocab = 100;
constexpr int kImStart = 7;  // the boundary token

DecodeState NewState() {
  DecodeState s;
  s.layers.resize(3);  // recurrent, attention, MTP
  s.layers[0].conv.assign(1, 0.0f);
  s.layers[0].ssm.assign(2, 0.0f);
  return s;
}

struct Toy {
  long tokens_run = 0;  // prompt tokens the model processed
  long mtp_pairs = 0;

  void Run(const int* tokens, int count, DecodeState& state, float* hidden,
           const qwen35::StateCapture* capture) {
    for (int i = 0; i < count; ++i) {
      auto& r = state.layers[0];
      r.ssm[0] =
          std::fmod(r.ssm[0] * 31.0f + static_cast<float>(tokens[i]), 65521.0f);
      r.ssm[1] += 1.0f;
      r.conv[0] = static_cast<float>(tokens[i]);
      auto& a = state.layers[1];
      a.keys.push_back(static_cast<std::uint16_t>(tokens[i]));
      a.keys.push_back(static_cast<std::uint16_t>(state.position));
      a.values.push_back(static_cast<std::uint16_t>(r.ssm[0]));
      a.values.push_back(1);
      hidden[i] = r.ssm[0] + 0.5f;
      ++state.position;
      ++tokens_run;
      if (capture != nullptr && capture->position == state.position) {
        capture->state->layers[0].conv = r.conv;
        capture->state->layers[0].ssm = r.ssm;
      }
    }
  }

  void FillMtp(const float* hidden, const int* tokens, int count, int position,
               DecodeState& mtp) {
    auto& m = mtp.layers[2];
    for (int i = 0; i < count; ++i) {
      m.keys.push_back(static_cast<std::uint16_t>(hidden[i]));
      m.keys.push_back(static_cast<std::uint16_t>(tokens[i]));
      m.keys.push_back(static_cast<std::uint16_t>(position + i));
      m.values.push_back(static_cast<std::uint16_t>(tokens[i]));
      m.values.push_back(0);
      m.values.push_back(0);
      ++mtp_pairs;
    }
  }
};

qwen35::PromptCacheModel Model(Toy& toy, int keep_last, int min_prefill) {
  qwen35::PromptCacheModel m;
  m.hidden = 1;
  m.vocab = kVocab;
  m.layers = 2;
  m.keep_last = keep_last;
  m.min_prefill = min_prefill;
  m.run = [&toy](const int* t, int n, DecodeState& s, float* h,
                 const qwen35::StateCapture* c) { toy.Run(t, n, s, h, c); };
  m.fill_mtp = [&toy](const float* h, const int* t, int n, int p,
                      DecodeState& mtp) { toy.FillMtp(h, t, n, p, mtp); };
  m.new_state = NewState;
  return m;
}

struct Outcome {
  int ran = 0;
  DecodeState state;
  DecodeState mtp;
  std::vector<float> hidden;
};

Outcome RunHook(qwen35::PromptCache& cache, const std::vector<int>& prompt,
                bool draft) {
  Outcome o;
  o.state = NewState();
  o.mtp = NewState();
  o.hidden.assign(prompt.size(), 0.0f);
  o.ran =
      cache.Hook()(prompt, o.state, draft ? &o.mtp : nullptr, o.hidden.data());
  return o;
}

bool SameLayer(const DecodeState::Layer& a, const DecodeState::Layer& b) {
  return a.conv == b.conv && a.ssm == b.ssm && a.keys == b.keys &&
         a.values == b.values;
}

// The cached run against the uncached one: what the generation reads
// afterwards (state, MTP cache, hidden rows from the first one the cache
// ran or restored).
void Compare(const Outcome& got, const Outcome& want, int from,
             const std::string& what) {
  Check(got.ran == want.ran, what + ": tokens run");
  Check(got.state.position == want.state.position, what + ": position");
  for (std::size_t il = 0; il < 2; ++il) {
    Check(SameLayer(got.state.layers[il], want.state.layers[il]),
          what + ": layer " + std::to_string(il));
  }
  Check(SameLayer(got.mtp.layers[2], want.mtp.layers[2]), what + ": MTP");
  for (int i = from; i < want.ran; ++i) {
    Check(got.hidden[static_cast<std::size_t>(i)] ==
              want.hidden[static_cast<std::size_t>(i)],
          what + ": hidden row " + std::to_string(i));
  }
}

// A chat-like prompt: system and user turns, then the generation prompt.
std::vector<int> Turn(const std::vector<int>& history, int user_len, int seed) {
  std::vector<int> p = history;
  p.push_back(kImStart);
  for (int i = 0; i < user_len; ++i) {
    p.push_back(10 + (seed * 13 + i * 7) % 80);
  }
  p.push_back(kImStart);  // the generation prompt
  p.push_back(3);
  p.push_back(4);
  return p;
}

// The next turn's history: the prompt up to its generation prompt, then the
// reply as the template re-renders it (without the generation prompt's
// think tokens).
std::vector<int> WithReply(const std::vector<int>& prompt, int reply_len,
                           int seed) {
  std::vector<int> h(prompt.begin(), prompt.end() - 3);
  h.push_back(kImStart);
  for (int i = 0; i < reply_len; ++i) {
    h.push_back(10 + (seed * 17 + i * 3) % 80);
  }
  h.push_back(5);
  return h;
}

void TestConversation(int keep_last, bool draft) {
  const std::string tag = "keep_last " + std::to_string(keep_last) +
                          (draft ? " draft" : " no draft");
  Toy toy;
  qwen35::PromptCacheOptions opt;
  opt.boundary_token = kImStart;
  opt.min_tokens = 4;
  qwen35::PromptCache cache(Model(toy, keep_last, 3), opt);
  Toy ref_toy;
  std::vector<int> history = {kImStart, 1, 2, 3, 4, 5, 6, 8, 9};
  long uncached_tokens = 0;
  for (int turn = 0; turn < 4; ++turn) {
    const auto prompt = Turn(history, 20 + turn, turn);
    qwen35::PromptCache fresh(Model(ref_toy, keep_last, 3), opt);
    const long before = ref_toy.tokens_run;
    const auto want = RunHook(fresh, prompt, draft);
    uncached_tokens += ref_toy.tokens_run - before;
    const auto got = RunHook(cache, prompt, draft);
    const int reused = cache.last_reused();
    Check(turn == 0 ? reused == 0 : reused > 0,
          tag + ": turn " + std::to_string(turn) + " reuse " +
              std::to_string(reused));
    Compare(got, want, reused == 0 ? 0 : reused - 1,
            tag + ": turn " + std::to_string(turn));
    history = WithReply(prompt, 30, turn);
  }
  Check(toy.tokens_run < uncached_tokens / 2,
        tag + ": cached runs " + std::to_string(toy.tokens_run) +
            " tokens of " + std::to_string(uncached_tokens));
  Check(cache.stats().hits == 3 && cache.stats().misses == 1,
        tag + ": hits/misses");
  Check(cache.size() == 2, tag + ": two checkpoints kept");
}

void TestShortPromptLeftToDecoder() {
  Toy toy;
  qwen35::PromptCacheOptions opt;
  opt.boundary_token = kImStart;
  qwen35::PromptCache cache(Model(toy, 0, 7), opt);
  const auto o = RunHook(cache, {1, 2, 3}, true);
  Check(o.ran == 0 && toy.tokens_run == 0 && cache.size() == 0,
        "short prompt left to the decoder");
}

void TestRepeatAndDedupe() {
  Toy toy;
  qwen35::PromptCacheOptions opt;
  opt.boundary_token = kImStart;
  opt.min_tokens = 4;
  qwen35::PromptCache cache(Model(toy, 0, 3), opt);
  const auto prompt = Turn({kImStart, 1, 2, 3, 4, 5}, 12, 1);
  const auto first = RunHook(cache, prompt, true);
  const long ran = toy.tokens_run;
  const auto again = RunHook(cache, prompt, true);
  // Only the generation prompt runs again; no second checkpoint.
  Check(toy.tokens_run - ran == 3, "repeat runs the generation prompt only");
  Check(cache.size() == 1, "repeat adds no checkpoint");
  Compare(again, first, cache.last_reused() - 1, "repeat");
}

void TestEviction() {
  Toy toy;
  qwen35::PromptCacheOptions opt;
  opt.boundary_token = kImStart;
  opt.min_tokens = 4;
  opt.max_checkpoints = 2;
  qwen35::PromptCache cache(Model(toy, 0, 3), opt);
  // Three unrelated conversations: the first's checkpoint is evicted.
  std::vector<std::vector<int>> prompts;
  for (int c = 0; c < 3; ++c) {
    prompts.push_back(Turn({kImStart, 20 + c, 21 + c, 22 + c, 23 + c}, 10, c));
    RunHook(cache, prompts.back(), true);
  }
  Check(cache.size() == 2 && cache.stats().evictions == 1, "LRU eviction");
  RunHook(cache, prompts[0], true);
  Check(cache.last_reused() == 0, "evicted conversation misses");
  RunHook(cache, prompts[2], true);
  Check(cache.last_reused() > 0, "recent conversation hits");

  // A byte budget below two checkpoints keeps one.
  Toy toy2;
  opt.max_checkpoints = 8;
  qwen35::PromptCache probe(Model(toy2, 0, 3), opt);
  RunHook(probe, prompts[0], true);
  opt.max_bytes = probe.bytes() + probe.bytes() / 2;
  qwen35::PromptCache budget(Model(toy2, 0, 3), opt);
  for (const auto& p : prompts) {
    RunHook(budget, p, true);
  }
  Check(budget.size() == 1 && budget.bytes() <= opt.max_bytes, "byte budget");
}

void TestDraftNeedsMtp() {
  Toy toy;
  qwen35::PromptCacheOptions opt;
  opt.boundary_token = kImStart;
  opt.min_tokens = 4;
  qwen35::PromptCache cache(Model(toy, 0, 3), opt);
  const auto prompt = Turn({kImStart, 1, 2, 3, 4, 5}, 12, 2);
  RunHook(cache, prompt, false);  // a checkpoint without the MTP cache
  RunHook(cache, prompt, true);
  Check(cache.last_reused() == 0, "a draft request skips MTP-less checkpoints");
  RunHook(cache, prompt, false);
  Check(cache.last_reused() > 0, "a plain request takes any checkpoint");
}

}  // namespace

int main() {
  for (const int keep_last : {0, 1}) {
    for (const bool draft : {true, false}) {
      TestConversation(keep_last, draft);
    }
  }
  TestShortPromptLeftToDecoder();
  TestRepeatAndDedupe();
  TestEviction();
  TestDraftNeedsMtp();
  if (failures > 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::puts("ok: prompt cache");
  return 0;
}
