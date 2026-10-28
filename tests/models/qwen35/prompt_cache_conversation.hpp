#pragma once

// The prompt cache on a model's NPU decoder and prefill, over chat
// conversations (shared by each model's prompt_cache_test):
//   1. State: for turns 2 and 3, a checkpoint from the previous turn's
//      prompt restored and the rest run through the prefill, against the
//      whole prompt through the model's own PrefillPrompt: the decode
//      state, the MTP layer's cache and the last hidden row, within the
//      model's prefill tolerances (the same prefill runs both; only the
//      split differs). Also an exact repeat of a prompt, whose remainder
//      (the generation prompt) goes through the decoder.
//   2. Output: three turns of greedy MTP decoding with the cache, against
//      the model's own PrefillPrompt: the same tokens every turn.
//   3. Time to the first token per turn, without the cache, on a cache
//      miss and on a hit, warm (a second round of each conversation), with
//      ~1k and ~4k tokens of history; and the memory per checkpoint at ~1k
//      and ~8k tokens.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "core/vec.hpp"
#include "models/qwen35/chat.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/prompt_cache.hpp"

namespace prompt_cache_test {

namespace qwen35 = gufo_npu::qwen35;
using qwen35::DecodeState;

struct Model {
  std::string name;
  const gufo_npu::gguf::File* file = nullptr;
  int layers = 0;
  int hidden = 0;
  // Builds the model's PromptCacheModel (decoder below the prefill
  // threshold, prefill above it).
  std::function<qwen35::PromptCacheModel()> cache_model;
  // The model's own PrefillPrompt hook (no cache).
  qwen35::PromptPrefill plain;
  // The whole prompt through the decoder alone (the same tokens `plain`
  // runs): the decode path's own distance from the prefill, which a short
  // remainder through the decoder inherits.
  qwen35::PromptPrefill decode_path;
  // Greedy MTP decoding with a prompt hook.
  std::function<std::vector<int>(const std::vector<int>& prompt, int max_tokens,
                                 const std::vector<int>& stop,
                                 const qwen35::PromptPrefill& hook,
                                 qwen35::SpeculativeStats* stats)>
      generate;
  std::function<DecodeState()> new_state;
  std::function<DecodeState()> new_mtp_state;
  // Limits for the restored-then-prefilled state against one prefill, and
  // for the repeat (decoder remainder) against it.
  double max_split_error = 1e-3;
  double max_hidden_error = 6e-2;
  double max_state_error = 8e-2;
};

inline double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

struct Error {
  double diff = 0;
  double norm = 0;
  void Add(double got, double want) {
    diff += (got - want) * (got - want);
    norm += want * want;
  }
  void Add(const float* got, const float* want, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      Add(got[i], want[i]);
    }
  }
  void AddHalf(const std::vector<std::uint16_t>& got,
               const std::vector<std::uint16_t>& want) {
    for (std::size_t i = 0; i < want.size(); ++i) {
      Add(gufo_npu::vec::FromHalf(got[i]), gufo_npu::vec::FromHalf(want[i]));
    }
  }
  double Relative() const { return norm > 0 ? std::sqrt(diff / norm) : 0; }
};

// A hook's result: what the generation reads afterwards.
struct HookRun {
  int ran = 0;
  DecodeState state;
  DecodeState mtp;
  std::vector<float> hidden;
};

inline HookRun RunHook(const Model& m, const qwen35::PromptPrefill& hook,
                       const std::vector<int>& prompt) {
  HookRun r;
  r.state = m.new_state();
  r.mtp = m.new_mtp_state();
  r.hidden.assign(prompt.size() * static_cast<std::size_t>(m.hidden), 0.0f);
  r.ran = hook(prompt, r.state, &r.mtp, r.hidden.data());
  return r;
}

struct Diff {
  bool shape = true;
  double hidden = 0, conv = 0, ssm = 0, kv = 0, mtp = 0;
  double Max() const { return std::max({hidden, conv, ssm, kv, mtp}); }
  double MaxState() const { return std::max({conv, ssm, kv, mtp}); }
};

inline Diff Compare(const Model& m, const HookRun& got, const HookRun& want) {
  Diff d;
  if (got.ran != want.ran || got.state.position != want.state.position) {
    d.shape = false;
    return d;
  }
  for (int il = 0; il <= m.layers; ++il) {
    const auto& g = il < m.layers ? got.state.layers[il] : got.mtp.layers[il];
    const auto& w = il < m.layers ? want.state.layers[il] : want.mtp.layers[il];
    if (g.conv.size() != w.conv.size() || g.ssm.size() != w.ssm.size() ||
        g.keys.size() != w.keys.size() || g.values.size() != w.values.size()) {
      d.shape = false;
      return d;
    }
    Error conv, ssm, kv;
    conv.Add(g.conv.data(), w.conv.data(), w.conv.size());
    ssm.Add(g.ssm.data(), w.ssm.data(), w.ssm.size());
    kv.AddHalf(g.keys, w.keys);
    kv.AddHalf(g.values, w.values);
    d.conv = std::max(d.conv, conv.Relative());
    d.ssm = std::max(d.ssm, ssm.Relative());
    if (il < m.layers) {
      d.kv = std::max(d.kv, kv.Relative());
    } else {
      d.mtp = kv.Relative();
    }
  }
  if (want.ran > 0) {
    const auto h = static_cast<std::size_t>(m.hidden);
    const auto row = static_cast<std::size_t>(want.ran - 1) * h;
    Error e;
    e.Add(got.hidden.data() + row, want.hidden.data() + row, h);
    d.hidden = e.Relative();
  }
  return d;
}

// A long, varied document of numbered notes, about `tokens` tokens.
inline std::string Document(const gufo_npu::Tokenizer& tokenizer, int tokens) {
  static const char* kColors[] = {"red",    "green", "blue", "amber",
                                  "violet", "grey",  "white"};
  static const char* kThings[] = {"lantern", "river",  "ledger", "orchard",
                                  "engine",  "bridge", "violin", "harbor",
                                  "glacier", "market", "library"};
  static const char* kVerbs[] = {"was repaired", "was moved",    "was counted",
                                 "was painted",  "was measured", "was sold",
                                 "was flooded",  "was rebuilt"};
  static const char* kPlaces[] = {"north of the mill", "near the station",
                                  "behind the school", "on the east ridge",
                                  "at the old quay",   "inside the tower"};
  std::string doc;
  for (int i = 1;; ++i) {
    doc += "Note " + std::to_string(i) + ": the " + kColors[i % 7] + " " +
           kThings[(i * 3) % 11] + " " + kVerbs[(i * 5) % 8] + " " +
           kPlaces[(i * 7) % 6] + " in year " + std::to_string(1800 + i * 7) +
           ".\n";
    if (i % 16 == 0 &&
        static_cast<int>(tokenizer.Encode(doc).size()) >= tokens) {
      return doc;
    }
  }
}

inline std::vector<int> Prompt(const gufo_npu::Tokenizer& tokenizer,
                               const std::vector<qwen35::Message>& messages) {
  return tokenizer.Encode(qwen35::RenderChat(messages, false));
}

inline std::vector<int> WithoutStop(std::vector<int> ids,
                                    const std::vector<int>& stop) {
  if (!ids.empty() &&
      std::find(stop.begin(), stop.end(), ids.back()) != stop.end()) {
    ids.pop_back();
  }
  return ids;
}

struct TurnTimes {
  double plain = 0;  // the model's own PrefillPrompt
  double miss = 0;   // through the cache, nothing to restore
  double hit = 0;    // through the cache, previous turn restored
  int prompt = 0;
  int reused = 0;
};

// One conversation of `turns` turns over a document of ~`doc_tokens`
// tokens. Returns failures; fills per-turn times (the last round's).
inline int Conversation(const Model& m, const gufo_npu::Tokenizer& tokenizer,
                        const std::vector<int>& stop, int doc_tokens,
                        int rounds, std::vector<TurnTimes>* times,
                        std::size_t* checkpoint_bytes, bool check_states) {
  constexpr int kTurns = 3;
  constexpr int kMaxTokens = 24;
  const char* questions[kTurns] = {
      "What happened to the item in note 3? Answer in one short sentence.",
      "And in note 7?", "And in note 11?"};
  int failures = 0;
  const std::string doc = Document(tokenizer, doc_tokens);
  qwen35::PromptCacheOptions options;
  options.boundary_token = tokenizer.Find(qwen35::kImStart);
  qwen35::PromptCache cache(m.cache_model(), options);
  for (int round = 0; round < rounds; ++round) {
    cache.Clear();
    times->assign(kTurns, {});
    std::vector<qwen35::Message> messages = {
        {"system", "You are a careful assistant. Answer briefly."},
        {"user", "Here are my notes:\n" + doc + "\n" + questions[0]}};
    std::vector<int> previous;
    for (int turn = 0; turn < kTurns; ++turn) {
      const auto prompt = Prompt(tokenizer, messages);
      auto& t = (*times)[static_cast<std::size_t>(turn)];
      t.prompt = static_cast<int>(prompt.size());

      qwen35::SpeculativeStats plain_stats;
      const auto want =
          m.generate(prompt, kMaxTokens, stop, m.plain, &plain_stats);
      t.plain = plain_stats.prefill_seconds;

      // A miss: the same prompt into a cache with nothing to restore.
      {
        qwen35::PromptCache fresh(m.cache_model(), options);
        qwen35::SpeculativeStats miss_stats;
        const auto got =
            m.generate(prompt, kMaxTokens, stop, fresh.Hook(), &miss_stats);
        t.miss = miss_stats.prefill_seconds;
        if (got != want) {
          std::printf(
              "FAIL: %s doc %d turn %d: output on a cache miss "
              "differs\n",
              m.name.c_str(), doc_tokens, turn + 1);
          ++failures;
        }
        if (checkpoint_bytes != nullptr && turn == 0 && fresh.size() == 1) {
          *checkpoint_bytes = fresh.bytes();
        }
      }

      qwen35::SpeculativeStats hit_stats;
      const auto got =
          m.generate(prompt, kMaxTokens, stop, cache.Hook(), &hit_stats);
      t.hit = hit_stats.prefill_seconds;
      t.reused = cache.last_reused();
      if (turn > 0 && t.reused == 0) {
        std::printf("FAIL: %s doc %d turn %d: no checkpoint reused\n",
                    m.name.c_str(), doc_tokens, turn + 1);
        ++failures;
      }
      if (got != want) {
        std::size_t i = 0;
        while (i < got.size() && i < want.size() && got[i] == want[i]) {
          ++i;
        }
        std::printf(
            "FAIL: %s doc %d turn %d: output with the cache differs "
            "from token %zu\n",
            m.name.c_str(), doc_tokens, turn + 1, i);
        ++failures;
      }

      if (check_states && round == 0 && turn > 0) {
        // The checkpoint from the previous prompt, restored and continued,
        // against the whole prompt in one prefill.
        qwen35::PromptCache split(m.cache_model(), options);
        RunHook(m, split.Hook(), previous);
        const auto restored = RunHook(m, split.Hook(), prompt);
        const auto whole = RunHook(m, m.plain, prompt);
        const Diff d = Compare(m, restored, whole);
        const bool ok =
            d.shape && d.Max() <= m.max_split_error && split.last_reused() > 0;
        std::printf(
            "%s  %s doc %d turn %d: %d of %d prompt tokens restored; vs one "
            "prefill: hidden %.2e, conv %.2e, ssm %.2e, kv %.2e, MTP kv "
            "%.2e\n",
            ok ? "ok  " : "FAIL", m.name.c_str(), doc_tokens, turn + 1,
            split.last_reused(), static_cast<int>(prompt.size()), d.hidden,
            d.conv, d.ssm, d.kv, d.mtp);
        failures += ok ? 0 : 1;
        // An exact repeat: only the generation prompt runs, through the
        // decoder.
        const auto repeat = RunHook(m, split.Hook(), prompt);
        const Diff r = Compare(m, repeat, whole);
        const bool rok = r.shape && r.hidden <= m.max_hidden_error &&
                         r.MaxState() <= m.max_state_error;
        std::printf(
            "%s  %s doc %d turn %d repeated: %d restored; vs one prefill: "
            "hidden %.2e, conv %.2e, ssm %.2e, kv %.2e, MTP kv %.2e\n",
            rok ? "ok  " : "FAIL", m.name.c_str(), doc_tokens, turn + 1,
            split.last_reused(), r.hidden, r.conv, r.ssm, r.kv, r.mtp);
        failures += rok ? 0 : 1;
        // Short remainders, below the prefill threshold, through the
        // decoder: this prompt's checkpoint, two tokens, a boundary,
        // two more (the decoder copies the DeltaNet state at the boundary
        // for a new checkpoint); then that checkpoint restored and the rest
        // of the prompt through the prefill.
        const int base = split.last_reused();
        const int boundary = tokenizer.Find(qwen35::kImStart);
        std::vector<int> shorter(prompt.begin(), prompt.begin() + base + 2);
        shorter.push_back(boundary);
        shorter.push_back(prompt[static_cast<std::size_t>(base) + 3]);
        shorter.push_back(prompt[static_cast<std::size_t>(base) + 4]);
        qwen35::PromptCache steps(m.cache_model(), options);
        RunHook(m, steps.Hook(), prompt);
        const auto stepped = RunHook(m, steps.Hook(), shorter);
        const auto shorter_whole = RunHook(m, m.plain, shorter);
        const Diff s = Compare(m, stepped, shorter_whole);
        const Diff base_diff =
            Compare(m, RunHook(m, m.decode_path, shorter), shorter_whole);
        std::vector<int> longer = shorter;
        longer.insert(longer.end(), prompt.begin() + base + 2, prompt.end());
        const auto continued = RunHook(m, steps.Hook(), longer);
        const Diff l = Compare(m, continued, RunHook(m, m.plain, longer));
        // The decoder's part may differ from the prefill as much as the
        // decode path does on the same prompt (one row's hidden error is
        // noisier than the prefill test's over every row).
        const double hidden_limit =
            std::max(m.max_hidden_error, 1.5 * base_diff.hidden);
        const bool sok = s.shape && l.shape && base_diff.shape &&
                         s.hidden <= hidden_limit &&
                         s.MaxState() <= m.max_state_error &&
                         l.hidden <= m.max_hidden_error &&
                         l.MaxState() <= m.max_state_error &&
                         steps.last_reused() == base + 2;
        std::printf(
            "%s  %s doc %d turn %d short remainder: decoder state vs one "
            "prefill: hidden %.2e, states %.2e (the whole prompt by the "
            "decode path: hidden %.2e, states %.2e); its checkpoint (%d "
            "tokens) continued: hidden %.2e, states %.2e\n",
            sok ? "ok  " : "FAIL", m.name.c_str(), doc_tokens, turn + 1,
            s.hidden, s.MaxState(), base_diff.hidden, base_diff.MaxState(),
            steps.last_reused(), l.hidden, l.MaxState());
        failures += sok ? 0 : 1;
      }
      previous = prompt;
      messages.push_back(
          {"assistant", tokenizer.Decode(WithoutStop(want, stop))});
      if (turn + 1 < kTurns) {
        messages.push_back({"user", questions[turn + 1]});
      }
    }
  }
  return failures;
}

// The whole test for one model.
inline int Run(const Model& m) {
  const gufo_npu::Tokenizer tokenizer(*m.file);
  const std::vector<int> stop = {tokenizer.Find(qwen35::kImEnd),
                                 tokenizer.Find(qwen35::kEndOfText)};
  int failures = 0;
  std::printf(
      "%s: time to first token (s), warm; plain = the model's own "
      "prefill, miss = through the cache with nothing to restore, "
      "hit = previous turn restored\n",
      m.name.c_str());
  for (const int doc_tokens : {1000, 4000}) {
    std::vector<TurnTimes> times;
    std::size_t bytes = 0;
    failures += Conversation(m, tokenizer, stop, doc_tokens, 2, &times, &bytes,
                             doc_tokens == 1000);
    for (std::size_t turn = 0; turn < times.size(); ++turn) {
      const auto& t = times[turn];
      std::printf(
          "  doc %d turn %zu: %d prompt tokens, %d restored: plain "
          "%.3f, miss %.3f, hit %.3f\n",
          doc_tokens, turn + 1, t.prompt, t.reused, t.plain, t.miss, t.hit);
    }
    std::printf("  doc %d: checkpoint after turn 1 %.1f MiB\n", doc_tokens,
                bytes / 1048576.0);
  }
  // Memory at ~8k tokens: one prompt, one checkpoint.
  {
    qwen35::PromptCacheOptions options;
    options.boundary_token = tokenizer.Find(qwen35::kImStart);
    qwen35::PromptCache cache(m.cache_model(), options);
    const auto prompt =
        Prompt(tokenizer,
               {{"user", "Here are my notes:\n" + Document(tokenizer, 8000) +
                             "\nWhat happened in note 3?"}});
    const auto start = std::chrono::steady_clock::now();
    RunHook(m, cache.Hook(), prompt);
    std::printf(
        "  %zu-token prompt: checkpoint %.1f MiB (prompt run in %.2f "
        "s)\n",
        prompt.size(), cache.bytes() / 1048576.0, Seconds(start));
  }
  if (failures > 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("ok: %s prompt cache\n", m.name.c_str());
  return 0;
}

}  // namespace prompt_cache_test
