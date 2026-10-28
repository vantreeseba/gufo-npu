// The Engine on Qwen3.5-2B (NPU): one load, many requests. Repeats give the
// same tokens, a cancelled request leaves the next one unchanged, MTP and
// plain decoding agree, and stop ids, max_tokens and the context limit end
// generations as reported.
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  the UD-Q4_K_XL GGUF
#include "serve/engine.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tokenizer.hpp"
#include "models/qwen35/chat.hpp"

namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
using gufo_npu::serve::Engine;
using gufo_npu::serve::EngineOptions;
using gufo_npu::serve::FinishReason;
using gufo_npu::serve::GenerateRequest;
using gufo_npu::serve::GenerateResult;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) {
    ++failures;
  }
}

std::vector<int> Chat(const Engine& engine, const std::string& text) {
  return engine.tokenizer().Encode(qwen35::RenderChat({{"user", text}}, false));
}

std::vector<int> StopIds(const Engine& engine) {
  return {engine.tokenizer().Find(qwen35::kImEnd),
          engine.tokenizer().Find(qwen35::kEndOfText)};
}

GenerateResult Run(Engine& engine, const std::vector<int>& prompt,
                   int max_tokens, bool draft = true) {
  GenerateRequest request;
  request.prompt = prompt;
  request.max_tokens = max_tokens;
  request.stop_ids = StopIds(engine);
  request.draft = draft;
  return engine.Generate(request, {});
}

std::string LongText(int items) {
  std::string text = "Here is some background text.";
  for (int i = 0; i < items; ++i) {
    text += " Item " + std::to_string(i) +
            ": the quick brown fox jumps over the lazy dog number " +
            std::to_string(i) + ".";
  }
  return text + " Now explain in a few sentences why the sky is blue.";
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_Q4_GGUF");
  if (model == nullptr || argc < 2) {
    std::puts("SKIP: set GUFO_NPU_QWEN35_2B_Q4_GGUF and pass the kernel dir");
    return 77;
  }
  if (!fs::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  EngineOptions options;
  options.model_path = model;
  options.kernels = argv[1];
  auto engine = gufo_npu::serve::LoadEngine(options);
  const auto& info = engine->info();
  Expect(!info.id.empty() && info.architecture == "qwen35" && info.can_draft &&
             info.context_length == 32768 && info.vocab > 0,
         "model info: id " + info.id + ", context " +
             std::to_string(info.context_length));

  const auto sky = Chat(*engine,
                        "Explain in a few sentences why the sky is "
                        "blue.");
  const auto first = Run(*engine, sky, 64);
  Expect(first.tokens.size() == 64 && first.finish == FinishReason::kLength,
         "64 tokens, finish length");

  // A long request, then two cancelled ones, between the repeats.
  const auto longer = Run(*engine, Chat(*engine, LongText(60)), 32);
  Expect(longer.prompt_tokens > 1000 && longer.tokens.size() == 32,
         "a " + std::to_string(longer.prompt_tokens) + "-token prompt");
  {
    GenerateRequest request;
    request.prompt = sky;
    request.max_tokens = 64;
    int seen = 0;
    const auto r = engine->Generate(request, [&](int) { return ++seen < 5; });
    Expect(r.finish == FinishReason::kCancelled && r.tokens.size() == 5,
           "the callback stops a generation after 5 tokens");
  }
  {
    std::atomic<bool> cancel{false};
    GenerateRequest request;
    request.prompt = sky;
    request.max_tokens = 64;
    request.cancel = &cancel;
    int seen = 0;
    const auto r = engine->Generate(request, [&](int) {
      if (++seen == 3) {
        cancel = true;
      }
      return true;
    });
    Expect(r.finish == FinishReason::kCancelled && r.tokens.size() == 3,
           "the cancel flag stops a generation");
  }
  const auto again = Run(*engine, sky, 64);
  Expect(again.tokens == first.tokens, "a repeat gives the same tokens");
  const auto plain = Run(*engine, sky, 64, false);
  Expect(plain.tokens == first.tokens && plain.accepted == 0,
         "decoding without drafts gives the same tokens");
  Expect(first.accepted > 0, "MTP drafts are accepted");

  const auto full = Run(*engine, sky, 1000);
  Expect(full.finish == FinishReason::kStop &&
             full.tokens.back() == StopIds(*engine)[0] &&
             full.tokens.size() < 1000,
         "a reply ends at <|im_end|> (" + std::to_string(full.tokens.size()) +
             " tokens), finish stop");

  // Sampling: a seed makes the draws repeatable.
  const auto sampled = [&](std::uint64_t seed) {
    GenerateRequest request;
    request.prompt = sky;
    request.max_tokens = 32;
    request.stop_ids = StopIds(*engine);
    request.sampling.temperature = 1.0f;
    request.sampling.seed = seed;
    return engine->Generate(request, {}).tokens;
  };
  const auto s1 = sampled(1);
  Expect(!s1.empty() && sampled(1) == s1,
         "the same seed samples the same tokens");
  Expect(sampled(2) != s1 || sampled(3) != s1,
         "other seeds sample other tokens");

  // The prompt cache: turn 2 of a chat reuses turn 1's history and decodes
  // the same tokens as an engine without the cache.
  const std::string question = LongText(4) + " Summarize it in one line.";
  const auto turn1 = Chat(*engine, question);
  const auto reply1 = Run(*engine, turn1, 32);
  std::vector<int> answer(reply1.tokens.begin(), reply1.tokens.end());
  if (!answer.empty() && answer.back() == StopIds(*engine)[0]) {
    answer.pop_back();
  }
  const std::vector<qwen35::Message> history = {
      {"user", question},
      {"assistant", engine->tokenizer().Decode(answer)},
      {"user", "Now in three words."}};
  const auto turn2 =
      engine->tokenizer().Encode(qwen35::RenderChat(history, false));
  const auto cached = Run(*engine, turn2, 32);
  Expect(cached.cached_prompt_tokens > 0 &&
             cached.cached_prompt_tokens < static_cast<int>(turn2.size()),
         "turn 2 reuses " + std::to_string(cached.cached_prompt_tokens) +
             " of " + std::to_string(turn2.size()) + " prompt tokens");

  bool threw = false;
  // A small context limit: generation stops at it, longer prompts are
  // refused.
  options.context_length = static_cast<int>(sky.size()) + 10;
  engine.reset();
  auto small = gufo_npu::serve::LoadEngine(options);
  const auto clipped = Run(*small, sky, 100);
  Expect(clipped.tokens.size() == 10 &&
             clipped.finish == FinishReason::kLength &&
             std::equal(clipped.tokens.begin(), clipped.tokens.end(),
                        first.tokens.begin()),
         "generation stops at the context limit");
  threw = false;
  try {
    Run(*small, Chat(*small, LongText(4)), 8);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "a prompt over the context limit is refused");

  options.context_length = 0;
  options.prompt_cache = 0;
  small.reset();
  auto uncached = gufo_npu::serve::LoadEngine(options);
  const auto fresh = Run(*uncached, turn2, 32);
  Expect(fresh.cached_prompt_tokens == 0 && fresh.tokens == cached.tokens,
         "without the cache turn 2 decodes the same tokens");

  return failures == 0 ? 0 : 1;
}
