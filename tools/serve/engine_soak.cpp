// Load once, generate many: reference prompts, then random requests of mixed
// prompt and output lengths through one Engine, then the reference prompts
// again. Writes each request's prompt text, max_tokens and output text (the
// generated tokens' bytes without stop ids, as `gufo-npu generate` prints
// them) to --out-dir, so a script can compare them with fresh processes, and
// prints per-request timing and memory.
//
// With --lock PATH it holds flock(PATH) only while loading and while each
// request runs, so other jobs sharing the NPU can run in between.
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer.hpp"
#include "models/qwen35/chat.hpp"
#include "serve/engine.hpp"

namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
namespace serve = gufo_npu::serve;

constexpr char kUsage[] =
    "usage: gufo_npu_engine_soak --model GGUF [--kernels DIR] [--device "
    "npu|cpu]\n"
    "         [--draft mtp|none] [--requests N] [--seconds S] [--seed N]\n"
    "         [--min-prompt N] [--max-prompt N] [--min-tokens N] "
    "[--max-tokens N]\n"
    "         [--context N] [--out-dir DIR] [--lock PATH]\n";

struct Memory {
  long rss_kb = 0;
  long anon_kb = 0;  // RSS without file pages (the GGUF's mapping)
  long locked_kb = 0;
  long pinned_kb = 0;
};

Memory ReadMemory() {
  Memory m;
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    const auto value = [&](std::string_view key, long& out) {
      if (line.rfind(key, 0) == 0) {
        out = std::atol(line.c_str() + key.size());
      }
    };
    value("VmRSS:", m.rss_kb);
    value("RssAnon:", m.anon_kb);
    value("VmLck:", m.locked_kb);
    value("VmPin:", m.pinned_kb);
  }
  return m;
}

class Lock {
public:
  explicit Lock(const std::string& path) {
    if (!path.empty()) {
      fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0644);
      if (fd_ < 0 || flock(fd_, LOCK_EX) != 0) {
        std::perror("lock");
        std::exit(1);
      }
    }
  }
  ~Lock() {
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
  }
  Lock(const Lock&) = delete;
  Lock& operator=(const Lock&) = delete;

private:
  int fd_ = -1;
};

std::string BackgroundPrompt(int items, int question) {
  // The benchmarks' ~1k prompt (items = 50, question 0).
  static const char* const kQuestions[] = {
      "Now explain in a few sentences why the sky is blue.",
      "Summarize the list above in two sentences.",
      "Which item number is the largest? Answer briefly.",
      "Write a short poem about the fox.",
  };
  std::string text = "Here is some background text.";
  for (int i = 0; i < items; ++i) {
    text += " Item " + std::to_string(i) +
            ": the quick brown fox jumps over the lazy dog number " +
            std::to_string(i) + ".";
  }
  return text + " " + kQuestions[question % 4];
}

std::string RandomPrompt(std::mt19937_64& rng, int min_tokens, int max_tokens,
                         const gufo_npu::Tokenizer& tokenizer) {
  static const char* const kShort[] = {
      "Explain in a few sentences why the sky is blue.",
      "Write a Python function that returns the n-th Fibonacci number, with "
      "a docstring and a short example.",
      "What is the capital of France?",
      "List three primary colors.",
  };
  const int target =
      std::uniform_int_distribution<int>(min_tokens, max_tokens)(rng);
  const int question = std::uniform_int_distribution<int>(0, 3)(rng);
  if (target < 40) {
    return kShort[question];
  }
  // ~19 tokens per item; then trim to the target.
  int items = std::max(1, (target - 20) / 19);
  for (;;) {
    const auto text = BackgroundPrompt(items, question);
    const int n = static_cast<int>(tokenizer.Encode(text).size());
    if (n <= target || items == 1) {
      return text;
    }
    items = std::max(1, items - std::max(1, (n - target) / 19));
  }
}

void WriteFile(const fs::path& path, const std::string& text) {
  std::ofstream(path, std::ios::binary) << text;
}

struct Request {
  std::string name;
  std::string prompt;
  int max_tokens = 0;
};

struct Row {
  double ttft = 0;
  double tok_s = 0;
};

double Mean(const std::vector<double>& v, std::size_t begin, std::size_t end) {
  double sum = 0;
  for (std::size_t i = begin; i < end; ++i) {
    sum += v[i];
  }
  return end > begin ? sum / static_cast<double>(end - begin) : 0;
}

}  // namespace

int main(int argc, char** argv) {
  serve::EngineOptions options;
  std::string draft = "mtp";
  std::string out_dir;
  std::string lock_path;
  int requests = 50;
  double seconds = 0;
  std::uint64_t seed = 1;
  int min_prompt = 10;
  int max_prompt = 4000;
  int min_tokens = 16;
  int max_tokens = 256;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (!has_value) {
      std::fputs(kUsage, stderr);
      return 2;
    }
    const std::string value = argv[++i];
    if (arg == "--model") {
      options.model_path = value;
    } else if (arg == "--kernels") {
      options.kernels = value;
    } else if (arg == "--device") {
      options.device = value;
    } else if (arg == "--context") {
      options.context_length = std::atoi(value.c_str());
    } else if (arg == "--draft") {
      draft = value;
    } else if (arg == "--requests") {
      requests = std::atoi(value.c_str());
    } else if (arg == "--seconds") {
      seconds = std::atof(value.c_str());
    } else if (arg == "--seed") {
      seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (arg == "--min-prompt") {
      min_prompt = std::atoi(value.c_str());
    } else if (arg == "--max-prompt") {
      max_prompt = std::atoi(value.c_str());
    } else if (arg == "--min-tokens") {
      min_tokens = std::atoi(value.c_str());
    } else if (arg == "--max-tokens") {
      max_tokens = std::atoi(value.c_str());
    } else if (arg == "--out-dir") {
      out_dir = value;
    } else if (arg == "--lock") {
      lock_path = value;
    } else {
      std::fputs(kUsage, stderr);
      return 2;
    }
  }
  if (options.model_path.empty() || (draft != "mtp" && draft != "none")) {
    std::fputs(kUsage, stderr);
    return 2;
  }
  if (!out_dir.empty()) {
    fs::create_directories(out_dir);
  }

  const Memory before = ReadMemory();
  std::unique_ptr<serve::Engine> engine;
  double load_seconds = 0;
  {
    const Lock lock(lock_path);
    const auto start = std::chrono::steady_clock::now();
    engine = serve::LoadEngine(options);
    load_seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  }
  const Memory loaded = ReadMemory();
  const auto& info = engine->info();
  std::printf("model %s (%s), context %d, draft %s; loaded in %.1f s; RSS "
              "%.2f GB (+%.2f), anonymous %.2f GB, locked %.2f GB, pinned "
              "%.2f GB\n",
              info.id.c_str(), info.architecture.c_str(), info.context_length,
              info.can_draft ? "available" : "none", load_seconds,
              loaded.rss_kb / 1048576.0,
              (loaded.rss_kb - before.rss_kb) / 1048576.0,
              loaded.anon_kb / 1048576.0, loaded.locked_kb / 1048576.0, loaded.pinned_kb / 1048576.0);
  std::fflush(stdout);

  const auto& tokenizer = engine->tokenizer();
  const std::vector<int> stop = {tokenizer.Find(qwen35::kImEnd),
                                 tokenizer.Find(qwen35::kEndOfText)};
  // The benchmarks' reference prompts (as the hash checks run them).
  const std::vector<Request> references = {
      {"sky", "Explain in a few sentences why the sky is blue.", 200},
      {"fib",
       "Write a Python function that returns the n-th Fibonacci number, "
       "with a docstring and a short example.",
       256},
      {"1k", BackgroundPrompt(50, 0), 128},
  };

  std::vector<double> ttfts;
  std::vector<double> rates;
  std::vector<long> rss;
  std::vector<long> anon;
  int done = 0;
  int failed = 0;
  const auto run = [&](const Request& r) {
    serve::GenerateRequest request;
    request.prompt =
        tokenizer.Encode(qwen35::RenderChat({{"user", r.prompt}}, false));
    request.max_tokens = r.max_tokens;
    request.stop_ids = stop;
    request.draft = draft == "mtp";
    std::string text;
    serve::GenerateResult result;
    try {
      const Lock lock(lock_path);
      result = engine->Generate(request, [&](int id) {
        if (std::find(stop.begin(), stop.end(), id) == stop.end()) {
          text += tokenizer.TokenBytes(id);
        }
        return true;
      });
    } catch (const std::exception& e) {
      std::printf("%-6s FAILED: %s\n", r.name.c_str(), e.what());
      ++failed;
      return std::string{};
    }
    const Memory m = ReadMemory();
    const double rate =
        result.decode_seconds > 0
            ? static_cast<double>(result.tokens.size() - 1) /
                  result.decode_seconds
            : 0;
    ttfts.push_back(result.prefill_seconds);
    rates.push_back(rate);
    rss.push_back(m.rss_kb);
    anon.push_back(m.anon_kb);
    ++done;
    std::printf("%-6s prompt %5d  out %4zu/%-4d %-6s ttft %7.3f s  "
                "%6.2f tok/s  drafts %3d/%-3d  RSS %.2f GB  anon %.2f GB\n",
                r.name.c_str(), result.prompt_tokens, result.tokens.size(),
                r.max_tokens,
                result.finish == serve::FinishReason::kStop     ? "stop"
                : result.finish == serve::FinishReason::kLength ? "length"
                                                                : "cancel",
                result.prefill_seconds, rate, result.accepted, result.steps,
                m.rss_kb / 1048576.0, m.anon_kb / 1048576.0);
    std::fflush(stdout);
    if (!out_dir.empty()) {
      const fs::path base = fs::path(out_dir) / r.name;
      WriteFile(base.string() + ".prompt", r.prompt);
      WriteFile(base.string() + ".max", std::to_string(r.max_tokens));
      WriteFile(base.string() + ".out", text);
    }
    return text;
  };

  std::vector<std::string> first;
  for (const auto& r : references) {
    first.push_back(run(r));
  }
  std::mt19937_64 rng(seed);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0;; ++i) {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    if (seconds > 0 ? elapsed >= seconds : i >= requests) {
      break;
    }
    Request r;
    char name[16];
    std::snprintf(name, sizeof name, "r%03d", i);
    r.name = name;
    r.prompt = RandomPrompt(rng, min_prompt, max_prompt, tokenizer);
    r.max_tokens =
        std::uniform_int_distribution<int>(min_tokens, max_tokens)(rng);
    run(r);
  }
  int mismatches = 0;
  for (std::size_t i = 0; i < references.size(); ++i) {
    Request again = references[i];
    again.name += "2";
    if (run(again) != first[i]) {
      std::printf("MISMATCH: %s differs after the soak\n",
                  references[i].name.c_str());
      ++mismatches;
    }
  }

  const std::size_t n = ttfts.size();
  const std::size_t q = std::max<std::size_t>(1, n / 4);
  std::printf("summary: %d requests, %d failed, %d reference mismatches; "
              "RSS after load %.2f GB, after the first request %.2f GB, at "
              "the end %.2f GB; anonymous %.2f, %.2f, %.2f GB\n",
              done, failed, mismatches, loaded.rss_kb / 1048576.0,
              rss.empty() ? 0 : rss.front() / 1048576.0,
              rss.empty() ? 0 : rss.back() / 1048576.0,
              loaded.anon_kb / 1048576.0,
              anon.empty() ? 0 : anon.front() / 1048576.0,
              anon.empty() ? 0 : anon.back() / 1048576.0);
  std::printf("drift: decode %.2f tok/s in the first quarter, %.2f in the "
              "last; mean TTFT %.3f s, %.3f s\n",
              Mean(rates, 0, q), Mean(rates, n - q, n), Mean(ttfts, 0, q),
              Mean(ttfts, n - q, n));
  return failed == 0 && mismatches == 0 ? 0 : 1;
}
