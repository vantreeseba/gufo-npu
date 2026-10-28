#include "serve/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "build_info.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen35/npu_prefill.hpp"
#include "models/qwen35/prompt_cache.hpp"
#include "models/qwen35/reference.hpp"
#include "models/qwen35_4b/npu_decoder.hpp"
#include "models/qwen35_4b/npu_prefill.hpp"
#include "models/qwen36_35b_a3b/npu_prefill.hpp"
#include "serve/model_registry.hpp"

namespace gufo_npu::serve {

namespace {

namespace fs = std::filesystem;

// --kernels, then $GUFO_NPU_KERNEL_DIR, then <prefix>/share/..., then the
// build tree this binary came from (as the CLI looks them up).
fs::path KernelDir(const std::optional<std::string>& option) {
  if (option) {
    return *option;
  }
  if (const char* env = std::getenv("GUFO_NPU_KERNEL_DIR")) {
    return env;
  }
  std::error_code error;
  const auto exe = fs::read_symlink("/proc/self/exe", error);
  if (!error) {
    const auto installed =
        exe.parent_path().parent_path() / build_info::kInstallKernelSubdir;
    if (fs::is_directory(installed)) {
      return installed;
    }
  }
  return build_info::kBuildKernelDir;
}

// One generation on a model package: its SpeculativeGenerate (or the CPU
// reference's generate loop) with that package's prompt processing, through
// the prompt cache when there is one. `on_token` returning false ends it.
using RunFn = std::function<void(
    const std::vector<int>& prompt, int max_tokens,
    const std::vector<int>& stop, bool draft, const SamplingParams& sampling,
    const std::function<bool(int)>& on_token, GenerateResult& result)>;

void CopyStats(const qwen35::SpeculativeStats& stats, GenerateResult& result) {
  result.prefill_seconds = stats.prefill_seconds;
  result.decode_seconds = stats.decode_seconds;
  result.steps = stats.steps;
  result.accepted = stats.accepted;
}

class ModelEngine final : public Engine {
public:
  ModelEngine(std::unique_ptr<gguf::File> file, ModelInfo info)
      : file_(std::move(file)), tokenizer_(*file_), info_(std::move(info)) {}

  // The package's runner and whatever it owns; set once after construction.
  // `can_draft`: the package runs MTP drafts for this GGUF.
  template<typename Owned>
  void Attach(std::shared_ptr<Owned> owned, RunFn run, bool can_draft) {
    owned_ = std::move(owned);
    run_ = std::move(run);
    info_.can_draft = can_draft;
  }

  const gguf::File& file() const { return *file_; }
  const ModelInfo& info() const override { return info_; }
  const Tokenizer& tokenizer() const override { return tokenizer_; }

  GenerateResult Generate(const GenerateRequest& request,
                          const std::function<bool(int)>& on_token) override {
    const std::lock_guard lock(mutex_);
    Validate(request);
    const int prompt_size = static_cast<int>(request.prompt.size());
    // Generation stops at the context limit (finish kLength).
    const int max_tokens =
        std::min(request.max_tokens, info_.context_length - prompt_size);
    GenerateResult result;
    result.prompt_tokens = prompt_size;
    bool stopped = false;
    const std::function<bool(int)> emit = [&](int id) {
      result.tokens.push_back(id);
      const bool go = on_token ? on_token(id) : true;
      const bool cancelled = request.cancel != nullptr &&
                             request.cancel->load(std::memory_order_relaxed);
      if (!go || cancelled) {
        stopped = true;
        return false;
      }
      return true;
    };
    run_(request.prompt, max_tokens, request.stop_ids,
         request.draft && info_.can_draft, request.sampling, emit, result);
    const bool at_stop =
        !result.tokens.empty() &&
        std::find(request.stop_ids.begin(), request.stop_ids.end(),
                  result.tokens.back()) != request.stop_ids.end();
    if (at_stop) {
      result.finish = FinishReason::kStop;
    } else if (stopped) {
      result.finish = FinishReason::kCancelled;
    } else {
      result.finish = FinishReason::kLength;
    }
    return result;
  }

private:
  void Validate(const GenerateRequest& request) const {
    const auto& prompt = request.prompt;
    if (prompt.empty()) {
      throw std::invalid_argument("empty prompt");
    }
    for (const int id : prompt) {
      if (id < 0 || id >= info_.vocab) {
        throw std::invalid_argument("prompt token id out of range");
      }
    }
    if (request.max_tokens < 1) {
      throw std::invalid_argument("max_tokens must be at least 1");
    }
    if (static_cast<int>(prompt.size()) >= info_.context_length) {
      throw std::invalid_argument(
          "prompt of " + std::to_string(prompt.size()) +
          " tokens leaves no room in the context length of " +
          std::to_string(info_.context_length));
    }
    const SamplingParams& s = request.sampling;
    if (s.temperature < 0.0f || s.top_p <= 0.0f || s.top_p > 1.0f ||
        s.top_k < 0 || s.min_p < 0.0f || s.min_p > 1.0f ||
        s.repetition_penalty <= 0.0f) {
      throw std::invalid_argument("sampling parameter out of range");
    }
  }

  std::unique_ptr<gguf::File> file_;
  Tokenizer tokenizer_;
  ModelInfo info_;
  std::shared_ptr<void> owned_;
  RunFn run_;
  std::mutex mutex_;
};

// The prompt cache's options for a loaded model: checkpoints before the
// chat template's last `<|im_start|>` (prompt_cache.hpp).
qwen35::PromptCacheOptions CacheOptions(const ModelEngine& engine,
                                        int checkpoints) {
  qwen35::PromptCacheOptions options;
  options.max_checkpoints = checkpoints;
  options.boundary_token = engine.FindToken("<|im_start|>");
  return options;
}

// A package's decoder and prefill, and the prompt cache over them (none with
// `checkpoints` 0). The cache's checkpoints are taken inside the prompt
// hook, before any token is reported, so a request cancelled or failing
// later leaves only checkpoints of fully processed prompt prefixes.
template<typename Decoder, typename Prefill>
struct Package {
  Decoder decoder;
  Prefill prefill;
  std::optional<qwen35::PromptCache> cache;

  template<typename MakeCacheModel>
  Package(const ModelEngine& engine, const fs::path& kernels, int checkpoints,
          const MakeCacheModel& make_cache_model)
      : decoder(engine.file(), kernels),
        prefill(MakePrefill(engine.file(), kernels)) {
    if (checkpoints > 0) {
      cache.emplace(make_cache_model(decoder, prefill),
                    CacheOptions(engine, checkpoints));
    }
  }

private:
  Prefill MakePrefill(const gguf::File& file, const fs::path& kernels) {
    if constexpr (std::is_constructible_v<Prefill, const gguf::File&, Decoder&,
                                          const fs::path&>) {
      return Prefill(file, decoder, kernels);
    } else {
      return Prefill(file, kernels);
    }
  }
};

// Runs one generation on a package with `generate` (its
// SpeculativeGenerate) and `prefill_hook` (its own PrefillPrompt), through
// the cache when the package has one.
template<typename P, typename GenerateFn, typename HookFn>
RunFn MakeRun(P* package, GenerateFn generate, HookFn prefill_hook) {
  return [package, generate, prefill_hook](
             const std::vector<int>& prompt, int max_tokens,
             const std::vector<int>& stop, bool draft,
             const SamplingParams& sampling,
             const std::function<bool(int)>& on_token, GenerateResult& result) {
    qwen35::SpeculativeStats stats;
    const qwen35::PromptPrefill hook =
        package->cache ? package->cache->Hook() : prefill_hook(*package);
    generate(package->decoder, prompt, max_tokens, stop, draft, on_token,
             &stats, hook, sampling);
    result.cached_prompt_tokens =
        package->cache ? package->cache->last_reused() : 0;
    CopyStats(stats, result);
  };
}

// Qwen3.5-2B: the family decoder and the family prefill.
void AttachQwen35_2B(ModelEngine& engine, const fs::path& kernels,
                     int checkpoints) {
  using P = Package<qwen35::NpuDecoder, qwen35::NpuPrefill>;
  auto m =
      std::make_shared<P>(engine, kernels, checkpoints,
                          [](qwen35::NpuDecoder& d, qwen35::NpuPrefill& p) {
                            return qwen35::MakePromptCacheModel(d, p);
                          });
  const bool can_draft = m->decoder.can_draft();
  RunFn run = MakeRun(
      m.get(),
      [](auto&&... args) {
        return qwen35::SpeculativeGenerate(
            std::forward<decltype(args)>(args)...);
      },
      [](P& p) { return qwen35::PrefillPrompt(p.prefill); });
  engine.Attach(std::move(m), std::move(run), can_draft);
}

// Qwen3.5-4B: its own decoder and prefill (src/models/qwen35_4b).
void AttachQwen35_4B(ModelEngine& engine, const fs::path& kernels,
                     int checkpoints) {
  using P = Package<qwen35_4b::NpuDecoder, qwen35_4b::NpuPrefill>;
  auto m = std::make_shared<P>(
      engine, kernels, checkpoints,
      [](qwen35_4b::NpuDecoder& d, qwen35_4b::NpuPrefill& p) {
        return qwen35_4b::MakePromptCacheModel(d, p);
      });
  const bool can_draft = m->decoder.config().mtp_layers > 0;
  RunFn run = MakeRun(
      m.get(),
      [](auto&&... args) {
        return qwen35_4b::SpeculativeGenerate(
            std::forward<decltype(args)>(args)...);
      },
      [](P& p) { return qwen35_4b::PrefillPrompt(p.prefill); });
  engine.Attach(std::move(m), std::move(run), can_draft);
}

// Qwen3.6-35B-A3B: the family decoder (MoE) and its own prefill, which
// reads the decoder's weights in place.
void AttachQwen36_35B_A3B(ModelEngine& engine, const fs::path& kernels,
                          int checkpoints) {
  using P = Package<qwen35::NpuDecoder, qwen36_35b_a3b::NpuPrefill>;
  auto m = std::make_shared<P>(
      engine, kernels, checkpoints,
      [](qwen35::NpuDecoder& d, qwen36_35b_a3b::NpuPrefill& p) {
        return qwen36_35b_a3b::MakePromptCacheModel(d, p);
      });
  const bool can_draft = m->decoder.can_draft();
  RunFn run = MakeRun(
      m.get(),
      [](auto&&... args) {
        return qwen35::SpeculativeGenerate(
            std::forward<decltype(args)>(args)...);
      },
      [](P& p) { return qwen36_35b_a3b::PrefillPrompt(p.prefill); });
  engine.Attach(std::move(m), std::move(run), can_draft);
}

// The CPU reference: no draft, no prompt cache. Slow; for tests.
void AttachReference(ModelEngine& engine) {
  auto m = std::make_shared<qwen35::Reference>(engine.file());
  qwen35::Reference* reference = m.get();
  engine.Attach(
      m,
      [reference](
          const std::vector<int>& prompt, int max_tokens,
          const std::vector<int>& stop, bool, const SamplingParams& sampling,
          const std::function<bool(int)>& on_token, GenerateResult& result) {
        auto state = reference->NewState();
        std::vector<float> logits;
        const auto vocab = static_cast<std::size_t>(reference->config().vocab);
        const qwen35::ForwardFn forward = [&](const std::vector<int>& step) {
          logits = reference->Forward(step, state).logits;
          return logits.data() + (step.size() - 1) * vocab;
        };
        qwen35::GenerateStats stats;
        qwen35::GreedyGenerate(forward, reference->config().vocab, prompt,
                               max_tokens, stop, on_token, &stats, sampling);
        result.prefill_seconds = stats.prefill_seconds;
        result.decode_seconds = stats.decode_seconds;
        result.steps = stats.generated_tokens;
      },
      false);
}

std::string ModelIdFor(const gguf::File& file, const fs::path& path) {
  if (file.Find("general.name") != nullptr) {
    const auto id = NormalizeModelId(file.GetString("general.name"));
    if (!id.empty()) {
      return id;
    }
  }
  return NormalizeModelId(path.stem().string());
}

int ContextLengthFor(const gguf::File& file, const qwen35::Config& config,
                     int requested) {
  const std::string key = config.architecture + ".context_length";
  const int trained = file.Find(key) != nullptr
                          ? static_cast<int>(file.GetUInt(key))
                          : kDefaultContextLength;
  if (requested < 0) {
    throw std::invalid_argument("context_length must not be negative");
  }
  if (requested > trained) {
    throw std::invalid_argument(
        "context_length " + std::to_string(requested) +
        " exceeds the model's trained context length of " +
        std::to_string(trained));
  }
  return requested > 0 ? requested : std::min(kDefaultContextLength, trained);
}

}  // namespace

std::unique_ptr<Engine> LoadEngine(const EngineOptions& options) {
  if (options.device != "npu" && options.device != "cpu") {
    throw std::invalid_argument("device must be npu or cpu, not " +
                                options.device);
  }
  if (options.prompt_cache < 0) {
    throw std::invalid_argument("prompt_cache must not be negative");
  }
  auto file = gguf::File::Open(options.model_path);
  const auto config = qwen35::ReadConfig(*file);
  const auto package = NpuPackageFor(config);
  if (options.device == "npu" && !package) {
    throw std::invalid_argument(
        "no NPU model package for this GGUF (" + config.architecture +
        ", hidden " + std::to_string(config.hidden) +
        "); gufo-npu runs Qwen3.5-2B, Qwen3.5-4B and Qwen3.6-35B-A3B");
  }
  ModelInfo info;
  info.id = options.model_id.empty() ? ModelIdFor(*file, options.model_path)
                                     : options.model_id;
  info.architecture = config.architecture;
  info.context_length = ContextLengthFor(*file, config, options.context_length);
  info.vocab = config.vocab;
  auto engine = std::make_unique<ModelEngine>(std::move(file), info);
  if (options.device == "cpu") {
    AttachReference(*engine);
    return engine;
  }
  const auto kernels = KernelDir(options.kernels);
  switch (*package) {
    case ModelPackage::kQwen35_2B:
      AttachQwen35_2B(*engine, kernels, options.prompt_cache);
      break;
    case ModelPackage::kQwen35_4B:
      AttachQwen35_4B(*engine, kernels, options.prompt_cache);
      break;
    case ModelPackage::kQwen36_35B_A3B:
      AttachQwen36_35B_A3B(*engine, kernels, options.prompt_cache);
      break;
  }
  return engine;
}

}  // namespace gufo_npu::serve
