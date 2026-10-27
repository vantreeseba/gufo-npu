#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "build_info.hpp"
#include "core/diagnose.hpp"
#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/chat.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/mtp.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen35/reference.hpp"
#include "npu/device.hpp"
#include "npu/gemv.hpp"

namespace {

using gufo_npu::diagnose::Check;
using gufo_npu::diagnose::Status;

constexpr char kUsage[] =
    "usage: gufo-npu <command> [options]\n"
    "\n"
    "commands:\n"
    "  diagnose                 check driver, device, limits and XRT\n"
    "  smoke [--iters N] [--kernels DIR]\n"
    "                           run the smoke kernel and time dispatches\n"
    "  bench dispatch [--rounds N] [--depths 1,8,32] [--kernels DIR]\n"
    "                           per-dispatch cost: sync, async and runlist\n"
    "  bench bandwidth [--iters N] [--kernels DIR]\n"
    "                           NPU DDR bandwidth: copy, and read only\n"
    "  bench gemv --model GGUF [--iters N] [--kernels DIR]\n"
    "                           q4nx GEMV on real Qwen3.5-2B decode shapes\n"
    "  bench mtp --model GGUF --prompt TEXT [--tokens N] [--kernels DIR]\n"
    "                           MTP draft acceptance on a greedy NPU decode\n"
    "  generate --model GGUF --prompt TEXT [--system TEXT] [--max-tokens N]\n"
    "           [--think] [--device cpu|npu] [--draft mtp|none] [--kernels "
    "DIR]\n"
    "                           greedy chat completion; cpu is the reference,\n"
    "                           npu decodes on the NPU, by default with MTP\n"
    "                           speculation (same output, fewer passes)\n"
    "  version                  print the version\n";

void PrintChecks(const std::vector<Check>& checks) {
  for (const auto& check : checks) {
    std::printf(
        "  [%-4s] %-16s %s\n",
        std::string(gufo_npu::diagnose::StatusLabel(check.status)).c_str(),
        check.name.c_str(), check.detail.c_str());
  }
}

int Diagnose() {
  std::printf("gufo-npu %s\n", gufo_npu::build_info::kVersion);
  auto checks = gufo_npu::diagnose::HostChecks();
  const auto runtime = gufo_npu::npu::RuntimeChecks();
  checks.insert(checks.end(), runtime.begin(), runtime.end());
  PrintChecks(checks);
  for (const auto& check : checks) {
    if (check.status == Status::kFail) {
      return 1;
    }
  }
  return 0;
}

std::filesystem::path ExecutableDir() {
  std::error_code error;
  const auto exe = std::filesystem::read_symlink("/proc/self/exe", error);
  return error ? std::filesystem::path{} : exe.parent_path();
}

// --kernels, then $GUFO_NPU_KERNEL_DIR, then <prefix>/share/..., then the
// build tree this binary came from.
std::filesystem::path KernelDir(const std::optional<std::string>& flag) {
  if (flag) {
    return *flag;
  }
  if (const char* env = std::getenv("GUFO_NPU_KERNEL_DIR")) {
    return env;
  }
  const auto installed = ExecutableDir().parent_path() /
                         gufo_npu::build_info::kInstallKernelSubdir;
  if (std::filesystem::is_directory(installed)) {
    return installed;
  }
  return gufo_npu::build_info::kBuildKernelDir;
}

int Smoke(const std::vector<std::string_view>& args) {
  int iterations = 100;
  std::optional<std::string> kernels;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--iters" && i + 1 < args.size()) {
      iterations = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--kernels" && i + 1 < args.size()) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  const auto dir = KernelDir(kernels);
  std::printf("kernels: %s\n", dir.c_str());
  const auto result = gufo_npu::npu::RunSmoke(dir, iterations);
  if (!result.ok) {
    std::fprintf(stderr, "smoke: FAIL: %s\n", result.error.c_str());
    return 1;
  }
  std::printf(
      "smoke: PASS, %d dispatches; first %.1f us, mean %.1f us, min %.1f us\n",
      result.iterations, result.first_us, result.mean_us, result.min_us);
  return 0;
}

std::vector<int> ParseInts(std::string_view text) {
  std::vector<int> values;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const auto comma = text.find(',', pos);
    const auto item = text.substr(pos, comma == std::string_view::npos
                                           ? std::string_view::npos
                                           : comma - pos);
    if (!item.empty()) {
      values.push_back(std::atoi(std::string(item).c_str()));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    pos = comma + 1;
  }
  return values;
}

int BenchDispatch(const std::vector<std::string_view>& args) {
  int rounds = 200;
  std::vector<int> depths = {1, 4, 8, 16, 32, 81};
  std::optional<std::string> kernels;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--rounds" && i + 1 < args.size()) {
      rounds = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--depths" && i + 1 < args.size()) {
      depths = ParseInts(args[++i]);
    } else if (args[i] == "--kernels" && i + 1 < args.size()) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  const auto result =
      gufo_npu::npu::BenchDispatch(KernelDir(kernels), rounds, depths);
  std::printf("%-8s %6s %11s %14s\n", "mode", "depth", "dispatches",
              "us/dispatch");
  for (const auto& t : result.timings) {
    std::printf("%-8s %6d %11d %14.1f\n", t.mode.c_str(), t.depth, t.dispatches,
                t.us_per_dispatch);
  }
  if (!result.ok) {
    std::fprintf(stderr, "bench dispatch: FAIL: %s\n", result.error.c_str());
    return 1;
  }
  return 0;
}

int BenchBandwidth(const std::vector<std::string_view>& args) {
  int iterations = 20;
  std::optional<std::string> kernels;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--iters" && i + 1 < args.size()) {
      iterations = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--kernels" && i + 1 < args.size()) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  const auto result =
      gufo_npu::npu::BenchBandwidth(KernelDir(kernels), iterations);
  if (!result.ok) {
    std::fprintf(stderr, "bench bandwidth: FAIL: %s\n", result.error.c_str());
    return 1;
  }
  std::printf(
      "bandwidth: %.1f MiB per run (read + write), mean %.0f us; "
      "mean %.1f GB/s, best %.1f GB/s\n",
      static_cast<double>(result.bytes_per_run) / (1 << 20), result.mean_us,
      result.mean_gb_s, result.best_gb_s);
  std::printf("read only: %.0f MiB per run; mean %.1f GB/s, best %.1f GB/s\n",
              static_cast<double>(result.read_bytes_per_run) / (1 << 20),
              result.read_mean_gb_s, result.read_best_gb_s);
  return 0;
}

int BenchGemv(const std::vector<std::string_view>& args) {
  int iterations = 200;
  std::optional<std::string> kernels;
  std::string model;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const bool has_value = i + 1 < args.size();
    if (args[i] == "--iters" && has_value) {
      iterations = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--kernels" && has_value) {
      kernels = std::string(args[++i]);
    } else if (args[i] == "--model" && has_value) {
      model = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (model.empty()) {
    std::cerr << kUsage;
    return 2;
  }
  namespace q4nx = gufo_npu::q4nx;
  namespace quant = gufo_npu::quant;
  const auto file = gufo_npu::gguf::File::Open(model);
  gufo_npu::npu::GemvRuntime runtime(KernelDir(kernels));

  using gufo_npu::npu::Epilogue;
  using gufo_npu::npu::Prologue;
  struct Case {
    const char* label;
    std::vector<const char*> tensors;  // stacked in order
    Prologue prologue;
    Epilogue epilogue;
  };
  // One stack per decode call of Qwen3.5-2B (layer 0 DeltaNet, layer 3 full
  // attention), in the input/output stages the decoder runs it with.
  const Case cases[] = {
      {"DeltaNet in (qkv+z)",
       {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"},
       Prologue::kResNorm,
       Epilogue::kStore},
      {"attention in (q+k+v)",
       {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"},
       Prologue::kResNorm,
       Epilogue::kStore},
      {"mixer out (ssm_out)",
       {"blk.0.ssm_out.weight"},
       Prologue::kPlain,
       Epilogue::kStore},
      {"ffn in (gate+up)",
       {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"},
       Prologue::kResNorm,
       Epilogue::kSwiGlu},
      {"ffn down", {"blk.0.ffn_down.weight"}, Prologue::kAct, Epilogue::kStore},
      {"LM head", {"token_embd.weight"}, Prologue::kResNorm, Epilogue::kStore},
  };
  std::printf("%-22s %6s %7s %8s %10s %10s %8s %9s\n", "stage", "dim", "rows",
              "MB", "1 tok us", "2 tok us", "GB/s(1)", "2 vs 1");
  for (const auto& c : cases) {
    std::vector<q4nx::Segment> segments;
    for (const char* name : c.tensors) {
      const auto& t = file->Tensor(name);
      segments.push_back({t.type, t.data, static_cast<std::size_t>(t.Rows())});
    }
    const std::size_t dim = file->Tensor(c.tensors[0]).RowElements();
    const auto packed = q4nx::PackRows(segments, dim);
    namespace npu = gufo_npu::npu;
    npu::Q4nxGemv gemv(runtime, packed, c.prologue, c.epilogue);
    // Unit inputs: timing only (tests/npu/q4nx_gemv_test.cpp checks values).
    npu::Buffer x(runtime, std::max<std::size_t>(npu::kGemvInputBytes,
                                                 4 * dim * sizeof(float)));
    std::fill_n(x.as<float>(), x.size() / sizeof(float), 1.0f);
    x.ToDevice();
    npu::Buffer y(runtime, gemv.y_bytes());
    npu::Buffer r(runtime, npu::kGemvResidualBytes);
    double mean_us[2];
    for (std::size_t tokens = 1; tokens <= 2; ++tokens) {
      gemv.Run(x, y, r, tokens);
      gemv.Time(20);  // warm up
      mean_us[tokens - 1] = gemv.Time(iterations).mean_us;
    }
    const double mb = static_cast<double>(gemv.weight_bytes()) / 1e6;
    std::printf("%-22s %6zu %7zu %8.1f %10.1f %10.1f %8.1f %8.2fx\n", c.label,
                dim, packed.rows, mb, mean_us[0], mean_us[1],
                mb * 1e6 / mean_us[0] / 1e3, mean_us[1] / mean_us[0]);
  }
  return 0;
}

// How often the MTP head's draft for position i + 2 (from the main model's
// hidden state at i and the token at i + 1) equals the main model's greedy
// token there: the speculative-decoding acceptance rate, per formulation.
int BenchMtp(const std::vector<std::string_view>& args) {
  std::optional<std::string> kernels;
  std::string model;
  std::string prompt;
  int tokens = 128;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const bool has_value = i + 1 < args.size();
    if (args[i] == "--model" && has_value) {
      model = std::string(args[++i]);
    } else if (args[i] == "--prompt" && has_value) {
      prompt = std::string(args[++i]);
    } else if (args[i] == "--tokens" && has_value) {
      tokens = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--kernels" && has_value) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (model.empty() || prompt.empty()) {
    std::cerr << kUsage;
    return 2;
  }
  namespace qwen35 = gufo_npu::qwen35;
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  auto seq = tokenizer.Encode(qwen35::RenderChat({{"user", prompt}}, false));
  const std::size_t prompt_len = seq.size();
  const int stop = tokenizer.Find(qwen35::kImEnd);

  // Greedy decode on the NPU, keeping the final residual of every position.
  std::vector<std::vector<float>> hidden;
  qwen35::NpuDecoder npu(*file, KernelDir(kernels));
  {
    auto state = npu.NewState();
    for (std::size_t i = 0; i < seq.size(); ++i) {
      const auto& logits = npu.Step(seq[i], state);
      hidden.emplace_back(npu.hidden(), npu.hidden() + npu.config().hidden);
      if (i + 1 == seq.size() &&
          static_cast<int>(seq.size() - prompt_len) < tokens) {
        const int next = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        seq.push_back(next);
        if (next == stop) {
          break;
        }
      }
    }
  }
  std::printf("%zu prompt tokens, %zu generated\n", prompt_len,
              seq.size() - prompt_len);

  // The NPU MTP head over the same pairs, one and two positions per draft.
  for (const int per_draft : {1, 2}) {
    auto state = npu.NewMtpState();
    std::size_t hits = 0;
    std::size_t total = 0;
    for (std::size_t i = 0; i + 2 < seq.size(); i += per_draft) {
      const int count = static_cast<int>(
          std::min<std::size_t>(per_draft, seq.size() - 2 - i));
      const float* hs[2] = {hidden[i].data(),
                            hidden[std::min(i + 1, seq.size() - 1)].data()};
      const int toks[2] = {seq[i + 1], seq[std::min(i + 2, seq.size() - 1)]};
      const float* logits =
          npu.Draft(hs, toks, count, static_cast<int>(i + 1), state);
      const std::size_t last = i + count - 1;
      const int draft = static_cast<int>(
          std::max_element(logits, logits + npu.config().vocab) - logits);
      if (last + 2 >= prompt_len) {
        hits += draft == seq[last + 2];
        ++total;
      }
    }
    std::printf(
        "NPU MTP, %d position(s) per draft: accepted %zu/%zu (%.1f%%)\n",
        per_draft, hits, total, total ? 100.0 * hits / total : 0.0);
  }
  for (const bool after_norm : {true, false}) {
    for (const bool token_position : {true, false}) {
      qwen35::MtpHead mtp(*file, {after_norm, token_position});
      auto state = mtp.NewState();
      std::size_t hits = 0;
      std::size_t total = 0;
      std::size_t prompt_hits = 0;
      std::size_t prompt_total = 0;
      for (std::size_t i = 0; i + 2 < seq.size(); ++i) {
        const auto& logits = mtp.Step(hidden[i].data(), seq[i + 1],
                                      static_cast<int>(i + 1), state);
        const int draft = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        const bool hit = draft == seq[i + 2];
        if (i + 2 >= prompt_len) {
          hits += hit;
          ++total;
        } else {
          prompt_hits += hit;
          ++prompt_total;
        }
      }
      std::printf(
          "h %s output_norm, RoPE at the %s position: accepted %zu/%zu "
          "generated (%.1f%%), %zu/%zu in the prompt\n",
          after_norm ? "after " : "before",
          token_position ? "token's" : "hidden's", hits, total,
          total ? 100.0 * hits / total : 0.0, prompt_hits, prompt_total);
    }
  }
  return 0;
}

int Generate(const std::vector<std::string_view>& args) {
  std::string model;
  std::string prompt;
  std::string system;
  int max_tokens = 128;
  bool thinking = false;
  std::string device = "cpu";
  std::string draft = "mtp";
  std::optional<std::string> kernels;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const bool has_value = i + 1 < args.size();
    if (args[i] == "--model" && has_value) {
      model = std::string(args[++i]);
    } else if (args[i] == "--prompt" && has_value) {
      prompt = std::string(args[++i]);
    } else if (args[i] == "--system" && has_value) {
      system = std::string(args[++i]);
    } else if (args[i] == "--max-tokens" && has_value) {
      max_tokens = std::atoi(std::string(args[++i]).c_str());
    } else if (args[i] == "--think") {
      thinking = true;
    } else if (args[i] == "--device" && has_value) {
      device = std::string(args[++i]);
    } else if (args[i] == "--draft" && has_value) {
      draft = std::string(args[++i]);
    } else if (args[i] == "--kernels" && has_value) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (model.empty() || prompt.empty() || (device != "cpu" && device != "npu") ||
      (draft != "mtp" && draft != "none")) {
    std::cerr << kUsage;
    return 2;
  }
  namespace qwen35 = gufo_npu::qwen35;
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  std::vector<qwen35::Message> messages;
  if (!system.empty()) {
    messages.push_back({"system", system});
  }
  messages.push_back({"user", prompt});
  const auto ids = tokenizer.Encode(qwen35::RenderChat(messages, thinking));
  const std::vector<int> stop = {tokenizer.Find(qwen35::kImEnd),
                                 tokenizer.Find(qwen35::kEndOfText)};
  const auto print = [&](int id) {
    if (std::find(stop.begin(), stop.end(), id) == stop.end()) {
      std::fputs(tokenizer.TokenBytes(id).c_str(), stdout);
      std::fflush(stdout);
    }
  };

  if (device == "npu") {
    qwen35::NpuDecoder npu(*file, KernelDir(kernels));
    qwen35::SpeculativeStats stats;
    qwen35::SpeculativeGenerate(npu, ids, max_tokens, stop, draft == "mtp",
                                print, &stats);
    std::fprintf(stderr,
                 "\n[%d prompt tokens in %.2f s; %d generated in %.2f s, "
                 "%.2f tok/s; NPU, %s]\n",
                 stats.prompt_tokens, stats.prefill_seconds,
                 stats.generated_tokens, stats.decode_seconds,
                 stats.decode_seconds > 0
                     ? stats.generated_tokens / stats.decode_seconds
                     : 0.0,
                 draft == "mtp" ? "MTP draft" : "no draft");
    std::fprintf(stderr,
                 "[%d decode steps, %d drafts accepted (%.0f%%); %ld NPU "
                 "submissions, %.2f s waiting on them]\n",
                 stats.steps, stats.accepted,
                 stats.steps ? 100.0 * stats.accepted / stats.steps : 0.0,
                 npu.submissions(), npu.npu_seconds());
    return 0;
  }

  const qwen35::Reference reference(*file);
  auto ref_state = reference.NewState();
  std::vector<float> logits;
  const qwen35::ForwardFn forward = [&](const std::vector<int>& step) {
    logits = reference.Forward(step, ref_state).logits;
    return logits.data() + (step.size() - 1) * static_cast<std::size_t>(
                                                   reference.config().vocab);
  };
  qwen35::GenerateStats stats;
  qwen35::GreedyGenerate(forward, reference.config().vocab, ids, max_tokens,
                         stop, print, &stats);
  std::fprintf(stderr,
               "\n[%d prompt tokens in %.2f s; %d generated in %.2f s, "
               "%.2f tok/s; CPU reference]\n",
               stats.prompt_tokens, stats.prefill_seconds,
               stats.generated_tokens, stats.decode_seconds,
               stats.decode_seconds > 0
                   ? stats.generated_tokens / stats.decode_seconds
                   : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  if (args.empty() || args[0] == "help" || args[0] == "--help") {
    std::cout << kUsage;
    return args.empty() ? 2 : 0;
  }
  const std::vector<std::string_view> rest(args.begin() + 1, args.end());
  if (args[0] == "version" || args[0] == "--version") {
    std::printf("gufo-npu %s\n", gufo_npu::build_info::kVersion);
    return 0;
  }
  if (args[0] == "diagnose") {
    return Diagnose();
  }
  if (args[0] == "generate") {
    return Generate(rest);
  }
  if (args[0] == "smoke") {
    return Smoke(rest);
  }
  if (args[0] == "bench" && !rest.empty() && rest[0] == "dispatch") {
    return BenchDispatch({rest.begin() + 1, rest.end()});
  }
  if (args[0] == "bench" && !rest.empty() && rest[0] == "bandwidth") {
    return BenchBandwidth({rest.begin() + 1, rest.end()});
  }
  if (args[0] == "bench" && !rest.empty() && rest[0] == "gemv") {
    return BenchGemv({rest.begin() + 1, rest.end()});
  }
  if (args[0] == "bench" && !rest.empty() && rest[0] == "mtp") {
    return BenchMtp({rest.begin() + 1, rest.end()});
  }
  std::cerr << "unknown command: " << args[0] << "\n" << kUsage;
  return 2;
}
