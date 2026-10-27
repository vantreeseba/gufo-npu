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
#include "models/qwen35/q4nx_projections.hpp"
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
    "  generate --model GGUF --prompt TEXT [--system TEXT] [--max-tokens N]\n"
    "           [--think] [--device cpu|npu] [--kernels DIR]\n"
    "                           greedy chat completion; cpu is the reference,\n"
    "                           npu runs the matrix products on the NPU\n"
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

// Relative L2 error of the NPU output over the first `rows` values.
double RelativeError(const std::vector<float>& y,
                     const std::vector<double>& ref, std::size_t rows) {
  double err2 = 0;
  double ref2 = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    const double e = y[r] - ref[r];
    err2 += e * e;
    ref2 += ref[r] * ref[r];
  }
  return std::sqrt(err2 / ref2);
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

  struct Case {
    const char* label;
    std::vector<const char*> tensors;  // stacked in order
  };
  // One stack per decode stage of Qwen3.5-2B (layer 0 DeltaNet, layer 3
  // full attention).
  const Case cases[] = {
      {"DeltaNet in (qkv+z)",
       {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"}},
      {"attention in (q+k+v)",
       {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"}},
      {"mixer out (ssm_out)", {"blk.0.ssm_out.weight"}},
      {"ffn in (gate+up)", {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"}},
      {"ffn down", {"blk.0.ffn_down.weight"}},
      {"LM head", {"token_embd.weight"}},
  };
  std::printf("%-22s %6s %7s %8s %9s %9s %7s %9s %9s\n", "stage", "dim", "rows",
              "MB", "mean us", "best us", "GB/s", "err npu", "err total");
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (const auto& c : cases) {
    std::vector<q4nx::Segment> segments;
    for (const char* name : c.tensors) {
      const auto& t = file->Tensor(name);
      segments.push_back({t.type, t.data, static_cast<std::size_t>(t.Rows())});
    }
    const std::size_t dim = file->Tensor(c.tensors[0]).RowElements();
    const auto packed = q4nx::PackRows(segments, dim);
    const std::size_t rows = packed.rows;

    std::vector<std::uint16_t> x_bits(dim);
    std::vector<float> x(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      x_bits[i] = quant::FloatToBf16(normal(rng));
      x[i] = quant::Bf16ToFloat(x_bits[i]);
    }
    gufo_npu::npu::Q4nxGemv gemv(runtime, packed);
    std::vector<float> y(packed.padded_rows);
    gemv.Run(x_bits.data(), y.data());
    gemv.Time(20);  // warm up
    const auto timing = gemv.Time(iterations);

    const double err_npu = RelativeError(
        y, q4nx::Reference(q4nx::Unpack(packed), packed, x.data()), rows);
    // Against the GGUF weights: kernel error plus requantization.
    std::vector<float> original(rows * dim);
    std::size_t row = 0;
    for (const auto& seg : segments) {
      quant::Dequantize(seg.type, seg.data, &original[row * dim],
                        seg.rows * dim);
      row += seg.rows;
    }
    q4nx::Matrix shape = packed;
    shape.padded_rows = rows;
    const double err_total =
        RelativeError(y, q4nx::Reference(original, shape, x.data()), rows);

    const double mb = static_cast<double>(gemv.weight_bytes()) / 1e6;
    std::printf("%-22s %6zu %7zu %8.1f %9.1f %9.1f %7.1f %9.2e %9.2e\n",
                c.label, dim, rows, mb, timing.mean_us, timing.best_us,
                mb * 1e6 / timing.mean_us / 1e3, err_npu, err_total);
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
    } else if (args[i] == "--kernels" && has_value) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (model.empty() || prompt.empty() || (device != "cpu" && device != "npu")) {
    std::cerr << kUsage;
    return 2;
  }
  namespace qwen35 = gufo_npu::qwen35;
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  const qwen35::Reference reference(*file);
  std::unique_ptr<qwen35::Q4nxProjections> npu;
  if (device == "npu") {
    npu = std::make_unique<qwen35::Q4nxProjections>(
        *file, qwen35::Q4nxProjections::Backend::kNpu, KernelDir(kernels));
  }

  std::vector<qwen35::Message> messages;
  if (!system.empty()) {
    messages.push_back({"system", system});
  }
  messages.push_back({"user", prompt});
  const auto ids = tokenizer.Encode(qwen35::RenderChat(messages, thinking));
  const std::vector<int> stop = {tokenizer.Find(qwen35::kImEnd),
                                 tokenizer.Find(qwen35::kEndOfText)};

  qwen35::GenerateStats stats;
  qwen35::GreedyGenerate(
      reference, ids, max_tokens, stop,
      [&](int id) {
        if (std::find(stop.begin(), stop.end(), id) == stop.end()) {
          std::fputs(tokenizer.TokenBytes(id).c_str(), stdout);
          std::fflush(stdout);
        }
      },
      &stats, npu.get());
  std::fprintf(stderr,
               "\n[%d prompt tokens in %.2f s; %d generated in %.2f s, "
               "%.2f tok/s; %s]\n",
               stats.prompt_tokens, stats.prefill_seconds,
               stats.generated_tokens, stats.decode_seconds,
               stats.decode_seconds > 0
                   ? stats.generated_tokens / stats.decode_seconds
                   : 0.0,
               npu ? "NPU projections" : "CPU reference");
  if (npu) {
    std::fprintf(stderr, "[%ld GEMVs, %.2f s on the NPU including copies]\n",
                 npu->gemv_calls(), npu->gemv_seconds());
  }
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
  std::cerr << "unknown command: " << args[0] << "\n" << kUsage;
  return 2;
}
