#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "models/qwen35/config.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/mtp.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen35/npu_prefill.hpp"
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
    "                           q4nx GEMV on the model's decode shapes\n"
    "  bench mtp --model GGUF --prompt TEXT [--tokens N] [--kernels DIR]\n"
    "                           MTP draft acceptance on a greedy NPU decode\n"
    "  bench prefill --model GGUF [--tokens 35,125,1024] [--runs N]\n"
    "                [--decode-path] [--kernels DIR]\n"
    "                           time to first token on the NPU per prompt\n"
    "                           length (prefill, MTP cache, first token), "
    "warm;\n"
    "                           --decode-path also times the prompt through\n"
    "                           the decoder alone\n"
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

// One MoE layer's FFN as the decoder runs it: the gate + up gather then the
// down gather in one submission, over layer 0's first routed experts and its
// shared expert. One token reads 9 experts; two tokens read their union, 9
// to 17 experts, with a variant per count.
void BenchMoeFfn(const gufo_npu::gguf::File& file,
                 gufo_npu::npu::GemvRuntime& runtime, int iterations) {
  namespace q4nx = gufo_npu::q4nx;
  namespace npu = gufo_npu::npu;
  const auto ffn = npu::kExpertFfn;
  const auto hidden = npu::kGemvHidden;
  const npu::ExpertArena arena{npu::kMaxGatherExperts};
  npu::Buffer arena_buf(runtime, arena.bytes());
  auto rows = [&](const char* name, std::size_t first, std::size_t count) {
    const auto& t = file.Tensor(name);
    return q4nx::Segment{
        t.type,
        t.data + first * gufo_npu::quant::RowBytes(t.type, t.RowElements()),
        count};
  };
  // Arena slots 0..15: routed experts 0..15; slot 16: the shared expert.
  for (std::size_t e = 0; e < arena.experts; ++e) {
    const bool shared = e + 1 == arena.experts;
    const auto up = q4nx::PackRows(
        shared ? std::vector<q4nx::Segment>{rows("blk.0.ffn_gate_shexp.weight",
                                                 0, ffn),
                                            rows("blk.0.ffn_up_shexp.weight", 0,
                                                 ffn)}
               : std::vector<q4nx::Segment>{rows("blk.0.ffn_gate_exps.weight",
                                                 e * ffn, ffn),
                                            rows("blk.0.ffn_up_exps.weight",
                                                 e * ffn, ffn)},
        hidden, q4nx::SwiGluOrder(ffn));
    const auto down = q4nx::PackRows(
        {shared ? rows("blk.0.ffn_down_shexp.weight", 0, hidden)
                : rows("blk.0.ffn_down_exps.weight", e * hidden, hidden)},
        ffn);
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.UpOffset(e),
                up.bytes.data(), up.bytes.size());
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.DownOffset(e),
                down.bytes.data(), down.bytes.size());
  }
  arena_buf.ToDevice();
  npu::Buffer x(runtime, npu::kGemvInputBytes);
  std::fill_n(x.as<float>(), x.size() / sizeof(float), 1.0f);
  x.ToDevice();
  npu::Buffer y(runtime, npu::kGemvResidualBytes);
  npu::Buffer r(runtime, npu::kGemvResidualBytes);
  // Mean us of one submission of the two gathers over `experts` experts.
  auto time = [&](std::size_t experts, std::size_t tokens) {
    npu::Q4nxGather gate_up(runtime, npu::Q4nxGather::Kind::kGateUp, arena,
                            experts, tokens);
    npu::Q4nxGather down(runtime, npu::Q4nxGather::Kind::kDown, arena, experts,
                         tokens);
    npu::Buffer mid(runtime, gate_up.y_bytes());
    std::vector<int> ids(experts);
    for (std::size_t i = 0; i + 1 < experts; ++i) {
      ids[i] = static_cast<int>(i);
    }
    ids.back() = static_cast<int>(arena.experts - 1);
    const std::vector<float> scales(experts * tokens, 0.5f);
    gate_up.Select(ids.data(), scales.data());
    down.Select(ids.data());
    npu::Chain chain(runtime);
    chain.Add(gate_up, x, arena_buf, mid, r);
    chain.Add(down, mid, arena_buf, y, r);
    double total = 0;
    for (int i = -20; i < iterations; ++i) {  // 20 warm-up runs
      const auto start = std::chrono::steady_clock::now();
      chain.Run();
      if (i >= 0) {
        total += std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - start)
                     .count();
      }
    }
    return total / std::max(iterations, 1);
  };
  const double expert_mb = static_cast<double>(npu::ExpertArena::kUpBytes +
                                               npu::ExpertArena::kDownBytes) /
                           1e6;
  const double one = time(npu::kGatherExperts, 1);
  const double two = time(npu::kGatherExperts, 2);
  std::printf("%-22s %6zu %7zu %8.1f %10.1f %10.1f %8.1f %8.2fx\n",
              "MoE FFN (9 experts)", hidden, npu::kGatherExperts * 2 * ffn,
              npu::kGatherExperts * expert_mb, one, two,
              npu::kGatherExperts * expert_mb * 1e6 / one / 1e3, two / one);
  std::printf("2-token MoE FFN by expert count (the pair's union + shared):\n");
  for (std::size_t experts = npu::kGatherExperts;
       experts <= npu::kMaxGatherExperts; experts += 2) {
    const double us = time(experts, 2);
    std::printf("  %2zu experts %5.1f MB %8.1f us %6.1f GB/s\n", experts,
                experts * expert_mb, us, experts * expert_mb * 1e3 / us);
  }
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
  namespace npu = gufo_npu::npu;
  const auto file = gufo_npu::gguf::File::Open(model);
  const bool moe = gufo_npu::qwen35::ReadConfig(*file).IsMoe();
  // The decoder's flow: MoE models need the instruction-buffer flow.
  npu::GemvRuntime runtime(
      KernelDir(kernels),
      moe ? npu::Flow::kInstructionBuffer : npu::Flow::kElf);

  using npu::Epilogue;
  using npu::Prologue;
  struct Case {
    const char* label;
    std::vector<const char*> tensors;  // stacked in order
    Prologue prologue;
    Epilogue epilogue;
  };
  // One stack per decode call (layer 0 DeltaNet, layer 3 full attention), in
  // the input/output stages the decoder runs it with: Qwen3.5-2B, or the
  // dense calls of an MoE model (its experts follow below).
  const std::vector<Case> cases =
      moe ? std::vector<Case>{
                {"DeltaNet in (qkv+z)",
                 {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"},
                 Prologue::kResNorm,
                 Epilogue::kStore},
                {"attention in (q+k+v)",
                 {"blk.3.attn_q.weight", "blk.3.attn_k.weight",
                  "blk.3.attn_v.weight"},
                 Prologue::kResNorm,
                 Epilogue::kStore},
                {"mixer out (ssm_out)",
                 {"blk.0.ssm_out.weight"},
                 Prologue::kPlain,
                 Epilogue::kStore},
                {"LM head",
                 {"output.weight"},
                 Prologue::kResNorm,
                 Epilogue::kStore},
            }
          : std::vector<Case>{
                {"DeltaNet in (qkv+z)",
                 {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"},
                 Prologue::kResNorm,
                 Epilogue::kStore},
                {"attention in (q+k+v)",
                 {"blk.3.attn_q.weight", "blk.3.attn_k.weight",
                  "blk.3.attn_v.weight"},
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
                {"ffn down",
                 {"blk.0.ffn_down.weight"},
                 Prologue::kAct,
                 Epilogue::kStore},
                {"LM head",
                 {"token_embd.weight"},
                 Prologue::kResNorm,
                 Epilogue::kStore},
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
    npu::Q4nxGemv gemv(runtime, packed, c.prologue, c.epilogue);
    // Unit inputs: timing only (tests/models/qwen35_2b/q4nx_gemv_test.cpp
    // checks values).
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
  if (moe) {
    BenchMoeFfn(*file, runtime, iterations);
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

  // The NPU MTP head over the same pairs, one and two positions per draft,
  // scoring the draft vocabulary (the CPU head below scores every token).
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
      const int draft =
          npu.Draft(hs, toks, count, static_cast<int>(i + 1), state);
      const std::size_t last = i + count - 1;
      if (last + 2 >= prompt_len) {
        hits += draft == seq[last + 2];
        ++total;
      }
    }
    std::printf(
        "NPU MTP (draft vocabulary), %d position(s) per draft: accepted "
        "%zu/%zu (%.1f%%)\n",
        per_draft, hits, total, total ? 100.0 * hits / total : 0.0);
  }
  // The CPU head's formulations (dense models with tied embeddings only).
  for (const bool after_norm : {true, false}) {
    if (npu.config().IsMoe()) {
      break;
    }
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

// Time to first token of `generate --device npu` per prompt length: prompt
// processing (NpuPrefill), the MTP cache over the prompt and the first
// token, as SpeculativeGenerate runs them with MTP drafting. The prompts are
// the first N tokens of one long chat prompt; each length runs once to warm
// up, then --runs times.
int BenchPrefill(const std::vector<std::string_view>& args) {
  std::optional<std::string> kernels;
  std::string model;
  std::vector<int> lengths = {35, 125, 1024, 2048, 4096};
  int runs = 3;
  bool decode_path = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const bool has_value = i + 1 < args.size();
    if (args[i] == "--model" && has_value) {
      model = std::string(args[++i]);
    } else if (args[i] == "--decode-path") {
      decode_path = true;
    } else if (args[i] == "--tokens" && has_value) {
      lengths = ParseInts(args[++i]);
    } else if (args[i] == "--runs" && has_value) {
      runs = std::max(1, std::atoi(std::string(args[++i]).c_str()));
    } else if (args[i] == "--kernels" && has_value) {
      kernels = std::string(args[++i]);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (model.empty() || lengths.empty()) {
    std::cerr << kUsage;
    return 2;
  }
  namespace qwen35 = gufo_npu::qwen35;
  const auto file = gufo_npu::gguf::File::Open(model);
  if (qwen35::ReadConfig(*file).IsMoe()) {
    std::fprintf(stderr, "bench prefill: MoE models have no NPU prefill yet\n");
    return 1;
  }
  const gufo_npu::Tokenizer tokenizer(*file);
  const int longest = *std::max_element(lengths.begin(), lengths.end());
  std::string text = "Here is some background text.";
  std::vector<int> ids;
  for (int item = 0; static_cast<int>(ids.size()) < longest; item += 100) {
    for (int i = item; i < item + 100; ++i) {
      text += " Item " + std::to_string(i) +
              ": the quick brown fox jumps over the lazy dog number " +
              std::to_string(i) + ".";
    }
    ids = tokenizer.Encode(qwen35::RenderChat(
        {{"user", text + " Now explain in a few sentences why the sky is "
                         "blue."}},
        false));
  }
  qwen35::NpuDecoder npu(*file, KernelDir(kernels));
  qwen35::NpuPrefill prefill(*file, KernelDir(kernels));
  const auto hook = qwen35::PrefillPrompt(prefill);
  std::printf("%8s %10s %8s %8s %12s %6s %8s %8s %8s %8s %8s\n", "tokens",
              "TTFT mean", "min", "max", "prompt tok/s", "GEMMs", "GEMM s",
              "sync s", "DN s", "attn s", "other s");
  for (const int length : lengths) {
    const std::vector<int> prompt(ids.begin(), ids.begin() + length);
    if (decode_path) {
      // The same prompt through the decoder alone, two tokens per pass.
      std::vector<double> ttft;
      for (int run = 0; run <= runs; ++run) {
        qwen35::SpeculativeStats stats;
        qwen35::SpeculativeGenerate(npu, prompt, 1, {}, true, {}, &stats);
        if (run > 0) {
          ttft.push_back(stats.prefill_seconds);
        }
      }
      double mean = 0;
      for (const double t : ttft) {
        mean += t / runs;
      }
      std::printf("%8d %10.3f %8.3f %8.3f %12.0f  (decode path)\n", length,
                  mean, *std::min_element(ttft.begin(), ttft.end()),
                  *std::max_element(ttft.begin(), ttft.end()), length / mean);
    }
    std::vector<double> ttft;
    qwen35::NpuPrefill::Profile first;
    qwen35::NpuPrefill::Profile last;
    for (int run = 0; run <= runs; ++run) {
      if (run == 1) {
        first = prefill.profile();
      }
      qwen35::SpeculativeStats stats;
      qwen35::SpeculativeGenerate(npu, prompt, 1, {}, true, {}, &stats, hook);
      if (run > 0) {
        ttft.push_back(stats.prefill_seconds);
      }
    }
    last = prefill.profile();
    double mean = 0;
    for (const double t : ttft) {
      mean += t / runs;
    }
    std::printf(
        "%8d %10.3f %8.3f %8.3f %12.0f %6ld %8.3f %8.3f %8.3f %8.3f %8.3f\n",
        length, mean, *std::min_element(ttft.begin(), ttft.end()),
        *std::max_element(ttft.begin(), ttft.end()), length / mean,
        (last.gemms - first.gemms) / runs,
        (last.gemm_seconds - first.gemm_seconds) / runs,
        (last.sync_seconds - first.sync_seconds) / runs,
        (last.deltanet_seconds - first.deltanet_seconds) / runs,
        (last.attention_seconds - first.attention_seconds) / runs,
        (last.other_seconds - first.other_seconds) / runs);
    std::fflush(stdout);
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
    const auto load_start = std::chrono::steady_clock::now();
    qwen35::NpuDecoder npu(*file, KernelDir(kernels));
    // MoE models have no NPU prefill yet: their prompt goes through the
    // decoder, two tokens per pass.
    std::optional<qwen35::NpuPrefill> prefill;
    if (!npu.config().IsMoe()) {
      prefill.emplace(*file, KernelDir(kernels));
    }
    const double load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      load_start)
            .count();
    // A GGUF without the MTP layer decodes without a draft.
    const bool speculate = draft == "mtp" && npu.can_draft();
    qwen35::SpeculativeStats stats;
    qwen35::SpeculativeGenerate(
        npu, ids, max_tokens, stop, speculate, print, &stats,
        prefill ? qwen35::PrefillPrompt(*prefill) : qwen35::PromptPrefill{});
    if (prefill) {
      const auto& pp = prefill->profile();
      std::fprintf(stderr,
                   "\n[prefill: %ld NPU GEMMs, %.3f s in them, %.3f s syncing "
                   "buffers; CPU %.3f s DeltaNet, %.3f s attention, %.3f s "
                   "other]",
                   pp.gemms, pp.gemm_seconds, pp.sync_seconds,
                   pp.deltanet_seconds, pp.attention_seconds, pp.other_seconds);
    }
    std::fprintf(stderr,
                 "\n[%d prompt tokens in %.2f s; %d generated in %.2f s, "
                 "%.2f tok/s; NPU, %s]\n",
                 stats.prompt_tokens, stats.prefill_seconds,
                 stats.generated_tokens, stats.decode_seconds,
                 stats.decode_seconds > 0
                     ? stats.generated_tokens / stats.decode_seconds
                     : 0.0,
                 speculate        ? "MTP draft"
                 : draft == "mtp" ? "no draft (no MTP layer in the GGUF)"
                                  : "no draft");
    std::fprintf(stderr,
                 "[%d decode steps, %d drafts accepted (%.0f%%); %ld NPU "
                 "submissions, %.2f s waiting on them; %.2f s in CPU mixers, "
                 "%.2f s routing; loaded in %.1f s]\n",
                 stats.steps, stats.accepted,
                 stats.steps ? 100.0 * stats.accepted / stats.steps : 0.0,
                 npu.submissions(), npu.npu_seconds(), npu.cpu_seconds(),
                 npu.router_seconds(), load_seconds);
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
  if (args[0] == "bench" && !rest.empty() && rest[0] == "prefill") {
    return BenchPrefill({rest.begin() + 1, rest.end()});
  }
  if (args[0] == "bench" && !rest.empty() && rest[0] == "mtp") {
    return BenchMtp({rest.begin() + 1, rest.end()});
  }
  std::cerr << "unknown command: " << args[0] << "\n" << kUsage;
  return 2;
}
