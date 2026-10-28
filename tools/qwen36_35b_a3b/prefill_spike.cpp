// Qwen3.6-35B-A3B prefill spike (docs/research/NPU-MOE-PREFILL.md), kept to
// re-measure the parts of src/models/qwen36_35b_a3b/npu_prefill.cpp. Measures
// the parts an MoE prefill on the NPU rests on, on random q4nx weights of the
// model's shapes, each checked against the CPU over the same q4nx weights:
//   dense    q4nx -> bfp16 (qwen36_35b_a3b_dequant) then the GEMM
//            (qwen36_35b_a3b_prefill)
//            at M = 256, 512, 1024 for the DeltaNet in, attention in and mixer
//            out shapes; TFLOPS of each and of the pair.
//   switch   the cost of alternating the two hardware contexts.
//   experts  one layer's routed experts for a chunk: dequantize the experts
//            (all in batches, or only those with tokens), group the chunk's
//            tokens by expert (slots padded to 256/512/1024 rows), gate + up
//            GEMM per expert, SwiGLU and router weight on the CPU, down GEMM
//            per expert, scatter-add on the CPU.
// Routing comes from a qwen36_route_dump file (real tokens) or, without one,
// 8 distinct uniform experts per token.
//
//   qwen36_prefill_spike <kernel-dir> [routing.bin]
// (routing.bin from qwen36_route_dump).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/bfp16.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/xrt_kernel.hpp"

namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace bfp16 = gufo_npu::bfp16;
namespace quant = gufo_npu::quant;
using npu::Clock;
using npu::MicrosSince;

constexpr int kLayers = 40;
constexpr int kExperts = 256;
constexpr int kUsed = 8;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kFfn = 512;
constexpr std::size_t kSlots[] = {256, 512, 1024};

std::size_t BfpBytes(std::size_t k, std::size_t n) {
  return k * n / 8 * 9;
}

std::string DequantName(std::size_t k, std::size_t n) {
  return "qwen36_35b_a3b_dequant_k" + std::to_string(k) + "_n" +
         std::to_string(n);
}
std::string GemmName(std::size_t m, std::size_t k, std::size_t n) {
  return "qwen36_35b_a3b_prefill_m" + std::to_string(m) + "_k" +
         std::to_string(k) + "_n" + std::to_string(n);
}

std::vector<float> RandomMatrix(std::mt19937& rng, std::size_t n, float sd) {
  std::normal_distribution<float> dist(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

void Upload(xrt::bo& bo, const void* data, std::size_t bytes,
            std::size_t offset = 0) {
  std::memcpy(bo.map<std::uint8_t*>() + offset, data, bytes);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset);
}

double RelL2(const std::vector<double>& want, const std::vector<double>& got) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    num += (want[i] - got[i]) * (want[i] - got[i]);
    den += want[i] * want[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

struct Timing {
  double mean = 0, best = 1e30;
};

template<typename F>
Timing Time(int iterations, F&& f) {
  for (int i = 0; i < 3; ++i) {
    f();
  }
  Timing t;
  double total = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    f();
    const double us = MicrosSince(start);
    total += us;
    t.best = std::min(t.best, us);
  }
  t.mean = total / iterations;
  return t;
}

void RunWait(xrt::run& run) {
  run.start();
  run.wait();
}

// A batch of runs in one context, executed as runlists of at most kChain.
struct Batch {
  static constexpr std::size_t kChain = 24;
  npu::HwContext& hw;
  std::vector<xrt::run> runs;
  std::vector<xrt::runlist> lists;

  explicit Batch(npu::HwContext& h) : hw(h) {}
  void Add(xrt::run run) { runs.push_back(std::move(run)); }
  void Seal() {
    lists.clear();
    for (std::size_t i = 0; i < runs.size(); i += kChain) {
      lists.emplace_back(*hw.context);
      for (std::size_t j = i; j < std::min(runs.size(), i + kChain); ++j) {
        lists.back().add(runs[j]);
      }
    }
  }
  void Execute() {
    // One runlist in flight at a time: each waits before the next starts.
    for (auto& l : lists) {
      l.execute();
      l.wait();
    }
  }
};

// ---- dense --------------------------------------------------------------

void Dense(const std::filesystem::path& dir, npu::HwContext& dq,
           npu::HwContext& gm, std::mt19937& rng) {
  std::printf("\n== dense: q4nx -> bfp16, then GEMM ==\n");
  std::printf(
      "%-14s %10s %8s %9s | %5s %10s %7s %10s %9s | %s\n", "K x N",
      "dequant us", "GB/s", "w err", "M", "GEMM us", "TFLOPS", "pair us",
      "pair TF", "C err");
  const std::pair<std::size_t, std::size_t> shapes[] = {
      {2048, 12288}, {2048, 9216}, {4096, 2048}};
  for (const auto& [k, n] : shapes) {
    const auto w = RandomMatrix(rng, n * k, 0.02f);
    const auto packed = q4nx::PackFloat(w.data(), n, k, q4nx::Fit::kSearch);
    const auto wq = q4nx::Unpack(packed);
    xrt::bo qw = dq.Buffer(packed.bytes.size());
    Upload(qw, packed.bytes.data(), packed.bytes.size());
    xrt::bo b = dq.Buffer(BfpBytes(k, n));
    npu::Sequence dseq(dq, dir, DequantName(k, n));
    xrt::run drun = dseq.Run({&qw, &b});
    const Timing dt = Time(20, [&] { RunWait(drun); });
    // The weights the GEMM multiplies, against the q4nx weights.
    b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<std::uint8_t> bytes(BfpBytes(k, n));
    std::memcpy(bytes.data(), b.map<void*>(), bytes.size());
    const auto wb = bfp16::UnpackGemmB(bytes, n, k);
    double num = 0, den = 0;
    for (std::size_t i = 0; i < n * k; ++i) {
      num += (wb[i] - wq[i]) * double(wb[i] - wq[i]);
      den += double(wq[i]) * wq[i];
    }
    const double werr = std::sqrt(num / den);
    // The same weights packed on the host (round to nearest), for scale.
    const auto host = bfp16::UnpackGemmB(bfp16::PackGemmB(wq.data(), n, k), n, k);
    double hnum = 0;
    for (std::size_t i = 0; i < n * k; ++i) {
      hnum += (host[i] - wq[i]) * double(host[i] - wq[i]);
    }
    std::printf("%5zu x %-6zu weights vs q4nx: NPU dequant %.2e, host bfp16 pack %.2e\n",
                k, n, werr, std::sqrt(hnum / den));
    const double gbs = (packed.bytes.size() + bytes.size()) / dt.mean / 1e3;

    for (const std::size_t m : kSlots) {
      xrt::bo a = gm.Buffer(m * k * 2);
      xrt::bo c = gm.Buffer(m * n * 2);
      std::vector<std::uint16_t> av(m * k);
      std::normal_distribution<float> x(0.0f, 1.0f);
      for (auto& v : av) {
        v = quant::FloatToBf16(x(rng));
      }
      Upload(a, av.data(), av.size() * 2);
      npu::Sequence gseq(gm, dir, GemmName(m, k, n));
      xrt::run grun = gseq.Run({&a, &b, &c});
      const Timing gt = Time(20, [&] { RunWait(grun); });
      const Timing pt = Time(10, [&] {
        RunWait(drun);
        RunWait(grun);
      });
      // Check 8 rows of C against the CPU over the q4nx weights.
      RunWait(grun);
      c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const auto* cv = c.map<const std::uint16_t*>();
      std::vector<double> want, got;
      for (std::size_t r = 0; r < m; r += m / 8) {
        for (std::size_t j = 0; j < n; ++j) {
          double s = 0;
          for (std::size_t i = 0; i < k; ++i) {
            s += double(quant::Bf16ToFloat(av[r * k + i])) * wq[j * k + i];
          }
          want.push_back(s);
          got.push_back(quant::Bf16ToFloat(cv[r * n + j]));
        }
      }
      const double flop = 2.0 * m * k * n;
      std::printf(
          "%5zu x %-6zu %10.1f %8.1f %9.2e | %5zu %10.1f %7.2f %10.1f %9.2f | "
          "%.2e\n",
          k, n, dt.mean, gbs, werr, m, gt.mean, flop / gt.mean / 1e6,
          pt.mean, flop / pt.mean / 1e6, RelL2(want, got));
    }
  }
}

// ---- context switches ---------------------------------------------------

void Switches(const std::filesystem::path& dir, npu::HwContext& dq,
              npu::HwContext& gm, std::mt19937& rng) {
  std::printf("\n== hardware context switches ==\n");
  const std::size_t k = 512, n = 2048, m = 256;
  const auto w = RandomMatrix(rng, n * k, 0.02f);
  const auto packed = q4nx::PackFloat(w.data(), n, k, q4nx::Fit::kSearch);
  xrt::bo qw = dq.Buffer(packed.bytes.size());
  Upload(qw, packed.bytes.data(), packed.bytes.size());
  xrt::bo b = dq.Buffer(BfpBytes(k, n));
  xrt::bo a = gm.Buffer(m * k * 2);
  xrt::bo c = gm.Buffer(m * n * 2);
  npu::Sequence dseq(dq, dir, DequantName(k, n));
  npu::Sequence gseq(gm, dir, GemmName(m, k, n));
  xrt::run drun = dseq.Run({&qw, &b});
  xrt::run grun = gseq.Run({&a, &b, &c});
  const Timing d2 = Time(50, [&] {
    RunWait(drun);
    RunWait(drun);
  });
  const Timing g2 = Time(50, [&] {
    RunWait(grun);
    RunWait(grun);
  });
  const Timing dg = Time(50, [&] {
    RunWait(drun);
    RunWait(grun);
  });
  // Both started before either is awaited: only overlaps when the two
  // contexts sit on separate columns.
  const Timing both = Time(50, [&] {
    drun.start();
    grun.start();
    drun.wait();
    grun.wait();
  });
  std::printf(
      "dequant+dequant %.1f us, GEMM+GEMM %.1f us, dequant+GEMM %.1f us: "
      "a switch costs ~%.1f us over staying; both in flight %.1f us\n",
      d2.mean, g2.mean, dg.mean, dg.mean - (d2.mean + g2.mean) / 2, both.mean);
}

// ---- experts ------------------------------------------------------------

struct Routing {
  int tokens = 0;
  std::vector<std::int16_t> ids;  // [token][layer][kUsed]
  const std::int16_t* At(int t, int l) const {
    return ids.data() + (static_cast<std::size_t>(t) * kLayers + l) * kUsed;
  }
};

Routing LoadRouting(const char* path, std::mt19937& rng, int tokens) {
  Routing r;
  if (path != nullptr) {
    std::ifstream f(path, std::ios::binary);
    const std::string s((std::istreambuf_iterator<char>(f)), {});
    r.ids.resize(s.size() / 2);
    std::memcpy(r.ids.data(), s.data(), r.ids.size() * 2);
    r.tokens = static_cast<int>(r.ids.size() / (kLayers * kUsed));
    return r;
  }
  r.tokens = tokens;
  r.ids.resize(static_cast<std::size_t>(tokens) * kLayers * kUsed);
  std::vector<int> all(kExperts);
  std::iota(all.begin(), all.end(), 0);
  for (int t = 0; t < tokens; ++t) {
    for (int l = 0; l < kLayers; ++l) {
      std::shuffle(all.begin(), all.end(), rng);
      for (int i = 0; i < kUsed; ++i) {
        r.ids[(static_cast<std::size_t>(t) * kLayers + l) * kUsed + i] =
            static_cast<std::int16_t>(all[i]);
      }
    }
  }
  return r;
}

float Silu(float x) {
  return x / (1.0f + std::exp(-x));
}

struct ExpertLayer {
  static constexpr int kTemplates = 8;
  static constexpr std::size_t kUpRows = 2 * kFfn;
  std::size_t up_q = 0, down_q = 0;  // q4nx bytes per expert
  std::vector<std::vector<float>> up_w, down_w;  // templates, q4nx weights
  xrt::bo up_qw, down_qw, up_b, down_b;
};

void Experts(const std::filesystem::path& dir, npu::HwContext& dq,
             npu::HwContext& gm, std::mt19937& rng, const Routing& routing,
             const char* source) {
  std::printf("\n== routed experts, one layer (%s routing, %d tokens) ==\n",
              source, routing.tokens);
  ExpertLayer L;
  std::vector<q4nx::Matrix> up_t, down_t;
  for (int t = 0; t < ExpertLayer::kTemplates; ++t) {
    const auto wu = RandomMatrix(rng, ExpertLayer::kUpRows * kHidden, 0.02f);
    up_t.push_back(q4nx::PackFloat(wu.data(), ExpertLayer::kUpRows, kHidden, q4nx::Fit::kSearch));
    L.up_w.push_back(q4nx::Unpack(up_t.back()));
    const auto wd = RandomMatrix(rng, kHidden * kFfn, 0.02f);
    down_t.push_back(q4nx::PackFloat(wd.data(), kHidden, kFfn, q4nx::Fit::kSearch));
    L.down_w.push_back(q4nx::Unpack(down_t.back()));
  }
  L.up_q = up_t[0].bytes.size();
  L.down_q = down_t[0].bytes.size();
  L.up_qw = dq.Buffer(L.up_q * kExperts);
  L.down_qw = dq.Buffer(L.down_q * kExperts);
  for (int e = 0; e < kExperts; ++e) {
    const int t = e % ExpertLayer::kTemplates;
    std::memcpy(L.up_qw.map<std::uint8_t*>() + e * L.up_q,
                up_t[t].bytes.data(), L.up_q);
    std::memcpy(L.down_qw.map<std::uint8_t*>() + e * L.down_q,
                down_t[t].bytes.data(), L.down_q);
  }
  L.up_qw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  L.down_qw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const std::size_t up_bfp = BfpBytes(kHidden, ExpertLayer::kUpRows);
  const std::size_t down_bfp = BfpBytes(kFfn, kHidden);
  L.up_b = dq.Buffer(up_bfp * kExperts);
  L.down_b = dq.Buffer(down_bfp * kExperts);

  // Dequant, all experts, one call per matrix (as the prefill converts the
  // experts with tokens).
  npu::Sequence up1(dq, dir, DequantName(kHidden, ExpertLayer::kUpRows));
  npu::Sequence down1(dq, dir, DequantName(kFfn, kHidden));
  std::vector<xrt::bo> subs;  // keep sub-buffers alive
  auto sub = [&](xrt::bo& parent, std::size_t bytes, std::size_t offset) {
    subs.emplace_back(parent, bytes, offset);
    return &subs.back();
  };
  subs.reserve(1 << 16);
  Batch all(dq);
  for (int e = 0; e < kExperts; ++e) {
    all.Add(up1.Run({sub(L.up_qw, L.up_q, e * L.up_q),
                     sub(L.up_b, up_bfp, e * up_bfp)}));
    all.Add(down1.Run({sub(L.down_qw, L.down_q, e * L.down_q),
                       sub(L.down_b, down_bfp, e * down_bfp)}));
  }
  all.Seal();
  const Timing all_t = Time(5, [&] { all.Execute(); });
  const double all_bytes =
      kExperts * double(L.up_q + L.down_q + up_bfp + down_bfp);
  std::printf(
      "dequant all 256 experts (%zu calls): %.1f ms, %.1f GB/s moved\n",
      all.runs.size(), all_t.mean / 1e3, all_bytes / all_t.mean / 1e3);

  // Per chunk size: group, GEMMs, CPU stages.
  const std::vector<int> layers = {0, 13, 26, 39};
  for (const int chunk : {256, 1024}) {
    if (chunk > routing.tokens) {
      continue;
    }
    for (const int layer : layers) {
      // Tokens per expert.
      std::vector<std::vector<std::pair<int, int>>> by_expert(kExperts);
      std::vector<float> rw(static_cast<std::size_t>(chunk) * kUsed);
      std::uniform_real_distribution<float> u(0.05f, 1.0f);
      for (int t = 0; t < chunk; ++t) {
        float s = 0;
        for (int i = 0; i < kUsed; ++i) {
          rw[t * kUsed + i] = u(rng);
          s += rw[t * kUsed + i];
        }
        const std::int16_t* ids = routing.At(t, layer);
        for (int i = 0; i < kUsed; ++i) {
          rw[t * kUsed + i] /= s;
          by_expert[ids[i]].push_back({t, i});
        }
      }
      // Slots.
      struct Slot {
        int expert;
        std::size_t rows, row0;
      };
      std::vector<Slot> slots;
      std::size_t total_rows = 0;
      int max_tokens = 0;
      for (int e = 0; e < kExperts; ++e) {
        const std::size_t n = by_expert[e].size();
        if (n == 0) {
          continue;
        }
        max_tokens = std::max<int>(max_tokens, static_cast<int>(n));
        std::size_t rows = 1024;
        for (const std::size_t s : kSlots) {
          if (n <= s) {
            rows = s;
            break;
          }
        }
        slots.push_back({e, rows, total_rows});
        total_rows += rows;
      }
      // Hidden states of the chunk (bf16), and grouped A.
      std::vector<std::uint16_t> xh(static_cast<std::size_t>(chunk) * kHidden);
      std::normal_distribution<float> xd(0.0f, 1.0f);
      for (auto& v : xh) {
        v = quant::FloatToBf16(xd(rng));
      }
      xrt::bo a_up = gm.Buffer(total_rows * kHidden * 2);
      xrt::bo c_up = gm.Buffer(total_rows * ExpertLayer::kUpRows * 2);
      xrt::bo a_dn = gm.Buffer(total_rows * kFfn * 2);
      xrt::bo c_dn = gm.Buffer(total_rows * kHidden * 2);
      std::vector<float> out(static_cast<std::size_t>(chunk) * kHidden);

      const auto t0 = Clock::now();
      auto* ap = a_up.map<std::uint16_t*>();
      for (const Slot& s : slots) {
        const auto& toks = by_expert[s.expert];
        for (std::size_t r = 0; r < toks.size(); ++r) {
          std::memcpy(ap + (s.row0 + r) * kHidden,
                      xh.data() + static_cast<std::size_t>(toks[r].first) *
                                      kHidden,
                      kHidden * 2);
        }
      }
      a_up.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      const double group_us = MicrosSince(t0);

      // Dequant only the experts with tokens (one call each, both matrices).
      Batch active(dq);
      for (const Slot& s : slots) {
        const std::size_t e = s.expert;
        active.Add(up1.Run({sub(L.up_qw, L.up_q, e * L.up_q),
                            sub(L.up_b, up_bfp, e * up_bfp)}));
        active.Add(down1.Run({sub(L.down_qw, L.down_q, e * L.down_q),
                              sub(L.down_b, down_bfp, e * down_bfp)}));
      }
      active.Seal();

      std::vector<std::unique_ptr<npu::Sequence>> seqs;
      auto gemm_seq = [&](std::size_t m, std::size_t k, std::size_t n) {
        seqs.push_back(
            std::make_unique<npu::Sequence>(gm, dir, GemmName(m, k, n)));
        return seqs.back().get();
      };
      std::map<std::size_t, npu::Sequence*> up_seq, dn_seq;
      for (const std::size_t m : kSlots) {
        up_seq[m] = gemm_seq(m, kHidden, ExpertLayer::kUpRows);
        dn_seq[m] = gemm_seq(m, kFfn, kHidden);
      }
      Batch ups(gm), downs(gm);
      for (const Slot& s : slots) {
        const std::size_t e = s.expert;
        ups.Add(up_seq[s.rows]->Run(
            {sub(a_up, s.rows * kHidden * 2, s.row0 * kHidden * 2),
             sub(L.up_b, up_bfp, e * up_bfp),
             sub(c_up, s.rows * ExpertLayer::kUpRows * 2,
                 s.row0 * ExpertLayer::kUpRows * 2)}));
        downs.Add(dn_seq[s.rows]->Run(
            {sub(a_dn, s.rows * kFfn * 2, s.row0 * kFfn * 2),
             sub(L.down_b, down_bfp, e * down_bfp),
             sub(c_dn, s.rows * kHidden * 2, s.row0 * kHidden * 2)}));
      }
      ups.Seal();
      downs.Seal();

      auto swiglu = [&] {
        c_up.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto* cu = c_up.map<const std::uint16_t*>();
        auto* ad = a_dn.map<std::uint16_t*>();
        for (const Slot& s : slots) {
          const auto& toks = by_expert[s.expert];
          for (std::size_t r = 0; r < toks.size(); ++r) {
            const float w = rw[toks[r].first * kUsed + toks[r].second];
            const std::uint16_t* row =
                cu + (s.row0 + r) * ExpertLayer::kUpRows;
            std::uint16_t* dst = ad + (s.row0 + r) * kFfn;
            // Packed rows: per 32, 16 gate then the same 16 up.
            for (std::size_t j = 0; j < kFfn; ++j) {
              const std::size_t tile = j / 16, i = j % 16;
              const float g = quant::Bf16ToFloat(row[tile * 32 + i]);
              const float up = quant::Bf16ToFloat(row[tile * 32 + 16 + i]);
              dst[j] = quant::FloatToBf16(Silu(g) * up * w);
            }
          }
        }
        a_dn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      };
      auto scatter = [&] {
        c_dn.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto* cd = c_dn.map<const std::uint16_t*>();
        std::fill(out.begin(), out.end(), 0.0f);
        for (const Slot& s : slots) {
          const auto& toks = by_expert[s.expert];
          for (std::size_t r = 0; r < toks.size(); ++r) {
            float* o = out.data() +
                       static_cast<std::size_t>(toks[r].first) * kHidden;
            const std::uint16_t* row = cd + (s.row0 + r) * kHidden;
            for (std::size_t j = 0; j < kHidden; ++j) {
              o[j] += quant::Bf16ToFloat(row[j]);
            }
          }
        }
      };

      all.Execute();  // every expert's bfp16 in place for the check
      const Timing active_t = Time(5, [&] { active.Execute(); });
      const Timing up_t = Time(5, [&] { ups.Execute(); });
      ups.Execute();
      const Timing sw_t = Time(5, swiglu);
      const Timing dn_t = Time(5, [&] { downs.Execute(); });
      downs.Execute();
      const Timing sc_t = Time(5, scatter);

      // Check 6 tokens against the CPU over the q4nx weights.
      std::vector<double> want, got;
      for (int t = 0; t < chunk; t += chunk / 6) {
        std::vector<double> y(kHidden, 0.0);
        const std::int16_t* ids = routing.At(t, layer);
        for (int i = 0; i < kUsed; ++i) {
          const int tp = ids[i] % ExpertLayer::kTemplates;
          const auto& wu = L.up_w[tp];
          const auto& wd = L.down_w[tp];
          std::vector<double> h(ExpertLayer::kUpRows);
          for (std::size_t r = 0; r < ExpertLayer::kUpRows; ++r) {
            double s = 0;
            for (std::size_t j = 0; j < kHidden; ++j) {
              s += quant::Bf16ToFloat(xh[t * kHidden + j]) * double(wu[r * kHidden + j]);
            }
            h[r] = s;
          }
          std::vector<double> act(kFfn);
          for (std::size_t j = 0; j < kFfn; ++j) {
            const std::size_t tile = j / 16, ii = j % 16;
            const double g = h[tile * 32 + ii], up = h[tile * 32 + 16 + ii];
            act[j] = g / (1 + std::exp(-g)) * up * rw[t * kUsed + i];
          }
          for (std::size_t r = 0; r < kHidden; ++r) {
            double s = 0;
            for (std::size_t j = 0; j < kFfn; ++j) {
              s += act[j] * wd[r * kFfn + j];
            }
            y[r] += s;
          }
        }
        for (std::size_t r = 0; r < kHidden; ++r) {
          want.push_back(y[r]);
          got.push_back(out[t * kHidden + r]);
        }
      }
      const double npu_ms = (up_t.mean + dn_t.mean) / 1e3;
      const double flop = 2.0 * chunk * kUsed *
                          (kHidden * ExpertLayer::kUpRows + kFfn * kHidden);
      std::printf(
          "chunk %4d layer %2d: %3zu experts, max %3d tokens, %4zu padded "
          "rows | dequant active %.1f ms | GEMMs %.1f + %.1f ms (%.2f useful "
          "TFLOPS, %zu calls) | CPU group %.2f, SwiGLU %.2f, scatter %.2f ms "
          "| err %.2e\n",
          chunk, layer, slots.size(), max_tokens, total_rows,
          active_t.mean / 1e3, up_t.mean / 1e3, dn_t.mean / 1e3,
          flop / (npu_ms * 1e9), ups.runs.size() + downs.runs.size(),
          group_us / 1e3, sw_t.mean / 1e3, sc_t.mean / 1e3,
          RelL2(want, got));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: qwen36_prefill_spike <kernel-dir> [routing.bin]\n");
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  std::mt19937 rng(7);
  npu::HwContext dq(dir, "qwen36_35b_a3b_dequant");
  npu::HwContext gm(dir, "qwen36_35b_a3b_prefill");
  const std::string only = std::getenv("SPIKE_ONLY") ? std::getenv("SPIKE_ONLY") : "";
  if (only.empty() || only == "dense") {
    Dense(dir, dq, gm, rng);
  }
  if (only.empty() || only == "switch") {
    Switches(dir, dq, gm, rng);
  }
  if (only.empty() || only == "experts") {
    const char* path = argc > 2 ? argv[2] : nullptr;
    const Routing routing = LoadRouting(path, rng, 1024);
    Experts(dir, dq, gm, rng, routing, path ? "recorded" : "uniform random");
  }
  return 0;
}
