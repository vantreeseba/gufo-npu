// Spike: Qwen3.6-35B-A3B MoE FFN on the NPU with experts chosen per call.
//  (c) gather: two calls per layer (9 experts' gate + up as 9 rounds, then
//      one down whose K is the 9 experts' outputs), weight offsets and router
//      scales patched into the instruction stream (legacy flow) before each
//      call. Checked against the CPU over the same q4nx weights.
//  (a) one ELF run per expert matrix (18 per layer), and (b) rebinding their
//      weight sub-BOs with set_arg between runlist executions.
// Not part of the build; how to build and run it, and the results, are in
// docs/research/NPU-MOE.md.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"
#include "npu/xrt_kernel.hpp"

using namespace gufo_npu;
using Clock = std::chrono::steady_clock;
static double Us(Clock::time_point s) {
  return std::chrono::duration<double, std::micro>(Clock::now() - s).count();
}

constexpr int kCols = 8, kTokens = 2, kExperts = 9, kFfn = 512, kHidden = 2048;
constexpr int kMaxDim = 6144;
constexpr uint32_t kSentinel = 0x7FA50000;

std::vector<uint32_t> ReadWords(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(in)), {});
  std::vector<uint32_t> w(b.size() / 4);
  std::memcpy(w.data(), b.data(), w.size() * 4);
  return w;
}

// DDR patches of argument `arg`: word positions, in stream order.
std::vector<size_t> Patches(const std::vector<uint32_t>& w, uint32_t arg) {
  std::vector<size_t> out;
  size_t pos = 4;
  while (pos < w.size()) {
    const uint32_t op = w[pos] & 0xFF;
    size_t len = 0;
    if (op == 0)
      len = 6;
    else if (op == 3 || op == 4)
      len = 7;
    else if (op == 1)
      len = w[pos + 3] / 4;
    else if (op == 0x80 || op == 0x81)
      len = w[pos + 1] / 4;
    else if (op == 8 || op == 10)
      len = 4;
    else if (op == 6)
      len = 1;
    else if (op == 12)
      len = 3;
    else {
      std::printf("unknown op %#x at %zu\n", op, pos);
      std::exit(1);
    }
    if (op == 0x81 && w[pos + 8] == arg)
      out.push_back(pos);
    pos += len;
  }
  return out;
}

uint64_t Plus(const uint32_t* w, size_t p) {
  return w[p + 10] | (uint64_t(w[p + 11]) << 32);
}
void SetPlus(uint32_t* w, size_t p, uint64_t v) {
  w[p + 10] = uint32_t(v);
  w[p + 11] = uint32_t(v >> 32);
}

float Bf16Round(float v) {
  return quant::Bf16ToFloat(quant::FloatToBf16(v));
}

// A patchable gather call: its own instruction BO, the pristine stream, and
// the patch sites.
struct Gather {
  std::vector<uint32_t> words;
  std::vector<size_t> sites;       // weight patches (arg 1), stream order
  std::vector<uint64_t> defaults;  // their placeholder offsets
  std::vector<std::pair<size_t, int>> scales;  // (word, round * TOKENS + t)
  xrt::bo instr;
  Gather(npu::HwContext& hw, xrt::kernel& k, const std::string& path) {
    words = ReadWords(path);
    sites = Patches(words, 1);
    for (size_t p : sites)
      defaults.push_back(Plus(words.data(), p));
    for (size_t i = 0; i < words.size(); ++i)
      if (words[i] >= kSentinel && words[i] < kSentinel + 256)
        scales.push_back({i, int(words[i] - kSentinel)});
    instr = xrt::bo(hw.device, words.size() * 4, XCL_BO_FLAGS_CACHEABLE,
                    k.group_id(1));
    std::memcpy(instr.map<void*>(), words.data(), words.size() * 4);
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  uint32_t* map() { return instr.map<uint32_t*>(); }
  void sync() { instr.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
};

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string dir = argv[1];
  const int n_experts = argc > 2 ? std::atoi(argv[2]) : 16;
  npu::HwContext hw(dir, "q4nx_gemv");
  xrt::kernel legacy(*hw.context, hw.kernel_name);
  std::mt19937 rng(5);
  std::normal_distribution<float> wd(0.f, 0.02f), ad(0.f, 1.f);

  // Experts: gate + up (1024 x 2048, SwiGLU order) and down (2048 x 512).
  std::vector<q4nx::Matrix> ups, downs;
  for (int e = 0; e < n_experts; ++e) {
    std::vector<uint16_t> bits(1024 * kHidden);
    for (auto& b : bits)
      b = quant::FloatToBf16(wd(rng));
    ups.push_back(
        q4nx::PackRows({{gguf::GgmlType::kBF16,
                         reinterpret_cast<const uint8_t*>(bits.data()), 1024}},
                       kHidden, q4nx::Fit::kSearch, q4nx::SwiGluOrder(kFfn)));
    std::vector<float> d(kHidden * kFfn);
    for (auto& v : d)
      v = wd(rng);
    downs.push_back(q4nx::PackFloat(d.data(), kHidden, kFfn, q4nx::Fit::kSearch));
  }
  const size_t up_bytes = ups[0].bytes.size(),
               down_bytes = downs[0].bytes.size();
  std::printf("expert: gate+up %zu B, down %zu B; arena of %d experts\n",
              up_bytes, down_bytes, n_experts);
  xrt::bo arena = hw.Buffer(n_experts * (up_bytes + down_bytes));
  for (int e = 0; e < n_experts; ++e) {
    std::memcpy(arena.map<char*>() + e * up_bytes, ups[e].bytes.data(),
                up_bytes);
    std::memcpy(arena.map<char*>() + n_experts * up_bytes + e * down_bytes,
                downs[e].bytes.data(), down_bytes);
  }
  arena.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto up_at = [&](int e) { return uint64_t(e) * up_bytes; };
  auto down_at = [&](int e) {
    return uint64_t(n_experts) * up_bytes + uint64_t(e) * down_bytes;
  };

  // Input: H, D and the norm weight (resnorm), one token.
  xrt::bo x = hw.Buffer(npu::kGemvInputBytes);
  float* xf = x.map<float*>();
  std::memset(xf, 0, npu::kGemvInputBytes);
  std::vector<float> h(kHidden), nw(kHidden);
  auto* nwb =
      reinterpret_cast<uint16_t*>(x.map<uint8_t*>() + npu::kGemvNormOffset);
  for (int i = 0; i < kHidden; ++i) {
    xf[i] = ad(rng) * 3;
    xf[kHidden + i] = ad(rng);
    h[i] = xf[i] + xf[kHidden + i];
    nwb[i] = quant::FloatToBf16(1.f + 0.2f * ad(rng));
    nw[i] = quant::Bf16ToFloat(nwb[i]);
  }
  x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  double ss = 0;
  for (float v : h)
    ss += double(v) * v;
  const float inv = 1.f / std::sqrt(float(ss / kHidden) + 1e-6f);
  std::vector<float> xn(kHidden);
  for (int i = 0; i < kHidden; ++i)
    xn[i] = Bf16Round(h[i] * inv * nw[i]);

  const size_t act_words = size_t(kTokens) * (kMaxDim / 2) * 4;  // gate+up Y
  xrt::bo act = hw.Buffer(act_words * 4);
  xrt::bo y = hw.Buffer((npu::kGemvSlot + kHidden) * 4),
          r = hw.Buffer(npu::kGemvResidualBytes);
  xrt::bo scratch = hw.Buffer(npu::kGemvResidualBytes);

  Gather up(
      hw, legacy,
      dir +
          "/q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_rounds9_t1.insts.bin");
  Gather down(hw, legacy,
              dir + "/q4nx_gemv_k4608_n2048_act_store_k9_t1.insts.bin");
  std::printf(
      "gate+up gather: %zu words, %zu weight sites, %zu scale words; down "
      "gather: %zu words, %zu sites\n",
      up.words.size(), up.sites.size(), up.scales.size(), down.words.size(),
      down.sites.size());
  const uint64_t up_seg = up_bytes / kCols;            // one column's slice
  const uint64_t down_seg = down_bytes / (2 * kCols);  // one (round, col) slice

  auto patch = [&](const std::vector<int>& experts,
                   const std::vector<float>& scales) {
    uint32_t* w = up.map();
    for (size_t i = 0; i < up.sites.size(); ++i) {
      const uint64_t idx = up.defaults[i] / up_seg;  // round * COLS + col
      SetPlus(w, up.sites[i],
              up_at(experts[idx / kCols]) + (idx % kCols) * up_seg);
    }
    for (auto [pos, i] : up.scales) {
      float s = (i / kTokens) < kExperts ? scales[i / kTokens] : 0.f;
      std::memcpy(&w[pos], &s, 4);
    }
    up.sync();
    w = down.map();
    for (size_t i = 0; i < down.sites.size(); ++i) {
      const uint64_t idx =
          down.defaults[i] / down_seg;  // (round*COLS+col)*E + e
      const uint64_t rc = idx / kExperts, e = idx % kExperts;
      SetPlus(w, down.sites[i], down_at(experts[e]) + rc * down_seg);
    }
    down.sync();
  };
  auto run_of = [&](Gather& g, xrt::bo& a0, xrt::bo& a2, xrt::bo& a3) {
    xrt::run run(legacy);
    run.set_arg(0, npu::kOpcodeRun);
    run.set_arg(1, g.instr);
    run.set_arg(2, static_cast<uint32_t>(g.words.size()));
    run.set_arg(3, a0);
    run.set_arg(4, arena);
    run.set_arg(5, a2);
    run.set_arg(6, a3);
    return run;
  };
  xrt::runlist layer(*hw.context);
  layer.add(run_of(up, x, act, r));
  layer.add(run_of(down, act, y, scratch));

  // Correctness over random selections.
  std::vector<std::vector<float>> up_w(n_experts), down_w(n_experts);
  for (int e = 0; e < n_experts; ++e) {
    up_w[e] = q4nx::Unpack(ups[e]);
    down_w[e] = q4nx::Unpack(downs[e]);
  }
  double worst = 0;
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<int> ex(n_experts);
    for (int i = 0; i < n_experts; ++i)
      ex[i] = i;
    std::shuffle(ex.begin(), ex.end(), rng);
    ex.resize(kExperts);
    std::vector<float> sc(kExperts);
    for (auto& s : sc)
      s = std::uniform_real_distribution<float>(0.05f, 1.f)(rng);
    patch(ex, sc);
    layer.execute();
    layer.wait();
    y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<double> want(kHidden, 0.0);
    for (int i = 0; i < kExperts; ++i) {
      const auto& U = up_w[ex[i]];
      std::vector<float> a(kFfn);
      for (int tile = 0; tile < 32; ++tile) {
        for (int j = 0; j < 16; ++j) {
          double g = 0, u = 0;
          const float* gr = &U[size_t(tile * 32 + j) * kHidden];
          const float* ur = &U[size_t(tile * 32 + 16 + j) * kHidden];
          for (int k = 0; k < kHidden; ++k) {
            g += double(gr[k]) * xn[k];
            u += double(ur[k]) * xn[k];
          }
          const double silu = g / (1 + std::exp(-g));
          a[tile * 16 + j] = Bf16Round(float(silu * u * sc[i]));
        }
      }
      const auto& Dn = down_w[ex[i]];
      for (int o = 0; o < kHidden; ++o) {
        double acc = 0;
        for (int k = 0; k < kFfn; ++k)
          acc += double(Dn[size_t(o) * kFfn + k]) * a[k];
        want[o] += acc;
      }
    }
    double diff = 0, norm = 0;
    const float* got = y.map<float*>();
    for (int o = 0; o < kHidden; ++o) {
      diff += (got[o] - want[o]) * (got[o] - want[o]);
      norm += want[o] * want[o];
    }
    worst = std::max(worst, std::sqrt(diff / norm));
  }
  std::printf(
      "%s gather MoE vs CPU over the same weights: worst rel err %.2e (20 "
      "selections)\n",
      worst < 1e-2 ? "ok  " : "FAIL", worst);

  // Patch cost, and layer / 40-layer timing.
  {
    std::vector<int> ex{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> sc(kExperts, 0.5f);
    double best = 1e9;
    for (int i = 0; i < 200; ++i) {
      auto s = Clock::now();
      patch(ex, sc);
      best = std::min(best, Us(s));
    }
    std::printf(
        "patch one layer (72 + 144 offsets, %zu scale words, 2 syncs): %.1f "
        "us\n",
        up.scales.size(), best);
    double lb = 1e9;
    for (int i = 0; i < 50; ++i) {
      auto s = Clock::now();
      layer.execute();
      layer.wait();
      lb = std::min(lb, Us(s));
    }
    const double bytes = kExperts * double(up_bytes + down_bytes);
    std::printf(
        "one MoE layer FFN (gate+up gather, down gather), one submission: %.1f "
        "us, %.1f GB/s\n",
        lb, bytes / lb / 1e3);
    // 40 layers chained: each layer its own patched instruction BOs.
    std::vector<Gather> ups40, downs40;
    ups40.reserve(40);
    downs40.reserve(40);
    xrt::runlist all(*hw.context);
    for (int l = 0; l < 40; ++l) {
      ups40.emplace_back(hw, legacy,
                         dir +
                             "/q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_"
                             "rounds9_t1.insts.bin");
      downs40.emplace_back(
          hw, legacy, dir + "/q4nx_gemv_k4608_n2048_act_store_k9_t1.insts.bin");
      all.add(run_of(ups40.back(), x, act, r));
      all.add(run_of(downs40.back(), act, y, scratch));
    }
    double ab = 1e9;
    for (int i = 0; i < 10; ++i) {
      auto s = Clock::now();
      all.execute();
      all.wait();
      ab = std::min(ab, Us(s));
    }
    std::printf(
        "40 MoE layers in one runlist (80 calls): %.2f ms, %.1f us per layer, "
        "%.1f GB/s\n",
        ab / 1e3, ab / 40, 40 * bytes / ab / 1e3);
  }

  // A layer's post-router chain in the legacy flow: gate+up gather, down
  // gather, then a dense input projection (next layer), one runlist; and the
  // 2-token gathers.
  {
    auto per = [&](xrt::runlist& l) {
      double b = 1e9;
      for (int i = 0; i < 10; ++i) {
        auto s = Clock::now();
        l.execute();
        l.wait();
        b = std::min(b, Us(s));
      }
      return b;
    };
    struct Dense {
      std::vector<uint32_t> words;
      xrt::bo instr;
    };
    auto dense_of = [&](const std::string& name) {
      Dense d{ReadWords(dir + "/" + name + ".insts.bin"), {}};
      d.instr = xrt::bo(hw.device, d.words.size() * 4, XCL_BO_FLAGS_CACHEABLE,
                        legacy.group_id(1));
      std::memcpy(d.instr.map<void*>(), d.words.data(), d.words.size() * 4);
      d.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      return d;
    };
    Dense in8192 = dense_of("q4nx_gemv_k2048_n8192_resnorm_store_t1");
    xrt::bo w_in = hw.Buffer(size_t(8192) * 2048 / 256 * 4736 / 32);
    xrt::bo y_in = hw.Buffer((npu::kGemvSlot + 8192) * 4 + 8192 * 4);
    auto dense_run = [&](Dense& d) {
      xrt::run run(legacy);
      run.set_arg(0, npu::kOpcodeRun);
      run.set_arg(1, d.instr);
      run.set_arg(2, static_cast<uint32_t>(d.words.size()));
      run.set_arg(3, x);
      run.set_arg(4, w_in);
      run.set_arg(5, y_in);
      run.set_arg(6, scratch);
      return run;
    };
    std::vector<Gather> gu, gd;
    gu.reserve(40);
    gd.reserve(40);
    xrt::runlist one(*hw.context), forty(*hw.context);
    for (int l = 0; l < 40; ++l) {
      gu.emplace_back(hw, legacy,
                      dir +
                          "/q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_"
                          "rounds9_t1.insts.bin");
      gd.emplace_back(hw, legacy,
                      dir + "/q4nx_gemv_k4608_n2048_act_store_k9_t1.insts.bin");
      if (l == 0) {
        one.add(run_of(gu.back(), x, act, r));
        one.add(run_of(gd.back(), act, y, scratch));
        one.add(dense_run(in8192));
      }
      forty.add(run_of(gu.back(), x, act, r));
      forty.add(run_of(gd.back(), act, y, scratch));
      forty.add(dense_run(in8192));
    }
    std::printf(
        "chain [gate+up gather, down gather, dense 8192x2048 resnorm], legacy "
        "flow: one submission %.1f us; "
        "40 in one runlist %.2f ms (%.1f us per layer)\n",
        per(one), per(forty) / 1e3, per(forty) / 40);
    Gather u2(hw, legacy,
              dir +
                  "/q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_rounds9_t2."
                  "insts.bin");
    Gather d2(hw, legacy,
              dir + "/q4nx_gemv_k4608_n2048_act_store_k9_t2.insts.bin");
    xrt::runlist two(*hw.context);
    two.add(run_of(u2, x, act, r));
    two.add(run_of(d2, act, y, scratch));
    std::printf(
        "2-token gathers (same 9 experts for both tokens): %.1f us per layer\n",
        per(two));
  }

  // Where the time goes: each gather alone, and dense ELF GEMVs of similar
  // size, each as a runlist of 20 calls.
  {
    auto per_call = [&](xrt::runlist& l, int n) {
      double b = 1e9;
      for (int i = 0; i < 10; ++i) {
        auto s = Clock::now();
        l.execute();
        l.wait();
        b = std::min(b, Us(s));
      }
      return b / n;
    };
    xrt::runlist lu(*hw.context), ld(*hw.context);
    std::vector<Gather> gu, gd;
    gu.reserve(20);
    gd.reserve(20);
    for (int i = 0; i < 20; ++i) {
      gu.emplace_back(hw, legacy,
                      dir +
                          "/q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_"
                          "rounds9_t1.insts.bin");
      gd.emplace_back(hw, legacy,
                      dir + "/q4nx_gemv_k4608_n2048_act_store_k9_t1.insts.bin");
      lu.add(run_of(gu.back(), x, act, r));
      ld.add(run_of(gd.back(), act, y, scratch));
    }
    const double tu = per_call(lu, 20), td = per_call(ld, 20);
    std::printf(
        "gate+up gather: %.1f us per call (%.1f GB/s); down gather: %.1f us "
        "(%.1f GB/s)\n",
        tu, 9 * up_bytes / tu / 1e3, td, 9 * down_bytes / td / 1e3);
    for (const char* name : {"q4nx_gemv_k2048_n8192_resnorm_store_t1",
                             "q4nx_gemv_k6144_n2048_act_store_t1",
                             "q4nx_gemv_k2048_n2048_plain_store_t1"}) {
      npu::Sequence s(hw, dir, name);
      std::string n = name;
      const size_t dim = std::stoul(n.substr(n.find("_k") + 2)),
                   rows = std::stoul(n.substr(n.find("_n") + 2));
      const size_t bytes = rows * dim / 256 * 4736 / 32;
      xrt::bo w = hw.Buffer(bytes),
              yy = hw.Buffer((npu::kGemvSlot + rows) * 4 + rows * 4);
      xrt::runlist l(*hw.context);
      std::vector<xrt::run> runs;
      for (int i = 0; i < 20; ++i) {
        runs.push_back(s.Run({n.find("act") != std::string::npos ? &act : &x,
                              &w, &yy, &scratch}));
        l.add(runs.back());
      }
      const double t = per_call(l, 20);
      std::printf("dense %s: %.1f us per call (%.1f GB/s)\n", name, t,
                  bytes / t / 1e3);
    }
  }

  // (a) one ELF run per expert matrix, 9 + 9 per layer, sub-BO weights.
  {
    npu::Sequence su(hw, dir, "q4nx_gemv_k2048_n1024_resnorm_swiglu_t1");
    npu::Sequence sd(hw, dir, "q4nx_gemv_k512_n2048_act_store_t1");
    std::vector<xrt::bo> subs;
    for (int e = 0; e < n_experts; ++e) {
      subs.emplace_back(arena, up_bytes, up_at(e));
      subs.emplace_back(arena, down_bytes, down_at(e));
    }
    xrt::runlist list(*hw.context);
    std::vector<xrt::run> runs;
    for (int i = 0; i < kExperts; ++i) {
      runs.push_back(su.Run({&x, &subs[2 * i], &act, &r}));
      list.add(runs.back());
      runs.push_back(sd.Run({&act, &subs[2 * i + 1], &y, &scratch}));
      list.add(runs.back());
    }
    double best = 1e9;
    for (int i = 0; i < 30; ++i) {
      auto s = Clock::now();
      list.execute();
      list.wait();
      best = std::min(best, Us(s));
    }
    std::printf(
        "(a) 18 ELF runs per layer (one per expert matrix): %.1f us per layer, "
        "%.1f us per run\n",
        best, best / 18);
    // (b) rebind every run's weight to other experts, then execute again.
    bool ok = true;
    double rebind = 1e9;
    try {
      for (int it = 0; it < 30; ++it) {
        auto s = Clock::now();
        for (int i = 0; i < kExperts; ++i) {
          const int e = (i + it) % n_experts;
          runs[2 * i].set_arg(4, subs[2 * e]);
          runs[2 * i + 1].set_arg(4, subs[2 * e + 1]);
        }
        rebind = std::min(rebind, Us(s));
        list.execute();
        list.wait();
      }
    } catch (const std::exception& e) {
      ok = false;
      std::printf("(b) rebinding failed: %s\n", e.what());
    }
    if (ok) {
      // Correctness of a rebound run: one expert's gate + up alone, against a
      // fresh run on the same sub-BO.
      act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::printf(
          "(b) set_arg on 18 runs in a runlist between executions: %.1f us per "
          "layer\n",
          rebind);
      runs[0].set_arg(4, subs[2 * 3]);
      list.execute();
      list.wait();
      xrt::bo act2 = hw.Buffer(act_words * 4);
      auto fresh = su.Run({&x, &subs[2 * 3], &act2, &scratch});
      fresh.start();
      fresh.wait();
      act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      act2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      // The last runs in the list overwrite act; compare only run 0's effect by
      // re-running it alone through a one-run list.
      xrt::runlist one(*hw.context);
      auto solo = su.Run({&x, &subs[0], &act, &scratch});
      one.add(solo);
      one.execute();
      one.wait();
      solo.set_arg(4, subs[2 * 3]);
      one.execute();
      one.wait();
      act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::printf(
          "(b) rebound run == fresh run on the new expert: %d\n",
          std::memcmp(act.map<void*>(), act2.map<void*>(), 512 * 2) == 0);
    }
  }
  return 0;
}
