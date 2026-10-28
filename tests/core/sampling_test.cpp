// Sampler: greedy equals std::max_element (ties included), seeds reproduce,
// each filter and penalty gives the documented distribution, draws follow it
// (chi-square), and the speculative loop's acceptance rule (draw from the
// target, keep the draft when they agree) samples the target exactly.
#include "core/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using gufo_npu::Sampler;
using gufo_npu::SamplingParams;

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// Chi-square critical value at p = 1e-6 (Wilson-Hilferty), so a correct
// sampler fails about once in a million runs per check.
double ChiSquareLimit(int df) {
  const double z = 4.753;
  const double a = 2.0 / (9.0 * df);
  return df * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

// Pearson's statistic of `counts` (n draws) against `expected` probabilities,
// pooling cells expected below 5 draws into one.
double ChiSquare(const std::vector<long>& counts,
                 const std::vector<double>& expected, long n, int* df) {
  double stat = 0;
  double pooled_e = 0;
  long pooled_c = 0;
  int cells = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const double e = expected[i] * static_cast<double>(n);
    if (e < 5) {
      pooled_e += e;
      pooled_c += counts[i];
      continue;
    }
    stat += (counts[i] - e) * (counts[i] - e) / e;
    ++cells;
  }
  if (pooled_e >= 5) {
    stat += (pooled_c - pooled_e) * (pooled_c - pooled_e) / pooled_e;
    ++cells;
  } else if (pooled_c > 0 && pooled_e == 0) {
    stat = 1e30;  // a draw where none can fall
  }
  *df = std::max(1, cells - 1);
  return stat;
}

std::vector<double> Softmax(const std::vector<float>& logits, float t) {
  const float m = *std::max_element(logits.begin(), logits.end());
  std::vector<double> p(logits.size());
  double sum = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp(static_cast<double>(logits[i] - m) / t);
    sum += p[i];
  }
  for (auto& v : p) {
    v /= sum;
  }
  return p;
}

void CheckDraws(Sampler& sampler, const std::vector<float>& logits,
                const std::vector<double>& expected, long n, const char* what) {
  std::vector<long> counts(logits.size(), 0);
  for (long i = 0; i < n; ++i) {
    ++counts[static_cast<std::size_t>(
        sampler.Sample(logits.data(), static_cast<int>(logits.size())))];
  }
  int df = 0;
  const double stat = ChiSquare(counts, expected, n, &df);
  const double limit = ChiSquareLimit(df);
  std::printf("  %s: chi-square %.1f (df %d, limit %.1f)\n", what, stat, df,
              limit);
  Check(stat < limit, what);
}

std::vector<double> Expected(Sampler& sampler, const std::vector<float>& l) {
  std::vector<double> p(l.size(), 0.0);
  for (const auto& [id, prob] :
       sampler.Distribution(l.data(), static_cast<int>(l.size()))) {
    p[static_cast<std::size_t>(id)] = prob;
  }
  return p;
}

void Greedy() {
  Sampler s;
  const std::vector<float> ties = {1, 3, 3, 2, 3};
  Check(s.Sample(ties.data(), 5) == 1, "greedy picks the first maximum");
  std::mt19937 rng(3);
  std::normal_distribution<float> normal(0, 4);
  std::vector<float> l(248320);
  for (int round = 0; round < 20; ++round) {
    for (auto& v : l) {
      v = normal(rng);
    }
    const int want =
        static_cast<int>(std::max_element(l.begin(), l.end()) - l.begin());
    Check(s.Sample(l.data(), static_cast<int>(l.size())) == want,
          "greedy equals std::max_element on a 248k vocab");
  }
  // Penalties apply under greedy too: a repeated best token loses its lead.
  SamplingParams p;
  p.repetition_penalty = 2.0f;
  Sampler pen(p);
  const std::vector<float> two = {5, 4, 1};
  Check(pen.Sample(two.data(), 3) == 0, "greedy with penalty, nothing seen");
  pen.Accept(0);
  Check(pen.Sample(two.data(), 3) == 1, "greedy with penalty, 0 seen");
  // Equal penalized logits break by id: 4/2 = 2 against a fresh 2 at id 2.
  const std::vector<float> tie = {1, 4, 2};
  Sampler pen2(p);
  pen2.Accept(1);
  Check(pen2.Sample(tie.data(), 3) == 1, "greedy penalized tie by id");
}

void Seeds() {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 42;
  const std::vector<float> l = {0.1f, 0.5f, 0.2f, 1.0f, -0.3f, 0.7f};
  Sampler a(p);
  Sampler b(p);
  bool same = true;
  for (int i = 0; i < 2000; ++i) {
    same = same && a.Sample(l.data(), 6) == b.Sample(l.data(), 6);
  }
  Check(same, "same seed, same draws");
  p.seed = 43;
  Sampler c(p);
  Sampler d(SamplingParams{1.0f, 1.0f, 0, 0.0f, 0, 0, 1.0f, 42});
  int differ = 0;
  for (int i = 0; i < 2000; ++i) {
    differ += c.Sample(l.data(), 6) != d.Sample(l.data(), 6);
  }
  Check(differ > 100, "different seeds, different draws");
}

void Temperature() {
  const std::vector<float> l = {2.0f, 1.0f, 0.5f, 0.0f, -1.0f, 1.5f, 3.0f};
  for (const float t : {0.3f, 0.7f, 1.0f, 2.0f}) {
    SamplingParams p;
    p.temperature = t;
    p.seed = 7;
    Sampler s(p);
    const auto want = Softmax(l, t);
    const auto got = Expected(s, l);
    double err = 0;
    for (std::size_t i = 0; i < l.size(); ++i) {
      err = std::max(err, std::fabs(got[i] - want[i]));
    }
    Check(err < 1e-6, "temperature distribution is softmax(l / t)");
    char what[64];
    std::snprintf(what, sizeof what, "draws at temperature %.1f", t);
    CheckDraws(s, l, want, 200000, what);
  }
}

void Filters() {
  const std::vector<float> l = {0.0f, 2.0f, 1.0f, 3.0f, 0.5f, 2.5f, -1.0f};
  // top_k = 3 keeps ids 3, 5, 1 in that order, renormalized.
  {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 3;
    p.seed = 1;
    Sampler s(p);
    const auto dist = s.Distribution(l.data(), 7);
    Check(dist.size() == 3 && dist[0].first == 3 && dist[1].first == 5 &&
              dist[2].first == 1,
          "top_k keeps the k best, best first");
    const double z = std::exp(0.0) + std::exp(-0.5) + std::exp(-1.0);
    Check(std::fabs(dist[0].second - 1.0 / z) < 1e-6, "top_k renormalizes");
    CheckDraws(s, l, Expected(s, l), 200000, "draws under top_k 3");
  }
  // top_p: the smallest best-first prefix holding at least top_p.
  {
    const auto full = Softmax(l, 1.0f);
    std::vector<int> order = {3, 5, 1, 2, 4, 0, 6};
    for (const float top_p : {0.3f, 0.6f, 0.9f}) {
      SamplingParams p;
      p.temperature = 1.0f;
      p.top_p = top_p;
      p.seed = 2;
      Sampler s(p);
      const auto dist = s.Distribution(l.data(), 7);
      double cum = 0;
      std::size_t keep = 0;
      while (keep < order.size()) {
        cum += full[static_cast<std::size_t>(order[keep])];
        ++keep;
        if (cum >= top_p) {
          break;
        }
      }
      Check(dist.size() == keep, "top_p keeps the smallest prefix");
      CheckDraws(s, l, Expected(s, l), 100000, "draws under top_p");
    }
  }
  // min_p: drops tokens below min_p times the best one's probability.
  {
    SamplingParams p;
    p.temperature = 1.0f;
    p.min_p = 0.3f;
    p.seed = 3;
    Sampler s(p);
    const auto dist = s.Distribution(l.data(), 7);
    // Relative to the best (logit 3): e^-0.5 = 0.61, e^-1 = 0.37 kept;
    // e^-2 = 0.14 dropped.
    Check(dist.size() == 3, "min_p drops tokens below min_p * best");
    CheckDraws(s, l, Expected(s, l), 100000, "draws under min_p 0.3");
  }
  // The tail: more than kTailNats below the best scaled logit never draws.
  {
    SamplingParams p;
    p.temperature = 1.0f;
    Sampler s(p);
    const std::vector<float> far = {10.0f, 10.0f - 26.0f, 10.0f - 24.0f};
    const auto dist = s.Distribution(far.data(), 3);
    Check(dist.size() == 2 && dist[1].first == 2,
          "tokens past kTailNats are not candidates");
  }
  // top_p on a large vocab (the histogram cut) keeps exactly the smallest
  // best-first prefix of the unfiltered distribution.
  {
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0, 2.0f);
    std::vector<float> big(248320);
    for (auto& v : big) {
      v = normal(rng);
    }
    for (int i = 0; i < 40; ++i) {
      big[static_cast<std::size_t>(i) * 4099] += 9.0f;
    }
    for (const float top_p : {0.5f, 0.9f, 0.99f}) {
      for (const float t : {0.6f, 1.0f}) {
        SamplingParams all;
        all.temperature = t;
        Sampler full(all);
        const auto ranked = full.Distribution(big.data(), 248320);
        SamplingParams p = all;
        p.top_p = top_p;
        Sampler s(p);
        auto got = s.Distribution(big.data(), 248320);
        double cum = 0;
        std::size_t keep = 0;
        while (keep < ranked.size()) {
          cum += ranked[keep].second;
          ++keep;
          if (cum >= top_p) {
            break;
          }
        }
        // The two totals round differently, so the boundary token may
        // differ by one.
        bool same = got.size() + 1 >= keep && got.size() <= keep + 1;
        for (std::size_t i = 0; same && i < std::min(got.size(), keep) - 1;
             ++i) {
          same = got[i].first == ranked[i].first;
        }
        std::printf("  top_p %.2f at temperature %.1f: %zu of %zu kept\n",
                    top_p, t, got.size(), ranked.size());
        Check(same, "top_p cut on a 248k vocab equals the sorted prefix");
      }
    }
  }
}

void Penalties() {
  const std::vector<float> l = {1.0f, -1.0f, 0.5f, 2.0f, 0.0f};
  SamplingParams p;
  p.temperature = 0.8f;
  p.repetition_penalty = 1.3f;
  p.presence_penalty = 0.4f;
  p.frequency_penalty = 0.25f;
  p.seed = 9;
  Sampler s(p);
  s.Accept(0);
  s.Accept(1);
  s.Accept(1);
  s.Accept(3);
  std::vector<float> want = l;
  want[0] = 1.0f / 1.3f - 0.4f - 1 * 0.25f;
  want[1] = -1.0f * 1.3f - 0.4f - 2 * 0.25f;
  want[3] = 2.0f / 1.3f - 0.4f - 1 * 0.25f;
  const auto expected = Softmax(want, 0.8f);
  const auto got = Expected(s, l);
  double err = 0;
  for (std::size_t i = 0; i < l.size(); ++i) {
    err = std::max(err, std::fabs(got[i] - expected[i]));
  }
  Check(err < 1e-6, "repetition, presence and frequency penalties");
  // Without a penalty set, Accept keeps nothing and draws are unchanged.
  SamplingParams q;
  q.temperature = 1.0f;
  Sampler plain(q);
  plain.Accept(3);
  const auto same = Expected(plain, l);
  const auto soft = Softmax(l, 1.0f);
  err = 0;
  for (std::size_t i = 0; i < l.size(); ++i) {
    err = std::max(err, std::fabs(same[i] - soft[i]));
  }
  Check(err < 1e-6, "no penalties: history does not matter");
}

// A toy model over 6 tokens: next-token logits depend on the previous
// token. The draft proposes the argmax of a different table (so it is often
// wrong). Sequences of 4 tokens after a fixed first token, drawn by plain
// sampling and by the speculative loop's rule, must have one distribution.
void SpeculativeRule() {
  constexpr int kV = 6;
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0, 1.2f);
  std::vector<std::vector<float>> target(kV, std::vector<float>(kV));
  std::vector<int> draft(kV);
  for (int a = 0; a < kV; ++a) {
    for (auto& v : target[static_cast<std::size_t>(a)]) {
      v = normal(rng);
    }
    draft[static_cast<std::size_t>(a)] = (a * 5 + 1) % kV;
  }
  constexpr int kLen = 4;
  auto key = [](const std::vector<int>& seq) {
    int k = 0;
    for (const int t : seq) {
      k = k * kV + t;
    }
    return k;
  };
  for (const bool penalties : {false, true}) {
    SamplingParams p;
    p.temperature = 0.9f;
    p.top_k = 5;
    if (penalties) {
      p.frequency_penalty = 0.5f;
    }
    // Exact distribution of the 4-token continuation by enumeration.
    std::map<int, double> exact;
    std::vector<int> seq;
    std::function<void(double, Sampler&)> walk = [&](double prob,
                                                     Sampler& history) {
      if (static_cast<int>(seq.size()) == kLen) {
        exact[key(seq)] += prob;
        return;
      }
      const int prev = seq.empty() ? 0 : seq.back();
      for (const auto& [id, q] : history.Distribution(
               target[static_cast<std::size_t>(prev)].data(), kV)) {
        Sampler next = history;
        next.Accept(id);
        seq.push_back(id);
        walk(prob * q, next);
        seq.pop_back();
      }
    };
    SamplingParams seeded = p;
    seeded.seed = 0;
    Sampler root(seeded);
    walk(1.0, root);

    // The speculative loop, as SpeculativeGenerate runs it.
    constexpr long kRuns = 300000;
    std::map<int, long> counts;
    long accepted = 0;
    long steps = 0;
    for (long run = 0; run < kRuns; ++run) {
      p.seed = static_cast<std::uint64_t>(run) + 1000;
      Sampler s(p);
      std::vector<int> out;
      auto emit = [&](int id) {
        out.push_back(id);
        s.Accept(id);
        return static_cast<int>(out.size()) < kLen;
      };
      int y1 = s.Sample(target[0].data(), kV);
      bool going = emit(y1);
      while (going) {
        const int y2 = draft[static_cast<std::size_t>(y1)];
        ++steps;
        const int z1 =
            s.Sample(target[static_cast<std::size_t>(y1)].data(), kV);
        if (z1 == y2) {
          ++accepted;
          going = emit(y2);
          if (!going) {
            break;
          }
          const int z2 =
              s.Sample(target[static_cast<std::size_t>(y2)].data(), kV);
          going = emit(z2);
          y1 = z2;
        } else {
          going = emit(z1);
          y1 = z1;
        }
      }
      ++counts[key(out)];
    }
    std::vector<long> c;
    std::vector<double> e;
    for (const auto& [k, prob] : exact) {
      e.push_back(prob);
      const auto it = counts.find(k);
      c.push_back(it == counts.end() ? 0 : it->second);
    }
    long outside = 0;
    for (const auto& [k, n] : counts) {
      outside += exact.count(k) == 0 ? n : 0;
    }
    Check(outside == 0, "speculative loop draws only possible sequences");
    int df = 0;
    const double stat = ChiSquare(c, e, kRuns, &df);
    const double limit = ChiSquareLimit(df);
    std::printf(
        "  speculative rule%s: chi-square %.1f (df %d, limit %.1f); %.0f%% "
        "of drafts accepted\n",
        penalties ? " with frequency penalty" : "", stat, df, limit,
        100.0 * accepted / steps);
    Check(stat < limit, "speculative loop samples the target distribution");
  }
}

}  // namespace

// logit_bias shifts the logits before the pick, greedy or not; without it
// greedy stays std::max_element.
void LogitBias() {
  const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f};
  SamplingParams p;
  p.logit_bias = {{2, 2.0f}};
  Sampler greedy(p);
  Check(greedy.Sample(logits.data(), 4) == 2, "logit_bias: greedy follows it");
  p.logit_bias = {{1, -100.0f}};
  Sampler banned(p);
  Check(banned.Sample(logits.data(), 4) == 2, "logit_bias: -100 bans");
  p.logit_bias = {{0, 100.0f}, {99, 5.0f}};  // 99 is outside: ignored
  p.temperature = 1.0f;
  p.seed = 3;
  Sampler forced(p);
  bool all = true;
  for (int i = 0; i < 200; ++i) {
    all = all && forced.Sample(logits.data(), 4) == 0;
  }
  Check(all, "logit_bias: +100 forces when sampling");
  // Sampling: the distribution is softmax of the biased logits.
  SamplingParams q;
  q.temperature = 1.0f;
  q.logit_bias = {{3, 1.5f}};
  Sampler biased(q);
  const auto dist = biased.Distribution(logits.data(), 4);
  double z = 0;
  const std::vector<double> shifted = {1.0, 3.0, 2.0, 2.0};
  for (const double v : shifted) {
    z += std::exp(v);
  }
  bool match = dist.size() == 4;
  for (const auto& [id, prob] : dist) {
    match = match && std::fabs(prob - std::exp(shifted[id]) / z) < 1e-5;
  }
  Check(match, "logit_bias: biased softmax");
  SamplingParams bad;
  bad.logit_bias = {{0, NAN}};
  bool threw = false;
  try {
    Sampler s(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "logit_bias: non-finite refused");
}

// The observer sees every pick with the raw logits; ComputeLogprobs is
// log-softmax with the top tokens best first.
void ObserverAndLogprobs() {
  struct Seen : gufo_npu::SampleObserver {
    void Sampled(const float* logits, int vocab, int id) override {
      ids.push_back(id);
      first.push_back(logits[0]);
      sizes.push_back(vocab);
    }
    std::vector<int> ids;
    std::vector<float> first;
    std::vector<int> sizes;
  } seen;
  const std::vector<float> logits = {0.25f, 3.0f, 2.0f, 2.0f, -1.0f};
  SamplingParams p;
  p.observer = &seen;
  p.logit_bias = {{0, 10.0f}};  // the observer still sees the raw logits
  Sampler sampler(p);
  const int a = sampler.Sample(logits.data(), 5);
  p.logit_bias.clear();
  Sampler plain(p);
  const int b = plain.Sample(logits.data(), 5);
  Check(seen.ids == std::vector<int>{a, b} && a == 0 && b == 1 &&
            seen.first == std::vector<float>{0.25f, 0.25f} &&
            seen.sizes == std::vector<int>{5, 5},
        "observer: every pick, raw logits");

  double z = 0;
  for (const float v : logits) {
    z += std::exp(static_cast<double>(v));
  }
  const auto lp = gufo_npu::ComputeLogprobs(logits.data(), 5, 4, 3);
  Check(lp.id == 4 && std::fabs(lp.logprob - (-1.0 - std::log(z))) < 1e-5,
        "logprobs: log-softmax");
  Check(lp.top.size() == 3 && lp.top[0].first == 1 && lp.top[1].first == 2 &&
            lp.top[2].first == 3 &&
            std::fabs(lp.top[1].second - lp.top[2].second) < 1e-7 &&
            std::fabs(lp.top[0].second - (3.0 - std::log(z))) < 1e-5,
        "logprobs: top tokens best first, ties in id order");
  Check(gufo_npu::ComputeLogprobs(logits.data(), 5, 1, 0).top.empty() &&
            gufo_npu::ComputeLogprobs(logits.data(), 5, 1, 9).top.size() == 5,
        "logprobs: top 0 and top above the vocab");
  // A 248k vocabulary of spread logits: probabilities sum to 1.
  std::vector<float> big(248320);
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  for (auto& v : big) {
    v = normal(rng);
  }
  double total = 0;
  for (int id = 0; id < 248320; id += 97) {
    total += std::exp(static_cast<double>(
        gufo_npu::ComputeLogprobs(big.data(), 248320, id, 0).logprob));
  }
  double direct = 0;
  const float best = *std::max_element(big.begin(), big.end());
  double zb = 0;
  for (const float v : big) {
    zb += std::exp(static_cast<double>(v - best));
  }
  for (int id = 0; id < 248320; id += 97) {
    direct += std::exp(static_cast<double>(big[id] - best)) / zb;
  }
  Check(std::fabs(total - direct) < 1e-5 * std::max(direct, 1e-3),
        "logprobs: agree with a double-precision softmax on 248k");
}

int main() {
  LogitBias();
  ObserverAndLogprobs();
  Greedy();
  Seeds();
  Temperature();
  Filters();
  Penalties();
  SpeculativeRule();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("ok: sampler\n");
  return EXIT_SUCCESS;
}
