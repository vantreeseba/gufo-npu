#include "core/sampling.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gufo_npu {
namespace {

// e^x for x in [-88, 0], relative error ~2e-7: 2^(x log2 e) as an integer
// power of two times a degree-6 polynomial for 2^f, f in [0, 1). Written
// so loops over it vectorize.
inline float FastExp(float x) {
  const float y = std::max(x * 1.44269504088896341f, -126.0f);
  const float fl = std::floor(y);
  const float f = y - fl;
  float p = 1.535336188319500e-4f;
  p = p * f + 1.339887440266574e-3f;
  p = p * f + 9.618437357674640e-3f;
  p = p * f + 5.550332471162809e-2f;
  p = p * f + 2.402264791363012e-1f;
  p = p * f + 6.931472028550421e-1f;
  p = p * f + 1.0f;
  const auto e = static_cast<std::int32_t>(fl) + 127;
  return p * std::bit_cast<float>(static_cast<std::uint32_t>(e) << 23);
}

}  // namespace

Sampler::Sampler(const SamplingParams& params) : params_(params) {
  if (!(params_.top_p > 0.0f && params_.top_p <= 1.0f)) {
    throw std::invalid_argument("sampling: top_p must be in (0, 1]");
  }
  if (params_.top_k < 0) {
    throw std::invalid_argument("sampling: top_k must be >= 0");
  }
  if (!(params_.min_p >= 0.0f && params_.min_p <= 1.0f)) {
    throw std::invalid_argument("sampling: min_p must be in [0, 1]");
  }
  if (!(params_.repetition_penalty > 0.0f)) {
    throw std::invalid_argument("sampling: repetition_penalty must be > 0");
  }
  if (!std::isfinite(params_.temperature) ||
      !std::isfinite(params_.presence_penalty) ||
      !std::isfinite(params_.frequency_penalty)) {
    throw std::invalid_argument("sampling: settings must be finite");
  }
  for (const auto& [id, bias] : params_.logit_bias) {
    if (!std::isfinite(bias)) {
      throw std::invalid_argument("sampling: logit_bias must be finite");
    }
  }
  if (params_.seed) {
    rng_.seed(*params_.seed);
  } else {
    std::random_device device;
    rng_.seed((static_cast<std::uint64_t>(device()) << 32) ^ device());
  }
}

double Sampler::Uniform() {
  return static_cast<double>(rng_() >> 11) * 0x1.0p-53;
}

void Sampler::Accept(int id) {
  if (HasPenalties() && id >= 0) {
    ++counts_[id];
  }
}

bool Sampler::HasPenalties() const {
  return params_.repetition_penalty != 1.0f ||
         params_.presence_penalty != 0.0f || params_.frequency_penalty != 0.0f;
}

float Sampler::Penalized(int id, float logit) const {
  const auto it = counts_.find(id);
  if (it == counts_.end()) {
    return logit;
  }
  const float r = params_.repetition_penalty;
  float v = logit > 0.0f ? logit / r : logit * r;
  v -= params_.presence_penalty +
       static_cast<float>(it->second) * params_.frequency_penalty;
  return v;
}

bool Sampler::Adjusts() const {
  return !params_.logit_bias.empty() || (HasPenalties() && !counts_.empty());
}

const float* Sampler::Source(const float* logits, int vocab) {
  if (!Adjusts()) {
    return logits;
  }
  penalized_.assign(logits, logits + vocab);
  if (HasPenalties()) {
    for (const auto& [id, count] : counts_) {
      if (id < vocab) {
        penalized_[static_cast<std::size_t>(id)] = Penalized(id, logits[id]);
      }
    }
  }
  for (const auto& [id, bias] : params_.logit_bias) {
    if (id >= 0 && id < vocab) {
      penalized_[static_cast<std::size_t>(id)] += bias;
    }
  }
  return penalized_.data();
}

void Sampler::Candidates(const float* logits, int vocab) {
  if (vocab <= 0) {
    throw std::invalid_argument("sampling: empty vocabulary");
  }
  list_.clear();
  dense_ = false;
  total_ = 0;
  const float* src = Source(logits, vocab);
  const int best = static_cast<int>(std::max_element(src, src + vocab) - src);
  const float best_v = src[best];
  if (params_.greedy()) {
    list_.emplace_back(best, 1.0);
    total_ = 1.0;
    return;
  }
  const float t = params_.temperature;
  const float inv_t = 1.0f / t;
  // Candidates: within kTailNats of the best, and within min_p of it
  // (p >= min_p * p_best <=> v >= best + t * ln min_p).
  float floor_v = best_v - kTailNats * t;
  if (params_.min_p > 0.0f) {
    floor_v = std::max(floor_v, best_v + t * std::log(params_.min_p));
  }
  // Every token's weight (0 below the floor), then their sum in 16 lanes
  // (a fixed order, so the same logits always give the same total).
  weights_.resize(static_cast<std::size_t>(vocab));
  float* w = weights_.data();
  for (int i = 0; i < vocab; ++i) {
    const float v = src[i];
    const float x = FastExp((v - best_v) * inv_t);
    w[i] = v >= floor_v ? x : 0.0f;
  }
  constexpr int kLanes = 16;
  float lanes[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= vocab; i += kLanes) {
    for (int j = 0; j < kLanes; ++j) {
      lanes[j] += w[i + j];
    }
  }
  for (; i < vocab; ++i) {
    lanes[0] += w[i];
  }
  for (const float lane : lanes) {
    total_ += lane;
  }
  // Best first, equal logits in id order.
  auto better = [src](int a, int b) {
    return src[a] > src[b] || (src[a] == src[b] && a < b);
  };
  if (params_.top_k > 0) {
    ids_.clear();
    for (int i = 0; i < vocab; ++i) {
      if (w[i] > 0.0f) {
        ids_.push_back(i);
      }
    }
    if (static_cast<std::size_t>(params_.top_k) < ids_.size()) {
      std::nth_element(ids_.begin(), ids_.begin() + params_.top_k, ids_.end(),
                       better);
      ids_.resize(static_cast<std::size_t>(params_.top_k));
    }
    std::sort(ids_.begin(), ids_.end(), better);
    total_ = 0;
    for (const int id : ids_) {
      list_.emplace_back(id, w[id]);
      total_ += w[id];
    }
  } else {
    dense_ = true;
  }
  if (params_.top_p >= 1.0f) {
    return;
  }
  // top_p: the smallest best-first set holding top_p of the mass.
  const double target = params_.top_p * total_;
  if (!dense_) {
    double mass = 0;
    std::size_t keep = 0;
    while (keep < list_.size()) {
      mass += list_[keep].second;
      ++keep;
      if (mass >= target) {
        break;
      }
    }
    list_.resize(keep);
  } else {
    // A histogram of the logits between the best and the floor finds the
    // bucket the cut falls in; only that bucket is sorted.
    constexpr int kBuckets = 1024;
    const float span = std::max(best_v - floor_v, 1e-30f);
    const float scale = kBuckets / span;
    auto bucket = [&](float v) {
      return std::min(kBuckets - 1, static_cast<int>((best_v - v) * scale));
    };
    mass_.assign(kBuckets, 0.0);
    for (int i = 0; i < vocab; ++i) {
      if (w[i] > 0.0f) {
        mass_[static_cast<std::size_t>(bucket(src[i]))] += w[i];
      }
    }
    int cut = 0;
    double above = 0;  // mass of the buckets before `cut`
    while (cut < kBuckets - 1 &&
           above + mass_[static_cast<std::size_t>(cut)] < target) {
      above += mass_[static_cast<std::size_t>(cut)];
      ++cut;
    }
    ids_.clear();
    for (int i = 0; i < vocab; ++i) {
      if (w[i] > 0.0f) {
        const int b = bucket(src[i]);
        if (b < cut) {
          list_.emplace_back(i, w[i]);
        } else if (b == cut) {
          ids_.push_back(i);
        }
      }
    }
    std::sort(ids_.begin(), ids_.end(), better);
    double mass = above;
    for (const int id : ids_) {
      list_.emplace_back(id, w[id]);
      mass += w[id];
      if (mass >= target) {
        break;
      }
    }
    dense_ = false;
  }
  total_ = 0;
  for (const auto& c : list_) {
    total_ += c.second;
  }
}

std::vector<std::pair<int, double>> Sampler::Distribution(const float* logits,
                                                          int vocab) {
  Candidates(logits, vocab);
  std::vector<std::pair<int, double>> out;
  if (dense_) {
    for (int i = 0; i < vocab; ++i) {
      if (weights_[static_cast<std::size_t>(i)] > 0.0f) {
        out.emplace_back(i, weights_[static_cast<std::size_t>(i)]);
      }
    }
  } else {
    out = list_;
  }
  for (auto& c : out) {
    c.second /= total_;
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.second > b.second || (a.second == b.second && a.first < b.first);
  });
  return out;
}

int Sampler::Sample(const float* logits, int vocab) {
  const int id = Pick(logits, vocab);
  if (params_.observer != nullptr) {
    params_.observer->Sampled(logits, vocab, id);
  }
  return id;
}

int Sampler::Pick(const float* logits, int vocab) {
  if (params_.greedy() && !Adjusts()) {
    return static_cast<int>(std::max_element(logits, logits + vocab) - logits);
  }
  Candidates(logits, vocab);
  const double u = Uniform() * total_;
  double cumulative = 0;
  if (dense_) {
    int last = -1;
    for (int i = 0; i < vocab; ++i) {
      const float wi = weights_[static_cast<std::size_t>(i)];
      if (wi > 0.0f) {
        cumulative += wi;
        last = i;
        if (u < cumulative) {
          return i;
        }
      }
    }
    return last;
  }
  if (list_.size() == 1) {
    return list_.front().first;
  }
  for (const auto& [id, wi] : list_) {
    cumulative += wi;
    if (u < cumulative) {
      return id;
    }
  }
  return list_.back().first;
}

TokenLogprobs ComputeLogprobs(const float* logits, int vocab, int id, int top) {
  if (vocab <= 0 || id < 0 || id >= vocab) {
    throw std::invalid_argument("logprobs: token id out of range");
  }
  const float best = *std::max_element(logits, logits + vocab);
  // log sum exp, summed in 16 lanes of float then in double.
  constexpr int kLanes = 16;
  float lanes[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= vocab; i += kLanes) {
    for (int j = 0; j < kLanes; ++j) {
      lanes[j] += FastExp(std::max(logits[i + j] - best, -87.0f));
    }
  }
  for (; i < vocab; ++i) {
    lanes[0] += FastExp(std::max(logits[i] - best, -87.0f));
  }
  double sum = 0;
  for (const float lane : lanes) {
    sum += lane;
  }
  const double log_z = best + std::log(sum);
  TokenLogprobs out;
  out.id = id;
  out.logprob = static_cast<float>(logits[id] - log_z);
  top = std::min(top, vocab);
  if (top > 0) {
    // The `top` best ids: a small sorted list, best first.
    auto better = [logits](int a, int b) {
      return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    };
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(top) + 1);
    for (int t = 0; t < vocab; ++t) {
      if (static_cast<int>(ids.size()) == top && !better(t, ids.back())) {
        continue;
      }
      ids.insert(std::upper_bound(ids.begin(), ids.end(), t, better), t);
      if (static_cast<int>(ids.size()) > top) {
        ids.pop_back();
      }
    }
    for (const int t : ids) {
      out.top.emplace_back(t, static_cast<float>(logits[t] - log_z));
    }
  }
  return out;
}

}  // namespace gufo_npu
