#include "models/qwen35/prompt_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace gufo_npu::qwen35 {
namespace {

template<typename T>
std::size_t VectorBytes(const std::vector<T>& v) {
  return v.size() * sizeof(T);
}

std::size_t LayerBytes(const DecodeState::Layer& l) {
  return VectorBytes(l.conv) + VectorBytes(l.ssm) + VectorBytes(l.keys) +
         VectorBytes(l.values);
}

// The first `rows` rows of an attention cache whose rows are `width` wide.
void KeepRows(const DecodeState::Layer& from, std::size_t rows,
              std::size_t width, DecodeState::Layer& to) {
  to.keys.assign(from.keys.begin(),
                 from.keys.begin() + static_cast<std::ptrdiff_t>(rows * width));
  to.values.assign(
      from.values.begin(),
      from.values.begin() + static_cast<std::ptrdiff_t>(rows * width));
}

}  // namespace

PromptCache::PromptCache(PromptCacheModel model, PromptCacheOptions options)
    : model_(std::move(model)), options_(options) {
  if (!model_.run || !model_.fill_mtp || !model_.new_state ||
      model_.hidden <= 0 || model_.vocab <= 0 || model_.keep_last < 0 ||
      options_.max_checkpoints < 1 || options_.min_tokens < 1) {
    throw std::invalid_argument("prompt cache: incomplete model or options");
  }
}

PromptPrefill PromptCache::Hook() {
  return [this](const std::vector<int>& prompt, DecodeState& state,
                DecodeState* mtp,
                float* hidden) { return Run(prompt, state, mtp, hidden); };
}

std::size_t PromptCache::Bytes(const Entry& e) {
  std::size_t bytes =
      VectorBytes(e.tokens) + VectorBytes(e.hidden) + LayerBytes(e.mtp);
  for (const auto& l : e.state.layers) {
    bytes += LayerBytes(l);
  }
  return bytes;
}

std::list<PromptCache::Entry>::iterator PromptCache::Find(
    const std::vector<int>& prompt, int end, bool need_mtp) {
  auto best = entries_.end();
  std::size_t best_size = 0;
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    const std::size_t size = it->tokens.size();
    // Later entries are more recent: on equal length the newest wins.
    if (size > static_cast<std::size_t>(end) || size < best_size ||
        (need_mtp && !it->has_mtp) ||
        !std::equal(it->tokens.begin(), it->tokens.end(), prompt.begin())) {
      continue;
    }
    best = it;
    best_size = size;
  }
  return best;
}

int PromptCache::Boundary(const std::vector<int>& prompt, int end) const {
  if (options_.boundary_token < 0) {
    return end;
  }
  // The last boundary token in [min_tokens, end): the state before it.
  for (int i = end - 1; i >= options_.min_tokens; --i) {
    if (prompt[static_cast<std::size_t>(i)] == options_.boundary_token) {
      return i;
    }
  }
  return 0;
}

void PromptCache::Insert(Entry entry) {
  entry.bytes = Bytes(entry);
  bytes_ += entry.bytes;
  entries_.push_back(std::move(entry));
  ++stats_.checkpoints;
  while (static_cast<int>(entries_.size()) > options_.max_checkpoints ||
         (options_.max_bytes > 0 && bytes_ > options_.max_bytes &&
          entries_.size() > 1)) {
    Drop(entries_.begin());
    ++stats_.evictions;
  }
  if (options_.max_bytes > 0 && bytes_ > options_.max_bytes) {
    // Even alone it is over the budget.
    Clear();
    ++stats_.evictions;
  }
}

int PromptCache::Run(const std::vector<int>& prompt, DecodeState& state,
                     DecodeState* mtp, float* hidden) {
  const int n = static_cast<int>(prompt.size());
  const auto d = static_cast<std::size_t>(model_.hidden);
  for (const int id : prompt) {
    if (id < 0 || id >= model_.vocab) {
      throw std::out_of_range("prompt cache: token id out of range");
    }
  }
  last_reused_ = 0;
  const int end = n - model_.keep_last;  // what the cache runs: [0, end)
  if (end <= 0) {
    ++stats_.misses;
    return 0;
  }

  int c = 0;  // restored prefix
  const auto hit = Find(prompt, end, mtp != nullptr);
  if (hit != entries_.end()) {
    ++stats_.hits;
    c = static_cast<int>(hit->tokens.size());
    // Fresh states from the caller: the checkpoint's buffers replace them.
    state = hit->state;
    if (mtp != nullptr) {
      auto& to = mtp->layers[static_cast<std::size_t>(model_.layers)];
      to.keys = hit->mtp.keys;
      to.values = hit->mtp.values;
    }
    std::copy(hit->hidden.begin(), hit->hidden.end(),
              hidden + static_cast<std::size_t>(c - 1) * d);
    entries_.splice(entries_.end(), entries_, hit);  // most recent
    last_reused_ = c;
  } else {
    ++stats_.misses;
    if (end < model_.min_prefill) {
      return 0;  // the decoder's, as without the cache
    }
  }

  // The new checkpoint, at b > c (else the restored one covers it).
  const int b = Boundary(prompt, end);
  const bool take = b > c && b >= options_.min_tokens;
  DecodeState captured;
  StateCapture capture;
  if (take) {
    captured = model_.new_state();
    capture = {b, &captured};
  }
  if (end > c) {
    model_.run(prompt.data() + c, end - c, state,
               hidden + static_cast<std::size_t>(c) * d,
               take ? &capture : nullptr);
  }
  // MTP pairs (h[p - 1], prompt[p]) at p = max(c, 1) .. min(end, n - 1).
  const int lo = std::max(c, 1);
  const int hi = std::min(end, n - 1);
  if (mtp != nullptr && hi >= lo) {
    model_.fill_mtp(hidden + static_cast<std::size_t>(lo - 1) * d,
                    prompt.data() + lo, hi - lo + 1, lo, *mtp);
  }

  if (take) {
    Entry e;
    e.tokens.assign(prompt.begin(), prompt.begin() + b);
    e.state = std::move(captured);
    e.state.position = b;
    for (std::size_t il = 0; il < state.layers.size(); ++il) {
      const auto& from = state.layers[il];
      auto& to = e.state.layers[il];
      if (!from.ssm.empty()) {
        // DeltaNet: copied by the capture; no speculative leftovers.
        continue;
      }
      if (!from.keys.empty()) {
        const std::size_t width = from.keys.size() / state.position;
        KeepRows(from, static_cast<std::size_t>(b), width, to);
      }
    }
    if (mtp != nullptr) {
      // An evicted checkpoint's buffers, emptied, take the rows.
      if (!spare_mtp_.empty()) {
        e.mtp = std::move(spare_mtp_.back());
        spare_mtp_.pop_back();
        e.mtp.keys.clear();
        e.mtp.values.clear();
      }
      // Rows for positions 1 .. hi; keep 1 .. b - 1.
      const auto& from = mtp->layers[static_cast<std::size_t>(model_.layers)];
      if (hi >= 1 && !from.keys.empty()) {
        const std::size_t width = from.keys.size() / hi;
        KeepRows(from, static_cast<std::size_t>(b - 1), width, e.mtp);
      }
      e.has_mtp = true;
    }
    e.hidden.assign(hidden + static_cast<std::size_t>(b - 1) * d,
                    hidden + static_cast<std::size_t>(b) * d);
    // A checkpoint of the same tokens is replaced.
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->tokens == e.tokens) {
        Drop(it);
        break;
      }
    }
    Insert(std::move(e));
  }
  return end;
}

std::vector<int> PromptCache::checkpoint_lengths() const {
  std::vector<int> out;
  for (const auto& e : entries_) {
    out.push_back(static_cast<int>(e.tokens.size()));
  }
  return out;
}

void PromptCache::Drop(std::list<Entry>::iterator it) {
  bytes_ -= it->bytes;
  if (model_.recycle) {
    model_.recycle(std::move(it->state));
  }
  if (it->has_mtp &&
      spare_mtp_.size() < static_cast<std::size_t>(options_.max_checkpoints)) {
    spare_mtp_.push_back(std::move(it->mtp));
  }
  entries_.erase(it);
}

void PromptCache::Clear() {
  while (!entries_.empty()) {
    Drop(entries_.begin());
  }
  bytes_ = 0;
}

}  // namespace gufo_npu::qwen35
