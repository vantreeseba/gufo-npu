#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace gufo_npu {

// A pool for bursts of short parallel jobs separated by waits elsewhere
// (prefill: CPU work between NPU calls of a millisecond or so). Workers spin
// for as long as a Burst is open, so a Run never waits for a thread to wake,
// and a Run waits only for its jobs, not for every worker to check in. Outside
// a Burst, workers sleep. Run() is not reentrant.
class BurstPool {
public:
  explicit BurstPool(std::size_t workers) {
    for (std::size_t i = 0; i < workers; ++i) {
      threads_.emplace_back([this] { Work(); });
    }
  }

  ~BurstPool() {
    stop_.store(true, std::memory_order_release);
    Wake();
    for (auto& t : threads_) {
      t.join();
    }
  }

  BurstPool(const BurstPool&) = delete;
  BurstPool& operator=(const BurstPool&) = delete;

  // Keeps the workers spinning while it lives.
  class Burst {
  public:
    explicit Burst(BurstPool& pool) : pool_(pool) {
      pool_.awake_.fetch_add(1, std::memory_order_acq_rel);
      pool_.Wake();
    }
    ~Burst() { pool_.awake_.fetch_sub(1, std::memory_order_acq_rel); }
    Burst(const Burst&) = delete;
    Burst& operator=(const Burst&) = delete;

  private:
    BurstPool& pool_;
  };

  // Runs fn(i) for i in [0, n) on the caller and the workers; returns when
  // every call has finished.
  void Run(std::size_t n, const std::function<void(std::size_t)>& fn) {
    if (n == 0) {
      return;
    }
    if (threads_.empty() || n == 1) {
      for (std::size_t i = 0; i < n; ++i) {
        fn(i);
      }
      return;
    }
    fn_ = &fn;
    done_.store(0, std::memory_order_relaxed);
    const std::uint64_t gen = ++generation_ & kGenMask;
    // Publishes fn_ and done_ with the new ticket.
    ticket_.store(Pack(gen, n, 0), std::memory_order_release);
    if (awake_.load(std::memory_order_acquire) == 0) {
      Wake();
    }
    Drain();
    while (done_.load(std::memory_order_acquire) != n) {
      Pause();
    }
  }

  std::size_t size() const { return threads_.size() + 1; }

private:
  // A ticket packs the generation, the job count and the next job index, so
  // a worker can only claim a job of the generation it read: one that lags
  // behind fails its compare-exchange instead of taking a later job.
  static constexpr int kIndexBits = 20;
  static constexpr std::uint64_t kIndexMask = (1ULL << kIndexBits) - 1;
  static constexpr std::uint64_t kGenMask = (1ULL << 24) - 1;

  static std::uint64_t Pack(std::uint64_t gen, std::uint64_t n,
                            std::uint64_t index) {
    return (gen << (2 * kIndexBits)) | (n << kIndexBits) | index;
  }

  static void Pause() {
#if defined(__x86_64__)
    _mm_pause();
#endif
  }

  void Wake() {
    sleep_word_.fetch_add(1, std::memory_order_release);
    sleep_word_.notify_all();
  }

  // Claims and runs jobs of the current ticket until none are left.
  void Drain() {
    std::uint64_t t = ticket_.load(std::memory_order_acquire);
    for (;;) {
      const std::uint64_t index = t & kIndexMask;
      const std::uint64_t n = (t >> kIndexBits) & kIndexMask;
      if (index >= n) {
        return;
      }
      if (ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        // The claim succeeded, so this generation is unfinished and fn_ is
        // still its function.
        (*fn_)(index);
        done_.fetch_add(1, std::memory_order_acq_rel);
        t = ticket_.load(std::memory_order_acquire);
      }
    }
  }

  void Work() {
    std::uint64_t seen = ticket_.load(std::memory_order_acquire);
    for (;;) {
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      const std::uint64_t now = ticket_.load(std::memory_order_acquire);
      if (now != seen) {
        // A later ticket of an exhausted generation drains to nothing.
        seen = now;
        Drain();
        continue;
      }
      if (awake_.load(std::memory_order_acquire) > 0) {
        Pause();
        continue;
      }
      // Sleep until woken; recheck the ticket first so a Run that started
      // before the sleep word was read is not missed.
      const std::uint32_t word = sleep_word_.load(std::memory_order_acquire);
      if (ticket_.load(std::memory_order_acquire) != seen ||
          awake_.load(std::memory_order_acquire) > 0 ||
          stop_.load(std::memory_order_acquire)) {
        continue;
      }
      sleep_word_.wait(word, std::memory_order_acquire);
    }
  }

  std::vector<std::thread> threads_;
  std::atomic<std::uint64_t> ticket_{0};
  std::atomic<std::size_t> done_{0};
  std::atomic<int> awake_{0};
  std::atomic<std::uint32_t> sleep_word_{0};
  std::atomic<bool> stop_{false};
  std::uint64_t generation_ = 0;
  const std::function<void(std::size_t)>* fn_ = nullptr;
};

}  // namespace gufo_npu
