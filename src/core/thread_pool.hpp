#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace gufo_npu {

// A small pool of persistent workers for the decode loop's per-token CPU
// work, where a job lasts tens of microseconds: starting threads per job
// (ParallelFor) costs more than the job. Workers spin briefly between jobs
// (a decode step submits the next job within a millisecond or so), then
// sleep until the next one. Run() is not reentrant.
class SpinPool {
public:
  explicit SpinPool(std::size_t workers) {
    for (std::size_t i = 0; i < workers; ++i) {
      threads_.emplace_back([this, i] { Work(i + 1); });
    }
  }

  ~SpinPool() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
  }

  SpinPool(const SpinPool&) = delete;
  SpinPool& operator=(const SpinPool&) = delete;

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
    n_ = n;
    next_.store(0, std::memory_order_relaxed);
    pending_.store(threads_.size(), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    Drain();
    while (pending_.load(std::memory_order_acquire) != 0) {
      Pause();
    }
  }

  std::size_t size() const { return threads_.size() + 1; }

private:
  // About 2 ms of pause instructions before a worker sleeps.
  static constexpr std::size_t kSpins = 50000;

  static void Pause() {
#if defined(__x86_64__)
    _mm_pause();
#endif
  }

  void Drain() {
    for (std::size_t i = next_.fetch_add(1, std::memory_order_relaxed); i < n_;
         i = next_.fetch_add(1, std::memory_order_relaxed)) {
      (*fn_)(i);
    }
  }

  void Work(std::size_t) {
    std::size_t seen = generation_.load(std::memory_order_acquire);
    for (;;) {
      std::size_t spins = 0;
      std::size_t now;
      while ((now = generation_.load(std::memory_order_acquire)) == seen) {
        if (++spins < kSpins) {
          Pause();
        } else {
          generation_.wait(seen, std::memory_order_acquire);
        }
      }
      seen = now;
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      Drain();
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  std::vector<std::thread> threads_;
  std::atomic<std::size_t> generation_{0};
  std::atomic<std::size_t> next_{0};
  std::atomic<std::size_t> pending_{0};
  std::atomic<bool> stop_{false};
  const std::function<void(std::size_t)>* fn_ = nullptr;
  std::size_t n_ = 0;
};

}  // namespace gufo_npu
