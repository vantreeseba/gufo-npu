#include "core/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

#include "core/burst_pool.hpp"

// A Run issued right after a pool is built must not wait on a worker that
// has not started yet. SpinPool workers used to take the generation they saw
// at thread start as already handled, so a worker that started after the
// first Run's increment skipped that Run, and Run spun forever on its
// check-in. Thread start-up takes far longer than the gap between the
// constructor and the first Run, so building and running pools many times
// hits the race; a watchdog turns a hang into a failure.
namespace {

using gufo_npu::BurstPool;
using gufo_npu::SpinPool;

constexpr int kPools = 300;
constexpr std::size_t kWorkers = 7;
constexpr std::size_t kJobs = 64;
constexpr auto kTimeout = std::chrono::seconds(60);

// Builds a pool, runs it at once, then a few more times; checks every job
// ran exactly once.
template<typename Pool>
bool RunAtOnce() {
  for (int p = 0; p < kPools; ++p) {
    Pool pool(kWorkers);
    for (int round = 0; round < 3; ++round) {
      std::vector<std::atomic<int>> hits(kJobs);
      pool.Run(kJobs, [&](std::size_t i) {
        hits[i].fetch_add(1, std::memory_order_relaxed);
      });
      for (std::size_t i = 0; i < kJobs; ++i) {
        if (hits[i].load() != 1) {
          std::printf("FAIL: pool %d round %d job %zu ran %d times\n", p, round,
                      i, hits[i].load());
          return false;
        }
      }
    }
  }
  return true;
}

// Runs fn on its own thread; a hang past the timeout fails the test (the
// hung thread cannot be joined, so the process exits).
template<typename Fn>
bool WithWatchdog(const char* name, Fn fn) {
  std::packaged_task<bool()> task(fn);
  std::future<bool> result = task.get_future();
  std::thread(std::move(task)).detach();
  if (result.wait_for(kTimeout) != std::future_status::ready) {
    std::printf("FAIL: %s hung (a Run never returned within %lld s)\n", name,
                static_cast<long long>(kTimeout.count()));
    std::fflush(stdout);
    std::_Exit(EXIT_FAILURE);
  }
  const bool ok = result.get();
  std::printf("%s %s: %d pools of %zu workers, run at once\n",
              ok ? "ok  " : "FAIL", name, kPools, kWorkers);
  return ok;
}

}  // namespace

int main() {
  bool ok = WithWatchdog("SpinPool", RunAtOnce<SpinPool>);
  ok = WithWatchdog("BurstPool", RunAtOnce<BurstPool>) && ok;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
