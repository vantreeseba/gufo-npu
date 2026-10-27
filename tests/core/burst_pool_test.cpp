#include "core/burst_pool.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// BurstPool runs every job of every Run exactly once, inside and outside a
// Burst, with job counts from 0 to more than the workers. A job that throws
// (on any thread) does not stop the others: Run rethrows the first exception
// after every job has finished, and the next Run is unaffected.
namespace {

int Check(gufo_npu::BurstPool& pool, std::mt19937& rng, int runs) {
  std::uniform_int_distribution<int> count(0, 300);
  for (int r = 0; r < runs; ++r) {
    const int n = count(rng);
    std::vector<std::atomic<int>> hits(n);
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t i) {
      hits[i].fetch_add(1, std::memory_order_relaxed);
    });
    for (int i = 0; i < n; ++i) {
      if (hits[i].load() != 1) {
        std::fprintf(stderr, "FAIL: run %d job %d ran %d times\n", r, i,
                     hits[i].load());
        return 1;
      }
    }
  }
  return 0;
}

// Every third job throws: Run throws one of them, after all jobs ran.
int CheckThrows(gufo_npu::BurstPool& pool, int runs) {
  for (int r = 0; r < runs; ++r) {
    constexpr int kJobs = 97;
    std::vector<std::atomic<int>> hits(kJobs);
    bool thrown = false;
    try {
      pool.Run(kJobs, [&](std::size_t i) {
        hits[i].fetch_add(1, std::memory_order_relaxed);
        if (i % 3 == 0) {
          throw std::runtime_error("job " + std::to_string(i));
        }
      });
    } catch (const std::runtime_error& e) {
      thrown = std::string(e.what()).rfind("job ", 0) == 0;
    }
    if (!thrown) {
      std::fprintf(stderr, "FAIL: run %d: no job exception rethrown\n", r);
      return 1;
    }
    for (int i = 0; i < kJobs; ++i) {
      if (hits[i].load() != 1) {
        std::fprintf(stderr, "FAIL: throwing run %d job %d ran %d times\n", r,
                     i, hits[i].load());
        return 1;
      }
    }
  }
  return 0;
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  gufo_npu::BurstPool pool(7);
  int failures = Check(pool, rng, 200);
  failures += CheckThrows(pool, 50);
  {
    const gufo_npu::BurstPool::Burst burst(pool);
    failures += Check(pool, rng, 5000);
    failures += CheckThrows(pool, 500);
    failures += Check(pool, rng, 500);
  }
  failures += Check(pool, rng, 200);
  gufo_npu::BurstPool lone(0);
  failures += Check(lone, rng, 20);
  failures += CheckThrows(lone, 5);
  if (failures == 0) {
    std::puts("burst_pool_test: ok");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
