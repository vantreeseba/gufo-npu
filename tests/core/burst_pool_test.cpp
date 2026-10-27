#include "core/burst_pool.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// BurstPool runs every job of every Run exactly once, inside and outside a
// Burst, with job counts from 0 to more than the workers.
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

}  // namespace

int main() {
  std::mt19937 rng(11);
  gufo_npu::BurstPool pool(7);
  int failures = Check(pool, rng, 200);
  {
    const gufo_npu::BurstPool::Burst burst(pool);
    failures += Check(pool, rng, 5000);
  }
  failures += Check(pool, rng, 200);
  gufo_npu::BurstPool lone(0);
  failures += Check(lone, rng, 20);
  if (failures == 0) {
    std::puts("burst_pool_test: ok");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
