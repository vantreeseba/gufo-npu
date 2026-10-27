#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace gufo_npu {

// Runs fn(begin, end) over [0, n) split into contiguous chunks, one per
// hardware thread. Blocks until all chunks finish.
template<typename Fn>
void ParallelFor(std::size_t n, Fn&& fn) {
  const std::size_t threads = std::max<std::size_t>(
      1, std::min<std::size_t>(std::thread::hardware_concurrency(), n));
  if (threads == 1) {
    fn(std::size_t{0}, n);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(threads);
  const std::size_t chunk = (n + threads - 1) / threads;
  for (std::size_t t = 0; t < threads; ++t) {
    const std::size_t begin = t * chunk;
    const std::size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }
    pool.emplace_back([&fn, begin, end] { fn(begin, end); });
  }
  for (auto& thread : pool) {
    thread.join();
  }
}

}  // namespace gufo_npu
