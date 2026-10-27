#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace gufo_npu {

namespace detail {

inline std::size_t CountUsableCpus() {
  std::size_t cpus =
      std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) {
    cpus = static_cast<std::size_t>(CPU_COUNT(&set));
  }
  // "max 100000" without a quota, else "<quota> <period>" in microseconds.
  std::ifstream max("/sys/fs/cgroup/cpu.max");
  long long quota = 0;
  long long period = 0;
  if (max >> quota >> period && quota > 0 && period > 0) {
    cpus = std::min<std::size_t>(
        cpus, std::max<long long>(1, (quota + period - 1) / period));
  }
#endif
  return cpus;
}

}  // namespace detail

// CPUs this process may use, at least 1: its affinity mask (taskset,
// cpusets, docker --cpuset-cpus), capped by a cgroup v2 CPU quota (docker
// --cpus) when one is set. std::thread::hardware_concurrency() sees neither.
// Read once, at the first call.
inline std::size_t UsableCpus() {
  static const std::size_t cpus = detail::CountUsableCpus();
  return cpus;
}

// Workers for one of the CPU decode's two spinning pools (the mixers' and
// the matrix products'): each gets at most half the usable CPUs, less the
// calling thread, so both pools and the caller fit together. Every
// SpinPool::Run waits for all its workers, and a worker that the scheduler
// parks costs a whole time slice per call.
inline std::size_t HalfCpuWorkers() {
  return std::max<std::size_t>(UsableCpus() / 2, 1) - 1;
}

}  // namespace gufo_npu
