#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/diagnose.hpp"

// NPU access through XRT. Builds without XRT link a stub that reports the
// runtime as missing, so CPU-only builds keep the same CLI.
namespace gufo_npu::npu {

// False when this binary was built without XRT.
bool RuntimeAvailable();

// Opens NPU device 0 through XRT and reports what it found.
std::vector<diagnose::Check> RuntimeChecks();

struct SmokeResult {
  bool ok = false;
  std::string error;
  int iterations = 0;
  // Wall time of one dispatch + wait, in microseconds.
  double first_us = 0;
  double mean_us = 0;
  double min_us = 0;
};

// Runs the smoke_add_one kernel `iterations` times, checks its output and
// times each dispatch. The per-dispatch time is the fixed cost every NPU
// call pays, which decides how many calls a decoded token can afford.
SmokeResult RunSmoke(const std::filesystem::path& kernel_dir, int iterations);

// Per-dispatch cost of the smoke kernel when issued three ways:
//   sync     dispatch, wait, repeat (the host waits on every call)
//   async    `depth` independent dispatches in flight before waiting
//   runlist  `depth` dependent dispatches (each reads the previous output)
//            chained in one xrt::runlist and submitted once
struct DispatchTiming {
  std::string mode;
  int depth = 0;
  int dispatches = 0;
  double us_per_dispatch = 0;
};

struct DispatchBenchResult {
  bool ok = false;
  std::string error;
  std::vector<DispatchTiming> timings{};
};

DispatchBenchResult BenchDispatch(const std::filesystem::path& kernel_dir,
                                  int rounds, const std::vector<int>& depths);

}  // namespace gufo_npu::npu
