#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>

#include "npu/device.hpp"
#include "npu/xrt_kernel.hpp"

namespace gufo_npu::npu {
namespace {

// Must match PROBLEM_SIZE in kernels/designs/smoke_add_one.py.
constexpr std::size_t kSmokeElements = 4096;
constexpr std::size_t kSmokeBytes = kSmokeElements * sizeof(std::int32_t);
// Must match SIZE in kernels/designs/bw_memcpy.py.
constexpr std::size_t kBandwidthElements = 16ULL * 1024 * 1024;
constexpr std::size_t kBandwidthBytes =
    kBandwidthElements * sizeof(std::int32_t);

// Must match SIZE in kernels/designs/bw_read.py.
constexpr std::size_t kReadElements = 64ULL * 1024 * 1024;
constexpr std::size_t kReadBytes = kReadElements * sizeof(std::int32_t);

xrt::run SmokeRun(LoadedKernel& kernel, const xrt::bo& input,
                  const xrt::bo& output) {
  return kernel.Run({&input, &output});
}

void Fill(xrt::bo& bo) {
  auto* data = bo.map<std::int32_t*>();
  for (std::size_t i = 0; i < kSmokeElements; ++i) {
    data[i] = static_cast<std::int32_t>(i * 7 - 1000);
  }
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void Clear(xrt::bo& bo) {
  std::memset(bo.map<void*>(), 0, kSmokeBytes);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// Empty when `output` holds `input` + `added` everywhere.
std::string Verify(xrt::bo& input, xrt::bo& output, int added) {
  output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const auto* in = input.map<const std::int32_t*>();
  const auto* out = output.map<const std::int32_t*>();
  for (std::size_t i = 0; i < kSmokeElements; ++i) {
    if (out[i] != in[i] + added) {
      return "wrong output at element " + std::to_string(i) + ": " +
             std::to_string(out[i]) + " != " + std::to_string(in[i] + added);
    }
  }
  return {};
}

}  // namespace

bool RuntimeAvailable() {
  return true;
}

std::vector<diagnose::Check> RuntimeChecks() {
  try {
    const xrt::device device(0U);
    return {{"XRT device", diagnose::Status::kOk,
             device.get_info<xrt::info::device::name>()}};
  } catch (const std::exception& error) {
    return {{"XRT device", diagnose::Status::kFail,
             std::string("cannot open NPU device 0: ") + error.what()}};
  }
}

SmokeResult RunSmoke(const std::filesystem::path& kernel_dir, int iterations) {
  SmokeResult result;
  result.iterations = std::max(iterations, 1);
  try {
    LoadedKernel smoke(kernel_dir, "smoke_add_one");
    auto input = smoke.Buffer(kSmokeBytes);
    auto output = smoke.Buffer(kSmokeBytes);
    Fill(input);
    auto run = SmokeRun(smoke, input, output);

    double total_us = 0;
    result.min_us = std::numeric_limits<double>::max();
    for (int i = 0; i < result.iterations; ++i) {
      Clear(output);
      const auto start = Clock::now();
      run.start();
      run.wait();
      const double us = MicrosSince(start);
      if (i == 0) {
        result.first_us = us;
      }
      total_us += us;
      result.min_us = std::min(result.min_us, us);
    }
    result.mean_us = total_us / result.iterations;
    result.error = Verify(input, output, 1);
    result.ok = result.error.empty();
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

DispatchBenchResult BenchDispatch(const std::filesystem::path& kernel_dir,
                                  int rounds, const std::vector<int>& depths) {
  DispatchBenchResult result;
  rounds = std::max(rounds, 1);
  try {
    LoadedKernel smoke(kernel_dir, "smoke_add_one");
    auto input = smoke.Buffer(kSmokeBytes);
    Fill(input);

    // Warm up: wakes the NPU and loads the context.
    {
      auto output = smoke.Buffer(kSmokeBytes);
      auto run = SmokeRun(smoke, input, output);
      for (int i = 0; i < 20; ++i) {
        run.start();
        run.wait();
      }
    }

    {
      auto output = smoke.Buffer(kSmokeBytes);
      auto run = SmokeRun(smoke, input, output);
      const auto start = Clock::now();
      for (int i = 0; i < rounds; ++i) {
        run.start();
        run.wait();
      }
      result.timings.push_back(
          {"sync", 1, rounds, MicrosSince(start) / rounds});
    }

    for (const int depth : depths) {
      if (depth < 1) {
        continue;
      }
      // async: independent runs, all writing their own output.
      {
        std::vector<xrt::bo> outputs;
        std::vector<xrt::run> runs;
        for (int i = 0; i < depth; ++i) {
          outputs.push_back(smoke.Buffer(kSmokeBytes));
        }
        for (int i = 0; i < depth; ++i) {
          runs.push_back(SmokeRun(smoke, input, outputs[i]));
        }
        const auto start = Clock::now();
        for (int r = 0; r < rounds; ++r) {
          for (auto& run : runs) {
            run.start();
          }
          for (auto& run : runs) {
            run.wait();
          }
        }
        const int dispatches = rounds * depth;
        result.timings.push_back(
            {"async", depth, dispatches, MicrosSince(start) / dispatches});
        if (auto error = Verify(input, outputs.back(), 1); !error.empty()) {
          result.error = "async: " + error;
          return result;
        }
      }
      // runlist: a dependent chain input -> b[0] -> b[1] -> ... -> b[depth-1].
      {
        std::vector<xrt::bo> chain;
        for (int i = 0; i < depth; ++i) {
          chain.push_back(smoke.Buffer(kSmokeBytes));
          Clear(chain.back());
        }
        xrt::runlist runlist = smoke.Runlist();
        std::vector<xrt::run> runs;
        for (int i = 0; i < depth; ++i) {
          runs.push_back(
              SmokeRun(smoke, i == 0 ? input : chain[i - 1], chain[i]));
        }
        for (auto& run : runs) {
          runlist.add(run);
        }
        const auto start = Clock::now();
        for (int r = 0; r < rounds; ++r) {
          runlist.execute();
          runlist.wait();
        }
        const int dispatches = rounds * depth;
        result.timings.push_back(
            {"runlist", depth, dispatches, MicrosSince(start) / dispatches});
        if (auto error = Verify(input, chain.back(), depth); !error.empty()) {
          result.error = "runlist: " + error;
          return result;
        }
      }
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

BandwidthResult BenchBandwidth(const std::filesystem::path& kernel_dir,
                               int iterations) {
  BandwidthResult result;
  result.bytes_per_run = 2 * kBandwidthBytes;
  iterations = std::max(iterations, 1);
  try {
    LoadedKernel copy(kernel_dir, "bw_memcpy");
    auto input = copy.Buffer(kBandwidthBytes);
    auto output = copy.Buffer(kBandwidthBytes);
    auto* in = input.map<std::int32_t*>();
    for (std::size_t i = 0; i < kBandwidthElements; ++i) {
      in[i] = static_cast<std::int32_t>(i * 2654435761U);
    }
    input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::memset(output.map<void*>(), 0, kBandwidthBytes);
    output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = SmokeRun(copy, input, output);
    for (int i = 0; i < 3; ++i) {
      run.start();
      run.wait();
    }
    double total_us = 0;
    double best_us = std::numeric_limits<double>::max();
    for (int i = 0; i < iterations; ++i) {
      const auto start = Clock::now();
      run.start();
      run.wait();
      const double us = MicrosSince(start);
      total_us += us;
      best_us = std::min(best_us, us);
    }
    const double mean_us = total_us / iterations;
    result.mean_us = mean_us;
    result.mean_gb_s =
        static_cast<double>(result.bytes_per_run) / mean_us / 1e3;
    result.best_gb_s =
        static_cast<double>(result.bytes_per_run) / best_us / 1e3;

    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto* out = output.map<const std::int32_t*>();
    if (std::memcmp(in, out, kBandwidthBytes) != 0) {
      result.error = "copied buffer differs from the input";
      return result;
    }

    LoadedKernel read(kernel_dir, "bw_read");
    auto source = read.Buffer(kReadBytes);
    auto sink = read.Buffer(4096);
    std::memset(source.map<void*>(), 1, kReadBytes);
    source.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto read_run = read.Run({&source, &sink});
    for (int i = 0; i < 3; ++i) {
      read_run.start();
      read_run.wait();
    }
    double read_total_us = 0;
    double read_best_us = std::numeric_limits<double>::max();
    for (int i = 0; i < iterations; ++i) {
      const auto start = Clock::now();
      read_run.start();
      read_run.wait();
      const double us = MicrosSince(start);
      read_total_us += us;
      read_best_us = std::min(read_best_us, us);
    }
    result.read_bytes_per_run = kReadBytes;
    result.read_mean_gb_s =
        static_cast<double>(kReadBytes) / (read_total_us / iterations) / 1e3;
    result.read_best_gb_s =
        static_cast<double>(kReadBytes) / read_best_us / 1e3;
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

}  // namespace gufo_npu::npu
