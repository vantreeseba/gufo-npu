#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>

#include "models/qwen36_35b_a3b/npu_prefill.hpp"
#include "npu/program.hpp"

// The NPU side of Qwen3.6-35B-A3B's prefill, shared by its core
// (npu_prefill.cpp) and its NPU expert step (npu_experts.cpp): the two
// programs (kernels/), the windows of device buffers their calls take, and
// the profile they charge.
namespace gufo_npu::qwen36_35b_a3b {

// The NPU programs (kernels/*_variants.py).
inline constexpr char kDequantProgram[] = "qwen36_35b_a3b_dequant";
inline constexpr char kGemmProgram[] = "qwen36_35b_a3b_prefill";

// GEMM rows per call (prefill_gemm_variants.py PREFILL_CHUNKS).
inline constexpr std::size_t kChunks[] = {256, 512, 1024};
inline constexpr std::size_t kMaxRows = 1024;

inline constexpr std::size_t kHidden = 2048;
inline constexpr std::size_t kExpertFfn = 512;

// bfp16ebs8 bytes of a rows x dim matrix (9 bytes per 8 values).
constexpr std::size_t BfpBytes(std::size_t rows, std::size_t dim) {
  return rows * dim / 8 * 9;
}

inline std::string DequantSequence(std::size_t k, std::size_t n) {
  return std::string(kDequantProgram) + "_k" + std::to_string(k) + "_n" +
         std::to_string(n);
}

inline std::string GemmSequence(std::size_t m, std::size_t k, std::size_t n) {
  return std::string(kGemmProgram) + "_m" + std::to_string(m) + "_k" +
         std::to_string(k) + "_n" + std::to_string(n);
}

// GEMM rows for a call over n <= kMaxRows tokens.
inline std::size_t PaddedRows(std::size_t n) {
  for (const std::size_t c : kChunks) {
    if (n <= c) {
      return c;
    }
  }
  return kMaxRows;
}

using Clock = std::chrono::steady_clock;

inline double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

class PrefillDevice {
public:
  PrefillDevice(const std::filesystem::path& kernel_dir,
                NpuPrefill::Profile& profile)
      : dequant(kernel_dir, kDequantProgram),
        gemm(kernel_dir, kGemmProgram),
        profile_(profile) {}

  npu::Program dequant;
  npu::Program gemm;

  NpuPrefill::Profile& profile() { return profile_; }

  // Runs `batch` of `program` and waits, counting a move between programs
  // (milliseconds each).
  void Run(npu::Batch& batch, const npu::Program& program) {
    if (batch.size() == 0) {
      return;
    }
    const auto start = Clock::now();
    if (current_ != &program) {
      ++profile_.switches;
      current_ = &program;
    }
    batch.Run();
    const double seconds = SecondsSince(start);
    if (&program == &dequant) {
      profile_.dequant_seconds += seconds;
      profile_.dequants += static_cast<long>(batch.size());
    } else {
      profile_.gemm_seconds += seconds;
      profile_.gemms += static_cast<long>(batch.size());
    }
  }

  // A window of `b` for one call. (Keeping the windows for reuse measured
  // slower than making them per call: EXPERIMENTS.md.)
  static npu::Buffer Window(const npu::Buffer& b, std::size_t offset,
                            std::size_t bytes) {
    return b.Sub(offset, bytes);
  }

  // Syncs the first `bytes` of `b` to or from the device.
  void Sync(npu::Buffer& b, std::size_t bytes, bool to_device) {
    if (bytes == 0) {
      return;
    }
    const auto start = Clock::now();
    npu::Buffer window = b.Sub(0, bytes);
    if (to_device) {
      window.ToDevice();
    } else {
      window.FromDevice();
    }
    profile_.sync_seconds += SecondsSince(start);
  }

private:
  NpuPrefill::Profile& profile_;
  const npu::Program* current_ = nullptr;  // the program the NPU ran last
};

}  // namespace gufo_npu::qwen36_35b_a3b
