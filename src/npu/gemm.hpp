#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

// The prefill GEMM design (kernels/designs/bfp16_gemm.py): C = A W^T for a
// chunk of m prompt tokens, A (m x k) and C (m x n) bf16 row-major, W the
// weights as bfp16 (src/core/bfp16.hpp), resident on the device. One xclbin
// in its own hardware context (separate from the decode GEMV's), one
// instruction sequence per (m, k, n).
namespace gufo_npu::npu {

// Prompt tokens per GEMM call (rows of A and C), smallest first; every
// matrix has a sequence for each (CHUNKS in kernels/designs/bfp16_gemm.py).
inline constexpr std::size_t kGemmChunks[] = {256, 512, 1024};
inline constexpr std::size_t kGemmMaxChunk = 1024;

// The instruction sequence name, e.g. "bfp16_gemm_m256_k2048_n8192".
std::string Bfp16GemmSequence(std::size_t m, std::size_t k, std::size_t n);

class GemmRuntime {
public:
  // Loads bfp16_gemm.xclbin from `kernel_dir`. Throws when it is missing or
  // the NPU cannot be opened.
  explicit GemmRuntime(const std::filesystem::path& kernel_dir);
  ~GemmRuntime();
  GemmRuntime(const GemmRuntime&) = delete;
  GemmRuntime& operator=(const GemmRuntime&) = delete;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

// Host-visible device memory for GEMM operands (zeroed).
class GemmBuffer {
public:
  GemmBuffer(GemmRuntime& runtime, std::size_t bytes);
  ~GemmBuffer();
  GemmBuffer(const GemmBuffer&) = delete;
  GemmBuffer& operator=(const GemmBuffer&) = delete;

  void* data();
  template<typename T>
  T* as() {
    return static_cast<T*>(data());
  }
  std::size_t size() const { return bytes_; }
  // Syncs the first `bytes` (all when 0).
  void ToDevice(std::size_t bytes = 0);
  void FromDevice(std::size_t bytes = 0);

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
  std::size_t bytes_ = 0;
};

// A started GEMM call; Wait() blocks until it has finished (also on
// destruction). Calls on one runtime run in the order they were started.
class GemmCall {
public:
  GemmCall();
  ~GemmCall();
  GemmCall(GemmCall&&) noexcept;
  GemmCall& operator=(GemmCall&&) noexcept;
  void Wait();

  struct Impl;
  explicit GemmCall(std::unique_ptr<Impl> impl);

private:
  std::unique_ptr<Impl> impl_;
};

class Bfp16Gemm {
public:
  // Packs W (`rows` out-features x `dim` in-features, row-major fp32) to
  // bfp16 and uploads it. Throws when a chunk size has no sequence for
  // (dim, rows).
  Bfp16Gemm(GemmRuntime& runtime, const float* w, std::size_t rows,
            std::size_t dim);
  ~Bfp16Gemm();
  Bfp16Gemm(const Bfp16Gemm&) = delete;
  Bfp16Gemm& operator=(const Bfp16Gemm&) = delete;

  // C (m x rows bf16) = A (m x dim bf16) W^T for m in kGemmChunks, on
  // device buffers of at least a_bytes(m) and c_bytes(m). Syncing them is
  // the caller's.
  void Run(GemmBuffer& a, GemmBuffer& c, std::size_t m);
  // The same call, started; the buffers must stay untouched until it is
  // waited for.
  GemmCall Start(GemmBuffer& a, GemmBuffer& c, std::size_t m);

  struct Timing {
    double mean_us = 0;
    double best_us = 0;
  };
  // Start + wait over `iterations` warm calls on the buffers of the last Run.
  Timing Time(int iterations);

  std::size_t dim() const { return dim_; }
  std::size_t rows() const { return rows_; }
  std::size_t a_bytes(std::size_t m) const {
    return m * dim_ * sizeof(std::uint16_t);
  }
  std::size_t c_bytes(std::size_t m) const {
    return m * rows_ * sizeof(std::uint16_t);
  }
  std::size_t weight_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::size_t dim_ = 0;
  std::size_t rows_ = 0;
};

}  // namespace gufo_npu::npu
