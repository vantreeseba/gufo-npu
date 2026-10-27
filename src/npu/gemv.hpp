#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/q4nx.hpp"

// The q4nx GEMV design (kernels/designs/q4nx_gemv.py): one xclbin in one
// hardware context, one instruction sequence per variant (shape, input stage,
// output stage). A Q4nxGemv keeps one matrix resident on the device; its
// calls read and write device Buffers, so the output of one call can be the
// input region of the next and a Chain can run them back to back.
namespace gufo_npu::npu {

// The longest plain input, in fp32 values. Every call reads 4 * kGemvMaxDim
// bytes of its input region (the tail past the input is ignored).
inline constexpr std::size_t kGemvMaxDim = 6144;
inline constexpr std::size_t kGemvInputBytes = 4 * kGemvMaxDim;
// Residual width of the resnorm input stage.
inline constexpr std::size_t kGemvHidden = 2048;

enum class Prologue {
  kPlain,    // dim fp32
  kAct,      // a SwiGLU output: the first 16 of every 32 fp32, dim of them
  kResNorm,  // [H | D | w]: dim fp32, dim fp32, dim bf16 -> rms(H + D) * w;
             // H + D also goes to the residual buffer
};
enum class Epilogue {
  kStore,   // padded_rows fp32
  kSwiGlu,  // per 32 rows (16 gate, then their 16 up): silu(g) * u, 16 zeros
};

// The instruction sequence name, e.g. "q4nx_gemv_k2048_n8192_resnorm_store".
std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows,
                             Prologue prologue, Epilogue epilogue);

class GemvRuntime {
public:
  // Loads q4nx_gemv.xclbin from `kernel_dir`. Throws when it is missing or
  // the NPU cannot be opened.
  explicit GemvRuntime(const std::filesystem::path& kernel_dir);
  ~GemvRuntime();
  GemvRuntime(const GemvRuntime&) = delete;
  GemvRuntime& operator=(const GemvRuntime&) = delete;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

// Host-visible device memory, or a window of it.
class Buffer {
public:
  Buffer(GemvRuntime& runtime, std::size_t bytes);
  ~Buffer();
  Buffer(const Buffer&);
  Buffer& operator=(const Buffer&);

  // `bytes` at `offset` of this buffer; offset must be a multiple of 64.
  Buffer Sub(std::size_t offset, std::size_t bytes) const;
  void* data();
  template<typename T>
  T* as() {
    return static_cast<T*>(data());
  }
  std::size_t size() const;
  void ToDevice();
  void FromDevice();

  struct Impl;
  Impl& impl() const { return *impl_; }

private:
  explicit Buffer(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

class Q4nxGemv {
public:
  // Uploads `weights`. Throws when no sequence was compiled for the variant.
  Q4nxGemv(GemvRuntime& runtime, const q4nx::Matrix& weights,
           Prologue prologue = Prologue::kPlain,
           Epilogue epilogue = Epilogue::kStore);
  ~Q4nxGemv();
  Q4nxGemv(const Q4nxGemv&) = delete;
  Q4nxGemv& operator=(const Q4nxGemv&) = delete;

  // One call on device buffers: x (kGemvInputBytes; 8 * dim for kAct), y
  // (padded_rows fp32), residual (kGemvHidden fp32, written by kResNorm).
  void Run(Buffer& x, Buffer& y, Buffer& residual);

  // Convenience for a plain/store GEMV on host memory: x has dim fp32
  // values (rounded to bf16 on the device), y gets `rows` values.
  void Run(const float* x, float* y);
  // x: dim bf16 bits (widened to fp32 on the host). y: padded_rows fp32.
  void Run(const std::uint16_t* x, float* y);

  // Mean and best wall time of start + wait, in microseconds, over
  // `iterations` warm dispatches on the buffers of the last Run.
  struct Timing {
    double mean_us = 0;
    double best_us = 0;
  };
  Timing Time(int iterations);

  std::size_t dim() const { return dim_; }
  std::size_t rows() const { return rows_; }
  std::size_t padded_rows() const { return padded_rows_; }
  std::size_t weight_bytes() const;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
  std::size_t dim_ = 0;
  std::size_t rows_ = 0;
  std::size_t padded_rows_ = 0;
};

// Calls submitted together and run back to back (an XRT runlist).
class Chain {
public:
  explicit Chain(GemvRuntime& runtime);
  ~Chain();
  Chain(const Chain&) = delete;
  Chain& operator=(const Chain&) = delete;

  // The buffers must outlive the chain.
  void Add(Q4nxGemv& gemv, Buffer& x, Buffer& y, Buffer& residual);
  // Runs every call in order and waits for the last.
  void Run();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::npu
